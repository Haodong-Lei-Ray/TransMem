#!/usr/bin/env python3
"""GRPO post-training for a frozen LLM with layered TransMem as the policy."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transmem.extract_features import build_chat_prompt_ids, load_records
from transmem.layered import LayeredRollout, TransMemLayered
from transmem.rl import (
    grpo_clipped_loss,
    group_relative_advantages,
    hotpot_answer_reward,
    sampled_reference_kl,
)
from transmem.train_inloop import (
    InLoopDataset,
    InLoopTrainer,
    collate_first,
    parse_args,
)


class RewardDataset(InLoopDataset):
    """Manifest-aligned raw records without loading unused Stage0 tensors."""

    def __init__(self, data_dir: str, data_path: str, data_format: str,
                 *, records: list | None = None, max_samples: int | None = None):
        super().__init__(
            data_dir, data_path, data_format, policy="tf",
            records=records, max_samples=max_samples)

    def __getitem__(self, index: int) -> dict:
        feature_file, sample_index = self.entries[index]
        record = self.records[sample_index]
        return {
            "context": record["context"],
            "question": record["question"],
            "ground_truth": record.get("ground_truth", ""),
            "sample_idx": sample_index,
            "feature_file": feature_file,
            "data_dir": str(self.data_dir),
        }


class GRPOTrainer(InLoopTrainer):
    """Critic-free group-relative policy optimization over answer strings."""

    def __init__(self, args, model=None, tokenizer=None):
        if getattr(args, "policy", None) != "grpo":
            raise ValueError("GRPOTrainer 要求 --policy grpo")
        if getattr(args, "group_size", 0) < 2:
            raise ValueError("--group_size 必须 >= 2")
        if getattr(args, "sample_temp", 0.0) <= 0.0:
            raise ValueError("GRPO 必须用正的 --sample_temp 探索")
        if getattr(args, "grpo_epochs", 0) < 2:
            raise ValueError("真正的 clipped GRPO 要求 --grpo_epochs >= 2")
        super().__init__(args, model=model, tokenizer=tokenizer)
        self.reference_mem: TransMemLayered | None = None
        self.reference_rollout: LayeredRollout | None = None
        self.reference_checkpoint: str | None = None
        self.reference_checkpoint_id = (
            getattr(args, "reference_id", None)
            or getattr(args, "reference_checkpoint", None))

    def checkpoint_metadata(self) -> dict:
        return {
            "reference_checkpoint_id": self.reference_checkpoint_id,
            "grpo_epochs": self.args.grpo_epochs,
            "group_size": self.args.group_size,
            "clip_eps": self.args.clip_eps,
            "reference_kl_beta": self.args.reference_kl_beta,
        }

    def validate_checkpoint_metadata(self, checkpoint: dict) -> None:
        expected = self.checkpoint_metadata()
        for key, value in expected.items():
            actual = checkpoint.get(key)
            if actual != value:
                raise ValueError(
                    f"GRPO resume provenance 不匹配: {key}="
                    f"{actual!r}, expected={value!r}")

    def set_reference(self, path: str) -> None:
        """Load the immutable pre-RL TransMem policy used by the KL guard."""
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != self.config.to_dict():
            raise ValueError("reference checkpoint config 与当前 TransMem 拓扑不一致")
        reference = TransMemLayered(self.config).to(
            self.device, dtype=next(self.mem.parameters()).dtype)
        reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        reference.eval().requires_grad_(False)
        self.reference_mem = reference
        self.reference_rollout = LayeredRollout(
            self.model, self.tok, self.device, reference)
        self.reference_checkpoint = str(checkpoint_path)
        if self.is_main:
            print(f"Fixed GRPO reference: {self.reference_checkpoint}")

    @staticmethod
    def _answer_log_probs(
        rollout: LayeredRollout,
        model,
        full_ids: torch.Tensor,
        answer_ids: list[int],
        *,
        len_cl: int,
        len_cq: int,
        temperature: float,
    ) -> torch.Tensor:
        hidden = rollout.teacher_forced_forward(
            full_ids, len_cl, len_cq, len(answer_ids))
        logits = model.lm_head(hidden).float() / temperature
        targets = torch.tensor(
            answer_ids, device=logits.device, dtype=torch.long)
        return torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)

    def collect_group(self, item: dict) -> dict:
        """Sample one fixed behavior group and cache immutable reference scores."""
        if self.reference_rollout is None:
            raise RuntimeError("GRPO reference 尚未初始化")
        context = item["context"]
        question = item["question"]
        ground_truth = str(item.get("ground_truth", ""))
        if not ground_truth:
            raise ValueError(
                f"GRPO 样本缺 ground_truth: sample_idx={item.get('sample_idx')}")
        cq_ids = build_chat_prompt_ids(
            self.tok, context, question, self.device)
        len_cq = cq_ids.shape[1]
        len_cl = self.tok(
            context, return_tensors="pt",
            add_special_tokens=False).input_ids.shape[1]

        candidates = []
        self.mem.eval()
        for _ in range(self.args.group_size):
            answer_ids, old_log_probs = self.rollout.generate_from_ids(
                cq_ids, len_cl, max_new=self.args.max_answer_tokens,
                sample=True, temperature=self.args.sample_temp,
                return_log_probs=True)
            prediction = self.tok.decode(
                answer_ids, skip_special_tokens=True).strip()
            reward = hotpot_answer_reward(
                prediction,
                ground_truth,
                len(answer_ids),
                em_weight=self.args.reward_em_weight,
                verbosity_weight=self.args.reward_verbosity_weight,
                verbosity_start=self.args.reward_verbosity_start,
                verbosity_cap=self.args.reward_verbosity_cap,
            )
            length = len(answer_ids)
            if length < 1:
                continue
            prefix = torch.tensor(
                [answer_ids[:-1]], device=self.device, dtype=cq_ids.dtype)
            full_ids = torch.cat([cq_ids, prefix], dim=1) if length > 1 else cq_ids
            with torch.no_grad():
                reference_log_probs = self._answer_log_probs(
                    self.reference_rollout, self.model, full_ids, answer_ids,
                    len_cl=len_cl, len_cq=len_cq,
                    temperature=self.args.sample_temp)
            candidates.append({
                "answer_ids": answer_ids,
                "old_log_probs": old_log_probs.detach().cpu(),
                "reference_log_probs": reference_log_probs.detach().cpu(),
                "reward": reward,
            })
        self.mem.train()

        if len(candidates) < 2:
            raise RuntimeError("GRPO rollout 未得到至少两个非空 response")

        reward_values = torch.tensor(
            [candidate["reward"].total for candidate in candidates],
            device=self.device, dtype=torch.float32)
        advantages, active = group_relative_advantages(reward_values)
        return {
            "cq_ids": cq_ids.detach().cpu(),
            "len_cl": len_cl,
            "len_cq": len_cq,
            "candidates": candidates,
            "advantages": advantages.detach().cpu(),
            "active": active,
            "diagnostic": {
                "question": question,
                "context": context,
                "sample_idx": item.get("sample_idx"),
                "feature_file": item.get("feature_file"),
                "data_dir": item.get("data_dir"),
            },
        }

    def backward_group(self, group: dict, *, loss_scale: float) -> dict[str, float]:
        """Re-score a fixed rollout group and backprop one clipped GRPO epoch."""
        cq_ids = group["cq_ids"].to(self.device)
        len_cl = int(group["len_cl"])
        len_cq = int(group["len_cq"])
        candidates = group["candidates"]
        advantages = group["advantages"].to(self.device)
        active = bool(group["active"])

        policy_total = kl_total = loss_total = 0.0
        ratio_total = clip_total = 0.0
        for index, candidate in enumerate(candidates):
            answer_ids = candidate["answer_ids"]
            old_log_probs = candidate["old_log_probs"].to(self.device)
            reference_log_probs = candidate["reference_log_probs"].to(self.device)
            length = len(answer_ids)
            prefix = torch.tensor(
                [answer_ids[:-1]], device=self.device, dtype=cq_ids.dtype)
            full_ids = torch.cat([cq_ids, prefix], dim=1) if length > 1 else cq_ids
            policy_log_probs = self._answer_log_probs(
                self.rollout, self.model, full_ids, answer_ids,
                len_cl=len_cl, len_cq=len_cq,
                temperature=self.args.sample_temp)
            valid = torch.ones(
                1, length, device=self.device, dtype=torch.bool)
            policy_loss = grpo_clipped_loss(
                policy_log_probs.unsqueeze(0), old_log_probs.unsqueeze(0),
                advantages[index:index + 1], valid,
                clip_eps=self.args.clip_eps)
            reference_kl = sampled_reference_kl(
                policy_log_probs, reference_log_probs)
            candidate_loss = (
                policy_loss + self.args.reference_kl_beta * reference_kl)
            (candidate_loss * loss_scale / self.args.group_size).backward()
            policy_total += float(policy_loss.detach())
            kl_total += float(reference_kl.detach())
            loss_total += float(candidate_loss.detach())
            with torch.no_grad():
                ratio = torch.exp(policy_log_probs - old_log_probs)
                ratio_total += float(ratio.mean())
                clip_total += float(
                    ((ratio - 1.0).abs() > self.args.clip_eps).float().mean())

        count = max(len(candidates), 1)
        rewards = [candidate["reward"] for candidate in candidates]
        reward_values = torch.tensor(
            [reward.total for reward in rewards], dtype=torch.float32)
        return {
            "loss": loss_total / count,
            "policy_loss": policy_total / count,
            "reference_kl": kl_total / count,
            "reward": float(reward_values.mean()),
            "reward_std": float(reward_values.std(unbiased=False)),
            "reward_active": float(active),
            "importance_ratio": ratio_total / count,
            "clip_fraction": clip_total / count,
            "f1": sum(reward.f1 for reward in rewards) / count,
            "em": sum(reward.em for reward in rewards) / count,
            "verbosity_penalty": (
                sum(reward.verbosity_penalty for reward in rewards) / count),
            "answer_tokens": (
                sum(len(candidate["answer_ids"]) for candidate in candidates) / count),
        }

    def run(self) -> None:
        args = self.args
        if "," in args.data_dir or "," in args.data_path or "," in args.data_format:
            raise ValueError("首版 GRPO 只接受一个训练数据源")
        records = load_records(args.data_path, args.data_format, None)
        train_ds = RewardDataset(
            args.data_dir, args.data_path, args.data_format,
            records=records, max_samples=args.max_samples)
        val_ds = (InLoopDataset(
            args.val_data_dir, args.val_data_path,
            args.val_data_format or args.data_format, policy="tf")
                  if args.val_data_dir else None)
        if self.world > 1:
            sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        else:
            sampler = torch.utils.data.RandomSampler(train_ds)
        loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_first,
            pin_memory=False, drop_last=True,
            persistent_workers=(args.num_workers > 0))
        steps_per_epoch = (
            len(loader) // args.grad_accum * args.grpo_epochs)
        total_steps = args.max_steps or args.epochs * steps_per_epoch
        if total_steps < 1:
            raise ValueError("GRPO total_steps 必须为正数")
        self.resolved_joint_finetune_steps = total_steps
        if self.is_main:
            print(
                f"GRPO: {len(train_ds)} prompts, group={args.group_size}, "
                f"global_prompt_batch={self.world}x{args.grad_accum}, "
                f"reuse_epochs={args.grpo_epochs}, total_steps={total_steps}, "
                f"temp={args.sample_temp}, "
                f"clip={args.clip_eps}, ref_kl_beta={args.reference_kl_beta}")

        rollout_buffer: list[dict] = []
        aggregate: dict[str, float] = {}
        aggregate_count = 0
        started = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self.epoch)
            for item in loader:
                if self.global_step >= total_steps:
                    break
                try:
                    rollout_buffer.append(self.collect_group(item))
                except torch.OutOfMemoryError:
                    self._log_fatal_oom(item, phase="grpo_rollout")
                    raise
                if len(rollout_buffer) < args.grad_accum:
                    continue
                for reuse_epoch in range(args.grpo_epochs):
                    if self.global_step >= total_steps:
                        break
                    self._set_lr(self.global_step, total_steps)
                    for group in rollout_buffer:
                        try:
                            metrics = self.backward_group(
                                group, loss_scale=1.0 / len(rollout_buffer))
                        except torch.OutOfMemoryError:
                            self._log_fatal_oom(
                                group["diagnostic"],
                                phase=f"grpo_backward[reuse={reuse_epoch}]")
                            raise
                        metrics["reuse_epoch"] = float(reuse_epoch)
                        for name, value in metrics.items():
                            aggregate[name] = aggregate.get(name, 0.0) + value
                        aggregate_count += 1
                    grad_norm, stepped = self.sync_and_step()
                    self.global_step += 1
                    if self.is_main and self.global_step % args.log_interval == 0:
                        means = {
                            name: value / max(aggregate_count, 1)
                            for name, value in aggregate.items()}
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"  step {self.global_step:5d}/{total_steps} | "
                            f"reward {means.get('reward', 0):.4f}±"
                            f"{means.get('reward_std', 0):.4f} | "
                            f"F1 {means.get('f1', 0):.3f} "
                            f"EM {means.get('em', 0):.3f} | "
                            f"loss {means.get('loss', 0):.4f} "
                            f"KLref {means.get('reference_kl', 0):.4f} | "
                            f"ratio {means.get('importance_ratio', 0):.3f} "
                            f"clip {means.get('clip_fraction', 0):.1%} | "
                            f"active {means.get('reward_active', 0):.1%} | "
                            f"grad {grad_norm:.3f} | "
                            f"{aggregate_count / elapsed:.3f} groups/s/rank")
                        if self.writer:
                            for name, value in means.items():
                                self.writer.add_scalar(
                                    f"train_grpo/{name}", value,
                                    self.global_step)
                            self.writer.add_scalar(
                                "train_grpo/grad_norm", grad_norm,
                                self.global_step)
                        aggregate.clear()
                        aggregate_count = 0
                        started = time.time()
                    if not stepped and self.is_main:
                        print(
                            f"  WARNING: non-finite grad norm {grad_norm}; "
                            "step skipped")
                    if val_ds and self.global_step % args.val_interval == 0:
                        validation = self.validate(val_ds)
                        improved = validation["val_loss"] < self.best_val
                        if improved:
                            self.best_val = validation["val_loss"]
                            self.best_step = self.global_step
                            self.save(args.output_dir, validation, kind="best")
                            self._save_result({"best_metrics": validation})
                        if self.is_main:
                            print(
                                f"  --- GRPO VAL step {self.global_step}: "
                                f"TF-KL={validation['val_loss']:.4f} "
                                f"best={self.best_val:.4f}@{self.best_step} ---")
                    if self.global_step % args.save_interval == 0:
                        self.save(args.output_dir)
                rollout_buffer.clear()

        final = self.validate(val_ds) if val_ds else {}
        self.save(args.output_dir, final, kind="final")
        self._save_result({
            "final_metrics": final,
            "done": True,
            "objective": "grpo",
            "reference_checkpoint": self.reference_checkpoint,
            "reference_checkpoint_id": self.reference_checkpoint_id,
            "group_size": args.group_size,
            "grpo_epochs": args.grpo_epochs,
            "clip_eps": args.clip_eps,
            "reference_kl_beta": args.reference_kl_beta,
        })
        if self.writer:
            self.writer.close()
        if self.world > 1:
            dist.barrier()
            dist.destroy_process_group()
        if self.is_main:
            print(f"GRPO complete: step={self.global_step}")


def main() -> None:
    args = parse_args()
    if args.policy != "grpo":
        raise ValueError("请显式传 --policy grpo")
    if args.resume and args.warm_start_checkpoint:
        raise ValueError("--resume 与 --warm_start_checkpoint 不能同时显式提供")
    trainer = GRPOTrainer(args)
    resume = trainer.resolve_resume_path(args.resume)
    if resume:
        trainer.load(resume)
        trainer.release_resume_path(resume)
    elif args.warm_start_checkpoint:
        trainer.warm_start(args.warm_start_checkpoint)
    else:
        raise ValueError("GRPO 必须提供 --warm_start_checkpoint 或可恢复的 latest.pt")
    reference = args.reference_checkpoint or args.warm_start_checkpoint
    if not reference:
        raise ValueError("恢复 GRPO 时仍必须提供固定的 --reference_checkpoint")
    trainer.set_reference(reference)
    trainer.run()
    if trainer.is_main and trainer.checkpoint_store is not None:
        trainer.checkpoint_store.assert_uploads_complete()


if __name__ == "__main__":
    main()
