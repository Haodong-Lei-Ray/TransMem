#!/usr/bin/env python3
"""
Stage 0 — 离线特征抽取: 用冻结 LLM forward, 产出 TransMem 训练数据.

与 Project3/diffusionmem/diffusion_mem/train_stage0.py 同一套已验证逻辑
(切 C_S、教师 rollout+hook HQ_tea、学生 teacher-forcing HM/HQ_stu、sharded .pt + manifest),
仅两点不同:
  1) 落盘 (hm_stu, hq_stu, hq_tea) —— TransMem off-policy 不需要 λ, 教师分布在 Stage1 现算;
  2) 额外 dump 一次冻结 lm_head 权重到 output_dir/lm_head.pt, 供 off-policy KD 穿它算 logits.

数据格式 (BytedTsinghua-SIA/hotpotqa):
  Train parquet 列: prompt(chat messages), context(长文 C_L), extra_info{index=golden 文档号},
                    reward_model{ground_truth}, ability.
  Eval  JSON 字段:  context, input(问题), index(golden 文档号), answers, num_docs.

C_S = context 中 "Document {index}:" 到下一个 "Document:" (或文末) 之间的文本.
Q   = prompt[0].content (train) 或 input (eval).  C_L = context 全文.

LLM 全程冻结. hidden 取 model.model.norm 输出 (最后 RMSNorm 之后、LM head 之前).

用法:
  python -m transmem.extract_features \
    --data_path ../Project3/data/hotpotqa/hotpotqa_train_32k.parquet \
    --data_format parquet \
    --model_path /path/to/Qwen3-4B-Instruct-2507 \
    --output_dir data/stage0_train \
    --N 4 --max_answer_tokens 50

并发: --num_workers K 开多进程数据并行 (每 worker 一份模型副本, 轮转绑定可见 GPU,
样本按 records[w::K] 交错切分); 产物布局与顺序版一致. 线程不可行: hook 在共享模型上
注册/摘除, 并发调用会互相污染捕获的 hidden state.
断点续抽: worker manifest 逐样本追加写 JSONL (output_dir/.worker_manifests/),
中途崩溃后用相同参数重跑即可跳过已完成样本; 参数 (含 num_workers) 变了则自动重抽.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage 0: 离线抽取 LLM hidden states (TransMem)")
    p.add_argument("--data_path", required=True, help="数据路径 (.parquet 或 .json)")
    p.add_argument("--data_format", default="parquet",
                   choices=["parquet", "json", "qasper", "hotpotqa-agentmem", "longmemeval"])
    p.add_argument("--model_path", required=True, help="Qwen3-4B-Instruct-2507 路径")
    p.add_argument("--output_dir", required=True, help="输出目录, 写 sharded .pt + lm_head.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--N", type=int, default=4, help="记忆分段数 (HM_stu = [N, dim])")
    p.add_argument("--pool_ns", default="",
                   help="记忆池消融: 逗号分隔 N 列表 (如 4,8,16,32,64,128,256,384). 非空时 "
                        "hm_stu 存全部 N 的取位并集 [P,dim] + hm_pos/hm_maps 索引表, --N 被忽略; "
                        "训练时按 config.n_mem 从池切片, 一次提取喂所有 N (须 --hm_mode frac)")
    p.add_argument("--hm_mode", default="floor", choices=["floor", "frac"],
                   help="HM 取位公式: floor=len_cl//N 分段 (历史默认); frac=ceil((i+1)*len_cl/N)-1 "
                        "(N 整除 N' 时位置嵌套, 记忆池只需存并集; 末槽恰为 C_L 末 token)")
    p.add_argument("--manifest_dir", default=None,
                   help="worker manifest 目录 (默认 output_dir/.worker_manifests). "
                        "output_dir 在 s3mount 上时必须指到本地盘: JSONL 追加写对象存储不支持")
    p.add_argument("--dump_layers", type=int, default=0,
                   help="v3 计划 6 (transmem-layer): 额外抽 LLM 最后 K 个 decoder 层输出的 "
                        "HM/HQ_stu/HQ_tea (每样本 hm_stu_layers [K,N,dim] / hq_{stu,tea}_layers "
                        "[K,M,dim]), 并 dump final_norm.pt (顶层 KL 用). 0=关 (行为与旧版完全一致). "
                        "与 --pool_ns 互斥")
    p.add_argument("--trajectory", default="teacher", choices=["teacher", "golden"],
                   help="轨迹来源: teacher=教师(C_S)rollout 当目标(off-policy KD, 默认); "
                        "golden=直接 teacher-force 数据集 golden 答案(SFT-on-golden, 无 teacher rollout, "
                        "存 answer_ids=golden 供 CE loss; 规避 dirty teacher)")
    p.add_argument("--max_answer_tokens", type=int, default=50)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--save_dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=1,
                   help="数据并行进程数: 每个 worker 一份模型副本, 轮转绑定可见 GPU "
                        "(worker w -> cuda:{w %% device_count}); 超过 GPU 数则多 worker 共卡, 注意显存. "
                        "样本按 records[w::K] 交错切分, 产物布局与顺序版一致")
    p.add_argument("--dump_lm_head", action="store_true", default=True,
                   help="是否 dump 冻结 lm_head (off-policy KD 需要)")
    p.add_argument("--thinking", action="store_true",
                   help="chat prompt 是否用 thinking 系统提示 (build_chat_prompt_ids 的 thinking 参数)")
    return p.parse_args()


def get_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def hm_positions(len_cl: int, N: int, mode: str = "floor") -> list[int]:
    """C_L 前 len_cl 个 token 位置中取 N 个记忆槽位置 (每段末位).

    floor: seg=len_cl//N, idx=(i+1)*seg-1 —— 历史公式, 段尾对不齐文末,
           且不同 N 的位置不互为子集 (floor 舍入).
    frac : idx=ceil((i+1)*len_cl/N)-1 —— 末槽=len_cl-1; N 整除 N' 时
           positions(N) ⊆ positions(N') 严格嵌套, 记忆池消融只存一份并集.
    训练 (stage0) 与推理 (OnPolicyRollout) 必须走同一 mode, 记录在 ckpt config.hm_mode.
    """
    if mode == "frac":
        return [max(math.ceil((i + 1) * len_cl / N) - 1, 0) for i in range(N)]
    seg = max(len_cl // N, 1)
    return [max(min((i + 1) * seg, len_cl) - 1, 0) for i in range(N)]


def _parse_pool_ns(args) -> list[int]:
    pool = getattr(args, "pool_ns", "") or ""
    return sorted({int(x) for x in str(pool).split(",") if x.strip()})


def atomic_save(obj, path, retries: int = 8) -> None:
    """torch.save 到本地临时文件, 再整块 copy 到目标 (带退避重试).

    目标在 s3mount 上时必须这样: torch.save 的 zip 写法需要 seek 回填目录,
    对象存储只支持顺序写新文件. 本地目标也顺带获得半写崩溃保护.
    重试: s3mount/Ceph 偶发瞬时 EIO (10209436 实测两 worker 同刻 Errno 5,
    其余正常) — 退避重跑 copy; 持续失败才抛出 (fail-fast 交给 watchdog)."""
    import shutil
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        for attempt in range(retries):
            try:
                shutil.copyfile(tmp, str(path))
                return
            except OSError as e:
                if attempt == retries - 1:
                    raise
                wait = (2, 10, 30, 60, 90, 120, 150, 180)[min(attempt, 7)]
                print(f"  [WARN] 写 {path} 失败 ({e}), {wait}s 后重试 "
                      f"({attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
    finally:
        os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════
# 数据适配: 统一 train parquet / eval JSON 字段访问
# ═══════════════════════════════════════════════════════════════════════════

def load_records(data_path: str, data_format: str, max_samples: Optional[int]):
    if data_format == "parquet":
        df = pd.read_parquet(data_path)
        if max_samples:
            df = df.head(max_samples)
        return [_parse_parquet_row(df.iloc[i], i) for i in range(len(df))]
    if data_format == "hotpotqa-agentmem":
        return _load_hotpotqa_agentmem(data_path, max_samples)
    if data_format == "longmemeval":
        return _load_longmemeval(data_path, max_samples)
    with open(data_path) as f:
        raw = json.load(f)
    if data_format == "qasper":
        # 单条记录 (含 selected_qa) 或记录列表
        items = raw if isinstance(raw, list) else [raw]
        if max_samples:
            items = items[:max_samples]
        return [_parse_qasper_record(item, i) for i, item in enumerate(items)]
    if max_samples:
        raw = raw[:max_samples]
    return [_parse_json_item(item, i) for i, item in enumerate(raw)]


def _parse_parquet_row(row, idx: int) -> dict:
    prompt_raw = row["prompt"]
    if isinstance(prompt_raw, str):
        question = prompt_raw.strip()
    elif hasattr(prompt_raw, "__getitem__") and len(prompt_raw) > 0:
        question = prompt_raw[0].get("content", str(prompt_raw[0])).strip()
    else:
        question = str(prompt_raw).strip()

    context = str(row["context"]).strip()
    extra_info = row.get("extra_info", {})
    golden_index = extra_info.get("index", None) if isinstance(extra_info, dict) else None

    reward = row.get("reward_model", {})
    ground_truth = ""
    if isinstance(reward, dict):
        gt_arr = reward.get("ground_truth", None)
        if gt_arr is not None and hasattr(gt_arr, "__iter__") and len(gt_arr) > 0:
            ground_truth = str(gt_arr[0]).strip()

    return {"question": question, "context": context, "golden_index": golden_index,
            "ground_truth": ground_truth, "sample_idx": idx}


# ── hotpotqa-agentmem: MemAgent 数据 (无 golden_index, 需 golden_titles_map 回填) ──

def _load_hotpotqa_agentmem(data_path: str, max_samples: Optional[int]):
    """加载 BytedTsinghua-SIA/hotpotqa parquet, 通过 golden_titles_map.json
    回填 golden_index (证据文档编号)."""
    import re
    import os

    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)

    # 加载 golden_titles_map (同目录下)
    map_path = os.path.join(os.path.dirname(data_path), "golden_titles_map.json")
    if not os.path.exists(map_path):
        print(f"  [WARN] 未找到 golden_titles_map.json ({map_path}), C_S 将为空")
        title_map = {}
    else:
        with open(map_path) as f:
            title_map = json.load(f)

    records = []
    doc_header_re = re.compile(r"^Document (\d+):\s*(.+)$", re.MULTILINE)
    missing_golden = 0
    for i in range(len(df)):
        row = df.iloc[i]
        rec = _parse_parquet_row(row, i)
        # MemAgent parquet 的 extra_info.index 是数据集行号而非证据文档号,
        # golden_index 只能来自 map 回填; 置 None 保证查不到时样本被跳过而非错切 C_S.
        rec["golden_index"] = None
        if not rec["context"]:
            records.append(rec)
            continue

        # 按 question 文本查 golden_titles_map
        q = rec["question"].strip()
        golden_titles = title_map.get(q) or title_map.get(q.lower()) or []
        if not golden_titles:
            missing_golden += 1
            records.append(rec)
            continue

        # 在 context 中找 "Document N: {title}" 匹配.
        # HotpotQA 每题恰有 2 篇 golden 文档 (多跳), C_S 必须是全部证据的并集 —
        # 只取第一篇会让教师只见半份证据, 实测教师反而比学生差 (dev contains
        # 0.375 vs 0.531), 蒸馏目标被污染 (48% 训练答案不含正确答案).
        golden_indices = []
        for m in doc_header_re.finditer(rec["context"]):
            doc_num = int(m.group(1))
            doc_title = m.group(2).strip()
            if any(gt.lower() in doc_title.lower() for gt in golden_titles):
                if doc_num not in golden_indices:
                    golden_indices.append(doc_num)

        if golden_indices:
            rec["golden_index"] = golden_indices[0]   # 仅供日志/调试
            rec["cs_text"] = "\n\n".join(
                cs for gi in golden_indices
                if (cs := extract_cs(rec["context"], gi)))
        records.append(rec)

    if missing_golden > 0:
        print(f"  hotpotqa-agentmem: {missing_golden}/{len(records)} 条找不到 golden 标题")
    print(f"  hotpotqa-agentmem: {len(records)} 条已加载")
    return records


# ── LongMemEval-S: 多 session 聊天史; C_S=answer_session_ids 对应 session 并集 ──

def _render_lme_session(sess: list, date: str) -> str:
    """一个 session 渲染成 '[Session time: ...]\\nUser: ...\\nAssistant: ...' 文本.
    时间戳必须保留: 133/470 是 temporal-reasoning 题."""
    lines = [f"[Session time: {date}]" if date else "[Session]"]
    for turn in sess:
        role = str(turn.get("role", "user")).strip().capitalize() or "User"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _load_longmemeval(data_path: str, max_samples: Optional[int]):
    """加载 longmemeval_{train,dev}.json (原 schema).
    C_L = 全部 haystack session 顺序拼接 (~122k token, 位置=时间序);
    C_S = answer_session_ids 命中的 session 并集 (hotpotqa 单 golden 事故教训:
          knowledge-update 等题有多个证据 session, 必须全取);
    Q   = question 前缀 (Current date: question_date) —— temporal 题需要今天日期."""
    with open(data_path) as f:
        raw = json.load(f)
    if max_samples:
        raw = raw[:max_samples]
    records = []
    n_no_ev = 0
    for i, item in enumerate(raw):
        sessions = item.get("haystack_sessions") or []
        sids = item.get("haystack_session_ids") or []
        dates = item.get("haystack_dates") or []
        ans_ids = set(item.get("answer_session_ids") or [])
        rendered, evid = [], []
        for j, sess in enumerate(sessions):
            date = dates[j] if j < len(dates) else ""
            txt = _render_lme_session(sess, date)
            rendered.append(txt)
            if j < len(sids) and sids[j] in ans_ids:
                evid.append(txt)
        q = str(item.get("question", "")).strip()
        qd = str(item.get("question_date", "")).strip()
        if not evid:
            n_no_ev += 1
        records.append({
            "question": f"(Current date: {qd}) {q}" if qd else q,
            "context": "\n\n".join(rendered).strip(),
            "cs_text": "\n\n".join(evid).strip(),
            "golden_index": None,
            "ground_truth": str(item.get("answer", "")).strip(),
            "sample_idx": i,
            "question_id": item.get("question_id", ""),
            "question_type": item.get("question_type", ""),
        })
    if n_no_ev:
        print(f"  longmemeval: {n_no_ev}/{len(records)} 条证据 session 为空 (将被跳过)")
    print(f"  longmemeval: {len(records)} 条已加载")
    return records


def _parse_json_item(item: dict, idx: int) -> dict:
    return {
        "question": str(item.get("input", "")).strip(),
        "context": str(item.get("context", "")).strip(),
        "golden_index": item.get("index", None),
        "ground_truth": str(item.get("answers", [""])[0]).strip() if item.get("answers") else "",
        "sample_idx": idx,
    }


# ── Qasper: C_L=全文, C_S=evidence(直接给, 无需正则切), Q=question ──────────

def _build_qasper_context(item: dict) -> str:
    """把 Qasper 一篇论文拼成长上下文 C_L: Title + Abstract + 各 section 的 paragraphs."""
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    if item.get("abstract"):
        parts.append(f"Abstract:\n{item['abstract']}")
    ft = item.get("full_text", {}) or {}
    names = ft.get("section_name", []) or []
    paras = ft.get("paragraphs", []) or []
    for i, sec_paras in enumerate(paras):
        name = names[i] if i < len(names) else ""
        body = "\n".join(p for p in sec_paras if p)
        if name or body:
            parts.append((f"{name}\n{body}" if name else body).strip())
    return "\n\n".join(parts).strip()


def _qasper_ground_truth(qa: dict) -> str:
    """取参考答案 (仅供 eval/参考; Stage0 训练特征用教师 rollout)."""
    if qa.get("free_form_answer"):
        return str(qa["free_form_answer"]).strip()
    spans = qa.get("extractive_spans") or []
    if spans:
        return "; ".join(str(s).strip() for s in spans)
    if qa.get("yes_no") is not None:
        return "Yes" if qa["yes_no"] else "No"
    if qa.get("unanswerable"):
        return "Unanswerable"
    return ""


def _parse_qasper_record(item: dict, idx: int) -> dict:
    """Qasper 记录(含 selected_qa) -> 统一字段; cs_text 由 evidence 直接给出."""
    qa = item.get("selected_qa", {}) or {}
    evidence = qa.get("evidence", []) or []
    cs_text = "\n\n".join(e for e in evidence if e).strip()
    return {
        "question": str(qa.get("question", "")).strip(),
        "context": _build_qasper_context(item),
        "cs_text": cs_text,                  # C_S 直供, process_sample 优先用它
        "golden_index": None,
        "ground_truth": _qasper_ground_truth(qa),
        "sample_idx": idx,
        "paper_id": item.get("id", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════
# C_S 提取: 从 context 中按 golden_index 切出 golden 文档
# ═══════════════════════════════════════════════════════════════════════════

_DOC_HEADER_RE = re.compile(r"^Document (\d+):\s*$", re.MULTILINE)


def extract_cs(context: str, golden_index: int) -> str:
    """从 context 中提取 Document {golden_index} 的正文 (不含标题行)."""
    if golden_index is None:
        return ""
    matches = list(_DOC_HEADER_RE.finditer(context))
    if not matches:
        return ""
    target_match = next_match = None
    for i, m in enumerate(matches):
        if int(m.group(1)) == golden_index:
            target_match = m
            if i + 1 < len(matches):
                next_match = matches[i + 1]
            break
    if target_match is None:                      # fallback: positional
        idx = golden_index - 1
        if 0 <= idx < len(matches):
            target_match = matches[idx]
            if idx + 1 < len(matches):
                next_match = matches[idx + 1]
    if target_match is None:
        return ""
    start = target_match.end()
    end = next_match.start() if next_match else len(context)
    return context[start:end].strip()


# ═══════════════════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════════════════

class Stage0Extractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model_dtype = get_dtype(args.dtype)
        self.save_dtype = get_dtype(args.save_dtype)
        self.N = args.N
        self.trajectory = getattr(args, "trajectory", "teacher")
        self.hm_mode = getattr(args, "hm_mode", "floor")
        self.pool_ns = _parse_pool_ns(args)
        if self.pool_ns:
            assert self.hm_mode == "frac", "--pool_ns 依赖 frac 取位的嵌套性, 须 --hm_mode frac"
        self.dump_layers = int(getattr(args, "dump_layers", 0) or 0)
        assert not (self.dump_layers and self.pool_ns), "--dump_layers 与 --pool_ns 互斥"
        self.layer_ids: list[int] = []      # load_model 后按模型层数填充

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        args = self.args
        print(f"加载 tokenizer: {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(torch_dtype=self.model_dtype, local_files_only=True,
                           trust_remote_code=True)
        if args.attn_impl:
            load_kwargs["attn_implementation"] = args.attn_impl
        print(f"加载模型: {args.model_path} (dtype={args.dtype}, attn={args.attn_impl})")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, **load_kwargs).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size
        self.lm_head = self.model.lm_head
        self._eos_ids = resolve_eos_ids(self.model)
        if self.dump_layers:
            n_layers = len(self.model.model.layers)
            assert self.dump_layers <= n_layers, (self.dump_layers, n_layers)
            self.layer_ids = list(range(n_layers - self.dump_layers, n_layers))
            print(f"  dump_layers={self.dump_layers} -> layer_ids={self.layer_ids}")
        print(f"  dim={self.dim}, 设备={self.device}, eos_ids={self._eos_ids}")

    def dump_lm_head(self, output_dir: Path):
        """dump 冻结 lm_head 权重 [vocab, dim] -> output_dir/lm_head.pt (off-policy KD 用).
        已存在且非空则跳过: 权重确定不变, 且 778MB 多段上传在桶临界抖动时必炸 (10210334)."""
        target = output_dir / "lm_head.pt"
        try:
            if target.exists() and target.stat().st_size > 700_000_000:
                print(f"  lm_head.pt 已存在 ({target.stat().st_size}B), 跳过 dump")
                return
        except OSError:
            pass
        w = self.model.lm_head.weight.detach().to(self.save_dtype).cpu()
        tied = bool(getattr(self.model.config, "tie_word_embeddings", False))
        atomic_save({"weight": w, "tied": tied,
                     "vocab_size": w.shape[0], "dim": w.shape[1]},
                    output_dir / "lm_head.pt")
        print(f"  dump lm_head.pt: weight{tuple(w.shape)} tied={tied}")

    def dump_final_norm(self, output_dir: Path):
        """dump 冻结 final RMSNorm 权重 (transmem-layer 顶层 KL: 层输出→final_norm→lm_head)."""
        target = output_dir / "final_norm.pt"
        try:
            if target.exists() and target.stat().st_size > 1000:
                print("  final_norm.pt 已存在, 跳过 dump")
                return
        except OSError:
            pass
        norm = self.model.model.norm
        atomic_save({"weight": norm.weight.detach().float().cpu(),
                     "eps": float(getattr(norm, "variance_epsilon", 1e-6))}, target)
        print(f"  dump final_norm.pt: weight{tuple(norm.weight.shape)}")

    def _register_layer_hooks(self, captured: dict, last_only: bool):
        """在 layer_ids 各层挂 forward hook.
        last_only=True: 每次前向追加末位 hidden [dim] (教师 generate 逐步捕获);
        last_only=False: 存整段 hidden [S, dim] (学生单次 teacher-forcing 前向).
        返回 handles (调用方负责 remove)."""
        handles = []
        for l in self.layer_ids:
            def mk(lid):
                def hook_fn(m, inp, out):
                    h = out[0] if isinstance(out, tuple) else out    # [B, S, dim]
                    if last_only:
                        captured.setdefault(lid, []).append(h[0, -1, :].detach())
                    else:
                        captured[lid] = h[0].detach()
                return hook_fn
            handles.append(self.model.model.layers[l].register_forward_hook(mk(l)))
        return handles

    # ── 教师生成答案 + 顺带捕获 HQ_tea_i (无需二次 forward) ──────────────
    @torch.inference_mode()
    def generate_answer(self, cs_text: str, question: str):
        """教师 (C_S, Q) greedy 自回归生成, 生成时 hook 末位隐状态拿 HQ_tea_i.
        返回 (answer_ids [M], answer_text, hq_tea [M, dim])."""
        cq_ids = build_chat_prompt_ids(self.tokenizer, cs_text, question, self.device,
                                        thinking=self.args.thinking)
        prompt_len = cq_ids.shape[1]

        captured: list[torch.Tensor] = []
        def hook_fn(m, inp, out):
            captured.append(out[0, -1, :].detach())     # 末位 = 下一 token 的预测隐状态
        handle = self.model.model.norm.register_forward_hook(hook_fn)
        layer_caps: dict[int, list[torch.Tensor]] = {}
        layer_handles = (self._register_layer_hooks(layer_caps, last_only=True)
                         if self.dump_layers else [])
        try:
            generated = self.model.generate(
                input_ids=cq_ids, attention_mask=torch.ones_like(cq_ids),
                max_new_tokens=self.args.max_answer_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._eos_ids)          # chat 模式: 答完吐 <|im_end|> 自然停
        finally:
            handle.remove()
            for hd in layer_handles:
                hd.remove()

        answer_ids = generated[0, prompt_len:].tolist()
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        M = len(answer_ids)
        if M > 0 and len(captured) >= M:
            hq_tea = torch.stack(captured[:M], dim=0)
        else:
            hq_tea = (torch.stack(captured, dim=0) if captured
                      else torch.empty(0, self.dim, device=self.device))
        hq_tea_layers = None
        if self.dump_layers:
            m_eff = hq_tea.shape[0]
            hq_tea_layers = torch.stack(
                [torch.stack(layer_caps[l][:m_eff], dim=0) for l in self.layer_ids],
                dim=0)                                   # [K, M, dim]
        return answer_ids, answer_text, hq_tea, hq_tea_layers

    # ── 学生 forward: HM_stu + HQ_stu_i ──────────────────────────────
    @torch.inference_mode()
    def student_forward(self, context_long: str, question: str, answer_ids: list[int]):
        """(C_L, Q, A_[1:M-1]) -> (HM_stu [N|P, dim], HQ_stu_i [M, dim], hm_extras).
        hm_extras: 池化模式下的 hm_pos/hm_maps/len_cl (单 N 模式为 {}).
        全程 token-id 空间拼接, 避免 BPE 边界 merge / 特殊 token 约定导致的位置错位."""
        M = len(answer_ids)
        cq_ids = build_chat_prompt_ids(self.tokenizer, context_long, question, self.device,
                                        thinking=self.args.thinking)
        len_cq = cq_ids.shape[1]
        cl_ids = self.tokenizer(context_long, return_tensors="pt",
                                add_special_tokens=False).input_ids.to(self.device)
        len_cl = cl_ids.shape[1]

        layer_store: Optional[dict] = {} if self.dump_layers else None
        if M <= 1:
            hidden = self._forward_ids_and_hook(cq_ids, layer_store)
            hm, hm_extras = self._extract_hm(hidden, len_cl)
            if self.dump_layers:
                self._pack_layer_feats(layer_store, len_cl,
                                       [hidden.shape[0] - 1], hm_extras)
            return hm, hidden[-1:, :], hm_extras

        prefix_ids = torch.tensor([answer_ids[:-1]], device=self.device, dtype=cq_ids.dtype)
        full_ids = torch.cat([cq_ids, prefix_ids], dim=1)
        hidden = self._forward_ids_and_hook(full_ids, layer_store)
        total_len = hidden.shape[0]
        hm, hm_extras = self._extract_hm(hidden, len_cl)

        positions = [len_cq - 1]                         # 预测 A[0]=HQ_stu_1
        for i in range(2, M + 1):
            pos = len_cq + (i - 2)                        # 预测 A[i-1]=HQ_stu_i
            if pos < total_len:
                positions.append(pos)
            else:
                break
        hq = hidden[torch.tensor(positions, device=hidden.device, dtype=torch.long)]
        if self.dump_layers:
            self._pack_layer_feats(layer_store, len_cl, positions, hm_extras)
        return hm, hq, hm_extras

    def _pack_layer_feats(self, layer_store: dict, len_cl: int,
                          positions: list[int], extras: dict) -> None:
        """把各层整段 hidden 切成 HM 槽 + 查询位, 塞进 extras (随 hm_extras 流入样本 .pt).
        取位与 final-norm 侧完全一致 (同 hm_positions / 同 positions)."""
        idx = hm_positions(len_cl, self.N, self.hm_mode)
        hm_layers, hq_layers = [], []
        for l in self.layer_ids:
            h = layer_store[l]                            # [S, dim]
            idx_t = torch.tensor(idx, device=h.device, dtype=torch.long)
            pos_t = torch.tensor(positions, device=h.device, dtype=torch.long)
            hm_layers.append(h[idx_t])
            hq_layers.append(h[pos_t])
        extras["hm_stu_layers"] = torch.stack(hm_layers, dim=0)   # [K, N, dim]
        extras["hq_stu_layers"] = torch.stack(hq_layers, dim=0)   # [K, M, dim]

    def _extract_hm(self, hidden: torch.Tensor, len_cl: int):
        """C_L 前 len_cl 个位置取记忆槽 hidden.

        单 N 模式: 返回 (hm [N,dim], {}).
        池化模式 (--pool_ns): 返回 (hm_pool [P,dim], extras), extras 存进 .pt:
          hm_pos  [P]  并集里每行对应的 token 位置 (调试/校验用)
          hm_maps {str(N): LongTensor[N]}  每个 N 在池里的行索引
          len_cl  int
        """
        if self.pool_ns:
            per_n = {n: hm_positions(len_cl, n, "frac") for n in self.pool_ns}
            union = sorted({p for pos in per_n.values() for p in pos})
            row = {p: r for r, p in enumerate(union)}
            hm = hidden[torch.tensor(union, device=hidden.device, dtype=torch.long)]
            extras = {
                "hm_pos": torch.tensor(union, dtype=torch.long),
                "hm_maps": {str(n): torch.tensor([row[p] for p in per_n[n]],
                                                 dtype=torch.long)
                            for n in self.pool_ns},
                "len_cl": len_cl,
            }
            return hm, extras
        idx = hm_positions(len_cl, self.N, self.hm_mode)
        return hidden[torch.tensor(idx, device=hidden.device, dtype=torch.long)], {}

    def _forward_ids_and_hook(self, ids: torch.Tensor,
                              layer_store: Optional[dict] = None) -> torch.Tensor:
        captured = {}
        def hook_fn(m, inp, out): captured["h"] = out.detach()
        handles = [self.model.model.norm.register_forward_hook(hook_fn)]
        if layer_store is not None and self.dump_layers:
            handles += self._register_layer_hooks(layer_store, last_only=False)
        try:
            # 只跑 base model, 不过 lm_head: 全长 logits [L,151936] 在 122k 上下文
            # (longmemeval) 是 ~37GB, 必 OOM; hook 挂在 model.model.norm 上照常触发.
            # attention_mask=ones 必须显式传: 本 venv (transformers 4.57.6) mask=None 时
            # 不走 is_causal skip, 会物化 S×S 因果 mask (125k 实测峰值 57.4GB vs 16.8GB,
            # 见 probe_longctx_mem.py, job 10216593), longmemeval 长样本必 OOM.
            self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
        return captured["h"][0]

    # ── 单样本处理 ────────────────────────────────────────────────────
    def process_sample(self, rec: dict) -> Optional[dict]:
        try:
            question = rec["question"]
            context_long = rec["context"]
            golden_index = rec.get("golden_index")
            sample_idx = rec["sample_idx"]
            if not question or not context_long:
                return None
            hq_tea_layers = None
            if self.trajectory == "golden":
                # SFT-on-golden: 直接 teacher-force 数据集 golden 答案, 不做 teacher rollout,
                # 规避 "教师看 evidence 生成的答案是脏的" 问题; answer_ids=golden 供 CE loss.
                golden = str(rec.get("ground_truth", "")).strip()
                if not golden:
                    print(f"  [WARN] sample {sample_idx}: golden 答案为空, 跳过")
                    return None
                gids = self.tokenizer(golden, add_special_tokens=False).input_ids
                gids = gids[:self.args.max_answer_tokens]
                if self._eos_ids:
                    gids = gids + [self._eos_ids[0]]        # 末尾 EOS 教模型停
                if len(gids) == 0:
                    return None
                answer_ids, answer_text = gids, golden
                hm_stu, hq_stu, hm_extras = self.student_forward(context_long, question, answer_ids)
                hq_tea = hq_stu    # dummy (CE 不用 teacher; 存同形保持 .pt 格式一致)
            else:
                # off-policy KD: C_S teacher rollout 当软目标 (原路径)
                cs_text = rec.get("cs_text") or (
                    extract_cs(context_long, golden_index) if golden_index is not None else "")
                if not cs_text:
                    print(f"  [WARN] sample {sample_idx}: C_S 为空 (golden_index={golden_index})")
                    return None
                answer_ids, answer_text, hq_tea, hq_tea_layers = \
                    self.generate_answer(cs_text, question)
                if len(answer_ids) == 0:
                    return None
                hm_stu, hq_stu, hm_extras = self.student_forward(context_long, question, answer_ids)

            actual_M = min(hq_tea.shape[0], hq_stu.shape[0], len(answer_ids))
            if actual_M < 1:
                return None
            out = {
                "hm_stu": hm_stu.to(dtype=self.save_dtype).cpu(),
                "hq_stu": hq_stu[:actual_M].to(dtype=self.save_dtype).cpu(),
                "hq_tea": hq_tea[:actual_M].to(dtype=self.save_dtype).cpu(),
                "answer_ids": torch.tensor(answer_ids[:actual_M], dtype=torch.long),
                "answer_text": answer_text,
                "sample_idx": sample_idx,
                "M": actual_M, "dim": self.dim,
                "N": (None if self.pool_ns else self.N),
            }
            if self.dump_layers:
                hqsl = hm_extras.pop("hq_stu_layers")[:, :actual_M]
                hqtl = (hqsl if hq_tea_layers is None
                        else hq_tea_layers[:, :actual_M])   # golden 轨迹: dummy 同形
                out["hm_stu_layers"] = hm_extras.pop("hm_stu_layers").to(self.save_dtype).cpu()
                out["hq_stu_layers"] = hqsl.to(self.save_dtype).cpu()
                out["hq_tea_layers"] = hqtl.to(self.save_dtype).cpu()
                out["layer_ids"] = list(self.layer_ids)
            out.update(hm_extras)     # 池化: hm_pos / hm_maps / len_cl
            return out
        except Exception as e:
            print(f"  [ERROR] sample {rec['sample_idx']}: {e}")
            traceback.print_exc()
            return None

    # ── 主循环 ────────────────────────────────────────────────────────
    def run(self):
        args = self.args
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.dump_lm_head:
            self.dump_lm_head(output_dir)
        if self.dump_layers:
            self.dump_final_norm(output_dir)

        records = load_records(args.data_path, args.data_format, args.max_samples)
        n_total = len(records)
        shard_size = args.samples_per_shard
        num_shards = math.ceil(n_total / shard_size)
        print(f"\n开始抽取: {n_total} 条 -> {num_shards} shards (每 {shard_size})")
        print(f"输出: {output_dir.resolve()}  N={args.N} max_ans={args.max_answer_tokens}")
        print("=" * 72)

        total_pairs = failed = 0
        manifest: list[dict] = []
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, n_total)
            shard_dir = output_dir / f"shard_{shard_idx:04d}"
            shard_dir.mkdir(exist_ok=True)
            shard_pairs = 0
            shard_samples = []
            pbar = tqdm(range(start, end), desc=f"Shard {shard_idx:04d}", unit="sample")
            for local_i, global_i in enumerate(pbar):
                result = self.process_sample(records[global_i])
                if result is None:
                    failed += 1
                    pbar.set_postfix({"fail": failed})
                    continue
                M = result.pop("M")
                fname = f"sample_{global_i:05d}.pt"
                atomic_save(result, shard_dir / fname)
                shard_samples.append(global_i)
                manifest.append({"sample_idx": global_i, "shard_idx": shard_idx,
                                 "file": f"shard_{shard_idx:04d}/{fname}", "M": M})
                shard_pairs += M
                total_pairs += M
                pbar.set_postfix({"pairs": shard_pairs, "fail": failed})
                if (local_i + 1) % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            with open(shard_dir / "meta.json", "w") as f:
                json.dump({"shard_idx": shard_idx, "num_samples": len(shard_samples),
                           "num_pairs": shard_pairs, "sample_indices": shard_samples}, f)
            print(f"  Shard {shard_idx:04d}: {len(shard_samples)} 样本, {shard_pairs} 对")

        meta = {
            "data_path": args.data_path, "data_format": args.data_format,
            "model_path": args.model_path,
            "N": (None if self.pool_ns else args.N), "dim": self.dim,
            "pool_ns": (self.pool_ns or None), "hm_mode": self.hm_mode,
            "dump_layers": (self.dump_layers or None),
            "layer_ids": (self.layer_ids or None),
            "save_dtype": args.save_dtype, "max_answer_tokens": args.max_answer_tokens,
            "total_records": n_total, "succeeded": n_total - failed, "failed": failed,
            "total_pairs": total_pairs, "num_shards": num_shards,
            "has_lm_head": bool(args.dump_lm_head), "samples": manifest,
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print("=" * 72)
        print(f"✅ Stage 0 完成: 成功 {meta['succeeded']}/{n_total}, 共 {total_pairs} (X, HQ_tea) 对")


# ═══════════════════════════════════════════════════════════════════════════
# 多进程数据并行 (--num_workers > 1)
# ═══════════════════════════════════════════════════════════════════════════
# 不能用线程: generate_answer/student_forward 在同一模型上注册/摘除 forward hook,
# 并发调用会互相捕获对方的 hidden state. 因此每 worker 一个进程 + 一份模型副本.

def _read_manifest_done(path) -> dict[int, dict]:
    """读一个 worker manifest JSONL, 返回成功样本 {sample_idx: entry}.
    只算成功: 失败可能是坏 GPU/CUDA 挂死等瞬态原因, 重跑时重试;
    真正的 C_S 为空失败在 GPU 前就返回, 重试代价毫秒级. 半行 (崩溃残留) 丢弃."""
    done: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sample_idx" in e and not e.get("failed"):
                    done[e["sample_idx"]] = e
    return done


def _worker_entry(worker_id: int, args, records: list, manifest_path: str,
                  global_done: Optional[set] = None):
    """单 worker: 处理自己的样本切片, .pt 直接落盘 (文件名含全局 sample_idx, 无冲突).
    manifest 逐样本追加写 JSONL 到 manifest_path (崩溃重跑时按已有行断点续抽),
    末行 {"done": true, "dim": ...} 标记本切片完成, 由父进程合并.
    global_done: 父进程汇总的全 worker 已完成集合 — 换 num_workers 重跑时,
    样本会重新交错分片, 自己 manifest 里没有但别的 worker 做过的样本也要跳过."""
    ngpu = torch.cuda.device_count()
    if ngpu > 0:
        args.device = f"cuda:{worker_id % ngpu}"

    done = _read_manifest_done(manifest_path)
    if global_done:
        done = {**dict.fromkeys(global_done, None), **done}
    todo = [r for r in records if r["sample_idx"] not in done]
    print(f"[W{worker_id}] 启动: device={args.device}, {len(records)} 条"
          + (f" (续抽: 已有 {len(done)}, 剩 {len(todo)})" if done else ""), flush=True)

    extractor = Stage0Extractor(args)
    extractor.load_model()
    output_dir = Path(args.output_dir)
    if worker_id == 0 and args.dump_lm_head:
        extractor.dump_lm_head(output_dir)
    if worker_id == 0 and extractor.dump_layers:
        extractor.dump_final_norm(output_dir)

    # 上次崩溃可能留下无换行的半行, 先补换行, 避免新条目黏连坏掉
    if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
        with open(manifest_path, "rb") as rf:
            rf.seek(-1, os.SEEK_END)
            needs_nl = rf.read(1) != b"\n"
        if needs_nl:
            with open(manifest_path, "a") as f:
                f.write("\n")

    shard_size = args.samples_per_shard
    failed = 0
    with open(manifest_path, "a") as mf:
        for k, rec in enumerate(todo):
            result = extractor.process_sample(rec)
            sid = rec["sample_idx"]
            if result is None:
                failed += 1
                entry = {"sample_idx": sid, "failed": True}
            else:
                M = result.pop("M")
                shard_idx = sid // shard_size
                shard_dir = output_dir / f"shard_{shard_idx:04d}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                fname = f"sample_{sid:05d}.pt"
                atomic_save(result, shard_dir / fname)
                entry = {"sample_idx": sid, "shard_idx": shard_idx,
                         "file": f"shard_{shard_idx:04d}/{fname}", "M": M}
            mf.write(json.dumps(entry) + "\n")
            mf.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                print(f"[W{worker_id}] {k + 1}/{len(todo)} fail={failed}", flush=True)
            if (k + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        mf.write(json.dumps({"done": True, "worker_id": worker_id,
                             "dim": extractor.dim,
                             "layer_ids": (extractor.layer_ids or None)}) + "\n")


def run_parallel(args):
    """父进程: 载数据 -> 交错切片 spawn K 个 worker -> 合并 manifest 写 meta.json.
    输出布局 (shard_XXXX/sample_XXXXX.pt + 每 shard meta.json + 顶层 meta.json)
    与顺序版 Stage0Extractor.run() 完全一致."""
    import multiprocessing as mp
    import shutil

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_path, args.data_format, args.max_samples)
    n_total = len(records)
    K = min(args.num_workers, max(n_total, 1))
    shard_size = args.samples_per_shard
    num_shards = math.ceil(n_total / shard_size)
    print(f"\n并行抽取: {n_total} 条 / {K} workers -> {num_shards} shards (每 {shard_size})")
    print(f"输出: {output_dir.resolve()}  N={args.N} max_ans={args.max_answer_tokens}")
    print("=" * 72)

    # 断点续抽: manifest 目录带 run 配置指纹, 一致则续抽, 不一致则清空重来.
    # output_dir 在 s3mount 上时用 --manifest_dir 指到本地盘 (JSONL 追加写对象存储不支持).
    pool_ns = _parse_pool_ns(args)
    tmp_dir = (Path(args.manifest_dir) if getattr(args, "manifest_dir", None)
               else output_dir / ".worker_manifests")
    # 指纹不含 num_workers: 样本按 sample_idx 全局标识, 换 worker 数续抽也安全
    # (global_done 跨切片跳过 + 合并 glob 全部 worker 文件).
    fingerprint = {"data_path": args.data_path, "data_format": args.data_format,
                   "max_samples": args.max_samples, "N": args.N,
                   "pool_ns": pool_ns, "hm_mode": getattr(args, "hm_mode", "floor"),
                   "max_answer_tokens": args.max_answer_tokens,
                   "thinking": bool(args.thinking),
                   "n_total": n_total}
    # 条件键: 只在开启时进指纹 — 保证旧 run (无此参数) 的指纹逐字节不变,
    # 正在跑/续抽中的作业升级代码后仍能无损断点续抽.
    if int(getattr(args, "dump_layers", 0) or 0):
        fingerprint["dump_layers"] = int(args.dump_layers)
    fp_path = tmp_dir / "run_config.json"
    if tmp_dir.exists():
        old_fp = None
        if fp_path.exists():
            with open(fp_path) as f:
                old_fp = json.load(f)
        if isinstance(old_fp, dict):
            old_fp.pop("num_workers", None)   # 兼容旧指纹 (曾含 num_workers)
        if old_fp == fingerprint:
            print("  检测到上次未完成的 worker manifest, 断点续抽")
        else:
            print("  上次 run 配置不同, 清空 worker manifest 重抽")
            shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f)

    # 全局断点集合: 汇总所有历史 worker 文件的成功样本 (换 K 后交错切片会变,
    # 自己文件里没有的完成样本也必须跳过). 顺带重写文件: 只留成功行, 清掉
    # 历史 done 标记 (否则换 K 后旧标记会骗过合并校验) 与半行/失败行.
    global_done: set = set()
    for wp in sorted(tmp_dir.glob("worker_*.jsonl")):
        d = _read_manifest_done(wp)
        global_done.update(d.keys())
        tmpf = str(wp) + ".tmp"
        with open(tmpf, "w") as f:
            for e in d.values():
                f.write(json.dumps(e) + "\n")
        os.replace(tmpf, wp)
    if global_done:
        print(f"  全局断点: 已完成 {len(global_done)} 样本")

    ctx = mp.get_context("spawn")           # CUDA 子进程必须 spawn
    procs = []
    for w in range(K):
        p = ctx.Process(target=_worker_entry,
                        args=(w, args, records[w::K], str(tmp_dir / f"worker_{w}.jsonl"),
                              global_done))
        p.start()
        procs.append(p)

    # watchdog: 任一 worker 异常退出或 30min 无 manifest 进展 (坏卡上 CUDA 调用会
    # 无限挂死, 不抛异常) → 立即终止全体 fail-fast; requeue 换节点后按 JSONL 续抽,
    # 比让好 worker 慢慢跑完再失败划算 (实测坏节点一次废 6/8 张卡).
    stall_sec = 1800
    t_start = time.time()
    abort_reason = None
    while True:
        alive = [w for w, p in enumerate(procs) if p.is_alive()]
        if not alive:
            break
        crashed = [w for w, p in enumerate(procs)
                   if not p.is_alive() and p.exitcode != 0]
        if crashed:
            abort_reason = f"worker {crashed} 异常退出 (exitcode={[procs[w].exitcode for w in crashed]})"
        else:
            stalled = []
            for w in alive:
                wp = tmp_dir / f"worker_{w}.jsonl"
                last = os.path.getmtime(wp) if wp.exists() else 0.0
                if time.time() - max(last, t_start) > stall_sec:
                    stalled.append(w)
            if stalled:
                abort_reason = f"worker {stalled} 超过 {stall_sec}s 无进展 (CUDA 挂死/坏卡?)"
        if abort_reason:
            print(f"[watchdog] {abort_reason}; 终止全部 worker (断点已在 JSONL)", flush=True)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(30)
            for p in procs:
                if p.is_alive():
                    p.kill()
            raise RuntimeError(f"[watchdog] {abort_reason}; "
                               f"相同参数重跑即可断点续抽 ({tmp_dir})")
        time.sleep(60)

    for p in procs:
        p.join()
    bad = [w for w, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"worker {bad} 退出码非 0; 不写顶层 meta.json. "
                           f"已完成样本记录在 {tmp_dir}, 相同参数重跑即可断点续抽")

    # 合并 glob 全部 worker 文件 (含换 K 前的高编号残留文件, 其样本行仍有效);
    # done 标记只可能来自本轮 (启动时历史标记已被重写清掉), 按 0..K-1 校验.
    entries: dict[int, dict] = {}
    failed = 0
    dim = None
    layer_ids = None
    done_markers: set[int] = set()
    for wp in sorted(tmp_dir.glob("worker_*.jsonl")):
        with open(wp) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("done"):
                    done_markers.add(e.get("worker_id", -1))
                    dim = e["dim"]
                    layer_ids = e.get("layer_ids") or layer_ids
                elif "sample_idx" in e:
                    prev = entries.get(e["sample_idx"])
                    if prev is None or (prev.get("failed") and not e.get("failed")):
                        entries[e["sample_idx"]] = e
    missing = [w for w in range(K) if w not in done_markers]
    if missing:
        raise RuntimeError(f"worker {missing} manifest 无完成标记 (退出码 0 但未写完?)")
    failed = sum(1 for e in entries.values() if e.get("failed"))
    manifest = sorted((e for e in entries.values() if not e.get("failed")),
                      key=lambda e: e["sample_idx"])
    total_pairs = sum(e["M"] for e in manifest)

    by_shard: dict[int, list[dict]] = {}
    for e in manifest:
        by_shard.setdefault(e["shard_idx"], []).append(e)
    for shard_idx in range(num_shards):
        entries = by_shard.get(shard_idx, [])
        shard_dir = output_dir / f"shard_{shard_idx:04d}"
        shard_dir.mkdir(exist_ok=True)
        with open(shard_dir / "meta.json", "w") as f:
            json.dump({"shard_idx": shard_idx, "num_samples": len(entries),
                       "num_pairs": sum(e["M"] for e in entries),
                       "sample_indices": [e["sample_idx"] for e in entries]}, f)

    meta = {
        "data_path": args.data_path, "data_format": args.data_format,
        "model_path": args.model_path,
        "N": (None if pool_ns else args.N), "dim": dim,
        "pool_ns": (pool_ns or None), "hm_mode": getattr(args, "hm_mode", "floor"),
        "dump_layers": (int(getattr(args, "dump_layers", 0) or 0) or None),
        "layer_ids": layer_ids,
        "save_dtype": args.save_dtype, "max_answer_tokens": args.max_answer_tokens,
        "total_records": n_total, "succeeded": n_total - failed, "failed": failed,
        "total_pairs": total_pairs, "num_shards": num_shards,
        "has_lm_head": bool(args.dump_lm_head), "num_workers": K, "samples": manifest,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(tmp_dir)
    print("=" * 72)
    print(f"✅ Stage 0 完成: 成功 {meta['succeeded']}/{n_total}, 共 {total_pairs} (X, HQ_tea) 对")


def _make_prompt(context: str, question: str) -> str:
    """chat user 消息正文 (Context + Question); assistant 生成提示交给 chat template."""
    return f"Context:\n{context}\n\nQuestion:\n{question}"


def _supports_enable_thinking(tokenizer) -> bool:
    """chat template 里是否原生带 enable_thinking 开关 (Qwen3 hybrid-thinking 系列,
    如 Qwen3-8B/14B/32B). Qwen3-4B-Instruct-2507 这类纯 instruct 模型没有这个开关,
    该模型不管 system prompt 写什么都不会自己吐 <think>, 靠 system prompt 兜底即可;
    hybrid 模型不看 system prompt, 必须靠这个模板开关才能真正关/开思考 (否则 128 token
    的 max_answer_tokens 会被 <think> 独吞, 答案都生成不出来)."""
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in chat_template


def build_chat_prompt_ids(tokenizer, context: str, question: str, device=None, thinking=False):
    """按 chat template 构造 prompt token ids (add_generation_prompt=True).

    这样是"对话"而非"文本续写", 模型答完会正常吐 <|im_end|>(151645) 停下,
    AN 自适应真实答案长度, 不再一路跑到 max_answer_tokens (官方推荐用法).
    teacher / student / 推理三处统一用它, 保证 prompt 格式一致 (对齐前提).
    """
    template_kwargs = {}
    if _supports_enable_thinking(tokenizer):
        # hybrid-thinking 模型: 用官方开关控制, 不走 system prompt 文字 hack.
        system_prompt_item = {"role": "system", "content": "You are a precise QA assistant."}
        template_kwargs["enable_thinking"] = thinking
    elif thinking:
        system_prompt_item = {"role": "system", "content": "You are a precise QA assistant. Answer with the short answer and thinking. If you encounter a math problem, you should calculate it step by step. Think it through several times and criticize yourself a few times."}
    else:
        system_prompt_item = {"role": "system", "content": "You are a precise QA assistant. Answer ONLY with the short answer. No explanation."}
    messages = [
        system_prompt_item,
        {"role": "user", "content": _make_prompt(context, question)}
        ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs)  # 官方示例写法
    # 模板已含全部特殊 token, add_special_tokens=False 避免再加 bos/eos (与 student 侧一致)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device) if device is not None else ids


def resolve_eos_ids(model) -> list[int]:
    """从 generation_config 取停止 token (Qwen3: [<|im_end|>=151645, <|endoftext|>=151643])."""
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = model.config.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def main():
    args = parse_args()
    if args.num_workers > 1:
        run_parallel(args)
    else:
        extractor = Stage0Extractor(args)
        extractor.load_model()
        extractor.run()


if __name__ == "__main__":
    main()
