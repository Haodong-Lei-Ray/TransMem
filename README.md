# TransMem

TransMem is a plug-in memory module for frozen decoder-only language models. It
compresses information from a long context into a small set of memory states and
injects a learned residual into selected LLM layers while the answer is decoded.
The backbone LLM remains frozen; only TransMem and, when enabled, its dynamic
gates are trained.

This repository contains training code, checkpoints, and evaluation adapters for
[LoCoMo](#locomo), [MemoryAgentBench](#memoryagentbench), and
[HotpotQA](#hotpotqa).

## How it works

For a layered TransMem checkpoint, a one-layer memory block is attached to each
of the last \(D\) backbone layers:

```text
long context ──► memory states HM
                       │
H^(l-1) ──► LLM layer l ──► H^l
    └─────► TransMem l ────► ΔH^l
                              │
                    H^l + gate^l ⊙ ΔH^l
```

The current default configuration is `D=4`, `N=4` memory slots, and a
token-scalar dynamic gate. A native `best.pt` stores the TransMem configuration
and adapter weights, but not the frozen backbone. Evaluation must therefore use
the same backbone family and hidden size as training.

## Repository layout

```text
transmem/                  Core modules, training, rollout, and checkpoint code
transmem/config*.json      Model- and ablation-specific configurations
scripts/stage0/            Teacher/student feature extraction
scripts/version*/          Training recipes and experiment entrypoints
scripts/eval/              Benchmark adapters and shard mergers
data/                      Prepared benchmark data and preprocessing tools
checkpoints/               Local checkpoint metadata or downloaded adapters
eval_results/              Benchmark outputs
```

## Installation

Python 3.10 and a recent CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/Haodong-Lei-Ray/Project4.git TransMem
cd TransMem

python -m venv .venv-transmem
source .venv-transmem/bin/activate
pip install torch transformers accelerate pandas pyarrow tqdm nltk rouge-score pytest

# The evaluators import the repository as a Python package.
export TRANSMEM_ROOT=$PWD
export PYTHONPATH="$TRANSMEM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Run the unit tests before using a new environment:

```bash
python -m pytest transmem/test_shapes.py \
  transmem/test_layered.py \
  transmem/test_dynamic_gate.py
```

The evaluation code loads backbones with `local_files_only=True`. Download the
backbone in advance and pass its local directory as `MODEL_PATH`.

Example released adapters:

- `Rayleihaodong/TransMem-qwen3-4B-D4`
- `Rayleihaodong/TransMem-qwen2.5-14B-D4`

After downloading an adapter, set:

```bash
export MODEL_PATH=/path/to/the/matching/backbone
export CKPT=/path/to/best.pt
```

## Evaluation modes

The adapters use deterministic greedy decoding.

- `student`: frozen backbone with the full context, without TransMem.
- `transmem`: frozen backbone plus a native TransMem checkpoint.
- `paired` (MemoryAgentBench): evaluates `student` and `transmem` on the same
  examples in one run.

Always compare a TransMem result with the matching student backbone, prompt,
context budget, and decoding settings.

All three evaluators are resumable. They append one record per completed
question to `*.progress.jsonl`; rerunning the same command continues from that
file. Delete the progress file, or use the benchmark's `--force` option where
available, only when intentionally starting over.

## LoCoMo

### Data and protocol dependency

Download `locomo10.json` from LoCoMo. The adapter reuses the LoCoMo protocol
implementation in the sibling `delta-Mem` repository for date handling,
category-specific canonicalization, and token-F1 scoring:

```bash
export DELTAMEM_ROOT=/path/to/delta-Mem
export LOCOMO_DATA=/path/to/locomo10.json
export PYTHONPATH="$TRANSMEM_ROOT:$DELTAMEM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

### TransMem evaluation

```bash
mkdir -p eval_results/locomo_transmem

CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_locomo.py \
  --data_file "$LOCOMO_DATA" \
  --model_path "$MODEL_PATH" \
  --mode transmem \
  --ckpt "$CKPT" \
  --N 4 \
  --categories 1 2 3 4 \
  --max_answer_tokens 50 \
  --attn_impl sdpa \
  --num_shards 1 \
  --shard_index 0 \
  --output_json eval_results/locomo_transmem/locomo_transmem.json
```

Run the matching student baseline by changing `--mode transmem` to
`--mode student` and removing `--ckpt`.

The final JSON contains:

- `summary.overall_f1`
- category F1 for multi-hop, temporal, open-domain, and single-hop questions
- all predictions and per-question scores

By default, category 5 is excluded. Do not compare this number with a result
that includes category 5 or uses a different answer prompt.

### Multi-GPU execution

Split questions round-robin with `--num_shards K` and launch one process per
`--shard_index 0 ... K-1`. Merge completed shards with:

```bash
python scripts/eval/merge_locomo_shards.py \
  --output eval_results/locomo_transmem/locomo_transmem.json \
  eval_results/locomo_transmem/shard_*.json
```

An internal Slurm/S3 example is available at
[`scripts/version4/run_eval_locomo_s32_parallel.sh`](scripts/version4/run_eval_locomo_s32_parallel.sh).
It contains site-specific absolute paths and should be copied and adapted rather
than used unchanged outside the original cluster.

## MemoryAgentBench

### Data

Clone MemoryAgentBench and download its processed parquet files:

```bash
export MAB_ROOT=/path/to/MemoryAgentBench
export MAB_DATA="$MAB_ROOT/processed_data"
```

The adapter evaluates the benchmark's main 13 sources. In `paired` mode it
shares the same examples and prompt budget between student and TransMem.

### Quick smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_memory_agent_bench.py \
  --model_path "$MODEL_PATH" \
  --mode paired \
  --ckpt "$CKPT" \
  --mab_root "$MAB_ROOT" \
  --data_dir "$MAB_DATA" \
  --sources ruler_qa1_197K eventqa_full \
  --max_questions_per_source 5 \
  --output_dir eval_results/mab_smoke
```

### Full main-13 evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_memory_agent_bench.py \
  --model_path "$MODEL_PATH" \
  --mode paired \
  --ckpt "$CKPT" \
  --checkpoint_id "$CKPT" \
  --mab_root "$MAB_ROOT" \
  --data_dir "$MAB_DATA" \
  --agent_input_tokens 128000 \
  --output_dir eval_results/mab_full
```

Results are written to:

```text
eval_results/mab_full/<source>.progress.jsonl
eval_results/mab_full/<source>.summary.json
eval_results/mab_full/summary.json
```

MemoryAgentBench defines scores per source; it does **not** define one official
cross-source overall score. Report each source's `student_primary` and
`transmem_primary`. `longmemeval_s*` and `infbench_sum_eng_shots2` require an
external LLM judge for their official score; lexical metrics in the first pass
are proxies only.

Judge one completed mode/source prediction file with:

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=<an-OpenAI-compatible-endpoint>  # optional

python scripts/eval/eval_memory_agent_bench_judge.py \
  --source 'longmemeval_s*' \
  --mode transmem \
  --input eval_results/mab_full/longmemeval_s.progress.jsonl \
  --output_dir eval_results/mab_full/judge_longmemeval_transmem \
  --mab_root "$MAB_ROOT" \
  --judge_model gpt-4o
```

For multi-GPU evaluation, use the source-level planner and launcher in
[`scripts/eval/Qwen3-4B-Instruct-2507/run_eval_memory_agent_bench_parallel.sh`](scripts/eval/Qwen3-4B-Instruct-2507/run_eval_memory_agent_bench_parallel.sh).
Each source has a single writer, so progress files remain resumable.

## HotpotQA

This repository evaluates the official HotpotQA distractor validation set:
7,405 questions with answer EM, token F1, and answer containment. This is
different from the synthetic long-context `hotpotqa-agent` training benchmark.

The prepared, audited evaluation file is:

```text
data/hotpotqa-benchmark/hotpot/hotpot_dev_distractor_v1_eval.json
```

It was generated from the official distractor validation parquet and checked
to have zero normalized-question overlap with the 32,768-question
`hotpotqa-agent` training split. To reproduce that audit:

```bash
python scripts/eval/prepare_hotpot_official_dev.py \
  --official-dev data/hotpotqa-benchmark/hotpot/hotpot_dev_distractor_v1.parquet \
  --agent-train /path/to/hotpotqa_train_32k.parquet \
  --output data/hotpotqa-benchmark/hotpot/hotpot_dev_distractor_v1_eval.json
```

### Single-GPU evaluation

```bash
mkdir -p eval_results/hotpot_official

CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_hotpot_official.py \
  --data-file data/hotpotqa-benchmark/hotpot/hotpot_dev_distractor_v1_eval.json \
  --model-path "$MODEL_PATH" \
  --mode transmem \
  --ckpt "$CKPT" \
  --output-json eval_results/hotpot_official/shard_0.json \
  --num-shards 1 \
  --shard-id 0 \
  --max-answer-tokens 50 \
  --attn-impl sdpa

python scripts/eval/merge_hotpot_official_shards.py \
  --input-dir eval_results/hotpot_official \
  --num-shards 1 \
  --output-json eval_results/hotpot_official/hotpot_official_results.json
```

The merger verifies exactly 7,405 unique question IDs and writes:

- `hotpot_official_results.json`: EM, F1, containment, and all records
- `hotpot_predictions.json`: answer predictions in HotpotQA submission shape

For the student baseline, use `--mode student` and omit `--ckpt`.

For multiple GPUs, launch `K` evaluator processes with
`--num-shards K`, assign `--shard-id 0 ... K-1`, and then merge. The shared
launcher is
[`scripts/eval/run_hotpot_official_parallel.sh`](scripts/eval/run_hotpot_official_parallel.sh);
model-specific Slurm examples are in the same directory.

## Reproducibility checklist

Record the following with every reported result:

1. Backbone model and exact revision.
2. TransMem checkpoint and checkpoint step.
3. `D`, trained `N`, and any inference-time `N` override.
4. Student or TransMem mode.
5. Context and answer-token budgets.
6. Thinking or non-thinking decoding.
7. Dataset split, included categories/sources, and contamination audit.
8. Number of questions completed and whether an external judge was used.

Never compare a partial/smoke run with a full benchmark. A run is complete only
after all expected shards or sources have been merged successfully.

## Training

Training has two stages:

1. **Stage 0:** extract frozen-backbone teacher/student trajectories.
2. **Stage 1:** train TransMem with off-policy distillation, in-loop
   distillation, or optional post-training.

See [`transmem/README.md`](transmem/README.md) for the internal data flow and
[`scripts/version5/run_train_inloop_generic.sh`](scripts/version5/run_train_inloop_generic.sh)
for the current layered dynamic-gate recipe.

## Citation

If you use TransMem, please cite the accompanying paper. Citation metadata will
be added here when the public version is available.

```text
@misc{lei2026transmemtransforminghiddenstates,
      title={TransMem: Transforming Hidden States into Memory for Large Language Models}, 
      author={Haodong Lei and Junming Liu and Yirong Chen and Pinlong Cai and Botian Shi and Ding Wang and Hongsong Wang},
      year={2026},
      eprint={2607.29032},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2607.29032}, 
}
```
