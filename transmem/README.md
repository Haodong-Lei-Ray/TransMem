# TransMem 实现

给冻结的 Qwen3-4B 外挂一个 **L 层 Qwen3 同款小 Transformer**,读「带干扰长文记忆 + 当前查询」,
一次前向回归出记忆偏置 `MS`,逐元素加到查询隐状态 `HQ_stu_i` 上,使带干扰的「学生」表现得像看了
golden 短文的「教师」。**LLM 全程冻结,唯一可训练的是 TransMem。**

设计文档: `../docs/plan.md`。姊妹方案(扩散版): `../../Project3/diffusionmem/diffusion_mem`(数据/教师-学生/OPSDL 框架完全一致,只把「扩散 DiT 多步采样 λ」换成「TransMem 一次前向回归 MS」)。

```
X_i = [HM_stu ; HQ_stu_i]  ──L层 Qwen3 block(causal,RoPE)──▶ 读末位查询槽 ─▶ MS_i
HQ'_i = HQ_stu_i + a·MS_i  ──冻结 LM head──▶ P_i^stu     ←对齐→  P_i^tea = softmax(LMhead(HQ_tea_i))
```

## 文件

| 文件 | 内容 | 验证 |
|---|---|---|
| `config.json` | 超参数(对齐 Qwen3-4B-Instruct-2507) | — |
| `layers.py` | 复用 HF `Qwen3DecoderLayer/RMSNorm/RotaryEmbedding`;造 Qwen3Config、因果 mask、热启动拷贝 | CPU ✓ |
| `transmem.py` | `TransMemConfig`、`TransMem`(L 层 block + 零初始化读出 + a)、`build_transmem` | CPU ✓ |
| `objectives.py` | `DistillLoss`(forward_kl/reverse_kl/jsd + 表征回归)、`FrozenLMHead` | CPU ✓ |
| `extract_features.py` | **Stage 0**:冻结 LLM forward → `HM_stu/HQ_stu/HQ_tea` + dump `lm_head` | 需 GPU+模型 |
| `train_offpolicy.py` | **Stage 1 off-policy**:Stage0 特征 + 穿 lm_head 逐位置蒸馏 | CPU 烟雾 ✓ |
| `train_onpolicy.py` | **Stage 1 OPD**:学生在线 rollout(TransMem 在环)+ 教师对齐 | 需 GPU+模型 |
| `evaluate.py` | 推理 + 长度外推评测;teacher/student/transmem 三模式 | 需 GPU+模型 |
| `test_shapes.py` | 无 GPU toy 验证 | CPU ✓ |

## 数据流

```
parquet ──extract_features(Stage0)──▶ data/stage0_{train,dev}/  (shard_*/sample_*.pt + meta.json + lm_head.pt)
                                          │
              ┌───────────────────────────┴────────────────────────────┐
   off-policy │ train_offpolicy: 读特征, 穿 lm_head 蒸馏 (不加载 LLM, 轻) │
   on-policy  │ train_onpolicy : 加载冻结 LLM, 学生在线 rollout + 教师对齐 │
              └───────────────────────────┬────────────────────────────┘
                                          ▼
                            checkpoints/*/latest.pt
                                          │
                            evaluate: eval_*.json 长度外推 accuracy
```

## 快速跑

```bash
PY=/mnt/petrelfs/leihaodong/Project1/delta-Mem/.venv-eval/bin/python

# 0) toy 验证 (无 GPU)
$PY -m transmem.test_shapes

# 1) Stage 0 抽特征 (GPU)            —— 见 scripts/run_stage0.sh
# 2a) off-policy 训练 (GPU, 轻)      —— 见 scripts/train/run_offpolicy.sh
# 2b) on-policy/OPD 训练 (GPU)       —— 见 scripts/train/run_onpolicy.sh
# 3) 评测 (GPU): 先 sanity check     —— 见 scripts/eval/run_eval.sh
$PY -m transmem.evaluate --eval_file <eval_50.json> --model_path <model> --mode teacher   # 上界
$PY -m transmem.evaluate --eval_file <eval_50.json> --model_path <model> --mode student   # 基线
```

集群提交: `sbatch scripts/run_stage0.sh` 等(分区 `DataFrontier_Explore`,reserved)。`MODEL_PATH` 指向
s3mount 挂载的 `Qwen3-4B-Instruct-2507`(挂载见 `~/.claude/CLAUDE.md` s3mount 模板,关代理)。

## 解耦开关(留作消融对比)

- **训练方式**(train.png「off-policy 和 OPD 俩种方式都做做」):`train_offpolicy` vs `train_onpolicy`,散度共用 `DistillLoss`。
- **散度** `--divergence`:`forward_kl`(OPSDL 默认,mode-covering)/ `reverse_kl` / `jsd`(GKD,on-policy 建议)。
- **位置注入** `pos_mode`:`none`(hidden 已烤进 backbone 绝对位置,RoPE 用全 0 = 恒等)/ `rope`(0..N)/ `learned`。HM_stu 已带位置,是否再注入留作对比。
- **注意力** `causal`:查询末尾因果 vs 记忆槽双向。
- **热启动** `warm_start`:用 backbone 顶部 L 层初始化(配合 pre-norm 残差流 hidden,见下)。
- **scale** `a` / `learnable_a`;**读出** `zero_init_out`(零初始化恒等启动)。

## 关键设计 & 对 plan 的修正

- **复用 HF Qwen3 block**(法则 4):与 backbone 逐位等同,热启动只需拷 `model.layers[-L:]`,头数/RoPE 不会写错。
- **off/on-policy 解耦**:二者只差「轨迹来源」(教师固定 rollout vs 当前策略在线采样),散度计算共享。
- **零初始化读出**:`out_proj` 零初始化 → 初始 `MS=0` → `HQ'=HQ_stu` 恒等,训练稳(文献一致)。
- 实测 backbone config 修正 plan §7:`rope_theta=5e6`(非 1e6),`intermediate_size=9728`,`tie_word_embeddings=true`(故 `lm_head.weight == embed_tokens.weight`,Stage0 dump 一次即可)。
- **热启动表征空间**(plan §4 的潜在不自洽):backbone decoder layer 工作在 **pre-norm 残差流**,而默认输入 `HM/HQ` 取自 `model.norm` **之后**(post-final-norm)。`warm_start=true` 时建议把 Stage0 的 hidden 源改成 pre-norm 残差流以匹配;默认冷启动(零假设、最稳)。

## 已验证 / 待 GPU 验证(诚实标注)

- **CPU 已验证**:`transmem` 前向/反传/形状、零初始化恒等、3×pos_mode×causal 组合、三种散度穿 `FrozenLMHead` 反传到 TransMem、真实 config(410M 参)构建;off-policy 训练全链路 overfit(loss 单调下降)。
- **需 GPU + 模型验证**:Stage 0 抽取(hook/切 C_S 逻辑复用自已验证的 Project3,但本仓未在 GPU 上重跑)、on-policy rollout(KV cache + TransMem 在环)、evaluate 长度外推。**上 GPU 第一件事**:`evaluate --mode teacher/student` 确认「教师 >> 学生」(plan §9.6),再大规模训。
