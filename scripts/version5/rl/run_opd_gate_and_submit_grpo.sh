#!/bin/bash
#SBATCH -J e09_v5opdgt
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved
#SBATCH -t 00:45:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v5opd_gate.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v5opd_gate.err

# Compare OPD best/final on LoCoMo.  Only a non-regressing checkpoint is copied
# to selected.pt and allowed to launch GRPO.  This avoids a permanently held
# DependencyNeverSatisfied GRPO job when OPD fails the quality gate.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED

PROJ=/mnt/petrelfs/leihaodong/Project4
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
OPD_RUN=${OPD_RUN:-v5_opd_qwen3_4b_hqa_d4_rkl_s250}
GRPO_RUN=${GRPO_RUN:-v5_grpo_qwen3_4b_hqa_d4_g4_e2_s200}
MIN_F1=${MIN_F1:-0.5411}
BEST_RESULT=${BEST_RESULT:-$PROJ/eval_results/locomo_${OPD_RUN}_best/locomo_transmem.json}
FINAL_RESULT=${FINAL_RESULT:-$PROJ/eval_results/locomo_${OPD_RUN}_final/locomo_transmem.json}
DECISION_DIR=$PROJ/eval_results/locomo_${OPD_RUN}_gate
DECISION_JSON=$DECISION_DIR/decision.json
mkdir -p "$DECISION_DIR" "$PROJ/logs/eval_locomo"
cd "$PROJ"

read -r selected_name selected_f1 < <(python - "$BEST_RESULT" "$FINAL_RESULT" <<'PY'
import json
import sys
from pathlib import Path

names = ("best.pt", "step_0000250.pt")
scores = []
for name, path in zip(names, sys.argv[1:]):
    payload = json.loads(Path(path).read_text())
    scores.append((float(payload["summary"]["overall_f1"]), name))
score, name = max(scores)
print(name, score)
PY
)

python - "$DECISION_JSON" "$MIN_F1" "$selected_name" "$selected_f1" \
  "$BEST_RESULT" "$FINAL_RESULT" <<'PY'
import json
import sys
from pathlib import Path

output, threshold, selected, score, best_path, final_path = sys.argv[1:]
payload = {
    "threshold": float(threshold),
    "selected_checkpoint": selected,
    "selected_f1": float(score),
    "passed": float(score) >= float(threshold),
    "best_result": best_path,
    "final_result": final_path,
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(payload, ensure_ascii=False))
PY

if ! python - "$selected_f1" "$MIN_F1" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
then
  echo "OPD gate failed: LoCoMo F1=$selected_f1 < baseline=$MIN_F1; GRPO 不提交"
  exit 0
fi

PARENT_REL=leihaodong/Project4/checkpoints/$OPD_RUN/$selected_name
PARENT_ETAG=$(aws s3api head-object --bucket datafrontier --key "$PARENT_REL" \
  --query ETag --output text --endpoint-url "$ENDPOINT")
PARENT_ETAG=${PARENT_ETAG//\"/}
REFERENCE_ID="s3://datafrontier/$PARENT_REL#etag=$PARENT_ETAG"

# Try Spot first for a bounded window.  Cancel it before falling back to
# Reserved, so two trainings can never write the same S3 prefix concurrently.
export_spec="ALL,RUN_NAME=$GRPO_RUN,PARENT_REL=$PARENT_REL,PARENT_ETAG=$PARENT_ETAG,REFERENCE_ID=$REFERENCE_ID"
spot_job=$(sbatch --parsable --quotatype=spot --job-name=e09_v5grp4s \
  --gres=gpu:4 --export="$export_spec" \
  "$PROJ/scripts/version5/rl/run_train_grpo_d4.sh")
grpo_job=""
for _ in $(seq 1 40); do
  state=$(squeue -h -j "$spot_job" -o '%T' | head -1 || true)
  if [[ "$state" == "RUNNING" ]]; then
    grpo_job=$spot_job
    break
  fi
  [[ -z "$state" ]] && break
  sleep 30
done
if [[ -z "$grpo_job" ]]; then
  scancel "$spot_job" 2>/dev/null || true
  # Never let a cancelling Spot allocation overlap the Reserved fallback on
  # the same checkpoint prefix.
  for _ in $(seq 1 40); do
    [[ -z "$(squeue -h -j "$spot_job" -o '%T' || true)" ]] && break
    sleep 3
  done
  if [[ -n "$(squeue -h -j "$spot_job" -o '%T' || true)" ]]; then
    echo "FATAL: Spot job $spot_job 未在取消后离开队列，拒绝双投" >&2
    exit 1
  fi
  grpo_job=$(sbatch --parsable --quotatype=reserved --job-name=e09_v5grp4r \
    --gres=gpu:4 --export="$export_spec" \
    "$PROJ/scripts/version5/rl/run_train_grpo_d4.sh")
fi
echo "GRPO submitted: job=$grpo_job parent=$PARENT_REL"

best_eval=$(sbatch --parsable --dependency="afterok:$grpo_job" \
  --quotatype=reserved --job-name=e09_v5gbev \
  --gres=gpu:4 --export="ALL,RUN_NAME=$GRPO_RUN,S3_CKPT_REL=leihaodong/Project4/checkpoints/$GRPO_RUN/best.pt,OUT_ROOT=$PROJ/eval_results/locomo_${GRPO_RUN}_best" \
  "$PROJ/scripts/version5/rl/run_eval_locomo_posttrain.sh")
final_eval=$(sbatch --parsable --dependency="afterok:$grpo_job" \
  --quotatype=reserved --job-name=e09_v5gfev \
  --gres=gpu:4 --export="ALL,RUN_NAME=$GRPO_RUN,S3_CKPT_REL=leihaodong/Project4/checkpoints/$GRPO_RUN/step_0000200.pt,OUT_ROOT=$PROJ/eval_results/locomo_${GRPO_RUN}_final" \
  "$PROJ/scripts/version5/rl/run_eval_locomo_posttrain.sh")

python - "$DECISION_JSON" "$grpo_job" "$best_eval" "$final_eval" \
  "$REFERENCE_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "grpo_job": sys.argv[2],
    "grpo_best_eval_job": sys.argv[3],
    "grpo_final_eval_job": sys.argv[4],
    "grpo_reference_id": sys.argv[5],
})
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
echo "GRPO LoCoMo eval queued: best=$best_eval final=$final_eval"
