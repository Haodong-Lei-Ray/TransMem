#!/usr/bin/env python3
"""诊断: 逐题对比 transmem 与 student 的 LoCoMo 预测, 定位 TransMem 到底做了什么.

回答三个问题 (决定改进方向):
  1) TransMem 改了多少题的答案 (vs student)? —— 偏置是"近似恒等"还是"实质改写"?
  2) 改了的题里, 帮忙 (f1↑) 多还是帮倒忙 (f1↓) 多? —— 偏置方向对不对?
  3) 逐类别的净增益 —— 哪类最受伤 (multi_hop? open_domain?).

用法:
  python scripts/eval/diagnose_transmem_vs_student.py \
    --transmem eval_outputs/locomo_offpolicy_v2_short128_forward_kl/locomo_transmem.transmem.progress.jsonl \
    --student   eval_outputs/locomo_offpolicy_short128_forward_kl_student/locomo_student.student.progress.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict

CAT = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transmem", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--show", type=int, default=8, help="每类展示几个改动样例")
    a = p.parse_args()

    tm, st = load(a.transmem), load(a.student)
    keys = sorted(set(tm) & set(st))
    print(f"共同题数: {len(keys)}  (transmem {len(tm)} / student {len(st)})")

    n_same_pred = n_diff = 0
    help_n = hurt_n = tie_n = 0
    cat_tm, cat_st, cat_cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    d_help = defaultdict(int); d_hurt = defaultdict(int); d_diff = defaultdict(int)
    examples_help, examples_hurt = [], []

    for k in keys:
        t, s = tm[k], st[k]
        c = int(t["category"])
        cat_tm[c] += t["score"]; cat_st[c] += s["score"]; cat_cnt[c] += 1
        same = t["prediction"].strip().lower() == s["prediction"].strip().lower()
        if same:
            n_same_pred += 1
            continue
        n_diff += 1; d_diff[c] += 1
        dt = t["score"] - s["score"]
        if dt > 1e-9:
            help_n += 1; d_help[c] += 1
            if len(examples_help) < a.show:
                examples_help.append((k, c, t, s, dt))
        elif dt < -1e-9:
            hurt_n += 1; d_hurt[c] += 1
            if len(examples_hurt) < a.show:
                examples_hurt.append((k, c, t, s, dt))
        else:
            tie_n += 1

    print("=" * 78)
    print(f"预测相同: {n_same_pred}/{len(keys)} ({n_same_pred/max(len(keys),1)*100:.1f}%) "
          f"—— 偏置在这些题上≈恒等/无效")
    print(f"预测不同: {n_diff}  其中  帮忙(f1↑) {help_n} | 帮倒忙(f1↓) {hurt_n} | 打平 {tie_n}")
    print(f"净: {help_n - hurt_n:+d} 题 (>0 说明 TransMem 有效, <0 说明总体有害)")
    print("-" * 78)
    print(f"{'cat':12s} {'n':>5s} {'student':>9s} {'transmem':>9s} {'Δoverall':>9s} "
          f"{'#changed':>9s} {'help':>5s} {'hurt':>5s}")
    tot_s = tot_t = tot_n = 0
    for c in sorted(cat_cnt):
        n = cat_cnt[c]; s_avg = cat_st[c]/n; t_avg = cat_tm[c]/n
        tot_s += cat_st[c]; tot_t += cat_tm[c]; tot_n += n
        print(f"{CAT.get(c,'?'):12s} {n:5d} {s_avg:9.4f} {t_avg:9.4f} "
              f"{t_avg-s_avg:+9.4f} {d_diff[c]:9d} {d_help[c]:5d} {d_hurt[c]:5d}")
    print(f"{'OVERALL':12s} {tot_n:5d} {tot_s/tot_n:9.4f} {tot_t/tot_n:9.4f} "
          f"{(tot_t-tot_s)/tot_n:+9.4f}")
    print("=" * 78)

    def dump(title, ex):
        print(f"\n### {title} ###")
        for k, c, t, s, dt in ex:
            print(f"[{CAT.get(c,'?')}] {k}  Δf1={dt:+.3f}")
            print(f"    Q   : {t['question'][:100]}")
            print(f"    gold: {t['answer']!r}")
            print(f"    stu : {s['prediction'][:80]!r} (f1={s['score']:.3f})")
            print(f"    mem : {t['prediction'][:80]!r} (f1={t['score']:.3f})")
    dump("TransMem 帮忙的样例", examples_help)
    dump("TransMem 帮倒忙的样例", examples_hurt)


if __name__ == "__main__":
    main()
