"""Recompute the manual-review table (Table: Manual review) from the two raters' answer files.

Usage: python manual_review/manual_review.py
Rater 1 rated all 49 distinct tasks; rater 2 rated a stratified subset of 20 tasks.
task_key.csv maps each instance to its sampling group(s) and to its row number in each rater's file.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
Q = {"Q1": "Q1_description_sufficient", "Q2": "Q2_tests_reject_valid_fix", "Q3": "Q3_fix_given_in_description"}


def load(path):
    df = pd.read_csv(path)
    ren = {c: k for c in df.columns for k, v in Q.items() if c.startswith(v)}
    df = df.rename(columns=ren)
    for k in Q:
        df[k] = df[k].astype(str).str.strip().str.lower()
    return df.set_index("instance_id")


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(a, b, n=1000, seed=20261116):
    """95% percentile interval from a bootstrap over tasks (1,000 resamples, fixed seed)."""
    rng = np.random.default_rng(seed)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.nanpercentile(np.array(ks, float), [2.5, 97.5])


r1 = load(os.path.join(HERE, "rater1", "2_FILL_answers.csv"))
r2 = load(os.path.join(HERE, "rater2", "2_FILL_answers.csv"))
key = pd.read_csv(os.path.join(HERE, "task_key.csv")).set_index("instance_id")

# Top panel: rater 1 by sampling group (a task in two groups is counted in both)
rows = []
for g in ["G1", "G2", "G3"]:
    ids = [i for i in key.index if g in str(key.loc[i, "groups"])]
    d = r1.loc[ids]
    rows.append({"group": g, "tasks": len(d),
                 "Q1_not_yes": int((d.Q1 != "yes").sum()),
                 "Q2_yes": int((d.Q2 == "yes").sum()),
                 "Q3_yes": int((d.Q3 == "yes").sum()),
                 "Q3_partly": int((d.Q3 == "partly").sum())})
print("Rater 1 by group\n", pd.DataFrame(rows).to_string(index=False))

# Bottom panel: agreement on the shared tasks
shared = [i for i in r2.index if i in r1.index]
print(f"\nShared tasks: {len(shared)}")
for k in Q:
    a, b = r1.loc[shared, k].values, r2.loc[shared, k].values
    same = int((a == b).sum())
    kp = kappa(a, b)
    # no interval when one rater used a single category or the raters agree on every task
    if len(set(a)) > 1 and len(set(b)) > 1 and same < len(shared):
        lo, hi = boot_ci(a, b)
        ci = f"[{lo:.2f}, {hi:.2f}]"
    else:
        ci = "--"
    print(f"{k}: same {same}/{len(shared)}, kappa {kp:.2f}, 95% interval {ci}")
