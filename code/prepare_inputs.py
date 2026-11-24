"""Builds the input files under sources/: the Lite and test-split verdict matrices and the SWE-bench Pro v1->V2 revision labels.
Usage: python code/prepare_inputs.py <repo_dir> <SWE-bench/experiments clone at 40f164d>. Needs the files listed in sources/README.md under sources/downloads/.  CPU; numpy, pandas, pyarrow."""
import json, os, re, sys
import numpy as np, pandas as pd
root, repo = sys.argv[1:3]; rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def fit_irt(M, two_pl, iters=1200, lr=0.05, lam=0.02):
    n, m = M.shape; th = np.zeros(n); b = np.zeros(m); a = np.ones(m)
    for _ in range(iters):
        R = M - sig(a * (th[:, None] - b))
        th += lr * ((R * a).sum(1) - th * lam * m) / m; b += lr * ((-R * a).sum(0) - lam * n * b) / n
        if two_pl: a += lr * ((R * (th[:, None] - b)).sum(0) - lam * n * (a - 1)) / n
        th -= th.mean()
    return th, b, a
def ranks(s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def auc(s, y):
    r = ranks(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def matched(s, p, y, w=0.05):
    win = tot = 0.0
    for i in np.where(y == 1)[0]:
        neg = np.where((y == 0) & (np.abs(p - p[i]) <= w))[0]
        win += (s[i] > s[neg]).sum() + .5 * (s[i] == s[neg]).sum(); tot += len(neg)
    return win / tot if tot else float("nan")
def boot(fn, y, B=600):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(B):
        i = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); v.append(fn(i))
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]
def pbis(M):
    tot = M.sum(1); out = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        x = M[:, j]; rest = tot - x
        if x.std() > 0 and rest.std() > 0: out[j] = np.corrcoef(x, rest)[0, 1]
    return out
def spearman(a, b):
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float)); return float(np.corrcoef(ra, rb)[0, 1])
def item_screens(M):
    p = M.mean(0); th, b, _ = fit_irt(M, False); P1 = sig(th[:, None] - b); w = P1 * (1 - P1)
    outfit = ((M - P1) ** 2 / np.clip(w, 1e-4, None)).mean(0); infit = ((M - P1) ** 2).sum(0) / np.clip(w.sum(0), 1e-6, None)
    _, _, a2 = fit_irt(M, True)
    return p, {"low_point_biserial": -pbis(M), "low_2pl_discrimination": -a2, "rasch_outfit": outfit, "rasch_infit": infit}
def evaluate(p, sc, y):
    res = {"positives": int(y.sum()), "negatives": int((1 - y).sum()),
           "mean_pass_rate_pos": round(float(p[y == 1].mean()), 3), "mean_pass_rate_neg": round(float(p[y == 0].mean()), 3),
           "low_pass_rate_auroc": round(auc(-p, y), 3), "low_pass_rate_ci95": boot(lambda i: auc(-p[i], y[i]), y), "screens": {}}
    for k, s in sc.items():
        res["screens"][k] = {"auroc": round(auc(s, y), 3), "matched_auroc": round(float(matched(s, p, y)), 3),
                             "matched_ci95": boot(lambda i: matched(s[i], p[i], y[i]), y, 300)}
    return res
def verdicts(split, tasks):
    base = os.path.join(repo, "evaluation", split); rows = []; names = []
    for d in sorted(os.listdir(base)):
        f = os.path.join(base, d, "results", "results.json")
        if not os.path.isfile(f): continue
        r = json.load(open(f)).get("resolved")
        if isinstance(r, list) and r: s = set(r); rows.append([float(t in s) for t in tasks]); names.append(d)
    return names, np.array(rows)

D = os.path.join(root, "sources/downloads"); out = {}
ann = pd.read_csv(os.path.join(root, "sources/labels/openai/ensembled_annotations_public.csv")).set_index("instance_id")
lite_ids = sorted(pd.read_parquet(os.path.join(D, "lite/test-00000-of-00001.parquet")).instance_id)
for split, universe in [("lite", lite_ids), ("test", sorted(ann.index))]:
    names, Mfull = verdicts(split, universe)
    keep = [i for i, t in enumerate(universe) if t in ann.index]; tasks = [universe[i] for i in keep]
    p_all, sc_all = item_screens(Mfull)                       # item statistics use every task in the split
    p = p_all[keep]; sc = {k: v[keep] for k, v in sc_all.items()}; a = ann.loc[tasks]
    r = {"submissions": len(names), "tasks_in_split_matrix": len(universe), "annotated_tasks": len(tasks),
         "never_resolved_annotated": int((p == 0).sum()), "labels": {}}
    for lab, y in [("filter_out (any severe issue)", a.filter_out.astype(bool).values.astype(int)),
                   ("tests_reject_valid_solutions (false_negative>=2)", (a.false_negative >= 2).values.astype(int)),
                   ("underspecified (>=2)", (a.underspecified >= 2).values.astype(int)),
                   ("tests_only (fn>=2 and underspecified<2)", ((a.false_negative >= 2) & (a.underspecified < 2)).values.astype(int))]:
        r["labels"][lab] = evaluate(p, sc, y)
    r["spearman_pass_rate_vs_false_negative_score"] = round(spearman(p, a.false_negative.values), 3)
    r["spearman_pass_rate_vs_underspecified_score"] = round(spearman(p, a.underspecified.values), 3)
    hard = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}
    r["spearman_pass_rate_vs_difficulty"] = round(spearman(p, a.difficulty.map(hard).values), 3)
    out["openai_" + split] = r
    pd.DataFrame(Mfull.astype(int), index=names, columns=universe).to_csv(os.path.join(root, f"sources/data/{split}_verdict_matrix.csv"))
    if split == "lite": Ml, keep_l, y_l = Mfull, keep, a.filter_out.astype(bool).values.astype(int)

# SWE-bench Pro: per-task revision labels from v1 -> V2
v1 = pd.read_parquet(os.path.join(D, "pro/v1/test-00000-of-00001.parquet")).set_index("instance_id")
v2 = pd.read_parquet(os.path.join(D, "pro/default/test-00000-of-00001.parquet")).set_index("instance_id")
def norm(s):
    s = s.lower(); s = s[9:] if s.startswith("instance_") else s
    m = re.match(r"(.*-[0-9a-f]{40})", s); return m.group(1) if m else s
rows = [json.loads(l) for l in open(os.path.join(root, "sources/data/pro_responses_agentpsychometrics.jsonl"))]
k2id = {norm(i): i for i in v1.index}; tasks = [t for t in sorted(rows[0]["responses"]) if norm(t) in k2id]
Mp = np.array([[float(r["responses"].get(t, 0) or 0) for t in tasks] for r in rows]); ids = [k2id[norm(t)] for t in tasks]
common = [i for i in ids if i in v2.index]
lab = pd.DataFrame(index=ids)
lab["dropped"] = [int(i not in v2.index) for i in ids]
for col, name in [("test_patch", "test_patch_revised"), ("patch", "gold_patch_revised"), ("problem_statement", "statement_rewritten"), ("fail_to_pass", "fail_to_pass_list_changed")]:
    lab[name] = [int(i in v2.index and str(v1.loc[i, col]).strip() != str(v2.loc[i, col]).strip()) for i in ids]
lab["dropped_or_tests_or_gold_revised"] = ((lab.dropped + lab.test_patch_revised + lab.gold_patch_revised) > 0).astype(int)
lab.to_csv(os.path.join(root, "sources/labels/pro_v1_to_v2_revision_labels.csv"))
