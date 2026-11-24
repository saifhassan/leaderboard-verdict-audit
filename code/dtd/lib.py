"""paper3 library: detectors, metrics and resampling used by run_confirmatory.py (protocol v1.0)."""
import hashlib, json, os, re
import numpy as np, pandas as pd
from scipy import stats as sps
from scipy.stats import rankdata

# ---------------------------------------------------------------- metrics
def auroc(score, y):
    """P(score of a positive > score of a negative), ties count half. Higher score = more suspicious."""
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def top_k(score, k):
    return np.argsort(-np.asarray(score), kind="mergesort")[:k]          # stable: ties keep task order

def precision_at(score, y, k):
    k = min(k, len(y)); return float(np.asarray(y)[top_k(score, k)].mean())

def sensitivity_at(score, y, k):
    y = np.asarray(y); k = min(k, len(y)); return float(y[top_k(score, k)].sum() / max(y.sum(), 1))

def strat_boot_idx(y, B, rng):
    y = np.asarray(y); i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    for _ in range(B):
        yield np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])

def ci(vals):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else [None, None]

def cell(score, y, B, rng, ks=(50,)):
    """AUROC, P@k, Sens@k with stratified-bootstrap 95% intervals over tasks."""
    score = np.asarray(score, float); y = np.asarray(y).astype(int); n = len(y)
    out = {"n": n, "positives": int(y.sum()), "base_rate": round(float(y.mean()), 4), "auroc": round(auroc(score, y), 4)}
    kk = sorted(set(list(ks) + [max(1, int(round(0.1 * n)))]))
    for k in kk:
        out[f"precision_at_{k}"] = round(precision_at(score, y, k), 4); out[f"sensitivity_at_{k}"] = round(sensitivity_at(score, y, k), 4)
    out["k_10pct"] = max(1, int(round(0.1 * n)))
    if B:
        au, pk = [], {k: [] for k in kk}
        for idx in strat_boot_idx(y, B, rng):
            au.append(auroc(score[idx], y[idx]))
            for k in kk: pk[k].append(precision_at(score[idx], y[idx], k))
        out["auroc_ci95"] = ci(au)
        for k in kk: out[f"precision_at_{k}_ci95"] = ci(pk[k])
    return out

def paired_diff(score_a, score_b, y, B, rng):
    """AUROC(a) - AUROC(b) on the same tasks, with a paired stratified bootstrap."""
    a = np.asarray(score_a, float); b = np.asarray(score_b, float); y = np.asarray(y).astype(int)
    d = [auroc(a[i], y[i]) - auroc(b[i], y[i]) for i in strat_boot_idx(y, B, rng)]
    lo, hi = ci(d)
    return {"n": len(y), "positives": int(y.sum()), "auroc_a": round(auroc(a, y), 4), "auroc_b": round(auroc(b, y), 4),
            "difference": round(auroc(a, y) - auroc(b, y), 4), "difference_ci95": [lo, hi],
            "equivalent_within_0.05": bool(lo is not None and lo >= -0.05 and hi <= 0.05)}

def matched_auroc(score, p, y, w=0.05):
    score = np.asarray(score, float); win = tot = 0.0
    for i in np.where(y == 1)[0]:
        neg = np.where((y == 0) & (np.abs(p - p[i]) <= w))[0]
        win += (score[i] > score[neg]).sum() + 0.5 * (score[i] == score[neg]).sum(); tot += len(neg)
    return float(win / tot) if tot else float("nan")

def cohen_kappa(a, b):
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int); po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

# ---------------------------------------------------------------- detectors
def fb_stats(X):
    """SRC-0057 statistics as defined in fantastic-bugs cf7f9a8 (src/metrics.py, src/analyze.py) and pyrelimri 2.2.3.
    X: agents x items with zero-variance columns already removed. Returns low-is-suspicious statistics."""
    X = np.asarray(X, float); n, m = X.shape
    S = np.cov(X, rowvar=False); v = np.var(X, axis=0, ddof=1); Sij = np.outer(v, v)
    Zij = np.divide(S * np.sqrt(n - 1), np.sqrt(Sij), out=np.zeros_like(S), where=Sij != 0); np.fill_diagonal(Zij, 0)
    scal = Zij.mean(0)
    tot = X.sum(1); Xc = X - X.mean(0); Tc = tot - tot.mean()
    itc = (Xc * Tc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (Tc ** 2).sum())
    A = (1 - X).T @ (1 - X); Bm = (1 - X).T @ X; C = X.T @ (1 - X); D = X.T @ X
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.cos(np.pi / (1 + np.sqrt(A * D / Bm / C)))
    T[(Bm == 0) | (C == 0)] = np.nan
    T[(Bm == 0) & (C == 0)] = 1.0
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); tet = np.nanmean(T, axis=1)
    out = {"tetrachoric": tet, "scalability": scal, "item_total": itc}
    gr = [sps.norm.ppf(pd.Series(out[k]).rank(method="first").values / (m + 1)) for k in ("tetrachoric", "scalability", "item_total")]
    out["ensemble"] = np.mean(gr, axis=0)
    return out

def corrected_point_biserial(X):
    X = np.asarray(X, float); tot = X.sum(1); out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        x = X[:, j]; rest = tot - x
        if x.std() > 0 and rest.std() > 0: out[j] = np.corrcoef(x, rest)[0, 1]
    return out

_sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def fit_irt(M, two_pl, iters=1500, lr=0.05, lam=0.02):
    """Regularised joint maximum likelihood by gradient ascent (deterministic)."""
    n, m = M.shape; th = np.zeros(n); b = np.zeros(m); a = np.ones(m)
    for _ in range(iters):
        R = M - _sig(a * (th[:, None] - b))
        th += lr * ((R * a).sum(1) - th * lam * m) / m; b += lr * ((-R * a).sum(0) - lam * n * b) / n
        if two_pl: a += lr * ((R * (th[:, None] - b)).sum(0) - lam * n * (a - 1)) / n
        th -= th.mean()
    return th, b, a

def irt_screens(M):
    th, b, _ = fit_irt(M, False); P1 = _sig(th[:, None] - b); w = P1 * (1 - P1)
    outfit = ((M - P1) ** 2 / np.clip(w, 1e-4, None)).mean(0); infit = ((M - P1) ** 2).sum(0) / np.clip(w.sum(0), 1e-6, None)
    _, _, a2 = fit_irt(M, True)
    return {"discrimination_2pl": a2, "rasch_outfit": outfit, "rasch_infit": infit}

def all_detectors(M, with_irt=True):
    """Returns (pass_rate, scorable mask, dict of suspicion scores). Every score: higher = more suspicious.
    SRC-0057 statistics and point-biserial are NaN on zero-variance tasks (dropped by the authors' pipeline)."""
    M = np.asarray(M, float); p = M.mean(0); var = (p > 0) & (p < 1); m = M.shape[1]
    det = {"low_pass_rate": -p, "high_pass_rate": p}
    st = fb_stats(M[:, var])
    for k, v in st.items():
        full = np.full(m, np.nan); full[var] = -v; det[f"low_{k}"] = full
    pb = np.full(m, np.nan); pb[var] = -corrected_point_biserial(M[:, var]); det["low_point_biserial"] = pb
    if with_irt:
        s = irt_screens(M); det["low_discrimination_2pl"] = -s["discrimination_2pl"]; det["high_rasch_outfit"] = s["rasch_outfit"]; det["high_rasch_infit"] = s["rasch_infit"]
    return p, var, det

FB = ["low_tetrachoric", "low_scalability", "low_item_total", "low_ensemble"]

# ---------------------------------------------------------------- data
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def repo_of(task): return re.sub(r"-\d+$", "", task)

def family_of(name):
    s = name.lower()
    for fam, keys in [("claude", ["claude", "sonnet", "opus", "haiku"]), ("gpt", ["gpt", "o1", "o3", "o4", "openai", "codex"]), ("gemini", ["gemini"]),
                      ("deepseek", ["deepseek"]), ("qwen", ["qwen", "lingma"]), ("llama", ["llama"]), ("other_named", ["kimi", "glm", "grok", "mistral", "devstral", "nova", "doubao"])]:
        if any(k in s for k in keys): return fam
    return "unknown"

class Data:
    def __init__(self, root):
        self.root = root; j = lambda *a: os.path.join(root, *a)
        self.files = {"lite_verdicts": j("sources/data/lite_verdict_matrix.csv"), "test_verdicts": j("sources/data/test_verdict_matrix.csv"),
                      "verified_verdicts": j("sources/data/verified_responses_agentpsychometrics.jsonl"), "pro_verdicts": j("sources/data/pro_responses_agentpsychometrics.jsonl"),
                      "labels_A_openai": j("sources/labels/openai/ensembled_annotations_public.csv"), "labels_B_agentless": j("sources/labels/agentless/swebench_lite_classifications.csv"),
                      "labels_C_utboost": j("sources/labels/utboost_augTest.json"), "labels_D_pro": j("sources/labels/pro_v1_to_v2_revision_labels.csv")}
        self.ann = pd.read_csv(self.files["labels_A_openai"]).set_index("instance_id")
        self.cls = pd.read_csv(self.files["labels_B_agentless"]).set_index("instance_id")
        self.ut = set(json.load(open(self.files["labels_C_utboost"])))
        self.pro_lab = pd.read_csv(self.files["labels_D_pro"], index_col=0)
    def hashes(self): return {k: sha256(v) for k, v in self.files.items()}
    def matrix(self, split):
        df = pd.read_csv(self.files[f"{split}_verdicts"], index_col=0); return list(df.index), list(df.columns), df.values.astype(float)
    def jsonl(self, key):
        rows = [json.loads(l) for l in open(self.files[key])]; tasks = sorted(rows[0]["responses"])
        return [r["subject_id"] for r in rows], tasks, np.array([[float(r["responses"].get(t, 0) or 0) for t in tasks] for r in rows])
    def labels_A(self, tasks):
        a = self.ann.loc[tasks]
        return {"A1_filter_out": a.filter_out.astype(bool).values.astype(int), "A2_tests_reject_valid": (a.false_negative >= 2).values.astype(int),
                "A3_underspecified": (a.underspecified >= 2).values.astype(int), "A4_tests_only": ((a.false_negative >= 2) & (a.underspecified < 2)).values.astype(int)}
    def labels_B(self, tasks):
        c = self.cls.loc[tasks]; ne = (c["description quality"] == "Not enough info").values; mi = (c["solution in description"] == "Misleading").values
        return {"B1_not_enough_info": ne.astype(int), "B2_misleading": mi.astype(int), "B3_not_enough_or_misleading": (ne | mi).astype(int),
                "B4_exact_patch_leak": (c["solution in description"] == "Exact patch").values.astype(int)}
    def pro(self):
        names, tasks, M = self.jsonl("pro_verdicts")
        def norm(s):
            s = s.lower(); s = s[9:] if s.startswith("instance_") else s
            m = re.match(r"(.*-[0-9a-f]{40})", s); return m.group(1) if m else s
        k2 = {norm(i): i for i in self.pro_lab.index}; keep = [i for i, t in enumerate(tasks) if norm(t) in k2]
        lab = self.pro_lab.loc[[k2[norm(tasks[i])] for i in keep]]
        return names, [tasks[i] for i in keep], M[:, keep], {"D1_dropped": lab.dropped.values, "D2_test_patch_revised": lab.test_patch_revised.values,
                "D3_gold_patch_revised": lab.gold_patch_revised.values, "D4_any": lab.dropped_or_tests_or_gold_revised.values}
