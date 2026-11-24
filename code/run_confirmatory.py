"""Main analysis run (EXP-00 to EXP-12). Usage: run_confirmatory.py <paper3_dir> <run_id> [--boot 2000] [--subsets 200] [--only EXP-01,EXP-05]"""
import argparse, json, os, platform, sys, time, subprocess, datetime
import numpy as np, pandas as pd, scipy, sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtd import lib
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("run_id"); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--subsets", type=int, default=200); ap.add_argument("--only", default=""); ap.add_argument("--seed", type=int, default=20261116)
a = ap.parse_args(); ROOT, B = a.root, a.boot
OUT = os.path.join(ROOT, "runs", a.run_id); os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
LOG = open(os.path.join(OUT, "run.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); LOG.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {s}\n"); LOG.flush()
D = lib.Data(ROOT); status = {}
def rng_for(exp): return np.random.default_rng([a.seed, int(exp.split("-")[1])])
def save(exp, obj):
    path = os.path.join(OUT, "results", f"{exp}.json")
    json.dump(obj, open(path, "w"), indent=1, default=float); return path

lite_names, lite_tasks, Ml_full = D.matrix("lite")
ann_idx = [i for i, t in enumerate(lite_tasks) if t in D.ann.index]; lite_ann_tasks = [lite_tasks[i] for i in ann_idx]; Ml_ann = Ml_full[:, ann_idx]
test_names, test_tasks, Mt = D.matrix("test")
ver_names, ver_tasks, Mv = D.jsonl("verified_verdicts"); yC = np.array([t in D.ut for t in ver_tasks], int)
SETS = {  # name -> (matrix, tasks, labels)   Statistics are computed on the labelled task set of each label family.
    "lite_A": (Ml_ann, lite_ann_tasks, D.labels_A(lite_ann_tasks)), "test_A": (Mt, test_tasks, D.labels_A(test_tasks)),
    "lite_B": (Ml_full, lite_tasks, D.labels_B(lite_tasks)), "verified_C": (Mv, ver_tasks, {"C1_weak_tests": yC})}
DET = {k: lib.all_detectors(v[0]) for k, v in SETS.items()}
def direction(label, det):  # leak label uses the high tail of pass rate only
    return True
def score_cells(setname, labels, dets, rng, ks=(50,)):
    M, tasks, labs = SETS[setname]; p, var, det = DET[setname]; res = {}
    for lab in labels:
        y = labs[lab]; r = {"_tasks": len(y), "_positives": int(y.sum()), "_zero_variance": int((~var).sum()),
                            "_positives_among_zero_variance": int(y[~var].sum()), "_never_resolved": int((p == 0).sum()),
                            "_defective_share_never_resolved": (round(float(y[p == 0].mean()), 4) if (p == 0).any() else None)}
        for d in dets:
            s = det[d]; ok = ~np.isnan(s)
            r[d + " | all tasks" if ok.all() else d + " | scorable tasks"] = lib.cell(s[ok], y[ok], B, rng, ks)
        r["low_pass_rate | scorable tasks"] = lib.cell(det["low_pass_rate"][var], y[var], B, rng, ks)
        res[lab] = r
    return res
ALLD = ["low_pass_rate", "high_pass_rate"] + lib.FB + ["low_point_biserial", "low_discrimination_2pl", "high_rasch_outfit", "high_rasch_infit"]

def exp00():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")); import fantastic_bugs_metrics as ref
    rng = rng_for("EXP-00"); out = {"reference": "verbatim copy of fantastic-bugs cf7f9a8 src/metrics.py functions", "checks": []}
    mats = {"lite_A_scorable": Ml_ann[:, DET["lite_A"][1]], "lite_B_scorable": Ml_full[:, DET["lite_B"][1]]}
    th = rng.normal(size=60); b = rng.normal(size=40); X = (rng.random((60, 40)) < 1 / (1 + np.exp(-(th[:, None] - b)))).astype(float); mats["synthetic_60x40"] = X[:, X.var(0) > 0]
    for name, X in mats.items():
        mine = lib.fb_stats(X); Xi = X.astype(int)
        d = {"scalability": float(np.nanmax(np.abs(mine["scalability"] - ref.scalability_coefs(Xi)["Zij"].mean(0)))),
             "item_total": float(np.nanmax(np.abs(mine["item_total"] - np.array(ref.calculate_item_total_corr(Xi))))),
             "tetrachoric": float(np.nanmax(np.abs(mine["tetrachoric"] - ref.calculate_tetrachoric(Xi, show_progress=False))))}
        out["checks"].append({"matrix": name, "shape": list(X.shape), "max_abs_difference": d, "pass": bool(max(d.values()) < 1e-9)})
    t = time.time(); lib.cell(DET["test_A"][2]["low_pass_rate"], SETS["test_A"][2]["A1_filter_out"], 200, rng); out["seconds_per_2000_resample_cell_test_split"] = round((time.time() - t) * 10, 2)
    out["all_pass"] = all(c["pass"] for c in out["checks"]); save("EXP-00", out); return out["all_pass"]
def exp01():
    rng = rng_for("EXP-01"); save("EXP-01", {"lite_A": score_cells("lite_A", list(SETS["lite_A"][2]), ALLD, rng), "test_A": score_cells("test_A", list(SETS["test_A"][2]), ALLD, rng),
                                             "verified_C": score_cells("verified_C", ["C1_weak_tests"], ALLD, rng)})
def exp02():
    rng = rng_for("EXP-02"); out = {}
    for setname, lab in [("lite_A", "A1_filter_out"), ("lite_B", "B3_not_enough_or_misleading")]:
        M, tasks, labs = SETS[setname]; y = labs[lab]; n = M.shape[0]; curve = []
        for k in [5, 10, 20, 30, 40, 60, n]:
            reps = 1 if k == n else a.subsets; acc = {}; zv = []
            for _ in range(reps):
                idx = np.arange(n) if k == n else rng.choice(n, k, replace=False); p, var, det = lib.all_detectors(M[idx], with_irt=False); zv.append(int((~var).sum()))
                vals = {"low_pass_rate | all tasks": (det["low_pass_rate"], np.ones(len(y), bool)), "low_pass_rate | scorable tasks": (det["low_pass_rate"], var)}
                for d in lib.FB: vals[d + " | scorable tasks"] = (det[d], var)
                # ensemble over all tasks: tasks the pipeline drops are ranked as not flagged
                e = det["low_ensemble"].copy(); e[~var] = -np.inf; vals["low_ensemble | all tasks, dropped = not flagged"] = (e, np.ones(len(y), bool))
                for name, (s, mk) in vals.items():
                    acc.setdefault(name, {"auroc": [], "p50": []}); acc[name]["auroc"].append(lib.auroc(s[mk], y[mk])); acc[name]["p50"].append(lib.precision_at(s[mk], y[mk], 50))
            curve.append({"agents": int(k), "subsets": reps, "mean_zero_variance_tasks": round(float(np.mean(zv)), 2),
                          **{nme: {m_: {"mean": round(float(np.nanmean(v)), 4), "p05": round(float(np.nanpercentile(v, 5)), 4), "p95": round(float(np.nanpercentile(v, 95)), 4)} for m_, v in dct.items()} for nme, dct in acc.items()}})
        out[f"{setname}:{lab}"] = curve
    save("EXP-02", out)
def exp03():
    rng = rng_for("EXP-03"); save("EXP-03", {"lite_B": score_cells("lite_B", list(SETS["lite_B"][2]), ALLD, rng)})
def exp04():
    rng = rng_for("EXP-04"); out = {}
    for setname in ["lite_A", "test_A"]:
        M, tasks, labs = SETS[setname]; p, var, det = DET[setname]; y = labs["A1_filter_out"][var]; pv = p[var]; r = {}
        for d in lib.FB + ["low_point_biserial", "low_discrimination_2pl"]:
            s = det[d][var]; iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(pv, s); res_iso = s - iso.predict(pv)
            q = np.polyfit(pv, s, 2); res_q = s - np.polyval(q, pv)
            r[d] = {"spearman_with_pass_rate": round(float(pd.Series(s).corr(pd.Series(pv), method="spearman")), 4), "r2_isotonic": round(float(1 - res_iso.var() / s.var()), 4),
                    "raw": lib.cell(s, y, B, rng), "residual_isotonic": lib.cell(res_iso, y, B, rng), "residual_quadratic": lib.cell(res_q, y, B, rng)}
        out[setname + ":A1_filter_out"] = r
    save("EXP-04", out)
def exp05():
    rng = rng_for("EXP-05"); out = {}
    for setname, lab in [("lite_A", "A1_filter_out"), ("test_A", "A1_filter_out"), ("lite_B", "B3_not_enough_or_misleading")]:
        M, tasks, labs = SETS[setname]; p, var, det = DET[setname]; y = labs[lab]; r = {"a_common_scorable_tasks": {}, "b_all_tasks_dropped_not_flagged": {}}
        for d in lib.FB:
            r["a_common_scorable_tasks"][f"{d} minus low_pass_rate"] = lib.paired_diff(det[d][var], det["low_pass_rate"][var], y[var], B, rng)
            s = det[d].copy(); s[~var] = -np.inf
            r["b_all_tasks_dropped_not_flagged"][f"{d} minus low_pass_rate"] = lib.paired_diff(s, det["low_pass_rate"], y, B, rng)
        r["dropped_tasks"] = int((~var).sum()); r["defective_share_among_dropped"] = round(float(y[~var].mean()), 4) if (~var).any() else None
        out[f"{setname}:{lab}"] = r
    save("EXP-05", out)
def exp06():
    rng = rng_for("EXP-06"); out = {}
    for setname, lab, names in [("lite_A", "A1_filter_out", lite_names), ("verified_C", "C1_weak_tests", ver_names)]:
        M, tasks, labs = SETS[setname]; y = labs[lab]; r = {}
        C = np.corrcoef(M[M.std(1) > 0]); ev = np.clip(np.linalg.eigvalsh(C), 0, None); r["agents"] = int(M.shape[0]); r["effective_agents_participation_ratio"] = round(float(ev.sum() ** 2 / (ev ** 2).sum()), 2)
        fam = np.array([lib.family_of(n) for n in names]); r["family_counts"] = {f: int((fam == f).sum()) for f in sorted(set(fam))}
        def quick(idx):
            p, var, det = lib.all_detectors(M[idx], with_irt=False); o = {"agents": int(len(idx)), "zero_variance": int((~var).sum()), "low_pass_rate | all tasks": round(lib.auroc(det["low_pass_rate"], y), 4),
                 "low_pass_rate | scorable tasks": round(lib.auroc(det["low_pass_rate"][var], y[var]), 4)}
            for d in lib.FB: o[d + " | scorable tasks"] = round(lib.auroc(det[d][var], y[var]), 4)
            return o
        r["leave_one_family_out"] = {f: quick(np.where(fam != f)[0]) for f in sorted(set(fam)) if (fam != f).sum() >= 10 and (fam == f).sum() >= 3}
        if setname == "lite_A":
            order = np.argsort(names); h = len(names) // 2
            r["dated_halves"] = {"earliest": {"range": [names[order[0]][:8], names[order[h - 1]][:8]], **quick(order[:h])}, "latest": {"range": [names[order[h]][:8], names[order[-1]][:8]], **quick(order[h:])}}
            ps = [M[order[i:i + 21]].mean(0) for i in range(0, 84, 21)]
            r["pass_rate_by_date_quarter"] = [{"submissions": [names[order[i]][:8], names[order[min(i + 20, 83)]][:8]], "mean_pass_rate_defective": round(float(q[y == 1].mean()), 4), "mean_pass_rate_clean": round(float(q[y == 0].mean()), 4)} for i, q in zip(range(0, 84, 21), ps)]
        out[f"{setname}:{lab}"] = r
    save("EXP-06", out)
def exp07():
    rng = rng_for("EXP-07"); out = {}; order = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}
    for setname in ["lite_A", "test_A"]:
        M, tasks, labs = SETS[setname]; p, var, det = DET[setname]; diff = D.ann.loc[tasks].difficulty.values; r = {}
        for lab in ["A1_filter_out", "A2_tests_reject_valid", "A3_underspecified"]:
            y = labs[lab]; rr = {"overall": lib.cell(-p, y, B, rng), "within_difficulty": {}}
            for b in order:
                mk = diff == b
                if mk.sum() >= 100 and 5 <= y[mk].sum() <= mk.sum() - 5: rr["within_difficulty"][b] = lib.cell(-p[mk], y[mk], B, rng)
                else: rr["within_difficulty"][b] = {"n": int(mk.sum()), "positives": int(y[mk].sum()), "note": "below 100 tasks: not analysed (protocol H8)"}
            Xd = np.column_stack([(p - p.mean()) / p.std()] + [(diff == b).astype(float) for b in list(order)[1:]])
            def coef(idx):
                return float(LogisticRegression(C=1e6, max_iter=2000).fit(Xd[idx], y[idx]).coef_[0][0])
            bs = [coef(i) for i in lib.strat_boot_idx(y, min(B, 500), rng)]
            rr["logistic_coefficient_standardised_pass_rate"] = {"estimate": round(coef(np.arange(len(y))), 4), "ci95": lib.ci(bs), "bootstrap": min(B, 500), "note": "negative = lower pass rate, higher odds of the label, adjusted for difficulty bucket"}
            r[lab] = rr
        out[setname] = r
    save("EXP-07", out)
def exp08():
    rng = rng_for("EXP-08"); out = {}
    for setname in ["lite_A", "test_A"]:
        M, tasks, labs = SETS[setname]; p, var, det = DET[setname]; y = labs["A1_filter_out"]; repo = np.array([lib.repo_of(t) for t in tasks]); r = {"per_repository": {}}
        num = den = 0.0
        for g in sorted(set(repo)):
            mk = repo == g; n1 = int(y[mk].sum()); n0 = int(mk.sum() - n1)
            au = lib.auroc(-p[mk], y[mk]) if n1 and n0 else None
            r["per_repository"][g] = {"n": int(mk.sum()), "positives": n1, "mean_pass_rate": round(float(p[mk].mean()), 4), "auroc": (round(au, 4) if au is not None else None)}
            if au is not None: num += au * n1 * n0; den += n1 * n0
        r["repository_stratified_auroc"] = round(num / den, 4); r["pooled_auroc"] = round(lib.auroc(-p, y), 4)
        groups = sorted(set(repo)); bs = []
        for _ in range(B):
            pick = rng.choice(len(groups), len(groups)); idx = np.concatenate([np.where(repo == groups[g])[0] for g in pick]); bs.append(lib.auroc(-p[idx], y[idx]))
        r["pooled_auroc_ci95_repository_cluster_bootstrap"] = lib.ci(bs); r["pooled_auroc_ci95_task_bootstrap"] = lib.cell(-p, y, B, rng)["auroc_ci95"]
        out[setname + ":A1_filter_out"] = r
    save("EXP-08", out)
def exp09():
    rng = rng_for("EXP-09"); M, tasks, labs = SETS["verified_C"]; p, var, det = DET["verified_C"]; y = labs["C1_weak_tests"]; r = {"positives": int(y.sum()), "tasks": len(y), "mean_pass_rate_pos": round(float(p[y == 1].mean()), 4), "mean_pass_rate_neg": round(float(p[y == 0].mean()), 4), "detectors": {}}
    for d in ALLD:
        s = det[d]; ok = ~np.isnan(s); ss, pp, yy = s[ok], p[ok], y[ok]
        bs = [lib.matched_auroc(ss[i], pp[i], yy[i]) for i in lib.strat_boot_idx(yy, min(B, 1000), rng)]
        r["detectors"][d] = {"n": int(ok.sum()), "auroc": round(lib.auroc(ss, yy), 4), "auroc_ci95": lib.cell(ss, yy, B, rng)["auroc_ci95"], "pass_rate_matched_auroc": round(lib.matched_auroc(ss, pp, yy), 4), "pass_rate_matched_ci95": lib.ci(bs)}
    save("EXP-09", r)
def exp11():
    rng = rng_for("EXP-11"); tasks = lite_ann_tasks; A = D.labels_A(tasks); Bl = D.labels_B(tasks); ann = D.ann.loc[tasks]; out = {"tasks": len(tasks), "pairs": {}}
    for la, lb in [("A3_underspecified", "B3_not_enough_or_misleading"), ("A3_underspecified", "B1_not_enough_info"), ("A1_filter_out", "B3_not_enough_or_misleading")]:
        x, z = A[la], Bl[lb]; ks = []
        for _ in range(B):
            i = rng.choice(len(x), len(x)); ks.append(lib.cohen_kappa(x[i], z[i]))
        out["pairs"][f"{la} vs {lb}"] = {"positives_a": int(x.sum()), "positives_b": int(z.sum()), "both": int((x & z).sum()), "cohen_kappa": round(lib.cohen_kappa(x, z), 4), "ci95": lib.ci(ks)}
    out["audit_as_detector"] = {"OpenAI underspecified score (0-3) predicting B3": lib.cell(ann.underspecified.values, Bl["B3_not_enough_or_misleading"], B, rng),
                                "OpenAI underspecified score (0-3) predicting B1": lib.cell(ann.underspecified.values, Bl["B1_not_enough_info"], B, rng),
                                "Agentless B3 (0/1) predicting A3": lib.cell(Bl["B3_not_enough_or_misleading"].astype(float), A["A3_underspecified"], B, rng),
                                "low pass rate predicting B3 (same 231 tasks)": lib.cell(-Ml_ann.mean(0), Bl["B3_not_enough_or_misleading"], B, rng),
                                "low pass rate predicting A3 (same 231 tasks)": lib.cell(-Ml_ann.mean(0), A["A3_underspecified"], B, rng)}
    save("EXP-11", out)
def exp12():
    rng = rng_for("EXP-12"); names, tasks, Mp, labs = D.pro(); p, var, det = lib.all_detectors(Mp); out = {"agents": int(Mp.shape[0]), "tasks": len(tasks), "zero_variance": int((~var).sum()), "labels": {}}
    for lab, y in labs.items():
        y = np.asarray(y).astype(int); r = {"positives": int(y.sum())}
        for d in ALLD:
            s = det[d]; ok = ~np.isnan(s); r[d + (" | all tasks" if ok.all() else " | scorable tasks")] = lib.cell(s[ok], y[ok], B, rng)
        out["labels"][lab] = r
    save("EXP-12", out)

EXPS = {"EXP-00": exp00, "EXP-01": exp01, "EXP-02": exp02, "EXP-03": exp03, "EXP-04": exp04, "EXP-05": exp05, "EXP-06": exp06, "EXP-07": exp07, "EXP-08": exp08, "EXP-09": exp09, "EXP-11": exp11, "EXP-12": exp12}
only = [e for e in a.only.split(",") if e] or list(EXPS)
start = datetime.datetime.now().isoformat(timespec="seconds"); log("start", a.run_id, "boot", B, "subsets", a.subsets, "only", only)
man_path = os.path.join(OUT, "manifest.json"); prev = json.load(open(man_path)) if os.path.exists(man_path) else {"experiments": {}}
for e in only:
    t = time.time()
    try:
        ok = EXPS[e](); st = "failed fidelity check" if (e == "EXP-00" and ok is False) else "completed"
    except Exception as ex:
        import traceback; st = f"error: {type(ex).__name__}: {ex}"; LOG.write(traceback.format_exc()); LOG.flush()
    prev["experiments"][e] = {"status": st, "seconds": round(time.time() - t, 1), "finished": datetime.datetime.now().isoformat(timespec="seconds")}; log(e, st, round(time.time() - t, 1), "s")
    if e == "EXP-00" and st != "completed": log("stopping: EXP-00 did not pass"); break
    prev.update({"run_id": a.run_id, "protocol": "plan/protocol.md v1.0", "command": " ".join(sys.argv), "seed": a.seed, "bootstrap_resamples": B, "subsets": a.subsets, "last_started": start,
                 "python": platform.python_version(), "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
                 "inputs_sha256": D.hashes(), "code_sha256": {f: lib.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ["run_confirmatory.py", "dtd/lib.py", "third_party/fantastic_bugs_metrics.py"]},
                 "sources": {"SWE-bench/experiments": "40f164d", "agent-psychometrics": "1c34d34", "Agentless": "5ce5888", "UTBoost": "4f32782", "fantastic-bugs": "cf7f9a8", "pyrelimri": "2.2.3"},
                 "denominators": {k: {"agents": int(v[0].shape[0]), "tasks": int(v[0].shape[1])} for k, v in SETS.items()}})
    json.dump(prev, open(man_path, "w"), indent=1)
# per-task detector scores for traceability
for k, (M, tasks, labs) in SETS.items():
    p, var, det = DET[k]; df = pd.DataFrame({"task": tasks, "pass_rate": p, "scorable": var, **{d: v for d, v in det.items()}, **labs}); df.to_csv(os.path.join(OUT, "results", f"per_task_{k}.csv"), index=False)
log("done")
