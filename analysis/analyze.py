"""paper3 results analysis. Reads only run outputs; writes analysis/results-summary.csv, tables/*.tex, figures/*.pdf (+png previews).
Usage: analyze.py <paper3_dir> [run_id]"""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
ROOT = sys.argv[1]; RUN = sys.argv[2] if len(sys.argv) > 2 else "RUN-20261116-09"
R = os.path.join(ROOT, "runs", RUN, "results"); A = os.path.join(ROOT, "analysis"); T = os.path.join(A, "tables"); F = os.path.join(A, "figures")
L = lambda e: json.load(open(os.path.join(R, f"{e}.json")))
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN, RED = "#2b6cb0", "#dd6b20", "#718096", "#2f855a", "#c53030"
rows = []
def add(metric, condition, value, lo=None, hi=None, n=None, label="Measured", src=RUN, exp=""):
    rows.append({"metric": metric, "condition": condition, "value": value, "ci_low": lo, "ci_high": hi, "n": n, "label": label, "run_id_or_source": src, "experiment": exp})
def addcell(exp, cond, c):
    add("AUROC", cond, c["auroc"], *(c.get("auroc_ci95") or [None, None]), c["n"], exp=exp)
    for k in [x for x in c if x.startswith("precision_at_") and not x.endswith("ci95")]:
        add("P@" + k.split("_")[-1], cond, c[k], *(c.get(k + "_ci95") or [None, None]), c["n"], exp=exp)

# ------------------------------------------------------------------ 1. independent recomputation from per-task outputs
checks = []; pt = {s: pd.read_csv(os.path.join(R, f"per_task_{s}.csv")) for s in ["lite_A", "test_A", "lite_B", "verified_C"]}
e1 = {**L("EXP-01"), **L("EXP-03")}
for s, df in pt.items():
    labs = [c for c in df.columns if c[:2] in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1")]
    for lab in labs:
        y = df[lab].values
        for det, key, mask in [("low_pass_rate", "low_pass_rate | all tasks", np.ones(len(df), bool)), ("low_ensemble", "low_ensemble | scorable tasks", df.scorable.values.astype(bool)),
                               ("low_tetrachoric", "low_tetrachoric | scorable tasks", df.scorable.values.astype(bool))]:
            mine = roc_auc_score(y[mask], df[det].values[mask]); rep = e1[s][lab][key]["auroc"]
            checks.append({"set": s, "label": lab, "detector": key, "recomputed": round(mine, 4), "reported": rep, "n": int(mask.sum()), "n_reported": e1[s][lab][key]["n"], "match": bool(abs(mine - rep) < 6e-5 and int(mask.sum()) == e1[s][lab][key]["n"])})
chk = pd.DataFrame(checks); chk.to_csv(os.path.join(A, "recomputation-check.csv"), index=False)
man = json.load(open(os.path.join(ROOT, "runs", RUN, "manifest.json")))
print("recomputation:", int(chk.match.sum()), "of", len(chk), "cells match; experiments:", {k: v["status"] for k, v in man["experiments"].items() if v["status"] != "completed"} or "all completed")

# ------------------------------------------------------------------ 2. results-summary.csv
for s in e1:
    for lab, r in e1[s].items():
        add("defective share among never-resolved tasks", f"{s}:{lab}", r["_defective_share_never_resolved"], n=r["_never_resolved"], exp="EXP-01/03")
        for k, c in r.items():
            if not k.startswith("_"): addcell("EXP-03" if s == "lite_B" else "EXP-01", f"{s}:{lab}:{k}", c)
for s, r in L("EXP-05").items():
    for part in ["a_common_scorable_tasks", "b_all_tasks_dropped_not_flagged"]:
        for k, c in r[part].items(): add("paired AUROC difference", f"{s}:{part}:{k}", c["difference"], *c["difference_ci95"], c["n"], exp="EXP-05")
for s, curve in L("EXP-02").items():
    for c in curve:
        for k, v in c.items():
            if isinstance(v, dict): add("mean AUROC over subsets (5th-95th percentile)", f"{s}:agents={c['agents']}:{k}", v["auroc"]["mean"], v["auroc"]["p05"], v["auroc"]["p95"], c["subsets"], exp="EXP-02")
for s, r in L("EXP-04").items():
    for d, c in r.items():
        add("Spearman with pass rate", f"{s}:{d}", c["spearman_with_pass_rate"], n=c["raw"]["n"], exp="EXP-04"); addcell("EXP-04", f"{s}:{d}:residual_isotonic", c["residual_isotonic"])
for s, r in L("EXP-07").items():
    for lab, rr in r.items():
        for b, c in rr["within_difficulty"].items():
            if "auroc" in c: addcell("EXP-07", f"{s}:{lab}:difficulty={b}", c)
        lc = rr["logistic_coefficient_standardised_pass_rate"]; add("logistic coefficient (standardised pass rate)", f"{s}:{lab}", lc["estimate"], *lc["ci95"], exp="EXP-07")
for s, r in L("EXP-08").items():
    add("repository-stratified AUROC", s, r["repository_stratified_auroc"], exp="EXP-08"); add("pooled AUROC, repository-cluster bootstrap", s, r["pooled_auroc"], *r["pooled_auroc_ci95_repository_cluster_bootstrap"], exp="EXP-08")
e9 = L("EXP-09")
for d, c in e9["detectors"].items(): add("pass-rate-matched AUROC", f"verified_C:C1:{d}", c["pass_rate_matched_auroc"], *c["pass_rate_matched_ci95"], c["n"], exp="EXP-09")
e11 = L("EXP-11")
for k, c in e11["pairs"].items(): add("Cohen kappa", k, c["cohen_kappa"], *c["ci95"], e11["tasks"], exp="EXP-11")
for k, c in e11["audit_as_detector"].items(): addcell("EXP-11", k, c)
e6 = L("EXP-06")
for s, r in e6.items(): add("effective number of agents (participation ratio)", s, r["effective_agents_participation_ratio"], n=r["agents"], exp="EXP-06")
e12 = L("EXP-12")
for lab, r in e12["labels"].items():
    for k, c in r.items():
        if isinstance(c, dict): addcell("EXP-12", f"pro:{lab}:{k}", c)
pd.DataFrame(rows).to_csv(os.path.join(A, "results-summary.csv"), index=False)

# ------------------------------------------------------------------ 3. tables
def ci(c): return f"{c['auroc']:.3f} [{c['auroc_ci95'][0]:.3f}, {c['auroc_ci95'][1]:.3f}]"
def tex(name, header, body, caption, label, note=""):
    cols = "l" * 1 + "r" * (len(header) - 1) if not isinstance(header, tuple) else header[1]; hd = header if not isinstance(header, tuple) else header[0]
    s = "%% Generated by analysis/analyze.py from runs/%s. Do not edit by hand.\n\\begin{table}[t]\n\\centering\\small\n\\caption{%s}\\label{%s}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n%s\\end{table}\n" % (
        RUN, caption, label, cols, " & ".join(hd), "\n".join(" & ".join(str(x) for x in r) + " \\\\" if r != "MID" else "\\midrule" for r in body), (f"\\\\[2pt]{{\\footnotesize {note}}}\n" if note else ""))
    open(os.path.join(T, name), "w").write(s)
NICE = {"A1_filter_out": "Any severe issue (OpenAI)", "A2_tests_reject_valid": "Tests reject valid solutions (OpenAI)", "A3_underspecified": "Underspecified (OpenAI)", "A4_tests_only": "Tests only (OpenAI)",
        "B1_not_enough_info": "Not enough information (Agentless)", "B2_misleading": "Misleading solution (Agentless)", "B3_not_enough_or_misleading": "Not enough info.\\ or misleading (Agentless)",
        "B4_exact_patch_leak": "Exact patch in description (Agentless)", "C1_weak_tests": "Weak tests (UTBoost)"}
SETN = {"lite_A": "Lite", "test_A": "Test", "lite_B": "Lite", "verified_C": "Verified"}
# Table 1: data
body = []
for s, df in pt.items():
    for lab in [c for c in df.columns if c in NICE]:
        body.append([SETN[s], man["denominators"][s]["agents"], NICE[lab], len(df), int(df[lab].sum()), f"{df[lab].mean():.3f}", int((~df.scorable.astype(bool)).sum()), int((df.pass_rate == 0).sum())])
    body.append("MID")
tex("tab_data.tex", (["Benchmark", "Agents", "Label", "Tasks", "Positive", "Base rate", "No variance", "Never solved"], "lrlrrrrr"), body[:-1],
    "Verdict matrices and audit labels. ``No variance'' counts tasks that every agent solves or every agent fails; the response-pattern statistics are undefined for them.", "tab:data")
# Table 2: main
body = []; order = [("lite_A", "A1_filter_out"), ("lite_A", "A2_tests_reject_valid"), ("lite_A", "A3_underspecified"), ("test_A", "A1_filter_out"), ("test_A", "A2_tests_reject_valid"), ("test_A", "A3_underspecified"),
                    ("lite_B", "B1_not_enough_info"), ("lite_B", "B3_not_enough_or_misleading")]
for s, lab in order:
    r = e1[s][lab]; body.append([SETN[s], NICE[lab], ci(r["low_pass_rate | all tasks"]), ci(r["low_pass_rate | scorable tasks"]), ci(r["low_tetrachoric | scorable tasks"]), ci(r["low_scalability | scorable tasks"]), ci(r["low_item_total | scorable tasks"]), ci(r["low_ensemble | scorable tasks"])])
tex("tab_main.tex", (["", "Label", "Pass rate (all)", "Pass rate", "Tetrachoric", "Scalability", "Item-total", "Ensemble"], "llrrrrrr"), body,
    "AUROC for detecting audited-defective tasks, with 95\\% stratified bootstrap intervals (2{,}000 resamples over tasks). The first score column ranks every labelled task; the remaining columns use only tasks with variance, the set on which the response-pattern statistics are defined.", "tab:main")
# Table 3: paired
e5 = L("EXP-05"); body = []
for s, r in e5.items():
    setn, lab = s.split(":")
    for d in ["low_tetrachoric", "low_scalability", "low_item_total", "low_ensemble"]:
        a_, b_ = r["a_common_scorable_tasks"][f"{d} minus low_pass_rate"], r["b_all_tasks_dropped_not_flagged"][f"{d} minus low_pass_rate"]
        f_ = lambda c: f"{c['difference']:+.3f} [{c['difference_ci95'][0]:+.3f}, {c['difference_ci95'][1]:+.3f}]"
        body.append([SETN[setn], NICE[lab], d[4:].replace("_", "-"), f_(a_), "yes" if a_["equivalent_within_0.05"] else "no", f_(b_)])
    body.append("MID")
tex("tab_paired.tex", (["", "Label", "Statistic", "Difference, common tasks", "Within $\\pm0.05$", "Difference, all tasks"], "lllrcr"), body[:-1],
    "Paired AUROC difference (statistic minus pass rate) with 95\\% paired bootstrap intervals. ``All tasks'' ranks the tasks that the response-pattern pipeline drops as not flagged, which is the pipeline's behaviour.", "tab:paired")
# Table 4: agents
e2 = L("EXP-02"); body = []
for s, curve in e2.items():
    for c in curve:
        g = lambda k: f"{c[k]['auroc']['mean']:.3f}"
        body.append([NICE[s.split(":")[1]], c["agents"], f"{c['mean_zero_variance_tasks']:.1f}", g("low_pass_rate | all tasks"), g("low_pass_rate | scorable tasks"), g("low_tetrachoric | scorable tasks"), g("low_item_total | scorable tasks"), g("low_ensemble | scorable tasks"), g("low_ensemble | all tasks, dropped = not flagged")])
    body.append("MID")
tex("tab_agents.tex", (["Label", "Agents", "No-variance tasks", "Pass rate (all)", "Pass rate", "Tetrachoric", "Item-total", "Ensemble", "Ensemble (all)"], "lrrrrrrrr"), body[:-1],
    "Mean AUROC over 200 random subsets of SWE-bench Lite submissions (one subset for all 84).", "tab:agents")
# Table 5: difficulty + repository
e7 = L("EXP-07"); e8 = L("EXP-08"); body = []
for s in ["lite_A", "test_A"]:
    for lab in ["A1_filter_out", "A2_tests_reject_valid", "A3_underspecified"]:
        rr = e7[s][lab]; w = rr["within_difficulty"]; cell = lambda b: ci(w[b]) if "auroc" in w[b] else "--"
        lc = rr["logistic_coefficient_standardised_pass_rate"]
        body.append([SETN[s], NICE[lab], cell("<15 min fix"), cell("15 min - 1 hour"), cell("1-4 hours"), f"{lc['estimate']:.2f} [{lc['ci95'][0]:.2f}, {lc['ci95'][1]:.2f}]"])
tex("tab_difficulty.tex", (["", "Label", "$<$15 min", "15 min--1 h", "1--4 h", "Logistic coefficient"], "llrrrr"), body,
    "Low pass rate as a detector within annotated-difficulty buckets of at least 100 tasks (AUROC with 95\\% intervals), and the coefficient of standardised pass rate in a logistic regression that also includes the difficulty bucket.", "tab:difficulty",
    f"Repository-stratified AUROC for the any-severe-issue label: Lite {e8['lite_A:A1_filter_out']['repository_stratified_auroc']:.3f}, test {e8['test_A:A1_filter_out']['repository_stratified_auroc']:.3f}; repository-cluster bootstrap intervals [{e8['lite_A:A1_filter_out']['pooled_auroc_ci95_repository_cluster_bootstrap'][0]:.3f}, {e8['lite_A:A1_filter_out']['pooled_auroc_ci95_repository_cluster_bootstrap'][1]:.3f}] and [{e8['test_A:A1_filter_out']['pooled_auroc_ci95_repository_cluster_bootstrap'][0]:.3f}, {e8['test_A:A1_filter_out']['pooled_auroc_ci95_repository_cluster_bootstrap'][1]:.3f}].")
# Table 6: other defect types
body = []
r = e1["lite_B"]["B4_exact_patch_leak"]; body.append(["Lite", NICE["B4_exact_patch_leak"], 13, ci(r["high_pass_rate | all tasks"]), ci(r["low_pass_rate | all tasks"]), ci(r["low_ensemble | scorable tasks"]), "--"])
r = e1["verified_C"]["C1_weak_tests"]; d9 = e9["detectors"]
body.append(["Verified", NICE["C1_weak_tests"], 26, ci(r["high_pass_rate | all tasks"]), ci(r["low_pass_rate | all tasks"]), ci(r["low_ensemble | scorable tasks"]), f"{d9['low_ensemble']['pass_rate_matched_auroc']:.3f} [{d9['low_ensemble']['pass_rate_matched_ci95'][0]:.3f}, {d9['low_ensemble']['pass_rate_matched_ci95'][1]:.3f}]"])
tex("tab_other_defects.tex", (["", "Label", "Positive", "High pass rate", "Low pass rate", "Ensemble", "Ensemble, pass-rate matched"], "llrrrrr"), body,
    "Defect types expected at the high end of the pass-rate axis (AUROC with 95\\% intervals).", "tab:otherdefects")

# ------------------------------------------------------------------ 4. figures
def savefig(fig, name):
    d = F
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(d, name + ".png"), dpi=220, bbox_inches="tight"); plt.close(fig)
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return c - h, c + h
# Fig A: defective share along the pass-rate axis
edges = [(-1, 0), (0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0)]; names = ["0", "(0,.1]", "(.1,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]"]
fig, axs = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True)
for ax, (s, lab, ttl, col) in zip(axs, [("lite_A", "A1_filter_out", "Lite, OpenAI: any severe issue", BLUE), ("test_A", "A1_filter_out", "Test split, OpenAI: any severe issue", BLUE), ("lite_B", "B3_not_enough_or_misleading", "Lite, Agentless labels", GREEN)]):
    df = pt[s]; bins = []
    series = [(lab, col, "defective")] + ([("B4_exact_patch_leak", RED, "exact patch in description"), ] if s == "lite_B" else [])
    for j, (lb, c_, nm) in enumerate(series):
        xs, ys, lo, hi, ns = [], [], [], [], []
        for i, (a, b) in enumerate(edges):
            m = (df.pass_rate > a) & (df.pass_rate <= b); n = int(m.sum()); k = int(df.loc[m, lb].sum())
            if n: l_, h_ = wilson(k, n); xs.append(i + (j - (len(series) - 1) / 2) * 0.22); ys.append(k / n); lo.append(max(0.0, k / n - l_)); hi.append(max(0.0, h_ - k / n)); ns.append(n)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-" if j == 0 else "s--", color=c_, ms=3, lw=1, capsize=2, label=nm if s == "lite_B" else None)
        if j == 0:
            nlab = ns
        if j == 0: ax.axhline(df[lb].mean(), color=c_, lw=.6, ls=":")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{a}\nn={b}" for a, b in zip(names, nlab)], fontsize=5.5); ax.set_title(ttl, fontsize=7); ax.set_ylim(0, 1.02); ax.set_xlabel("task pass rate (bin)")
    if s == "lite_B": ax.legend(fontsize=5.5, frameon=False, loc="upper center")
axs[0].set_ylabel("share of tasks with the label")
savefig(fig, "fig_defect_share_by_pass_rate")
# Fig B: agent-count curve
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
ser = [("low_pass_rate | all tasks", BLUE, "-", "o", "Pass rate (all tasks)"), ("low_pass_rate | scorable tasks", BLUE, "--", "o", "Pass rate (tasks with variance)"), ("low_tetrachoric | scorable tasks", ORANGE, "-", "s", "Tetrachoric"),
       ("low_item_total | scorable tasks", ORANGE, "--", "^", "Item-total"), ("low_ensemble | scorable tasks", ORANGE, ":", "D", "Ensemble"), ("low_ensemble | all tasks, dropped = not flagged", GREY, "-", "x", "Ensemble (all tasks)")]
for ax, (s, curve) in zip(axs, e2.items()):
    x = [c["agents"] for c in curve]
    for k, col, ls, mk, nm in ser:
        m = [c[k]["auroc"]["mean"] for c in curve]; ax.plot(x, m, ls, color=col, marker=mk, ms=3, lw=1, label=nm)
        if k in ("low_pass_rate | all tasks", "low_tetrachoric | scorable tasks"): ax.fill_between(x, [c[k]["auroc"]["p05"] for c in curve], [c[k]["auroc"]["p95"] for c in curve], color=col, alpha=.12, lw=0)
    ax.axhline(.5, color=GREY, lw=.5, ls=":"); ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.minorticks_off(); ax.set_xlabel("number of agents"); ax.set_title(NICE[s.split(":")[1]].replace("\\ ", " "), fontsize=7)
axs[0].set_ylabel("AUROC"); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2)); fig.subplots_adjust(wspace=0.08)
savefig(fig, "fig_agent_count")
# Fig C: forest plot
fig, ax = plt.subplots(figsize=(7, 3.3)); yy = 0; ticks = []; lbls = []
for s, lab in order + [("lite_B", "B4_exact_patch_leak"), ("verified_C", "C1_weak_tests")]:
    r = e1[s][lab]; hi_tail = lab in ("B4_exact_patch_leak", "C1_weak_tests")
    items = [("high_pass_rate | all tasks" if hi_tail else "low_pass_rate | all tasks", BLUE, "o"), ("low_tetrachoric | scorable tasks", ORANGE, "s"), ("low_ensemble | scorable tasks", ORANGE, "D")]
    for j, (k, col, mk) in enumerate(items):
        c = r[k]; y_ = yy + (j - 1) * 0.25; ax.errorbar(c["auroc"], y_, xerr=[[c["auroc"] - c["auroc_ci95"][0]], [c["auroc_ci95"][1] - c["auroc"]]], fmt=mk, color=col, ms=3.5, lw=1, capsize=2, mfc=("white" if mk == "D" else col))
    ticks.append(yy); lbls.append(f"{SETN[s]}: {NICE[lab]}".replace("\\ ", " ") + (" [high tail]" if hi_tail else "")); yy -= 1
ax.axvline(.5, color=GREY, lw=.6, ls=":"); ax.axvline(.6, color=GREY, lw=.6, ls="--"); ax.set_yticks(ticks); ax.set_yticklabels(lbls, fontsize=6.3); ax.set_xlabel("AUROC (95% bootstrap interval)"); ax.set_xlim(0.0, 1.0)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], marker="o", color=BLUE, lw=0, ms=4), Line2D([], [], marker="s", color=ORANGE, lw=0, ms=4), Line2D([], [], marker="D", color=ORANGE, mfc="white", lw=0, ms=4)], ["Pass rate, all tasks", "Tetrachoric, tasks with variance", "Ensemble, tasks with variance"], fontsize=6, frameon=False, loc="upper left")
ax.text(0.602, ticks[0] + 0.55, "detection bar 0.60", fontsize=5.5, color=GREY)
savefig(fig, "fig_forest")
# Fig D: statistics track pass rate
fig, axs = plt.subplots(1, 3, figsize=(7, 2.2)); fig.subplots_adjust(wspace=0.42); df = pt["lite_A"]; m = df.scorable.astype(bool); e4 = L("EXP-04")["lite_A:A1_filter_out"]
for ax, (d, nm) in zip(axs, [("low_tetrachoric", "Mean tetrachoric correlation"), ("low_scalability", "Scalability"), ("low_item_total", "Item-total correlation")]):
    y = df.loc[m, "A1_filter_out"].values; ax.scatter(df.loc[m, "pass_rate"][y == 0], -df.loc[m, d][y == 0], s=6, color=GREY, alpha=.7, lw=0, label="clean"); ax.scatter(df.loc[m, "pass_rate"][y == 1], -df.loc[m, d][y == 1], s=6, color=RED, alpha=.7, lw=0, label="defective (OpenAI)")
    ax.set_xlabel("task pass rate"); ax.set_ylabel(nm); ax.set_title(f"Spearman {e4[d]['spearman_with_pass_rate']*-1:+.2f}", fontsize=7)
axs[0].legend(fontsize=5.5, frameon=False)
savefig(fig, "fig_statistics_vs_pass_rate")
print("tables:", sorted(os.listdir(T)), "\nfigures:", sorted(f for f in os.listdir(F) if f.endswith(".pdf")), "\nrows in results-summary:", len(rows))
