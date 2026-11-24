# Can Leaderboard Verdicts Find Defective Tasks?

Code, data and results for the paper

> **Can Leaderboard Verdicts Find Defective Tasks? Testing Response-Pattern Statistics Against Independent Audits of Agentic Coding Benchmarks**
> Saif Hassan, Khalid Hussain, Umair Khan, Muhammad Arslan

We test whether public pass/fail verdicts from coding-agent leaderboards identify tasks that independent audits found defective (over-strict tests, insufficient descriptions, leaked patches, weak tests). We compare the plain task pass rate with response-pattern statistics proposed for static benchmarks (item-total correlation, scalability, mean tetrachoric correlation and their ensemble) on SWE-bench Lite, the SWE-bench test split, SWE-bench Verified and SWE-bench Pro.

## Main results

All values are AUROC with 95% stratified bootstrap intervals (2,000 resamples, seed 20261116), from `runs/RUN-20261116-09`.

| Finding | Value |
|---|---|
| Low pass rate, Lite, any severe issue (OpenAI audit, 231 tasks) | 0.746 [0.682, 0.808] |
| Low pass rate, test split, tests reject valid solutions (1,699 tasks) | 0.758 [0.735, 0.781] |
| Low pass rate, Lite, not enough information (Agentless, 300 tasks) | 0.825 [0.757, 0.884] |
| High pass rate, Lite, exact patch in description (13 tasks) | 0.915 [0.809, 0.986] |
| Response-pattern statistics vs. pass rate over all labelled tasks | pass rate higher in 12 of 12 cells |
| Defective share among never-solved tasks (Lite / test) | 0.885 (26 tasks) / 0.893 (643 tasks) |
| Agents needed: pass rate at 10 agents vs. all 84 (Lite) | 0.727 vs. 0.746 |
| Agreement of two audits on the same 231 tasks (Cohen's κ) | 0.181 [0.061, 0.311] |
| Weak-test tasks (UTBoost, Verified), best detector | 0.661 [0.570, 0.745]; not detected |

Every number reported in the paper's main tables (Tables 1–7) is in `analysis/results-summary.csv` (803 rows), each with its interval, denominator and source experiment.

## Repository layout

```
code/
  run_confirmatory.py       main analysis run, experiments EXP-00 to EXP-12
  dtd/lib.py                detectors, AUROC, bootstrap, IRT/Rasch fits, data loading
  tests/test_lib.py         unit tests, incl. fidelity check against the reference implementation
  third_party/              verbatim reference functions from fantastic-bugs (MIT), used only for the fidelity check
  prepare_inputs.py         builds the Lite/test verdict matrices and the Pro revision labels from the public sources
analysis/
  analyze.py                builds results-summary.csv, the LaTeX tables and the figures from the run outputs
  results-summary.csv       every reported value with interval, n and source
  recomputation-check.csv   39 headline values recomputed independently from the per-task files (39/39 match)
  tables/                   LaTeX tables (numbers identical to the paper; the paper's versions differ only in layout)
  figures/                  figures as PDF and PNG
runs/RUN-20261116-09/
  manifest.json             command, seeds, package versions, input and code SHA-256 hashes, denominators
  run.log                   run log
  results/EXP-xx.json       all cells per experiment
  results/per_task_*.csv    every detector score and audit label for every task
manual_review/
  manual_review.py          recomputes the manual-review table (counts and Cohen's κ)
  sampling_record.csv       the 50 sampled rows (49 distinct tasks) and their sampling groups
  task_key.csv              task-to-group and task-to-row mapping
  rater1/, rater2/          instructions, task packet, answers and time log of each rater
sources/
  data/, labels/            input verdict matrices and audit labels (see sources/README.md)
```

## Experiments

| ID | What it measures |
|---|---|
| EXP-00 | Fidelity of our statistics against the reference implementation (max. abs. difference 1.1e-16) |
| EXP-01 | Main detection results: AUROC and P@50 for every detector and label |
| EXP-02 | Agent-count curves (5 to 84 agents, 200 random subsets each) |
| EXP-03 | Agentless labels on Lite |
| EXP-04 | Dependence of the statistics on pass rate; residual after an isotonic fit |
| EXP-05 | Paired differences statistic − pass rate, on common tasks and over all tasks |
| EXP-06 | Non-independent agents (effective number of agents, leave-one-family-out) and time |
| EXP-07 | Difficulty control (within annotated difficulty buckets, logistic regression) |
| EXP-08 | Repository mix (repository-stratified AUROC, repository-cluster bootstrap) |
| EXP-09 | Weak-test tasks on Verified (UTBoost), with pass-rate-matched AUROC |
| EXP-11 | Agreement between the two audits (Cohen's κ) and one audit as a detector of the other |
| EXP-12 | SWE-bench Pro v1 → V2 revisions, 14 agents |

## Reproduce

CPU only, no GPU and no model API. The full run takes about 2 minutes on 2 CPU cores.

```bash
pip install -r requirements.txt

cd code
python -m pytest -q tests/test_lib.py                       # 6 tests
python run_confirmatory.py .. MYRUN --boot 2000 --subsets 200   # writes runs/MYRUN/
cd ..
python analysis/analyze.py . RUN-20261116-09                 # tables, figures, results-summary.csv
python manual_review/manual_review.py                        # manual-review table
```

Seeds are fixed per experiment (base seed 20261116), so a rerun reproduces the stored results. The original run used Python 3.10.12. On Python 3.13 we obtained identical results for EXP-01, 04, 05, 07, 08, 09, 11 and 12 and identical per-task detector scores to within 3e-15. A few percentile cells of the agent-count curves (EXP-02) and of the Rasch outfit (EXP-03) differed in the third or fourth decimal, because of floating-point ties in the scalability and ensemble statistics. To run one experiment, add `--only EXP-05`.

The original run was launched in two parts (EXP-00 to EXP-09, then EXP-11 and EXP-12) with the same code and seeds; see `runs/RUN-20261116-09/run.log`. Appendix A of the paper is not part of this repository.

## Manual review

Rater 1 (first author) rated all 49 distinct tasks; rater 2 (second author) independently rated a stratified subset of 20 tasks on a separate computer. Pass rates, sampling groups and audit labels were hidden from both. Each rater's packet, answers and time log are in `manual_review/`. The review was not used to change any label or metric.

## Data

All inputs are public and come from other teams. Sources, commits and licences are listed in [`sources/README.md`](sources/README.md). Please cite the original sources when you use them.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`). The paper is under review; the citation will be updated on publication.

## Licence

The code in this repository is released under the MIT licence (`LICENSE`). Files under `sources/` and `code/third_party/` keep the licences of their original authors.
