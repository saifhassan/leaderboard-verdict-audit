# Input data

All inputs were produced by other teams and are redistributed here unchanged (or, where noted, derived by `code/prepare_inputs.py`) so that the analysis can be rerun. SHA-256 hashes of every input file are in `runs/RUN-20261116-09/manifest.json`. Please cite the original sources and respect their licences.

| File | Content | Source |
|---|---|---|
| `data/lite_verdict_matrix.csv` | 84 submissions × 300 SWE-bench Lite tasks (1 = resolved) | Derived from SWE-bench/experiments, commit 40f164d (`evaluation/lite/*/results/results.json`) |
| `data/test_verdict_matrix.csv` | 24 submissions × 2,294 SWE-bench test-split tasks | Derived from SWE-bench/experiments, commit 40f164d (`evaluation/test`) |
| `data/verified_responses_agentpsychometrics.jsonl` | 134 submissions × 500 SWE-bench Verified tasks | agent-psychometrics (Ge et al., 2026), commit 1c34d34, `data/swebench_verified` |
| `data/pro_responses_agentpsychometrics.jsonl` | 14 agents × 730 SWE-bench Pro tasks | agent-psychometrics, commit 1c34d34, `data/swebench_pro` |
| `labels/openai/ensembled_annotations_public.csv` | Audit A: OpenAI annotations of 1,699 SWE-bench tasks | OpenAI SWE-bench Verified annotation release (`swe-bench-annotation-results.zip`) |
| `labels/agentless/swebench_lite_classifications.csv` | Audit B: Agentless classification of the 300 Lite tasks | OpenAutoCoder/Agentless, commit 5ce5888 |
| `labels/utboost_augTest.json` | Audit C: tasks whose tests accept incorrect patches | UTBoost (Yu et al., ACL 2025), commit 4f32782 |
| `labels/pro_v1_to_v2_revision_labels.csv`, `labels/pro_v2_kept_task_dirs.txt` | Label D: SWE-bench Pro v1 → V2 revisions (dropped, test patch revised, gold patch revised) | Derived from ScaleAI/SWE-bench_Pro v1 and default parquet files |

## Rebuilding the derived files

`code/prepare_inputs.py` rebuilds the two verdict matrices and the Pro revision labels. It needs a clone of SWE-bench/experiments at commit 40f164d and these files under `sources/downloads/` (not included; download them from Hugging Face):

- `lite/test-00000-of-00001.parquet` (princeton-nlp/SWE-bench_Lite)
- `pro/v1/test-00000-of-00001.parquet`, `pro/default/test-00000-of-00001.parquet` (ScaleAI/SWE-bench_Pro)
