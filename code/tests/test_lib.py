import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dtd import lib
def test_auroc_basic():
    assert lib.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and lib.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert lib.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5
def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 300); s = np.round(rng.normal(size=300) + y, 1)
    assert abs(lib.auroc(s, y) - roc_auc_score(y, s)) < 1e-12
def test_precision_sensitivity():
    y = np.array([1, 0, 1, 0, 0]); s = np.array([5, 4, 3, 2, 1])
    assert lib.precision_at(s, y, 2) == 0.5 and lib.sensitivity_at(s, y, 3) == 1.0
def test_kappa():
    assert lib.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0 and abs(lib.cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-12
def test_fb_stats_match_reference():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))
    import fantastic_bugs_metrics as ref
    rng = np.random.default_rng(1); th = rng.normal(size=60); b = rng.normal(size=40)
    X = (rng.random((60, 40)) < 1 / (1 + np.exp(-(th[:, None] - b)))).astype(int); X = X[:, X.var(0) > 0]
    mine = lib.fb_stats(X)
    np.testing.assert_allclose(mine["scalability"], ref.scalability_coefs(X)["Zij"].mean(0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mine["item_total"], np.array(ref.calculate_item_total_corr(X)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mine["tetrachoric"], ref.calculate_tetrachoric(X, show_progress=False), rtol=1e-10, atol=1e-12)
def test_zero_variance_is_nan():
    X = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)
    p, var, det = lib.all_detectors(X, with_irt=False)
    assert list(var) == [False, False, True, True] and np.isnan(det["low_ensemble"][0]) and not np.isnan(det["low_pass_rate"][0])
