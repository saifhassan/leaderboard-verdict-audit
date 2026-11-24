"""Verbatim copy of three functions from github.com/sangttruong/fantastic-bugs, commit cf7f9a8, src/metrics.py (MIT License,
Copyright (c) 2025 Sang Truong, Yuheng Tu, Michael Hardy, Anka Reuel, Zeyu Tang, and co-authors; see the repository LICENSE).
Used only as the reference in the EXP-00 fidelity check. The tqdm progress bar import is replaced by a no-op."""
import numpy as np
import pandas as pd
from typing import Dict, Union, Any, List
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
def tqdm(x, **k): return x

def scalability_coefs(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compute item-level scalability coefficients (Hi and Zi) using simplified approach,
    which does not include standard errors or confidence intervals.
    (Loevinger, 1948; Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)

    This function computes:
    - Hi: Item-level H coefficients (scalability of each item with rest of scale)
    - Zi: Item-level Z-scores (standardized Hi coefficients)
    - H: Overall scale H coefficient (scalar)
    - Z: Overall scale Z-score (scalar)
    - Hij: Item-pair H coefficients (matrix of shape (n_items, n_items))
    - Zij: Item-pair Z-scores (matrix of shape (n_items, n_items))

    Parameters
    ----------
    X : array-like of shape (n_subjects, n_items)
        Data matrix containing item responses. Should be integer-valued.
        Missing values are handled by listwise deletion.

    Returns
    -------
    dict
        Dictionary containing:
        - 'Hi': Item-level H coefficients (array of length n_items)
        - 'Zi': Item-level Z-scores (array of length n_items)
        - 'H': Overall scale H coefficient (scalar)
        - 'Z': Overall scale Z-score (scalar)
        - 'Hij': Item-pair H coefficients (matrix of shape (n_items, n_items))
        - 'Zij': Item-pair Z-scores (matrix of shape (n_items, n_items))

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(0, 4, (100, 5))
    >>> result = scalability_coefs(X)
    >>> print(f"Item coefficients: {result['Hi']}")
    >>> print(f"Overall coefficient: {result['H']:.3f}")
    """
    # Convert input to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)

    # Convert to integers
    X = X.astype(int)

    # Validate input
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 items")
    if X.shape[0] < 5:
        raise ValueError("X must have at least 5 subjects")

    n_subjects, n_items = X.shape

    # Compute H scaling (Loevinger, 1948; Mokken, 1971) using simple method
    # Compute covariance matrices
    S = np.cov(X, rowvar=False)  # Item covariance matrix
    X_sorted = np.sort(X, axis=0)  # Sort each item independently
    Smax = np.cov(X_sorted, rowvar=False)  # Maximum possible covariance

    # Compute Hij matrix (item-pair coefficients)
    Hij = S / Smax
    np.fill_diagonal(Hij, 0)  # Zero out diagonal

    # Compute Hi coefficients (item-level)
    S_offdiag = S.copy()
    Smax_offdiag = Smax.copy()
    np.fill_diagonal(S_offdiag, 0)
    np.fill_diagonal(Smax_offdiag, 0)

    Hij = np.divide(S_offdiag, Smax_offdiag,
                   out=np.zeros_like(S_offdiag), where=Smax_offdiag != 0)
    Hi = np.sum(Hij, axis=1)

    # Compute overall H coefficient
    H = np.sum(S_offdiag) / np.sum(Smax_offdiag)

    # Compute Z-standardized scaling using simple method
    # (Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)
    # Only appropriate for testing lowerbound = 0.
    var_vec = np.var(X, axis=0, ddof=1)  # Item variances, unweighted and unbiased
    Sij = np.outer(var_vec, var_vec)  # Outer product of variances

    # Item-pair Z-standardized scaling coefficients
    Zij = np.divide(S * np.sqrt(n_subjects - 1), np.sqrt(Sij),
                   out=np.zeros_like(S_offdiag), where=Sij != 0)
    np.fill_diagonal(Zij, 0)  # Zero diagonal

    # Item-level Z-standardized scaling
    Sij_for_z = Sij.copy()
    np.fill_diagonal(Sij_for_z, 0)

    Zi = np.divide(np.sum(S_offdiag, axis=1) * np.sqrt(n_subjects - 1),
                  np.sqrt(np.sum(Sij_for_z, axis=1)),
                  out=np.zeros(n_items), where=np.sum(Sij_for_z, axis=1) != 0)

    # Overall Z-standardized scaling (divided by 2 because the matrix is symmetric)
    sum_S = np.sum(S_offdiag) / 2.0
    sum_Sij = np.sum(Sij_for_z) / 2.0
    Z = (sum_S * np.sqrt(n_subjects - 1)) / np.sqrt(sum_Sij) if sum_Sij != 0 else 0.0

    return {
        'Hi': Hi,
        'Zi': Zi,
        'H': H,
        'Z': Z,
        'Hij': Hij,
        'Zij': Zij
    }


def calculate_tetrachoric(data: np.ndarray, show_progress: bool = True) -> np.ndarray:
    """
    Calculate mean tetrachoric correlation for each item.

    Tetrachoric correlation estimates the correlation between two latent
    continuous variables from their observed binary (0/1) values.

    Parameters
    ----------
    data : np.ndarray of shape (n_subjects, n_items)
        Binary response matrix (0/1 values).
    show_progress : bool, default=True
        Whether to show a progress bar.

    Returns
    -------
    np.ndarray of shape (n_items,)
        Mean tetrachoric correlation of each item with all other items.

    Examples
    --------
    >>> data = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1]])
    >>> corrs = calculate_tetrachoric(data, show_progress=False)
    >>> corrs.shape
    (3,)
    """
    n_questions = data.shape[1]
    corr_matrix = np.zeros((n_questions, n_questions))

    iterator = range(n_questions)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing tetrachoric correlations")

    for i in iterator:
        for j in range(i, n_questions):
            r = tetrachoric_corr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = r

    return np.nanmean(corr_matrix, axis=1)


def calculate_item_total_corr(X: np.ndarray) -> List[float]:
    """
    Calculate item-total correlation for each item.

    Item-total correlation measures how well each item correlates with
    the total score (sum of all items). Higher values indicate items
    that are more consistent with the overall scale.

    Parameters
    ----------
    X : np.ndarray of shape (n_subjects, n_items)
        Response matrix.

    Returns
    -------
    list of float
        Item-total correlation for each item.

    Examples
    --------
    >>> X = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0]])
    >>> corrs = calculate_item_total_corr(X)
    >>> len(corrs)
    3
    """
    total = X.sum(axis=1)
    Xc = X - X.mean(axis=0)
    Tc = total - total.mean()
    numer = (Xc * Tc[:, None]).sum(axis=0)
    denom = np.sqrt((Xc**2).sum(axis=0) * (Tc**2).sum())
    raw_r = numer / denom
    return raw_r.tolist()
