"""Shared computations of the shift statistics (scripts/10a_shift_statistics.py): missing
groups, out-of-fold domain-classifier AUC, MMD, and the residual shift of a regressor trained on A.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 2027
N_FOLDS = 5
MMD_MAX_PER_SIDE = 1000
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)


def missing_groups(df, num_cols):
    """Groups of numerical columns with exactly the same missing positions over the whole data. Returns group id -> column list
    (columns without missing values are left out)."""
    miss = df[num_cols].isna()
    cols = [c for c in num_cols if miss[c].any()]
    groups, key_to_id = {}, {}
    for c in cols:
        k = miss[c].to_numpy().tobytes()
        gid = key_to_id.setdefault(k, len(key_to_id))
        groups.setdefault(gid, []).append(c)
    return groups


def oof_auc(X, label, groups, cat_mask):
    """ROC-AUC of out-of-fold predictions over 5 folds stratified and grouped by lot."""
    oof = np.zeros(len(label))
    for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, label, groups):
        clf = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB).fit(X.iloc[tr], label[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(label, oof))


def mmd_rbf(XA, XB, rng):
    """Unbiased MMD^2 (RBF, bandwidth = median distance of the pooled sample)."""
    a = XA[rng.choice(len(XA), min(len(XA), MMD_MAX_PER_SIDE), replace=False)]
    b = XB[rng.choice(len(XB), min(len(XB), MMD_MAX_PER_SIDE), replace=False)]
    z = np.vstack([a, b])
    d2 = (z ** 2).sum(1)[:, None] + (z ** 2).sum(1)[None, :] - 2 * z @ z.T
    d2 = np.maximum(d2, 0)
    sigma2 = np.median(d2[np.triu_indices(len(z), 1)]) or 1.0
    k = np.exp(-d2 / sigma2)
    na = len(a)
    kaa, kbb, kab = k[:na, :na], k[na:, na:], k[:na, na:]
    return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1))
                 - 2 * kab.mean())


def resid_shift(X, y, is_b, groups, cat_mask):
    """Mean residual on B of a regressor trained on the A rows minus the mean lot-level out-of-fold residual within A, and the RMSE ratio."""
    a = np.where(~is_b)[0]
    oof = np.zeros(len(a))
    strat = (y[a] > 0).astype(int)
    for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(a, strat, groups[a]):
        m = HistGradientBoostingRegressor(categorical_features=cat_mask, **HGB).fit(X.iloc[a[tr]], y[a[tr]])
        oof[te] = m.predict(X.iloc[a[te]])
    full = HistGradientBoostingRegressor(categorical_features=cat_mask, **HGB).fit(X.iloc[a], y[a])
    b = np.where(is_b)[0]
    rb, ra = y[b] - full.predict(X.iloc[b]), y[a] - oof
    return float(rb.mean() - ra.mean()), float(np.sqrt((rb ** 2).mean()) / np.sqrt((ra ** 2).mean()))
