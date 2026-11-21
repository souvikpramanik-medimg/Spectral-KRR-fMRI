import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_ridge(X_tr, y_tr, X_te, ridge_alphas=RIDGE_ALPHAS):
    return RidgeCV(alphas=ridge_alphas).fit(X_tr, y_tr).predict(X_te)


def fit_logreg(X_tr, y_tr, X_te):
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), 0.5, dtype=np.float32)
    n_pos = int(y_tr.sum())
    n_neg = int(y_tr.size - n_pos)
    cv_folds = max(2, min(3, min(n_pos, n_neg)))
    try:
        m = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=cv_folds,
                                 max_iter=2000, penalty="l2").fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    except Exception:
        return np.full(len(X_te), 0.5, dtype=np.float32)


def metrics_reg(y_true, mu):
    mae = float(np.mean(np.abs(y_true - mu)))
    r = float(pearsonr(y_true, mu)[0]) if np.std(mu) > 1e-8 else 0.0
    return dict(mae=mae, r=r)


def metrics_cls(y_true, proba):
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        auc = 0.5
    return dict(auc=auc)


def select_alpha_nested(cache, idx_train, y_train, alphas, fixed_alpha,
                        ridge_alphas=RIDGE_ALPHAS, cv=3):
    """Choose alpha by nested CV on training subjects only."""
    if len(idx_train) < cv * 2:
        return fixed_alpha
    kf = KFold(cv, shuffle=True, random_state=42)
    best_a, best_score = fixed_alpha, -np.inf
    for a in alphas:
        F = cache[a][idx_train]
        scores = []
        for tr, val in kf.split(F):
            sc = StandardScaler().fit(F[tr])
            try:
                m = RidgeCV(alphas=ridge_alphas).fit(sc.transform(F[tr]), y_train[tr])
                pred = m.predict(sc.transform(F[val]))
                if np.std(pred) > 1e-8:
                    scores.append(pearsonr(y_train[val], pred)[0])
            except Exception:
                pass
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_a = s, a
    return best_a
