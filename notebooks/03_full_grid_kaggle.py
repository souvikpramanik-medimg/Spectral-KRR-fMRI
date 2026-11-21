# ============================================================
# NOTEBOOK 3 v5 — Age + FIQ + ASD with fixed & adaptive α
# Kaggle: GPU T4 ON, Internet OFF
# Datasets: abide-cc200-asd, abide-mae-ddpm-continuous
# ============================================================
import os, glob, time, zipfile, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

np.random.seed(42)

def find(pat, label):
    hits = glob.glob(pat, recursive=True)
    if not hits: raise FileNotFoundError(f"{label} not found: {pat}")
    return hits[0]

DATA_PATH  = find('/kaggle/input/**/abide_cc200_asd.npz', 'abide_cc200_asd.npz')
Z_PATH     = find('/kaggle/input/**/Z_all.npy', 'Z_all.npy')
SYNTH_PATH = find('/kaggle/input/**/synthetic_latents.npz', 'synthetic_latents.npz')
print(f"Data:  {DATA_PATH}")
print(f"Z:     {Z_PATH}")
print(f"Synth: {SYNTH_PATH}")

d = np.load(DATA_PATH, allow_pickle=True)
X    = np.nan_to_num(d['connectomes'].astype(np.float32), nan=0.0)
site = np.asarray(d['site'], dtype=str)
age  = d['age'].astype(np.float32)
fiq  = d['fiq'].astype(np.float32)
asd  = d['asd'].astype(np.float32)
N, D = X.shape

Z_all = np.load(Z_PATH).astype(np.float32)
syn   = np.load(SYNTH_PATH)
synth_Z    = syn['synth_Z'].astype(np.float32)
synth_ages = syn['synth_ages'].astype(np.float32)

print(f"\nX: {X.shape}")
print(f"  age valid: {(~np.isnan(age)).sum()}")
print(f"  fiq valid: {(~np.isnan(fiq)).sum()}")
print(f"  asd valid: {(~np.isnan(asd)).sum()}  "
      f"(ASD={int(np.nansum(asd==1))}, Ctrl={int(np.nansum(asd==0))})")
assert len(np.unique(asd[~np.isnan(asd)])) == 2, "ASD needs both classes"

# ============================================================
# PRECOMPUTE eigendecompositions + spectral features (all α, all subjects)
# ============================================================
print("\nPrecomputing eigendecompositions...")
t0 = time.time()
TRIU_IDX = np.triu_indices(200, k=1)
eigvals_all = np.zeros((N, 200), dtype=np.float32)
eigvecs_all = np.zeros((N, 200, 200), dtype=np.float32)
for i in range(N):
    M = np.zeros((200, 200), dtype=np.float64)
    M[TRIU_IDX] = X[i]
    M = M + M.T
    w, v = np.linalg.eigh(M)
    eigvals_all[i] = np.clip(w, 1e-6, None).astype(np.float32)
    eigvecs_all[i] = v.astype(np.float32)
print(f"  done ({time.time()-t0:.1f}s)")

def spectral_features(alpha, idx):
    feats = np.zeros((len(idx), len(TRIU_IDX[0])), dtype=np.float32)
    for j, i in enumerate(idx):
        M = (eigvecs_all[i] * (eigvals_all[i] ** alpha)) @ eigvecs_all[i].T
        feats[j] = M[TRIU_IDX]
    return feats

ADAPTIVE_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
FIXED_ALPHA = 0.3

print("\nPrecomputing spectral features for all α (cache)...")
t0 = time.time()
F_CACHE = {}
for a in ADAPTIVE_ALPHAS:
    F_CACHE[a] = spectral_features(a, np.arange(N))
    print(f"  α={a:.1f}: {F_CACHE[a].shape}  ({time.time()-t0:.1f}s)")
assert FIXED_ALPHA in F_CACHE, "FIXED_ALPHA must be in ADAPTIVE_ALPHAS"

# ============================================================
# HELPERS
# ============================================================
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def fit_ridge(X_tr, y_tr, X_te):
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_tr, y_tr)
    return m.predict(X_te)

def fit_logreg(X_tr, y_tr, X_te):
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), 0.5, dtype=np.float32)
    n_pos = int(y_tr.sum()); n_neg = int(y_tr.size - n_pos)
    cv_folds = max(2, min(3, min(n_pos, n_neg)))
    if cv_folds < 2:
        return np.full(len(X_te), 0.5, dtype=np.float32)
    try:
        m = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=cv_folds,
                                  max_iter=2000, penalty='l2').fit(X_tr, y_tr)
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

def select_alpha_nested(idx_train, y_train, cv=3):
    """Nested CV on training subjects only."""
    n = len(idx_train)
    if n < cv * 2:
        return FIXED_ALPHA
    kf = KFold(cv, shuffle=True, random_state=42)
    best_a, best_score = FIXED_ALPHA, -np.inf
    for a in ADAPTIVE_ALPHAS:
        F_tr_all = F_CACHE[a][idx_train]
        scores = []
        for tr, val in kf.split(F_tr_all):
            sc = StandardScaler().fit(F_tr_all[tr])
            try:
                m = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(F_tr_all[tr]), y_train[tr])
                pred = m.predict(sc.transform(F_tr_all[val]))
                if np.std(pred) > 1e-8:
                    scores.append(pearsonr(y_train[val], pred)[0])
            except Exception:
                pass
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_a = s, a
    return best_a

# ============================================================
# TARGET LOOP
# ============================================================
TARGETS = {
    'age': dict(y=age, valid=~np.isnan(age), task='reg'),
    'fiq': dict(y=fiq, valid=~np.isnan(fiq), task='reg'),
    'asd': dict(y=asd, valid=~np.isnan(asd), task='cls'),
}
N_VALUES = [10, 15, 25, 50, 75, 100]
SEEDS = [0, 1, 2, 3, 4]
all_rows = []
t_start = time.time()

for tname, tconf in TARGETS.items():
    print(f"\n{'='*65}")
    print(f"TARGET: {tname.upper()}  ({tconf['task']}, "
          f"n_valid={tconf['valid'].sum()})")
    print(f"{'='*65}")
    y_all = tconf['y']
    hold_idx = np.where(tconf['valid'] & (site == 'NYU'))[0]
    y_hold = y_all[hold_idx]
    Z_hold = Z_all[hold_idx]
    X_hold = X[hold_idx]
    print(f"NYU held-out with valid {tname}: n={len(y_hold)}"
          + (f"  (pos={int(y_hold.sum())})" if tname == 'asd' else ""))

    rows = []
    for n in N_VALUES:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(y_hold))
            tr, te = idx[:n], idx[n:]
            tr_global = hold_idx[tr]
            te_global = hold_idx[te]
            y_tr, y_te = y_hold[tr], y_hold[te]
            Z_tr, Z_te = Z_hold[tr], Z_hold[te]
            X_tr, X_te = X_hold[tr], X_hold[te]

            # Adaptive α (only for regression)
            if tconf['task'] == 'reg':
                a_star = select_alpha_nested(tr_global, y_tr)
            else:
                a_star = FIXED_ALPHA

            # Spectral features at fixed and adaptive α
            Ff_tr = F_CACHE[FIXED_ALPHA][tr_global]; Ff_te = F_CACHE[FIXED_ALPHA][te_global]
            Fa_tr = F_CACHE[a_star][tr_global];        Fa_te = F_CACHE[a_star][te_global]

            def record(method, metrics):
                row = dict(target=tname, n=n, seed=seed, method=method, **metrics)
                if tconf['task'] == 'reg':
                    row['alpha_star'] = a_star
                rows.append(row)

            # 1. KRR raw
            sc = StandardScaler().fit(X_tr)
            if tconf['task'] == 'reg':
                mu = fit_ridge(sc.transform(X_tr), y_tr, sc.transform(X_te))
                record('krr_raw', metrics_reg(y_te, mu))
            else:
                proba = fit_logreg(sc.transform(X_tr), y_tr, sc.transform(X_te))
                record('krr_raw', metrics_cls(y_te, proba))

            # 2. Spectral-KRR fixed α=0.3
            sc_f = StandardScaler().fit(Ff_tr)
            if tconf['task'] == 'reg':
                mu = fit_ridge(sc_f.transform(Ff_tr), y_tr, sc_f.transform(Ff_te))
                record('spec_fixed', metrics_reg(y_te, mu))
            else:
                proba = fit_logreg(sc_f.transform(Ff_tr), y_tr, sc_f.transform(Ff_te))
                record('spec_fixed', metrics_cls(y_te, proba))

            # 3. Spectral-KRR adaptive α
            sc_a = StandardScaler().fit(Fa_tr)
            if tconf['task'] == 'reg':
                mu = fit_ridge(sc_a.transform(Fa_tr), y_tr, sc_a.transform(Fa_te))
                record('spec_adaptive', metrics_reg(y_te, mu))
            else:
                proba = fit_logreg(sc_a.transform(Fa_tr), y_tr, sc_a.transform(Fa_te))
                record('spec_adaptive', metrics_cls(y_te, proba))

            # 4. Latent noaug
            if tconf['task'] == 'reg':
                mu = fit_ridge(Z_tr, y_tr, Z_te)
                record('latent_noaug', metrics_reg(y_te, mu))
            else:
                proba = fit_logreg(Z_tr, y_tr, Z_te)
                record('latent_noaug', metrics_cls(y_te, proba))

            # 5. Diffusion augmentation
            aug_Z, aug_y = [Z_tr], [y_tr]
            rr = np.random.RandomState(seed * 137)
            for yv in y_tr:
                cand = np.argsort(np.abs(synth_ages - yv))[:20]
                pick = rr.choice(cand, 1, replace=False)
                aug_Z.append(synth_Z[pick]); aug_y.append(synth_ages[pick])
            Zd = np.concatenate(aug_Z, 0); yd = np.concatenate(aug_y, 0)
            if tconf['task'] == 'reg':
                mu = fit_ridge(Zd, yd, Z_te)
                record('diffaug_cont', metrics_reg(y_te, mu))
            else:
                proba = fit_logreg(Zd, yd, Z_te)
                record('diffaug_cont', metrics_cls(y_te, proba))

        # Progress per n
        if tconf['task'] == 'reg':
            sub = [r for r in rows if r['n'] == n]
            m_k = np.mean([r['r'] for r in sub if r['method'] == 'krr_raw'])
            m_f = np.mean([r['r'] for r in sub if r['method'] == 'spec_fixed'])
            m_a = np.mean([r['r'] for r in sub if r['method'] == 'spec_adaptive'])
            a_m = np.mean([r['alpha_star'] for r in sub if r['method'] == 'spec_adaptive'])
            print(f"  n={n:3d}  krr_r={m_k:+.3f}  spec_fixed={m_f:+.3f}  "
                  f"spec_adapt={m_a:+.3f}  α*={a_m:.2f}")
        else:
            sub = [r for r in rows if r['n'] == n]
            m_k = np.mean([r['auc'] for r in sub if r['method'] == 'krr_raw'])
            m_f = np.mean([r['auc'] for r in sub if r['method'] == 'spec_fixed'])
            print(f"  n={n:3d}  krr_auc={m_k:.3f}  spec_auc={m_f:.3f}")

    all_rows.extend(rows)

print(f"\nTotal grid: {(time.time()-t_start)/60:.1f} min, {len(all_rows)} rows")

# ============================================================
# SUMMARY TABLES
# ============================================================
df = pd.DataFrame(all_rows)
df.to_csv('/kaggle/working/results_v5.csv', index=False)

for tname in ['age', 'fiq', 'asd']:
    sub = df[df['target'] == tname]
    print(f"\n{'='*65}\n=== {tname.upper()} SUMMARY ===\n{'='*65}")
    if tname != 'asd':
        piv = sub.groupby(['n','method'])['r'].mean().unstack().round(3)
        a_by_n = sub[sub['method']=='spec_adaptive'].groupby('n')['alpha_star'].mean()
        piv['alpha*'] = a_by_n.round(3)
    else:
        piv = sub.groupby(['n','method'])['auc'].mean().unstack().round(3)
    print(piv.to_string())
    piv.to_csv(f'/kaggle/working/summary_{tname}.csv')

# ============================================================
# VERDICTS
# ============================================================
print(f"\n{'='*65}\nVERDICTS\n{'='*65}")

age_df = df[df['target'] == 'age']
alpha_by_n = age_df[age_df['method'] == 'spec_adaptive'].groupby('n')['alpha_star'].mean()
print(f"\n1. Adaptive α for AGE (mean per n):")
for n_, a in alpha_by_n.items():
    print(f"   n={n_:3d}: α* = {a:.3f}")
alpha_range = float(alpha_by_n.max() - alpha_by_n.min())
shifts = alpha_range > 0.10
print(f"   range = {alpha_range:.3f}  →  {'✓ SHIFTS with n (NOVEL)' if shifts else '✗ roughly constant'}")

age_piv = age_df.groupby(['n','method'])['r'].mean().unstack()
d_adapt = (age_piv['spec_adaptive'] - age_piv['spec_fixed']).round(3)
print(f"\n2. Δr(adaptive − fixed) for AGE:")
print(d_adapt.to_string())
best_adapt = float(d_adapt.max())
print(f"   best Δr = {best_adapt:+.3f}  →  "
      f"{'✓ adaptive helps' if best_adapt > 0.02 else '~ adaptive ≈ fixed'}")

print(f"\n3. Spectral-KRR vs vanilla KRR (Δ metric):")
for tname in ['age', 'fiq', 'asd']:
    tsub = df[df['target'] == tname]
    metric = 'r' if tname != 'asd' else 'auc'
    piv = tsub.groupby(['n','method'])[metric].mean().unstack()
    d_fixed = (piv['spec_fixed'] - piv['krr_raw'])
    d_adapt = (piv['spec_adaptive'] - piv['krr_raw'])
    print(f"\n   {tname.upper()} (metric = {metric}):")
    for n_ in N_VALUES:
        print(f"     n={n_:3d}:  Δ(fixed) = {d_fixed[n_]:+.3f}   Δ(adaptive) = {d_adapt[n_]:+.3f}")

print(f"\n{'='*65}\nHEADLINE\n{'='*65}")
if shifts and best_adapt > 0.02:
    print(">>> METHOD CONTRIBUTION: Adaptive α is both n-dependent AND improves prediction")
elif shifts:
    print(">>> DIAGNOSTIC: α is n-dependent but adaptive selection is too noisy to help")
else:
    print(">>> BENCHMARK: α is roughly constant — fixed α generalizes across n")
    print("    (Your paper becomes a 3-target benchmark: age + FIQ + ASD)")

# Save verdicts
with open('/kaggle/working/verdicts_v5.json', 'w') as f:
    json.dump({
        'alpha_by_n': {int(k): float(v) for k, v in alpha_by_n.items()},
        'alpha_range': alpha_range,
        'alpha_shifts': bool(shifts),
        'best_adapt_delta_r': best_adapt,
    }, f, indent=2)

# Bundle
with zipfile.ZipFile('/kaggle/working/results_bundle_v5.zip', 'w') as z:
    for f in ['results_v5.csv', 'summary_age.csv', 'summary_fiq.csv',
              'summary_asd.csv', 'verdicts_v5.json']:
        p = f'/kaggle/working/{f}'
        if os.path.exists(p): z.write(p, arcname=f)
print(f"\nBundle: {os.path.getsize('/kaggle/working/results_bundle_v5.zip')/1e6:.2f} MB")
print("DONE.")
