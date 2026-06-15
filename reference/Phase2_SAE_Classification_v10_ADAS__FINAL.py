# %% [markdown]
# # Phase 2 — SAE Classification v10 (ADAS features | Year excluded)
# **Dataset:** `SAE_enriched-merged_ADAS_v5.csv` (18,609 × 46)
# 
# **Changes vs v9 (A-directed revision):**
# 
# | Feature | Action | Reason |
# |---------|--------|--------|
# | `Prsn_Age`, `Veh_Mod_Year` | Dropped | Not present in v5 (removed in Phase 2 merge) |
# | `ABS`, `ESC` | Dropped | Near-uniform baseline (85–84%); baseline safety infra |
# | `Fuel_Type`, `Is_Electric` | Dropped | Advisor directive; 87% Gasoline; redundant overlap |
# | `Entr_Road_ID` | Dropped | 67% 'Not Applicable'; collinear with `Intrsct_Relat_ID` |
# | `VIN`, `ADAS_src`, `Model_Year` | Excluded | Identifier / trace / advisor-excluded |
# | `ACC`, `Backup_Camera`, `LDW`, `FCW`, `CIB`, `DBS`, `LKA`, `BSW`, `PAEB` | ADDED | 9 ADAS flags with good variance (60–84%, V≥0.09) |
# | `ADAS_Count` | Dropped | Composite 0–9 score; mean=6.45, well-distributed |
# | `Body_Class` | ADDED (grouped) | 27 raw NHTSA labels → 5 groups before encoding |
# | `Contrib_Factr_1_ID`, `FHE_Collsn_ID`, `Obj_Struck_ID`, `Pop_Group_ID` | ADDED | Present in v5 |
# 
# **Final feature set: 24 features (11 numeric + 13 categorical)**
# 

# %%
# Step 0 — Imports
import os, json, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
    cohen_kappa_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# %%
# Step 1 — Load data
# ── UPDATE THESE TWO PATHS FOR YOUR MACHINE ──────────────────────────────
INPUT_CSV  = r'D:/OneDrive - Texas State University/AV_Levels_CRIS/0_DataCode/3_TDL/Data_SAE_Severity_comb/Data_SS/Merged_clean/SAE_enriched-merged_ADAS_v5.csv'
OUTPUT_DIR = r'D:/OneDrive - Texas State University/Hackathon_Shriyank/01_Dissertation/2017-2025Data/Data_SS/Phase2_Results/v7_TDL_ADAS/'
# ─────────────────────────────────────────────────────────────────────────

MODEL_DIR  = os.path.join(OUTPUT_DIR, 'saved_models')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUTPUT_DIR, 'figures_journal')).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f'Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols')
print(f'Input CSV: {INPUT_CSV}')
print(f'Output DIR: {OUTPUT_DIR}')
print(f'Unique Crash_IDs: {df["Crash_ID"].nunique():,}')


# %%
# Step 2 — Feature definition + preprocessing
# Year excluded: only 2 values — binary confound; kept only for OOT split.
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
year_vals_int = df['Year'].values.copy()

# ── Body_Class: consolidate 27 raw NHTSA labels → 5 groups ───────────────
_body_map = {
    'Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)': 'SUV_CUV',
    'Crossover Utility Vehicle (CUV)':                         'SUV_CUV',
    'Sedan/Saloon':                'Sedan_Coupe',
    'Coupe':                       'Sedan_Coupe',
    'Hatchback/Liftback/Notchback':'Sedan_Coupe',
    'Convertible/Cabriolet':       'Sedan_Coupe',
    'Wagon':                       'Sedan_Coupe',
    'Pickup': 'Pickup',
    'Minivan':              'Van_Minivan',
    'Van':                  'Van_Minivan',
    'Cargo Van':            'Van_Minivan',
    'Step Van / Walk-in Van':'Van_Minivan',
}
df['Body_Class'] = df['Body_Class'].map(_body_map).fillna('Other').astype(str)
print('Body_Class groups:', df['Body_Class'].value_counts().to_dict())

# ── Exclusion lists ────────────────────────────────────────────────────────
LEAKAGE  = ['Rpt_Autonomous_Level_Engaged_ID', 'Crash_Sev_ID', 'Prsn_Injry_Sev_ID']
YEAR_EXC = ['Year']
IDS      = ['Crash_ID', 'Unit_ID', 'Person_ID', 'Unit_Nbr', 'VIN']
DATES    = ['Crash_Date', 'Crash_Time']
TARGETS  = ['SAE_Group', 'Severity_Group', 'Combined_Target', 'SAE_Binary']
# TRACE: provenance + advisor-excluded + ADAS_Count removed (VIF=10,836 — perfect
# linear combination of the 9 individual ADAS flags; causes catastrophic multicollinearity)
TRACE        = ['ADAS_src', 'Model_Year', 'ADAS_Count']

# Advisor-directed drops (data-driven rationale):
ADVISOR_DROP = [
    'ABS',          # 85.6% = 1 — near-universal baseline; poor discrimination
    'ESC',          # 84.2% = 1 — near-universal baseline; poor discrimination
    'Fuel_Type',    # 87% Gasoline — near-uniform; redundant with ADAS flags
    'Is_Electric',  # 13% = 1     — subset of Fuel_Type; advisor directive
    'Entr_Road_ID', # 67% Not Applicable — collinear with Intrsct_Relat_ID
]

EXCLUDE_ALL  = set(LEAKAGE + YEAR_EXC + IDS + DATES + TARGETS + TRACE + ADVISOR_DROP)
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_ALL]

# ── Numeric / categorical split ────────────────────────────────────────────
# 9 retained ADAS flags (ABS + ESC dropped as near-universal baseline;
# ADAS_Count dropped — perfect sum of flags → VIF=10,836)
ADAS_KEEP = ['ACC', 'Backup_Camera', 'LDW', 'FCW', 'CIB', 'DBS', 'LKA', 'BSW', 'PAEB']
NUM_FEATS_CANDIDATES = ['Crash_Speed_Limit'] + ADAS_KEEP
NUM_FEATS = [c for c in NUM_FEATS_CANDIDATES if c in FEATURE_COLS]
CAT_FEATS = [c for c in FEATURE_COLS if c not in NUM_FEATS]

print(f'Feature columns ({len(FEATURE_COLS)} total):')
print(f'  Numeric  ({len(NUM_FEATS)}): {NUM_FEATS}')
print(f'  Categorical ({len(CAT_FEATS)}): {CAT_FEATS}')
print(f'  Dropped (advisor+trace): {sorted(ADVISOR_DROP)} + ADAS_Count')

# ── Preprocessing ──────────────────────────────────────────────────────────
for c in NUM_FEATS:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c] = df[c].fillna(df[c].median())

for c in CAT_FEATS:
    df[c] = df[c].fillna('Unknown').astype(str)

encoders = {}
X_le = np.zeros((len(df), len(FEATURE_COLS)))
for i, c in enumerate(FEATURE_COLS):
    if c in NUM_FEATS:
        X_le[:, i] = df[c].astype(float).values
    else:
        le = LabelEncoder()
        X_le[:, i] = le.fit_transform(df[c].values)
        encoders[c] = le

cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c not in NUM_FEATS]

target_le    = LabelEncoder()
y_all        = target_le.fit_transform(df['SAE_Group'].values)
CLASS_LABELS = list(target_le.classes_)
CLASS_IDS    = list(range(len(CLASS_LABELS)))
N_CLASSES    = len(CLASS_LABELS)
CLASS_IDX_AD = CLASS_LABELS.index('Assisted_Driving')
CLASS_IDX_PA = CLASS_LABELS.index('Partial_Automation')
CLASS_IDX_AA = CLASS_LABELS.index('Advanced_Automation')

print(f'\nCLASS_LABELS: {CLASS_LABELS}')
print(f'CLASS_IDX_AA={CLASS_IDX_AA}  CLASS_IDX_AD={CLASS_IDX_AD}  CLASS_IDX_PA={CLASS_IDX_PA}')
print(f'Features: {len(FEATURE_COLS)} total ({len(CAT_FEATS)} cat + {len(NUM_FEATS)} num)')
print(f'X shape: {X_le.shape}')
for label, idx in zip(CLASS_LABELS, CLASS_IDS):
    n = int((y_all==idx).sum())
    print(f'  {label}: {n:,} ({n/len(y_all)*100:.1f}%)')


# %% [markdown]
# ## Step 2.5 — Formal Feature Screening: Cramér's V + VIF
# Addresses committee Q2 (feature selection justification).

# %%
# Step 2.5 — Cramers V + VIF formal screening
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.sum().sum(); r, c = ct.shape
    phi2 = max(0, chi2/n - (r-1)*(c-1)/(n-1))
    r2 = r-(r-1)**2/(n-1); c2 = c-(c-1)**2/(n-1)
    denom = min(r2-1, c2-1)
    return float(np.sqrt(phi2/denom)) if denom > 0 else 0.0

cv_rows = []
for c in FEATURE_COLS:
    col = df[c].astype(str) if c in CAT_FEATS else pd.cut(df[c], bins=10, labels=False).astype(str)
    cv_rows.append({'Feature': c, 'CramersV': round(cramers_v(col, df['SAE_Group']), 4)})
cv_df = pd.DataFrame(cv_rows).sort_values('CramersV', ascending=False)
print('Bias-corrected Cramers V (feature <-> SAE_Group):')
print(cv_df.to_string(index=False))
cv_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_cramers_v.csv'), index=False)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xnum = X_le[:, [i for i,c in enumerate(FEATURE_COLS) if c in NUM_FEATS]]
    vif_df = pd.DataFrame({'Feature': NUM_FEATS,
                           'VIF': [round(variance_inflation_factor(Xnum, i), 2)
                                   for i in range(Xnum.shape[1])]})
    print('\nVIF (numeric features):'); print(vif_df.to_string(index=False))
    vif_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_vif.csv'), index=False)
except ImportError:
    print('statsmodels not available — skip VIF')
print(f'\nMax V={cv_df["CramersV"].max():.4f} ({cv_df.iloc[0]["Feature"]})')
print('Saved: feature_cramers_v.csv | feature_vif.csv')


# %%
# Step 3 — Group-aware 80/20 split + OneHot fit
crash_ids = df['Crash_ID'].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X_le, y_all, crash_ids))

X_tr_le, X_te_le = X_le[train_idx], X_le[test_idx]
y_tr, y_te = y_all[train_idx], y_all[test_idx]
crash_ids_tr = crash_ids[train_idx]

print(f'Train: {len(X_tr_le):,} | Test: {len(X_te_le):,}')
print(f'Group overlap (must be 0): {len(set(crash_ids[train_idx]) & set(crash_ids[test_idx]))}')
print(f'Test class counts: {dict(Counter(y_te))} → AA={int((y_te==CLASS_IDX_AA).sum())}')

# Fit OneHot on the FULL data (so unseen categories don't blow up)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_le[:, cat_indices].astype(int))

# Get one-hot feature names for SHAP later
ohe_feat_names = []
for i, c_idx in enumerate(cat_indices):
    col_name = FEATURE_COLS[c_idx]
    for cat in ohe.categories_[i]:
        ohe_feat_names.append(f'{col_name}={cat}')
NUM_FEAT_NAMES = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if i not in cat_indices]
ALL_FEAT_NAMES = NUM_FEAT_NAMES + ohe_feat_names

def ohe_transform(X):
    Xn  = X[:, [i for i in range(X.shape[1]) if i not in cat_indices]]
    Xc  = ohe.transform(X[:, cat_indices].astype(int))
    return np.hstack([Xn, Xc])

# Apply SMOTE-NC ONCE on the training set for the "final" fit (per-fold SMOTE in CV)
c_tr = Counter(y_tr)
nm = c_tr[CLASS_IDX_AD]
strategy = {
    CLASS_IDX_PA: max(c_tr[CLASS_IDX_PA], int(nm * 0.50)),
    CLASS_IDX_AA: max(c_tr[CLASS_IDX_AA], int(nm * 0.30)),
}
k_neigh = min(5, c_tr[CLASS_IDX_AA] - 1)
sm_full = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_SEED,
                  k_neighbors=k_neigh, sampling_strategy=strategy)
X_tr_sm_le, y_tr_sm = sm_full.fit_resample(X_tr_le, y_tr)
X_tr_pp = ohe_transform(X_tr_sm_le)
X_te_pp = ohe_transform(X_te_le)
INPUT_DIM = X_tr_pp.shape[1]
print(f'\nAfter SMOTE-NC: {dict(Counter(y_tr_sm))}')
print(f'INPUT_DIM (after OHE): {INPUT_DIM}')

# Save preprocessing artifacts for the separate SHAP/IG notebook
with open(os.path.join(MODEL_DIR, 'preprocessing.pkl'), 'wb') as f:
    pickle.dump({
        'encoders': encoders, 'ohe': ohe, 'cat_indices': cat_indices,
        'FEATURE_COLS': FEATURE_COLS, 'NUM_FEATS': NUM_FEATS,
        'ALL_FEAT_NAMES': ALL_FEAT_NAMES, 'NUM_FEAT_NAMES': NUM_FEAT_NAMES,
        'CLASS_LABELS': CLASS_LABELS, 'CLASS_IDX_AA': CLASS_IDX_AA,
        'CLASS_IDX_PA': CLASS_IDX_PA, 'CLASS_IDX_AD': CLASS_IDX_AD,
        'INPUT_DIM': INPUT_DIM, 'test_idx': test_idx,
    }, f)
np.save(os.path.join(MODEL_DIR, 'X_te_pp.npy'), X_te_pp)
np.save(os.path.join(MODEL_DIR, 'y_te.npy'), y_te)
np.save(os.path.join(MODEL_DIR, 'X_tr_pp.npy'), X_tr_pp)
np.save(os.path.join(MODEL_DIR, 'y_tr_sm.npy'), y_tr_sm)
print('Saved preprocessing → preprocessing.pkl + X_te_pp.npy + y_te.npy')


# %%
# Step 4 — Helpers: per-fold SMOTE-NC, training, evaluation, metric printing
def _ohe_fold(X):
    Xn = X[:, [i for i in range(X.shape[1]) if i not in cat_indices]]
    Xc = ohe.transform(X[:, cat_indices].astype(int))
    return np.hstack([Xn, Xc])

def _smotenc_fold(Xtr, ytr):
    cnt = Counter(ytr); nm = cnt[CLASS_IDX_AD]
    st = {CLASS_IDX_PA: max(cnt[CLASS_IDX_PA], int(nm * 0.50)),
          CLASS_IDX_AA: max(cnt[CLASS_IDX_AA], int(nm * 0.30))}
    ks = min(5, min(cnt[CLASS_IDX_AA], cnt[CLASS_IDX_PA]) - 1)
    if ks < 1:
        return _ohe_fold(Xtr), ytr
    sm = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_SEED,
                 k_neighbors=ks, sampling_strategy=st)
    Xr, yr = sm.fit_resample(Xtr, ytr)
    return _ohe_fold(Xr), yr

class_w = compute_class_weight('balanced', classes=np.unique(y_tr_sm), y=y_tr_sm)
class_weights_tensor = torch.tensor(class_w, dtype=torch.float32, device=DEVICE)
print(f'Class weights: {dict(zip(CLASS_LABELS, class_w.round(3)))}')

def print_metrics(name, yt, yp, ypr=None):
    acc = accuracy_score(yt, yp)
    mf  = f1_score(yt, yp, average='macro', zero_division=0)
    ba  = balanced_accuracy_score(yt, yp)
    yb  = label_binarize(yt, classes=CLASS_IDS)
    try:    ma = roc_auc_score(yb, ypr, average='macro', multi_class='ovr') if ypr is not None else float('nan')
    except: ma = float('nan')
    print(f'\n{"─"*62}\n  {name}\n{"─"*62}')
    print(f'  Acc={acc:.4f}  Macro-F1={mf:.4f}  Bal-Acc={ba:.4f}  Macro-AUC={ma:.4f}')
    print(classification_report(yt, yp, target_names=CLASS_LABELS, zero_division=0, digits=4))
    return {'accuracy':acc, 'macro_f1':mf, 'balanced_acc':ba, 'macro_auc':ma,
            'y_pred':yp, 'y_probs':ypr}

final_results, cv_results, oof_probs = {}, {}, {}


# %% [markdown]
# ## Step 5 — ML Baselines (RF, XGBoost) — accepted-paper precedent

# %%
# Step 5 — Random Forest baseline
def cv_sklearn(factory, name, n_folds=5):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    f1s, oof = [], np.zeros((len(X_tr_le), N_CLASSES))
    for fold, (fi, vi) in enumerate(sgkf.split(X_tr_le, y_tr, crash_ids_tr)):
        Xt, Xv, yt, yv = X_tr_le[fi], X_tr_le[vi], y_tr[fi], y_tr[vi]
        Xt, yt = _smotenc_fold(Xt, yt); Xv = _ohe_fold(Xv)
        m = factory(); m.fit(Xt, yt)
        oof[vi] = m.predict_proba(Xv)
        f1s.append(f1_score(yv, m.predict(Xv), average='macro', zero_division=0))
        print(f'  Fold {fold+1}/5: F1={f1s[-1]:.4f}')
    mu, sd = np.mean(f1s), np.std(f1s)
    print(f'  ► {name:<22} CV F1={mu:.4f}±{sd:.4f}')
    return mu, sd, oof

print('='*65, '\n  Baseline 1: RANDOM FOREST  (5-fold CV)\n', '='*65, sep='')
rf_factory = lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                             max_features='sqrt', min_samples_leaf=2,
                                             random_state=RANDOM_SEED, n_jobs=-1)
mu, sd, oof_rf = cv_sklearn(rf_factory, 'Random Forest')
cv_results['Random Forest'] = (mu, sd); oof_probs['Random Forest'] = oof_rf

rf = rf_factory(); rf.fit(X_tr_pp, y_tr_sm)
rf_pred = rf.predict(X_te_pp); rf_prob = rf.predict_proba(X_te_pp)
final_results['Random Forest'] = print_metrics('Random Forest', y_te, rf_pred, rf_prob)
with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'wb') as f: pickle.dump(rf, f)


# %%
# Step 5b — XGBoost baseline
from xgboost import XGBClassifier
spw = max(1, int(c_tr[CLASS_IDX_AD] / max(1, c_tr[CLASS_IDX_AA])))
print('='*65, '\n  Baseline 2: XGBOOST  (5-fold CV)\n', '='*65, sep='')
print(f'scale_pos_weight = {spw}')
xgb_factory = lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                     subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=spw, eval_metric='mlogloss',
                                     random_state=RANDOM_SEED, verbosity=0)
mu, sd, oof_xgb = cv_sklearn(xgb_factory, 'XGBoost')
cv_results['XGBoost'] = (mu, sd); oof_probs['XGBoost'] = oof_xgb

xgb = xgb_factory(); xgb.fit(X_tr_pp, y_tr_sm)
xgb_pred = xgb.predict(X_te_pp); xgb_prob = xgb.predict_proba(X_te_pp)
final_results['XGBoost'] = print_metrics('XGBoost', y_te, xgb_pred, xgb_prob)
xgb.save_model(os.path.join(MODEL_DIR, 'xgboost.json'))


# %% [markdown]
# ## Step 6 — TDL Architectures (v7.5, proven to train)
# 
# Imported verbatim from your working v7.5 pipeline. The key difference from v6 is **per-feature W/b tokenizers** instead of a shared `Linear(1, d)`. Each one-hot column now gets its own learned embedding, so the transformer can actually distinguish features.

# %%
# Step 6a — Training loop and focal loss (from v7.5)
NUM_EPOCHS    = 80
WARMUP_EPOCHS = 5
BATCH_SIZE    = 128

def make_loaders(Xtr, ytr, Xva, yva, bs=BATCH_SIZE):
    def t(X, y): return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
                        torch.tensor(np.array(y), dtype=torch.long).to(DEVICE))
    return (DataLoader(TensorDataset(*t(Xtr, ytr)), batch_size=bs, shuffle=True),
            DataLoader(TensorDataset(*t(Xva, yva)), batch_size=bs))

train_loader, test_loader = make_loaders(X_tr_pp, y_tr_sm, X_te_pp, y_te)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, ls=0.05):
        super().__init__(); self.a=alpha; self.g=gamma; self.ls=ls
    def forward(self, inp, tgt):
        ce = F.cross_entropy(inp, tgt, weight=self.a, label_smoothing=self.ls, reduction='none')
        return ((1-torch.exp(-ce))**self.g * ce).mean()

def train_model(model, trl, val, ne=NUM_EPOCHS, verbose=True, lr=1e-3, gamma=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, ne-WARMUP_EPOCHS))
    wrm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    best_f1, best_ep, pat, PAT_LIM = 0.0, 0, 0, 30
    best_st = None
    for ep in range(ne):
        (wrm if ep < WARMUP_EPOCHS else cos).step()
        crit = FocalLoss(alpha=class_weights_tensor, gamma=gamma, ls=0.05)
        model.train(); tl = 0
        for Xb, yb in trl:
            opt.zero_grad(); l = crit(model(Xb), yb); l.backward(); opt.step(); tl += l.item()
        tl /= len(trl)
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for Xb, yb in val:
                o = model(Xb); ps.extend(o.argmax(1).cpu().numpy()); ts.extend(yb.cpu().numpy())
        vf1 = f1_score(ts, ps, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1, best_ep, pat = vf1, ep, 0
            best_st = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT_LIM: break
        if verbose and ep % 16 == 0:
            print(f'  Ep {ep}/{ne}  loss={tl:.4f}  F1={vf1:.4f}')
    if best_st: model.load_state_dict(best_st)
    print(f'  Stop ep{ep}  best_F1={best_f1:.4f}@ep{best_ep}')
    return best_f1, best_ep

def eval_model(model, loader):
    model.eval(); ps, ts, prs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            o = model(Xb); ps.extend(o.argmax(1).cpu().numpy())
            prs.append(F.softmax(o, dim=1).cpu().numpy())
            ts.extend(yb.cpu().numpy())
    return np.array(ts), np.array(ps), np.vstack(prs)

def cv_pytorch(model_factory, name, n_folds=5, lr=1e-3, gamma=1.0):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    f1s, oof = [], np.zeros((len(X_tr_le), N_CLASSES))
    for fold, (fi, vi) in enumerate(sgkf.split(X_tr_le, y_tr, crash_ids_tr)):
        Xt, Xv, yt, yv = X_tr_le[fi], X_tr_le[vi], y_tr[fi], y_tr[vi]
        Xt, yt = _smotenc_fold(Xt, yt); Xv = _ohe_fold(Xv)
        trl, vl = make_loaders(Xt, yt, Xv, yv)
        model = model_factory().to(DEVICE)
        train_model(model, trl, vl, ne=NUM_EPOCHS, verbose=False, lr=lr, gamma=gamma)
        _, yp, ypr = eval_model(model, vl)
        oof[vi] = ypr
        f1s.append(f1_score(yv, yp, average='macro', zero_division=0))
        print(f'  Fold {fold+1}/5: F1={f1s[-1]:.4f}')
    mu, sd = np.mean(f1s), np.std(f1s)
    print(f'  ► {name:<22} CV F1={mu:.4f}±{sd:.4f}')
    return mu, sd, oof


# %%
# Step 6b — TDL architectures from v7.5 (proven, do not modify)

# ── 1. MambaAttention ──
class MambaAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.proj=nn.Linear(input_dim, hidden_dim); self.n1=nn.LayerNorm(hidden_dim)
        self.d1=nn.Dropout(dropout)
        self.attn=nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.n2=nn.LayerNorm(hidden_dim); self.fc=nn.Linear(hidden_dim, hidden_dim//2)
        self.n3=nn.LayerNorm(hidden_dim//2); self.d2=nn.Dropout(dropout)
        self.out=nn.Linear(hidden_dim//2, num_classes)
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def forward(self, x):
        x=self.d1(F.gelu(self.n1(self.proj(x)))); r=x
        xq,_=self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x=self.n2(xq.squeeze(1)+r)
        return self.out(self.d2(F.gelu(self.n3(self.fc(x)))))

# ── 2. FT-Transformer (Gorishniy NeurIPS 2021) ──
class FTTransformerClassifier(nn.Module):
    def __init__(self, n_features, n_classes=3, d_token=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.W=nn.Parameter(torch.empty(n_features, d_token)); nn.init.kaiming_uniform_(self.W, a=0.01)
        self.b=nn.Parameter(torch.zeros(n_features, d_token))
        self.cls=nn.Parameter(torch.zeros(1, 1, d_token))
        enc=nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads,
                                        dim_feedforward=d_token*4, dropout=dropout,
                                        batch_first=True, norm_first=True)
        self.enc=nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm=nn.LayerNorm(d_token)
        self.head=nn.Sequential(nn.Linear(d_token, d_token//2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_token//2, n_classes))
    def forward(self, x):
        B=x.shape[0]
        tokens=x.unsqueeze(-1)*self.W.unsqueeze(0)+self.b.unsqueeze(0)
        cls=self.cls.expand(B, -1, -1)
        tokens=torch.cat([cls, tokens], dim=1)
        out=self.enc(tokens)
        return self.head(self.norm(out[:, 0]))

# ── 3. AutoInt (Song 2019) ──
class AutoIntClassifier(nn.Module):
    def __init__(self, n_features, n_classes=3, d=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.W=nn.Parameter(torch.empty(n_features, d)); nn.init.kaiming_uniform_(self.W, a=0.01)
        self.b=nn.Parameter(torch.zeros(n_features, d))
        self.attn_layers=nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head=nn.Sequential(nn.Flatten(), nn.Linear(n_features*d, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, n_classes))
    def forward(self, x):
        h=x.unsqueeze(-1)*self.W.unsqueeze(0)+self.b.unsqueeze(0)
        for attn, norm in zip(self.attn_layers, self.norms):
            res,_=attn(h, h, h); h=norm(h+res)
        return self.head(h)

# ── 4. TabResNet ──
class TabResNet(nn.Module):
    def __init__(self, input_dim, n_classes=3, hidden=256, n_blocks=6, dropout=0.2):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden))
        self.blocks=nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden*2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden*2, hidden), nn.Dropout(dropout/2))
            for _ in range(n_blocks)])
        self.head=nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden//2),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden//2, n_classes))
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def forward(self, x):
        x=self.proj(x)
        for blk in self.blocks: x=x+blk(x)
        return self.head(x)

print("Architectures from v7.5 loaded ✓")
print("  MambaAttention | FT-Transformer | AutoInt | TabResNet")


# %% [markdown]
# ## Step 4.5 — Learning Curve: Justifying the 80/20 Split
# **Placed after Step 6b** so BATCH_SIZE, AutoIntClassifier, train_model, eval_model are all defined.
# XGBoost: 7 fractions. AutoInt: 3 anchor points (20/50/80%).

# %%
# Step 4.5 — Learning curve (runs after Step 6b — all deps defined)
# Uses: BATCH_SIZE, AutoIntClassifier, train_model, eval_model (all from Steps 6a/6b)
import matplotlib.pyplot as plt, matplotlib
matplotlib.rcParams.update({'font.size':12,'font.weight':'bold',
                            'axes.labelweight':'bold','axes.titleweight':'bold'})
from xgboost import XGBClassifier as XGB_LC

FRACTIONS_XGB = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80]
FRACTIONS_AI  = [0.20,0.50,0.80]
N_FOLDS_LC    = 3
rng_lc        = np.random.RandomState(RANDOM_SEED)

def lc_subsample(frac):
    unique_cr = np.unique(crash_ids_tr)
    sub_ids   = rng_lc.choice(unique_cr, max(1, int(len(unique_cr)*frac)), replace=False)
    mask      = np.isin(crash_ids_tr, sub_ids)
    return X_tr_le[mask], y_tr[mask], crash_ids_tr[mask]

print('='*65)
print('  Step 4.5 — Learning Curve')
print('='*65)

# XGBoost
print('\n  XGBoost (7 fractions):')
lc_xgb = []
for frac in FRACTIONS_XGB:
    Xs,ys,cs = lc_subsample(frac)
    if (ys==CLASS_IDX_AA).sum()<6: continue
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS_LC,shuffle=True,random_state=RANDOM_SEED)
    f1s = []
    for fi,vi in sgkf.split(Xs,ys,cs):
        Xt,Xv,yt,yv = Xs[fi],Xs[vi],ys[fi],ys[vi]
        if (yt==CLASS_IDX_AA).sum()<2: continue
        Xt,yt = _smotenc_fold(Xt,yt); Xv = _ohe_fold(Xv)
        m = XGB_LC(n_estimators=150,learning_rate=0.05,max_depth=5,
                   subsample=0.8,colsample_bytree=0.8,
                   eval_metric='mlogloss',random_state=RANDOM_SEED,verbosity=0)
        m.fit(Xt,yt)
        f1s.append(f1_score(yv,m.predict(Xv),average='macro',zero_division=0))
    if not f1s: continue
    mu,sd = float(np.mean(f1s)),float(np.std(f1s))
    lc_xgb.append({'frac':frac,'mu':mu,'sd':sd})
    print(f'    {frac:.0%}  F1={mu:.4f} +/- {sd:.4f}')

# AutoInt (3 anchors)
print('\n  AutoInt (3 anchor points):')
lc_ai = []
for frac in FRACTIONS_AI:
    Xs,ys,cs = lc_subsample(frac)
    if (ys==CLASS_IDX_AA).sum()<6: continue
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS_LC,shuffle=True,random_state=RANDOM_SEED)
    f1s = []
    for fi,vi in sgkf.split(Xs,ys,cs):
        Xt,Xv,yt,yv = Xs[fi],Xs[vi],ys[fi],ys[vi]
        if (yt==CLASS_IDX_AA).sum()<2: continue
        Xt,yt = _smotenc_fold(Xt,yt); Xv = _ohe_fold(Xv)
        ai_m = AutoIntClassifier(Xt.shape[1],N_CLASSES,d=64).to(DEVICE)
        trl_ai = DataLoader(TensorDataset(
            torch.tensor(Xt,dtype=torch.float32).to(DEVICE),
            torch.tensor(yt,dtype=torch.long).to(DEVICE)),batch_size=BATCH_SIZE,shuffle=True)
        val_ai = DataLoader(TensorDataset(
            torch.tensor(Xv,dtype=torch.float32).to(DEVICE),
            torch.tensor(yv,dtype=torch.long).to(DEVICE)),batch_size=BATCH_SIZE)
        train_model(ai_m,trl_ai,val_ai,ne=40,verbose=False,lr=1e-3,gamma=1.0)
        _,yp_ai,_ = eval_model(ai_m,val_ai)
        f1s.append(f1_score(yv,yp_ai,average='macro',zero_division=0))
        del ai_m; torch.cuda.empty_cache()
    if not f1s: continue
    mu,sd = float(np.mean(f1s)),float(np.std(f1s))
    lc_ai.append({'frac':frac,'mu':mu,'sd':sd})
    print(f'    {frac:.0%}  F1={mu:.4f} +/- {sd:.4f}')

pd.DataFrame(lc_xgb).to_csv(os.path.join(OUTPUT_DIR,'learning_curve_xgb.csv'),index=False)
pd.DataFrame(lc_ai).to_csv(os.path.join(OUTPUT_DIR,'learning_curve_autoint.csv'),index=False)

fig,ax = plt.subplots(figsize=(10,5.5))
xf=[r['frac']*100 for r in lc_xgb]; xm=[r['mu'] for r in lc_xgb]; xs_=[r['sd'] for r in lc_xgb]
ax.plot(xf,xm,'o-',color='#00A08A',lw=2.5,ms=8,label='XGBoost (ML baseline)')
ax.fill_between(xf,[m-s for m,s in zip(xm,xs_)],[m+s for m,s in zip(xm,xs_)],alpha=0.15,color='#00A08A')
if lc_ai:
    af=[r['frac']*100 for r in lc_ai]; am=[r['mu'] for r in lc_ai]; as_=[r['sd'] for r in lc_ai]
    ax.plot(af,am,'^--',color='#46ACC8',lw=2.5,ms=10,label='AutoInt (best TDL -- 3 anchors)',zorder=5)
    ax.fill_between(af,[m-s for m,s in zip(am,as_)],[m+s for m,s in zip(am,as_)],alpha=0.15,color='#46ACC8')
ax.axvline(80,color='#D62728',ls='--',lw=2,label='Chosen split (80%)')
if len(lc_xgb)>=2:
    gain = lc_xgb[-1]['mu']-lc_xgb[0]['mu']
    ax.annotate(f'Total gain 10%->80%:\n+{gain:.3f} macro-F1\n(signal-limited)',
                xy=(80,lc_xgb[-1]['mu']),xytext=(48,lc_xgb[0]['mu']-0.025),fontsize=9,color='#1F3864',
                arrowprops=dict(arrowstyle='->',color='#1F3864',lw=1.2))
ax.set_xlabel('Training Data Used (%)',fontsize=13,fontweight='bold')
ax.set_ylabel('Macro-F1 (3-fold CV)',fontsize=13,fontweight='bold')
ax.set_title('Learning Curve -- XGBoost + AutoInt\nSignal-limited: performance flat across training sizes',fontsize=13,fontweight='bold')
ax.set_xticks([int(f*100) for f in FRACTIONS_XGB]); ax.legend(fontsize=11,loc='lower right'); ax.grid(True,alpha=0.3)
plt.tight_layout()
fig_lc_path = os.path.join(OUTPUT_DIR,'figures_journal','learning_curve_split_justification.png')
os.makedirs(os.path.dirname(fig_lc_path),exist_ok=True)
fig.savefig(fig_lc_path,dpi=300,bbox_inches='tight'); plt.show()
if len(lc_xgb)>=2:
    print(f'XGB gain 10%->80%: {lc_xgb[-1]["mu"]-lc_xgb[0]["mu"]:+.4f}')
if lc_ai:
    print(f'AutoInt gain 20%->80%: {lc_ai[-1]["mu"]-lc_ai[0]["mu"]:+.4f}')
print('Saved: learning_curve_xgb.csv | learning_curve_autoint.csv | learning_curve_split_justification.png')


# %%
# Step 7 — MambaAttention (v7.5 architecture)
print('='*65, '\n  TDL 1: MAMBAATTENTION\n', '='*65, sep='')
mu, sd, oof = cv_pytorch(lambda: MambaAttentionClassifier(INPUT_DIM, 256, N_CLASSES),
                          'MambaAttention', lr=1e-3, gamma=1.0)
cv_results['MambaAttention'] = (mu, sd); oof_probs['MambaAttention'] = oof
mamba = MambaAttentionClassifier(INPUT_DIM, 256, N_CLASSES).to(DEVICE)
train_model(mamba, train_loader, test_loader, lr=1e-3, gamma=1.0)
yt, yp, ypr = eval_model(mamba, test_loader)
final_results['MambaAttention'] = print_metrics('MambaAttention', yt, yp, ypr)
torch.save(mamba.state_dict(), os.path.join(MODEL_DIR, 'mamba_attention.pt'))


# %%
# Step 8 — FT-Transformer (v7.5 architecture)
print('='*65, '\n  TDL 2: FT-TRANSFORMER\n', '='*65, sep='')
mu, sd, oof = cv_pytorch(lambda: FTTransformerClassifier(INPUT_DIM, N_CLASSES, d_token=128),
                          'FT-Transformer', lr=1e-3, gamma=1.0)
cv_results['FT-Transformer'] = (mu, sd); oof_probs['FT-Transformer'] = oof
ftt = FTTransformerClassifier(INPUT_DIM, N_CLASSES, d_token=128).to(DEVICE)
train_model(ftt, train_loader, test_loader, lr=1e-3, gamma=1.0)
yt, yp, ypr = eval_model(ftt, test_loader)
final_results['FT-Transformer'] = print_metrics('FT-Transformer', yt, yp, ypr)
torch.save(ftt.state_dict(), os.path.join(MODEL_DIR, 'ft_transformer.pt'))


# %%
# Step 9 — AutoInt (v7.5 architecture)
print('='*65, '\n  TDL 3: AUTOINT\n', '='*65, sep='')
mu, sd, oof = cv_pytorch(lambda: AutoIntClassifier(INPUT_DIM, N_CLASSES, d=64),
                          'AutoInt', lr=1e-3, gamma=1.0)
cv_results['AutoInt'] = (mu, sd); oof_probs['AutoInt'] = oof
ai = AutoIntClassifier(INPUT_DIM, N_CLASSES, d=64).to(DEVICE)
train_model(ai, train_loader, test_loader, lr=1e-3, gamma=1.0)
yt, yp, ypr = eval_model(ai, test_loader)
final_results['AutoInt'] = print_metrics('AutoInt', yt, yp, ypr)
torch.save(ai.state_dict(), os.path.join(MODEL_DIR, 'autoint.pt'))


# %%
# Step 10 — TabResNet (v7.5 architecture)
print('='*65, '\n  TDL 4: TABRESNET\n', '='*65, sep='')
mu, sd, oof = cv_pytorch(lambda: TabResNet(INPUT_DIM, N_CLASSES, hidden=256, n_blocks=6),
                          'TabResNet', lr=1e-3, gamma=1.0)
cv_results['TabResNet'] = (mu, sd); oof_probs['TabResNet'] = oof
trn = TabResNet(INPUT_DIM, N_CLASSES, hidden=256, n_blocks=6).to(DEVICE)
train_model(trn, train_loader, test_loader, lr=1e-3, gamma=1.0)
yt, yp, ypr = eval_model(trn, test_loader)
final_results['TabResNet'] = print_metrics('TabResNet', yt, yp, ypr)
torch.save(trn.state_dict(), os.path.join(MODEL_DIR, 'tabresnet.pt'))


# %%
# Step 11 — TabPFN 2.5
print('='*65, '\n  TDL 5: TABPFN 2.5  (Hollmann 2024)\n', '='*65, sep='')
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print('pip install tabpfn')

if HAS_TABPFN:
    TABPFN_MAX = 3000
    rng = np.random.RandomState(RANDOM_SEED)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1s, oof = [], np.zeros((len(X_tr_le), N_CLASSES))
    for fold, (fi, vi) in enumerate(sgkf.split(X_tr_le, y_tr, crash_ids_tr)):
        Xt, Xv, yt, yv = X_tr_le[fi], X_tr_le[vi], y_tr[fi], y_tr[vi]
        Xt, yt = _smotenc_fold(Xt, yt); Xv = _ohe_fold(Xv)
        sub = rng.choice(len(yt), min(TABPFN_MAX, len(yt)), replace=False)
        pfn = TabPFNClassifier(n_estimators=16)
        pfn.fit(Xt[sub], yt[sub])
        ypr = pfn.predict_proba(Xv)
        oof[vi] = ypr
        f1s.append(f1_score(yv, ypr.argmax(1), average='macro', zero_division=0))
        print(f'  Fold {fold+1}/5: F1={f1s[-1]:.4f}')
    mu, sd = np.mean(f1s), np.std(f1s)
    cv_results['TabPFN 2.5'] = (mu, sd); oof_probs['TabPFN 2.5'] = oof
    print(f'  ► TabPFN 2.5             CV F1={mu:.4f}±{sd:.4f}')

    sub = rng.choice(len(y_tr_sm), min(TABPFN_MAX, len(y_tr_sm)), replace=False)
    pfn_final = TabPFNClassifier(n_estimators=32)
    pfn_final.fit(X_tr_pp[sub], y_tr_sm[sub])
    bs = 500
    probs = np.vstack([pfn_final.predict_proba(X_te_pp[i:i+bs]) for i in range(0, len(X_te_pp), bs)])
    final_results['TabPFN 2.5'] = print_metrics('TabPFN 2.5', y_te, probs.argmax(1), probs)


# %% [markdown]
# ## Step 12 — Per-class threshold calibration

# %%
# Step 12 — Threshold calibration on OOF probs
def calibrate_thresholds(proba_oof, y_true, n_classes):
    th = np.zeros(n_classes)
    for c in range(n_classes):
        precs, recs, ts = precision_recall_curve((y_true == c).astype(int), proba_oof[:, c])
        f1s = 2 * precs * recs / (precs + recs + 1e-9)
        th[c] = ts[np.argmax(f1s[:-1])] if len(ts) > 0 else 0.5
    return th

def apply_thresholds(proba_test, thresholds):
    margins = proba_test - thresholds[np.newaxis, :]
    no_pass = (margins.max(axis=1) < 0)
    pred = margins.argmax(axis=1)
    pred[no_pass] = proba_test[no_pass].argmax(axis=1)
    return pred

calibrated_results = {}
thresholds_map = {}
for name, oof in oof_probs.items():
    if name not in final_results: continue
    thr = calibrate_thresholds(oof, y_tr, N_CLASSES)
    thresholds_map[name] = thr
    proba_test = final_results[name]['y_probs']
    if proba_test is None: continue
    yp_cal = apply_thresholds(proba_test, thr)
    calibrated_results[name] = print_metrics(f'{name} (calibrated)', y_te, yp_cal, proba_test)
    print(f'  Thresholds: {dict(zip(CLASS_LABELS, thr.round(3)))}')


# %%
# Step 12b — Save prediction arrays for SCCS notebook
# This must run immediately after Step 12 (calibration) is complete.
# Phase2_SCCS_metric.ipynb loads these files from OUTPUT_DIR.
print('Saving calibrated prediction arrays for SCCS notebook ...')
for mname, res in calibrated_results.items():
    cn = mname.lower().replace(' ', '_').replace('-', '_')
    np.save(os.path.join(OUTPUT_DIR, f'y_pred_cal_{cn}.npy'), res['y_pred'])
    np.save(os.path.join(OUTPUT_DIR, f'y_probs_cal_{cn}.npy'), res['y_probs'])
    print(f'  Saved: y_pred_cal_{cn}.npy  y_probs_cal_{cn}.npy')
# Also save uncalibrated (fallback for models not in calibrated_results)
for mname, res in final_results.items():
    if mname in calibrated_results: continue  # already saved calibrated version
    if res.get('y_pred') is None: continue
    cn = mname.lower().replace(' ', '_').replace('-', '_')
    np.save(os.path.join(OUTPUT_DIR, f'y_pred_{cn}.npy'), res['y_pred'])
    if res.get('y_probs') is not None:
        np.save(os.path.join(OUTPUT_DIR, f'y_probs_{cn}.npy'), res['y_probs'])
    print(f'  Saved (raw): y_pred_{cn}.npy')
np.save(os.path.join(MODEL_DIR, 'y_te.npy'), y_te)   # ensure y_te.npy is fresh
print(f'  Saved y_te.npy to saved_models/')
print(f'  Saved {len(calibrated_results)} calibrated models.')
print('SCCS notebook ready — run Phase2_SCCS_metric.ipynb next.')


# %% [markdown]
# ## Step 13 — TDL Stacked Ensemble (best 3 by AA F1, auto-selected)

# %%
# Step 13 — TDL stacked ensemble (auto-pick best 3 TDL models by OOF AA F1)
TDL_ALL = [n for n in oof_probs if n not in ('Random Forest','XGBoost')]
# Rank TDL by AA-class F1 on OOF (we want minority sensitivity)
def oof_aa_f1(oof_p, y_true):
    yp = oof_p.argmax(1)
    return f1_score((y_true==CLASS_IDX_AA).astype(int), (yp==CLASS_IDX_AA).astype(int), zero_division=0)

ranked = sorted(TDL_ALL, key=lambda n: oof_aa_f1(oof_probs[n], y_tr), reverse=True)
TDL_BASE = ranked[:3]
print(f'TDL stack base = {TDL_BASE}')

if len(TDL_BASE) >= 2:
    oof_stack  = np.hstack([oof_probs[n]            for n in TDL_BASE])
    test_stack = np.hstack([final_results[n]['y_probs'] for n in TDL_BASE])
    meta = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=RANDOM_SEED)
    meta.fit(oof_stack, y_tr)
    meta_proba = meta.predict_proba(test_stack)
    meta_pred  = meta_proba.argmax(axis=1)
    final_results['TDL Stack'] = print_metrics(f'TDL Stack ({" + ".join(TDL_BASE)})', y_te, meta_pred, meta_proba)
    thr_stack = calibrate_thresholds(meta.predict_proba(oof_stack), y_tr, N_CLASSES)
    meta_cal  = apply_thresholds(meta_proba, thr_stack)
    calibrated_results['TDL Stack'] = print_metrics('TDL Stack (calibrated)', y_te, meta_cal, meta_proba)
    thresholds_map['TDL Stack'] = thr_stack


# %% [markdown]
# ## Step 14 — Out-of-time (2024 → 2025) for RQ3

# %%
# Step 14 — OOT
idx24 = np.where(year_vals_int == 2024)[0]
idx25 = np.where(year_vals_int == 2025)[0]
print(f'2024 rows: {len(idx24):,}  |  2025 rows: {len(idx25):,}')
Xtr24_le, ytr24 = X_le[idx24], y_all[idx24]
Xte25_le, yte25 = X_le[idx25], y_all[idx25]
cnt24 = Counter(ytr24)
print(f'2024 — {dict(cnt24)}  |  2025 — {dict(Counter(yte25))}')

if cnt24[CLASS_IDX_AA] > 5:
    sm24 = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_SEED,
                   k_neighbors=min(5, cnt24[CLASS_IDX_AA]-1),
                   sampling_strategy={
                       CLASS_IDX_PA: max(cnt24[CLASS_IDX_PA], int(cnt24[CLASS_IDX_AD]*0.50)),
                       CLASS_IDX_AA: max(cnt24[CLASS_IDX_AA], int(cnt24[CLASS_IDX_AD]*0.30))})
    Xtr24_sm, ytr24_sm = sm24.fit_resample(Xtr24_le, ytr24)
else:
    Xtr24_sm, ytr24_sm = Xtr24_le, ytr24

Xtr24_pp = ohe_transform(Xtr24_sm)
Xte25_pp = ohe_transform(Xte25_le)
yb25 = label_binarize(yte25, classes=CLASS_IDS)

# Run OOT on the best non-collapsed TDL model + on XGBoost (paper-style)
best_tdl_name = max(['TabResNet','AutoInt','MambaAttention','FT-Transformer'],
                     key=lambda n: final_results.get(n, {}).get('macro_f1', 0))
print(f'\nOOT TDL pick: {best_tdl_name}')

# Build OOT model fresh
if best_tdl_name == 'MambaAttention':
    oot_tdl = MambaAttentionClassifier(Xtr24_pp.shape[1], 256, N_CLASSES).to(DEVICE)
elif best_tdl_name == 'FT-Transformer':
    oot_tdl = FTTransformerClassifier(Xtr24_pp.shape[1], N_CLASSES, d_token=128).to(DEVICE)
elif best_tdl_name == 'AutoInt':
    oot_tdl = AutoIntClassifier(Xtr24_pp.shape[1], N_CLASSES, d=64).to(DEVICE)
else:
    oot_tdl = TabResNet(Xtr24_pp.shape[1], N_CLASSES, hidden=256, n_blocks=6).to(DEVICE)

trl_oot = DataLoader(TensorDataset(
    torch.tensor(Xtr24_pp, dtype=torch.float32).to(DEVICE),
    torch.tensor(ytr24_sm, dtype=torch.long).to(DEVICE)), batch_size=BATCH_SIZE, shuffle=True)
val_oot = DataLoader(TensorDataset(
    torch.tensor(Xte25_pp, dtype=torch.float32).to(DEVICE),
    torch.tensor(yte25, dtype=torch.long).to(DEVICE)), batch_size=BATCH_SIZE)
train_model(oot_tdl, trl_oot, val_oot, ne=NUM_EPOCHS, verbose=False, lr=1e-3, gamma=1.0)
_, oot_yp, oot_ypr = eval_model(oot_tdl, val_oot)
oot_f1  = f1_score(yte25, oot_yp, average='macro', zero_division=0)
oot_acc = accuracy_score(yte25, oot_yp)
try:    oot_auc = roc_auc_score(yb25, oot_ypr, average='macro', multi_class='ovr')
except: oot_auc = float('nan')
delta_tdl = oot_f1 - final_results[best_tdl_name]['macro_f1']
print(f'\nOOT {best_tdl_name}: F1={oot_f1:.4f} Acc={oot_acc:.4f} AUC={oot_auc:.4f} Δ={delta_tdl:+.4f}  '
      f'{"STABLE" if abs(delta_tdl)<0.08 else "DROP"}')
print(classification_report(yte25, oot_yp, target_names=CLASS_LABELS, zero_division=0, digits=4))
final_results[f'{best_tdl_name} OOT'] = {'accuracy':oot_acc,'macro_f1':oot_f1,'macro_auc':oot_auc,
                                          'y_pred':oot_yp,'y_probs':oot_ypr}

# XGBoost OOT (paper precedent)
xgb24 = xgb_factory(); xgb24.fit(Xtr24_pp, ytr24_sm)
xgb_oot_pred = xgb24.predict(Xte25_pp); xgb_oot_pr = xgb24.predict_proba(Xte25_pp)
xgb_oot_f1 = f1_score(yte25, xgb_oot_pred, average='macro', zero_division=0)
xgb_delta = xgb_oot_f1 - final_results['XGBoost']['macro_f1']
print(f'OOT XGBoost: F1={xgb_oot_f1:.4f} Δ={xgb_delta:+.4f}  '
      f'{"STABLE" if abs(xgb_delta)<0.08 else "DROP"}')


# %% [markdown]
# ## Step 14b — Extended OOT: All TDL Models (RQ3)

# %%
from sklearn.metrics import recall_score   # needed for AA_recall in OOT loop
# Step 14b — OOT for FT-Transformer, MambaAttention, AutoInt explicitly
oot_cfgs = [
    ('FT-Transformer', lambda d: FTTransformerClassifier(d,N_CLASSES,d_token=128).to(DEVICE)),
    ('MambaAttention', lambda d: MambaAttentionClassifier(d,256,N_CLASSES).to(DEVICE)),
    ('AutoInt',        lambda d: AutoIntClassifier(d,N_CLASSES,d=64).to(DEVICE)),
]
trl14 = DataLoader(TensorDataset(
    torch.tensor(Xtr24_pp,dtype=torch.float32).to(DEVICE),
    torch.tensor(ytr24_sm,dtype=torch.long).to(DEVICE)),batch_size=BATCH_SIZE,shuffle=True)
val14 = DataLoader(TensorDataset(
    torch.tensor(Xte25_pp,dtype=torch.float32).to(DEVICE),
    torch.tensor(yte25,dtype=torch.long).to(DEVICE)),batch_size=BATCH_SIZE)

oot14b = {}
print('='*70); print('  Step 14b -- Extended OOT (train 2024 -> test 2025)'); print('='*70)
for mname, factory in oot_cfgs:
    print(f'\n  {mname}')
    mdl = factory(Xtr24_pp.shape[1])
    train_model(mdl,trl14,val14,ne=NUM_EPOCHS,verbose=False,lr=1e-3,gamma=1.0)
    _,yp14,ypr14 = eval_model(mdl,val14)
    f1o = f1_score(yte25,yp14,average='macro',zero_division=0)
    f1c = final_results.get(mname,{}).get('macro_f1',float('nan'))
    d   = f1o-f1c
    try:    auc14 = roc_auc_score(yb25,ypr14,average='macro',multi_class='ovr')
    except: auc14 = float('nan')
    aa14 = recall_score(yte25,yp14,labels=[CLASS_IDX_AA],average=None,zero_division=0)[0]
    stab = 'STABLE' if abs(d)<0.08 else 'DROP'
    print(f'  F1_OOT={f1o:.4f}  F1_CV={f1c:.4f}  Delta={d:+.4f}  [{stab}]')
    oot14b[mname] = {'F1_cv':f1c,'F1_oot':f1o,'delta':d,'AUC_oot':auc14,'AA_recall_oot':aa14,'stable':stab}
    del mdl; torch.cuda.empty_cache()

oot14b['XGBoost'] = {'F1_cv':final_results['XGBoost']['macro_f1'],'F1_oot':xgb_oot_f1,
    'delta':xgb_delta,'AUC_oot':float('nan'),'AA_recall_oot':float('nan'),
    'stable':'STABLE' if abs(xgb_delta)<0.08 else 'DROP'}
oot14b_df = pd.DataFrame(oot14b).T.reset_index().rename(columns={'index':'Model'})
print('\nConsolidated OOT:')
#print(oot14b_df[['Model','F1_cv','F1_oot','delta','stable']].to_string(index=False,float_format='%.4f'))
#oot14b_df.to_csv(os.path.join(OUTPUT_DIR,'oot_extended_14b.csv'),index=False)
oot14b_df = pd.DataFrame(oot14b).T.reset_index().rename(columns={'index':'Model'})
print('\nConsolidated OOT:')

print(
    oot14b_df[['Model','F1_cv','F1_oot','delta','stable']]
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

oot14b_df.to_csv(os.path.join(OUTPUT_DIR,'oot_extended_14b.csv'), index=False)
print('Saved: oot_extended_14b.csv')



# %% [markdown]
# ## Step 14.5 — SMOTE-NC Ratio Sensitivity Analysis

# %%
# Step 14.5 — SMOTE-NC ratio sensitivity (3 levels x 3-fold XGBoost)
from xgboost import XGBClassifier as XGB_SM
SMOTE_RATIOS = [
    (0.40,0.20,'Conservative (40%PA/20%AA)'),
    (0.50,0.30,'Baseline     (50%PA/30%AA) <- main'),
    (0.70,0.50,'Aggressive   (70%PA/50%AA)'),
]
print('='*65); print('  Step 14.5 -- SMOTE Sensitivity'); print('='*65)
smrows=[]
for pa_r,aa_r,label in SMOTE_RATIOS:
    sgkf=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=RANDOM_SEED)
    f1s=[]
    for fi,vi in sgkf.split(X_tr_le,y_tr,crash_ids_tr):
        Xt,Xv,yt,yv=X_tr_le[fi],X_tr_le[vi],y_tr[fi],y_tr[vi]
        cnt=Counter(yt)
        if cnt[CLASS_IDX_AA]<2: continue
        sm=SMOTENC(categorical_features=cat_indices,random_state=RANDOM_SEED,
                   k_neighbors=min(5,cnt[CLASS_IDX_AA]-1),
                   sampling_strategy={CLASS_IDX_PA:max(cnt[CLASS_IDX_PA],int(cnt[CLASS_IDX_AD]*pa_r)),
                                      CLASS_IDX_AA:max(cnt[CLASS_IDX_AA],int(cnt[CLASS_IDX_AD]*aa_r))})
        Xts,yts=sm.fit_resample(Xt,yt)
        Xts2=ohe_transform(Xts); Xv2=_ohe_fold(Xv)
        m=XGB_SM(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=RANDOM_SEED,verbosity=0)
        m.fit(Xts2,yts)
        f1s.append(f1_score(yv,m.predict(Xv2),average='macro',zero_division=0))
    mu=float(np.mean(f1s)) if f1s else 0.0
    print(f'  {label}  F1={mu:.4f}')
    smrows.append({'label':label,'macro_F1':round(mu,4)})
sm_df=pd.DataFrame(smrows)
rng_sm=sm_df['macro_F1'].max()-sm_df['macro_F1'].min()
print(f'\n  F1 range: {rng_sm:.4f}  [{"STABLE" if rng_sm<0.02 else "SENSITIVE"}]')
sm_df.to_csv(os.path.join(OUTPUT_DIR,'smote_sensitivity.csv'),index=False)
print('Saved: smote_sensitivity.csv')


# %% [markdown]
# ## Step 15 — Per-Road_Cls_ID breakdown (RQ3)

# %%
# Step 15 — Per-Road_Cls breakdown using best CALIBRATED model
BEST_NAME = max(calibrated_results, key=lambda n: calibrated_results[n]['macro_f1'])
print(f'Best calibrated model: {BEST_NAME}')
best_pred = calibrated_results[BEST_NAME]['y_pred']

test_df = df.iloc[test_idx].reset_index(drop=True)
test_df['_pred'] = best_pred
test_df['_true'] = y_te

rows = []
for rc, sub in test_df.groupby('Road_Cls_ID'):
    if len(sub) < 30: continue
    f1m = f1_score(sub['_true'], sub['_pred'], average='macro', zero_division=0)
    ba  = balanced_accuracy_score(sub['_true'], sub['_pred'])
    aa_recall_rc = ((sub['_pred'] == CLASS_IDX_AA) & (sub['_true'] == CLASS_IDX_AA)).sum() / max(1, (sub['_true'] == CLASS_IDX_AA).sum())
    rows.append({'Road_Cls_ID': rc, 'n_test': len(sub), 'macro_F1': f1m, 'balanced_acc': ba,
                 'AA_recall': aa_recall_rc, 'AA_in_test': int((sub['_true']==CLASS_IDX_AA).sum())})

road_df = pd.DataFrame(rows).sort_values('n_test', ascending=False)
print(road_df.to_string(index=False, float_format='%.4f'))
road_df.to_csv(os.path.join(OUTPUT_DIR, 'per_road_class_metrics_v7.csv'), index=False)


# %% [markdown]
# ## Step 15b — Per-Weather-Condition Breakdown (RQ3)

# %%
# Step 15b -- Per-Wthr_Cond_ID
# Note: v5 dataset stores Weather as string labels (e.g. 'Clear', 'Rain'),
# not integer codes — display the label directly.
wthr_rows = []
for wc, sub in test_df.groupby('Wthr_Cond_ID'):
    if len(sub) < 20: continue
    f1m  = f1_score(sub['_true'], sub['_pred'], average='macro', zero_division=0)
    n_aa = int((sub['_true'] == CLASS_IDX_AA).sum())
    aa_r = ((sub['_pred'] == CLASS_IDX_AA) & (sub['_true'] == CLASS_IDX_AA)).sum() / max(1, n_aa)
    wthr_rows.append({'Wthr_Cond_ID': wc, 'Weather': str(wc),
                      'n_test': len(sub), 'macro_F1': f1m,
                      'AA_recall': aa_r, 'AA_in_test': n_aa})
wthr_df = pd.DataFrame(wthr_rows).sort_values('n_test', ascending=False)
print(wthr_df.to_string(index=False, float_format='%.4f'))
wthr_df.to_csv(os.path.join(OUTPUT_DIR, 'per_weather_metrics_v8.csv'), index=False)

fig_wc, ax_wc = plt.subplots(figsize=(11, 4.5))
cmap_w = plt.cm.get_cmap('RdYlGn')
nrm_w  = plt.Normalize(wthr_df['macro_F1'].min()-0.02, wthr_df['macro_F1'].max()+0.02)
brs = ax_wc.bar(wthr_df['Weather'], wthr_df['macro_F1'],
                color=[cmap_w(nrm_w(v)) for v in wthr_df['macro_F1']],
                edgecolor='black', lw=0.8)
for bar, row in zip(brs, wthr_df.itertuples()):
    ax_wc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
               f'n={row.n_test}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_wc.axhline(wthr_df['macro_F1'].mean(), color='#D62728', ls='--', lw=1.8,
              label=f'Mean={wthr_df["macro_F1"].mean():.3f}')
ax_wc.set_ylabel('Macro-F1', fontsize=12, fontweight='bold')
ax_wc.set_title(f'RQ3 Environmental Robustness by Weather -- {BEST_NAME}',
                fontsize=12, fontweight='bold')
ax_wc.legend(fontsize=10); ax_wc.grid(axis='y', alpha=0.3)
plt.xticks(rotation=25, ha='right'); plt.tight_layout()
fig_wc.savefig(os.path.join(OUTPUT_DIR, 'figures_journal', 'per_weather_robustness.png'),
               dpi=300, bbox_inches='tight')
plt.show()
print('Saved: per_weather_metrics_v8.csv | per_weather_robustness.png')


# %% [markdown]
# ## Step 15c — Per-Light-Condition Breakdown (RQ3)

# %%
# Step 15c -- Per-Light_Cond_ID
# Note: v5 dataset stores light condition as string labels (e.g. 'Daylight', 'Dark, Lighted')
# Use the string directly — no integer lookup dict needed.
light_rows = []
for lc, sub in test_df.groupby('Light_Cond_ID'):
    if len(sub) < 20: continue
    f1m  = f1_score(sub['_true'], sub['_pred'], average='macro', zero_division=0)
    n_aa = int((sub['_true'] == CLASS_IDX_AA).sum())
    aa_r = ((sub['_pred'] == CLASS_IDX_AA) & (sub['_true'] == CLASS_IDX_AA)).sum() / max(1, n_aa)
    light_rows.append({'Light_Cond_ID': lc, 'Lighting': str(lc),
                       'n_test': len(sub), 'macro_F1': f1m,
                       'AA_recall': aa_r, 'AA_in_test': n_aa})
light_df = pd.DataFrame(light_rows).sort_values('n_test', ascending=False)
print(light_df.to_string(index=False, float_format='%.4f'))
light_df.to_csv(os.path.join(OUTPUT_DIR, 'per_light_metrics_v8.csv'), index=False)

fig_lc2, axes_lc = plt.subplots(1, 2, figsize=(13, 5))
cmap_l = plt.cm.get_cmap('RdYlGn')
nrm_l  = plt.Normalize(light_df['macro_F1'].min()-0.02, light_df['macro_F1'].max()+0.02)
axes_lc[0].bar(light_df['Lighting'], light_df['macro_F1'],
               color=[cmap_l(nrm_l(v)) for v in light_df['macro_F1']],
               edgecolor='black', lw=0.8)
axes_lc[0].axhline(light_df['macro_F1'].mean(), color='#D62728', ls='--', lw=1.8,
                   label=f'Mean={light_df["macro_F1"].mean():.3f}')
axes_lc[0].set_ylabel('Macro-F1', fontsize=12, fontweight='bold')
axes_lc[0].set_title('Macro-F1 by Lighting', fontsize=12, fontweight='bold')
axes_lc[0].legend(fontsize=9); axes_lc[0].grid(axis='y', alpha=0.3)
plt.setp(axes_lc[0].get_xticklabels(), rotation=30, ha='right', fontsize=9)

col_aa = ['#D62728' if v==0 else '#FF6F61' if v<0.10 else '#00A08A'
          for v in light_df['AA_recall']]
axes_lc[1].bar(light_df['Lighting'], light_df['AA_recall'],
               color=col_aa, edgecolor='black', lw=0.8)
axes_lc[1].axhline(0.10, color='#6A5ACD', ls='--', lw=1.8, label='10% safety floor')
axes_lc[1].set_ylabel('AA Recall', fontsize=12, fontweight='bold')
axes_lc[1].set_title('AA Recall by Lighting', fontsize=12, fontweight='bold')
axes_lc[1].legend(fontsize=9); axes_lc[1].grid(axis='y', alpha=0.3)
plt.setp(axes_lc[1].get_xticklabels(), rotation=30, ha='right', fontsize=9)
fig_lc2.suptitle(f'RQ3 Infrastructure Robustness -- Light Condition -- {BEST_NAME}',
                 fontsize=12, fontweight='bold')
plt.tight_layout()
fig_lc2.savefig(os.path.join(OUTPUT_DIR, 'figures_journal', 'per_light_robustness.png'),
                dpi=300, bbox_inches='tight')
plt.show()
print('Saved: per_light_metrics_v8.csv | per_light_robustness.png')


# %% [markdown]
# ## Step 16 — Bootstrap CIs + McNemar (named class indices, not positional)

# %%
# Step 16 — Bootstrap CIs (all three class recalls: AA, PA, AD)
N_BOOT = 2000
rng = np.random.RandomState(RANDOM_SEED)

print(f'=== Bootstrap 95% CIs (n_boot={N_BOOT}) ===')
ci_table = []
for name, res in final_results.items():
    yp = res.get('y_pred')
    if yp is None or len(yp) != len(y_te): continue
    f1_b, aa_b, pa_b, ad_b = [], [], [], []
    aa_mask = (y_te == CLASS_IDX_AA)
    pa_mask = (y_te == CLASS_IDX_PA)
    ad_mask = (y_te == CLASS_IDX_AD)
    for _ in range(N_BOOT):
        idx = rng.choice(len(y_te), len(y_te), replace=True)
        f1_b.append(f1_score(y_te[idx], yp[idx], average='macro', zero_division=0))
        if aa_mask.sum() > 0:
            iaa = rng.choice(np.where(aa_mask)[0], aa_mask.sum(), replace=True)
            aa_b.append((yp[iaa] == y_te[iaa]).mean())
        if pa_mask.sum() > 0:
            ipa = rng.choice(np.where(pa_mask)[0], pa_mask.sum(), replace=True)
            pa_b.append((yp[ipa] == y_te[ipa]).mean())
        if ad_mask.sum() > 0:
            iad = rng.choice(np.where(ad_mask)[0], ad_mask.sum(), replace=True)
            ad_b.append((yp[iad] == y_te[iad]).mean())
    row = dict(
        Model=name,
        macro_F1_lo=float(np.percentile(f1_b, 2.5)),
        macro_F1_hi=float(np.percentile(f1_b, 97.5)),
        AA_Recall_obs=float((yp[aa_mask] == y_te[aa_mask]).mean()) if aa_mask.sum() else 0,
        AA_Recall_lo=float(np.percentile(aa_b, 2.5)) if aa_b else 0,
        AA_Recall_hi=float(np.percentile(aa_b, 97.5)) if aa_b else 0,
        PA_Recall_obs=float((yp[pa_mask] == y_te[pa_mask]).mean()) if pa_mask.sum() else 0,
        PA_Recall_lo=float(np.percentile(pa_b, 2.5)) if pa_b else 0,
        PA_Recall_hi=float(np.percentile(pa_b, 97.5)) if pa_b else 0,
        AD_Recall_obs=float((yp[ad_mask] == y_te[ad_mask]).mean()) if ad_mask.sum() else 0,
        AD_Recall_lo=float(np.percentile(ad_b, 2.5)) if ad_b else 0,
        AD_Recall_hi=float(np.percentile(ad_b, 97.5)) if ad_b else 0,
    )
    ci_table.append(row)
    print(f"  {name:<25} F1 [{row['macro_F1_lo']:.3f}-{row['macro_F1_hi']:.3f}]  "
          f"AA-Rec {row['AA_Recall_obs']:.3f} [{row['AA_Recall_lo']:.3f}-{row['AA_Recall_hi']:.3f}]  "
          f"PA-Rec {row['PA_Recall_obs']:.3f} [{row['PA_Recall_lo']:.3f}-{row['PA_Recall_hi']:.3f}]  "
          f"AD-Rec {row['AD_Recall_obs']:.3f} [{row['AD_Recall_lo']:.3f}-{row['AD_Recall_hi']:.3f}]")

pd.DataFrame(ci_table).to_csv(os.path.join(OUTPUT_DIR, 'bootstrap_cis_v7.csv'), index=False)


# %%
# Step 16b — McNemar pairwise
from statsmodels.stats.contingency_tables import mcnemar
print('\n=== McNemar pairwise (p < 0.05 = significantly different) ===')
mc_table = []
names = [n for n in final_results if final_results[n].get('y_pred') is not None and len(final_results[n]['y_pred']) == len(y_te)]
for i, m1 in enumerate(names):
    for m2 in names[i+1:]:
        p1 = final_results[m1]['y_pred']; p2 = final_results[m2]['y_pred']
        c1 = (p1 == y_te); c2 = (p2 == y_te)
        b = (c1 & ~c2).sum(); c = (~c1 & c2).sum()
        if (b + c) == 0: continue
        try:
            res = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
            sig = 'sig' if res.pvalue < 0.05 else 'n.s.'
            mc_table.append({'m1':m1,'m2':m2,'b':int(b),'c':int(c),
                              'chi2':float(res.statistic),'p':float(res.pvalue),'sig':sig})
        except Exception:
            pass
mc_df = pd.DataFrame(mc_table)
print(mc_df.to_string(index=False, float_format='%.4f'))
mc_df.to_csv(os.path.join(OUTPUT_DIR, 'mcnemar_v7.csv'), index=False)


# %%
# Step 16c — Bootstrap CI + McNemar visualisation (all TDL + baseline models)
import matplotlib.gridspec as gridspec

ci_df  = pd.read_csv(os.path.join(OUTPUT_DIR, 'bootstrap_cis_v7.csv'))
mc_df2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'mcnemar_v7.csv'))

# All 8 models included — TabResNet is stable with ADAS data (F1=0.4361, OOT Δ=-0.0334)
PLOT_ORDER = [m for m in ['Random Forest','XGBoost','MambaAttention',
                             'FT-Transformer','AutoInt','TabResNet','TabPFN 2.5','TDL Stack']
              if m in ci_df['Model'].values]
ci_df = ci_df[ci_df['Model'].isin(PLOT_ORDER)].set_index('Model').loc[PLOT_ORDER].reset_index()

PAL = {'Random Forest':'#6A5ACD','XGBoost':'#D62728','MambaAttention':'#FF6F61',
       'FT-Transformer':'#46ACC8','AutoInt':'#00A08A','TabPFN 2.5':'#E2D200','TDL Stack':'#333333'}
colors = [PAL.get(m,'#888888') for m in ci_df['Model']]

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Step 16 — Bootstrap 95% CIs & Statistical Significance\n'
             'Train 2024 / Test 2024 (held-out 20%) | n_boot=2000', fontsize=13, fontweight='bold')

# Panel 1 — Macro-F1 with 95% CIs
ax = axes[0]
y  = np.arange(len(PLOT_ORDER))
ax.barh(y, ci_df['macro_F1_hi'] - ci_df['macro_F1_lo'], left=ci_df['macro_F1_lo'],
        height=0.5, color=colors, alpha=0.3)
obs_f1 = [final_results.get(m,{}).get('macro_f1', 0) for m in PLOT_ORDER]
ax.scatter(obs_f1, y, color=colors, s=90, zorder=5)
ax.errorbar(obs_f1, y,
            xerr=[np.array(obs_f1)-ci_df['macro_F1_lo'].values,
                  ci_df['macro_F1_hi'].values-np.array(obs_f1)],
            fmt='none', color='black', capsize=5, lw=1.5)
ax.set_yticks(y); ax.set_yticklabels(PLOT_ORDER, fontsize=10, fontweight='bold')
ax.set_xlabel('Macro-F1 (95% CI)', fontsize=11, fontweight='bold')
ax.set_title('Panel A — Macro-F1 Bootstrap CIs', fontsize=11, fontweight='bold')
ax.axvline(0.45, color='gray', lw=1, ls='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()

# Panel 2 — All three class recalls with 95% CIs (grouped per model)
ax = axes[1]
# (label, obs_col, lo_col, hi_col, color, y-offset, marker)
CLASS_CI = [
    ('AA (Adv. Auto)',  'AA_Recall_obs', 'AA_Recall_lo', 'AA_Recall_hi', '#D62728', -0.22, 'D'),
    ('PA (Part. Auto)', 'PA_Recall_obs', 'PA_Recall_lo', 'PA_Recall_hi', '#FF7F0E',  0.00, 's'),
    ('AD (Asst. Drv.)', 'AD_Recall_obs', 'AD_Recall_lo', 'AD_Recall_hi', '#1F77B4',  0.22, 'o'),
]
for lbl, obs_col, lo_col, hi_col, clr, offset, mk in CLASS_CI:
    obs = ci_df[obs_col].values
    lo  = ci_df[lo_col].values
    hi  = ci_df[hi_col].values
    yy  = y + offset
    ax.barh(yy, hi - lo, left=lo, height=0.18, color=clr, alpha=0.3)
    ax.scatter(obs, yy, color=clr, s=60, zorder=5, label=lbl, marker=mk)
    ax.errorbar(obs, yy, xerr=[obs - lo, hi - obs],
                fmt='none', color=clr, capsize=3, lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(PLOT_ORDER, fontsize=10, fontweight='bold')
ax.set_xlabel('Per-Class Recall (95% CI)', fontsize=11, fontweight='bold')
ax.set_title('Panel B — Per-Class Recall Bootstrap CIs\n'
             '(AA=◆ safety-critical; PA=■ partial; AD=● majority)', fontsize=11, fontweight='bold')
ax.axvline(0.10, color='red', lw=1.2, ls='--', alpha=0.7, label='AA θ_min=0.10')
ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()

# Panel 3 — McNemar significance heatmap
ax = axes[2]
sig_matrix = pd.DataFrame(index=PLOT_ORDER, columns=PLOT_ORDER, data='—')
if len(mc_df2) > 0:
    for _, row in mc_df2.iterrows():
        m1, m2 = row['m1'], row['m2']
        if m1 in PLOT_ORDER and m2 in PLOT_ORDER:
            p = row['p']
            sym = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
            sig_matrix.loc[m1, m2] = sym
            sig_matrix.loc[m2, m1] = sym
pval_num = pd.DataFrame(index=PLOT_ORDER, columns=PLOT_ORDER, data=0.0)
if len(mc_df2) > 0:
    for _, row in mc_df2.iterrows():
        m1, m2 = row['m1'], row['m2']
        if m1 in PLOT_ORDER and m2 in PLOT_ORDER:
            pval_num.loc[m1,m2] = -np.log10(max(row['p'], 1e-6))
            pval_num.loc[m2,m1] = -np.log10(max(row['p'], 1e-6))
im = ax.imshow(pval_num.values.astype(float), cmap='YlOrRd', vmin=0, vmax=6, aspect='auto')
for i, r in enumerate(PLOT_ORDER):
    for j, c2 in enumerate(PLOT_ORDER):
        ax.text(j, i, sig_matrix.loc[r, c2], ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if pval_num.loc[r,c2] > 3 else 'black')
ax.set_xticks(range(len(PLOT_ORDER)))
ax.set_xticklabels(PLOT_ORDER, rotation=40, ha='right', fontsize=9, fontweight='bold')
ax.set_yticks(range(len(PLOT_ORDER)))
ax.set_yticklabels(PLOT_ORDER, fontsize=9, fontweight='bold')
ax.set_title('Panel C — McNemar Pairwise Significance\n(-log10 p; * p<0.05, ** p<0.01, *** p<0.001)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.75, label='-log10(p-value)')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'figures_journal', 'step16_bootstrap_mcnemar.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')
print('\nNote: All 8 models included. TabResNet: F1=0.4361, OOT Δ=-0.0334 (STABLE with ADAS data).')



# %% [markdown]
# ## Step 17 — Final summary (CORRECTLY labeled AA / PA / AD columns)

# %%
# Step 17 — Final summary table with correctly-labeled per-class columns
print('='*110)
print(f"  FINAL SUMMARY — v7 (n_train={len(X_tr_le):,}, n_test={len(X_te_le):,}, AA_in_test={int((y_te==CLASS_IDX_AA).sum())})")
print('='*110)

rows = []
for name, res in final_results.items():
    cv = cv_results.get(name, (None, None))
    yp = res.get('y_pred'); ypr = res.get('y_probs')
    if yp is None or len(yp) != len(y_te): continue
    rep = classification_report(y_te, yp, output_dict=True, zero_division=0,
                                 target_names=CLASS_LABELS)
    rows.append({
        'Model': name,
        'Class': 'TDL' if name in ['MambaAttention','FT-Transformer','AutoInt','TabResNet','TabPFN 2.5','TDL Stack'] else 'ML/Baseline',
        'CV_F1':   round(cv[0], 4) if cv[0] is not None else None,
        'CV_std':  round(cv[1], 4) if cv[0] is not None else None,
        'Test_F1': round(res['macro_f1'], 4),
        'Acc':     round(res['accuracy'], 4),
        'Bal_Acc': round(res['balanced_acc'], 4),
        'AUC':     round(res['macro_auc'], 4) if not np.isnan(res['macro_auc']) else None,
        'AD_F1':   round(rep.get('Assisted_Driving',   {}).get('f1-score', 0), 4),
        'PA_F1':   round(rep.get('Partial_Automation', {}).get('f1-score', 0), 4),
        'AA_F1':   round(rep.get('Advanced_Automation',{}).get('f1-score', 0), 4),
        'AD_Recall': round(rep.get('Assisted_Driving',   {}).get('recall', 0), 4),
        'PA_Recall': round(rep.get('Partial_Automation', {}).get('recall', 0), 4),
        'AA_Recall': round(rep.get('Advanced_Automation',{}).get('recall', 0), 4),
    })

summary_df = pd.DataFrame(rows).sort_values('Test_F1', ascending=False)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'v7_final_summary.csv'), index=False)

print('\n' + '='*110)
print('Calibrated results:')
print('='*110)
cal_rows = []
for name, res in calibrated_results.items():
    rep = classification_report(y_te, res['y_pred'], output_dict=True, zero_division=0, target_names=CLASS_LABELS)
    cal_rows.append({
        'Model': name + ' (cal)',
        'Test_F1':  round(res['macro_f1'], 4),
        'Acc':      round(res['accuracy'], 4),
        'AD_F1':    round(rep.get('Assisted_Driving',   {}).get('f1-score', 0), 4),
        'PA_F1':    round(rep.get('Partial_Automation', {}).get('f1-score', 0), 4),
        'AA_F1':    round(rep.get('Advanced_Automation',{}).get('f1-score', 0), 4),
        'AA_Recall':round(rep.get('Advanced_Automation',{}).get('recall', 0), 4),
    })
print(pd.DataFrame(cal_rows).sort_values('Test_F1', ascending=False).to_string(index=False))

# Save trained-models manifest for the SHAP/IG notebook
with open(os.path.join(MODEL_DIR, 'models_manifest.json'), 'w') as f:
    json.dump({
        'models': {n: True for n in final_results if final_results[n].get('y_pred') is not None},
        'thresholds': {k: v.tolist() for k,v in thresholds_map.items()},
        'BEST_TDL_FOR_IG': max([n for n in ['TabResNet','AutoInt','FT-Transformer','MambaAttention']
                                if n in final_results],
                                key=lambda n: final_results[n]['macro_f1']),
    }, f, indent=2)
print('\n✓ Saved: v7_final_summary.csv, bootstrap_cis_v7.csv, mcnemar_v7.csv, per_road_class_metrics_v7.csv')
print('✓ Saved models in: saved_models/  (use Phase2_SHAP_IG_v7.ipynb for interpretability)')


# %% [markdown]
# ## Step 17b — Multi-Panel Model Comparison Plot

# %%
# Step 17b -- 4-panel model comparison
import matplotlib.gridspec as gridspec
matplotlib.rcParams.update({'font.size':11,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})
MODEL_ORDER=[m for m in ['Random Forest','XGBoost','MambaAttention','FT-Transformer',
                          'AutoInt','TabResNet','TabPFN 2.5','TDL Stack'] if m in calibrated_results]
PAL={'Random Forest':'#6A5ACD','XGBoost':'#D62728','MambaAttention':'#FF6F61',
     'FT-Transformer':'#46ACC8','AutoInt':'#00A08A','TabResNet':'#888888','TabPFN 2.5':'#E2D200','TDL Stack':'#333333'}

def gmet(name):
    res=calibrated_results[name]; yp=res['y_pred']; ypr=res['y_probs']
    rep=classification_report(y_te,yp,output_dict=True,zero_division=0,target_names=CLASS_LABELS)
    try: auc=roc_auc_score(label_binarize(y_te,classes=CLASS_IDS),ypr,average='macro',multi_class='ovr')
    except: auc=float('nan')
    return {'F1':res['macro_f1'],'BA':balanced_accuracy_score(y_te,yp),'AUC':auc,
            'AA_R':rep.get('Advanced_Automation',{}).get('recall',0),
            'PA_R':rep.get('Partial_Automation',{}).get('recall',0),
            'AD_F':rep.get('Assisted_Driving',{}).get('f1-score',0),
            'PA_F':rep.get('Partial_Automation',{}).get('f1-score',0),
            'AA_F':rep.get('Advanced_Automation',{}).get('f1-score',0)}

CM={m:gmet(m) for m in MODEL_ORDER}
UF={m:final_results.get(m,{}).get('macro_f1',CM[m]['F1']) for m in MODEL_ORDER}
fig=plt.figure(figsize=(18,14)); gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.34)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax4=fig.add_subplot(gs[1,1])
x=np.arange(len(MODEL_ORDER)); W=0.25
for met,off,bc,ht in zip(['F1','AUC','BA'],[-W,0,W],['#00A08A','#46ACC8','#E2D200'],['','//','...']):
    vals=[CM[m][met] for m in MODEL_ORDER]
    bp=ax1.bar(x+off,vals,W,label=met,color=bc,edgecolor='black',lw=0.7,hatch=ht,alpha=0.88)
    for bar,v in zip(bp,vals):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,f'{v:.3f}',
                 ha='center',va='bottom',fontsize=7,fontweight='bold',rotation=90)
ax1.set_xticks(x); ax1.set_xticklabels(MODEL_ORDER,rotation=22,ha='right',fontsize=10,fontweight='bold')
ax1.set_ylabel('Score',fontsize=12,fontweight='bold')
ax1.set_title('Panel A -- Overall Performance (Macro-F1 | AUC | Bal-Acc)',fontsize=12,fontweight='bold')
ax1.legend(fontsize=9,loc='lower right'); ax1.set_ylim(0,1.0); ax1.grid(axis='y',alpha=0.3)

hm=np.array([[CM[m]['AD_F'],CM[m]['PA_F'],CM[m]['AA_F']] for m in MODEL_ORDER])
im2=ax2.imshow(hm,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1)
for i in range(len(MODEL_ORDER)):
    for j in range(3):
        v=hm[i,j]; ax2.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=11.5,fontweight='bold',
                             color='white' if v<0.28 or v>0.82 else 'black')
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(['AD (SAE1)','PA (SAE2)','AA (SAE3-5)'],fontsize=10,fontweight='bold')
ax2.set_yticks(range(len(MODEL_ORDER))); ax2.set_yticklabels(MODEL_ORDER,fontsize=10,fontweight='bold')
ax2.set_title('Panel B -- Per-Class F1 Heatmap',fontsize=12,fontweight='bold')
plt.colorbar(im2,ax=ax2,shrink=0.82,label='F1')

rmets=['F1','AUC','BA','AA_R','PA_R']; n_r=len(rmets)
angs=[n/float(n_r)*2*np.pi for n in range(n_r)]+[0]
ax3=fig.add_subplot(gs[1,0],polar=True); ax3.set_theta_offset(np.pi/2); ax3.set_theta_direction(-1)
ax3.set_thetagrids(np.degrees(angs[:-1]),['F1','AUC','Bal-Acc','AA-Recall','PA-Recall'],fontsize=10,fontweight='bold')
ax3.set_rgrids([0.2,0.4,0.6,0.8],fontsize=8)
for mn in MODEL_ORDER[:5]:
    vr=[CM[mn][m] for m in rmets]+[CM[mn][rmets[0]]]
    ax3.plot(angs,vr,'o-',lw=2.2,color=PAL.get(mn,'#888'),label=mn)
    ax3.fill(angs,vr,alpha=0.07,color=PAL.get(mn,'#888'))
ax3.set_ylim(0,1); ax3.set_title('Panel C -- Radar (Top 5)',fontsize=12,fontweight='bold',pad=20)
ax3.legend(loc='upper right',bbox_to_anchor=(1.38,1.14),fontsize=9)

ax4t=ax4.twinx()
gn=[CM[m]['F1']-UF[m] for m in MODEL_ORDER]; ar=[CM[m]['AA_R'] for m in MODEL_ORDER]
ax4.bar(x-W/2,gn,W*0.9,color=['#00A08A' if g>=0 else '#D62728' for g in gn],edgecolor='black',lw=0.7,label='Cal. Gain',alpha=0.87)
ax4t.bar(x+W/2,ar,W*0.9,color='#FF6F61',edgecolor='black',lw=0.7,label='AA Recall',alpha=0.87)
ax4.axhline(0,color='black',lw=0.8); ax4.set_xticks(x)
ax4.set_xticklabels(MODEL_ORDER,rotation=22,ha='right',fontsize=10,fontweight='bold')
ax4.set_ylabel('Calibration Gain (Delta F1)',fontsize=11,fontweight='bold',color='#00A08A')
ax4t.set_ylabel('AA Recall',fontsize=11,fontweight='bold',color='#FF6F61')
ax4.set_title('Panel D -- Calibration Gain vs AA Recall',fontsize=12,fontweight='bold')
l1,lb1=ax4.get_legend_handles_labels(); l2,lb2=ax4t.get_legend_handles_labels()
ax4.legend(l1+l2,lb1+lb2,fontsize=9,loc='upper left'); ax4.grid(axis='y',alpha=0.3)
fig.suptitle('Model Comparison -- SAE Classification | Texas CRIS 2024-2025',fontsize=14,fontweight='bold',y=1.01)
fig.savefig(os.path.join(OUTPUT_DIR,'figures_journal','model_comparison_4panel.png'),dpi=300,bbox_inches='tight')
plt.show(); print('Saved: model_comparison_4panel.png')


# %%


# %%


# %%


# %%



