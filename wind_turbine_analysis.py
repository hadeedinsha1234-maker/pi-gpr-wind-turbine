"""
=====================================================================
Wind Turbine Power Output Prediction — Kelmarsh 1 (Senvion MM92)
=====================================================================
Paper: Surrogate Model Comparison for Wind Turbine Power Output
       Prediction Using SCADA Data: GPR, Random Forest, and ANN
Data:  Kelmarsh SCADA 2017-2021 (10-minute resolution)

Models:
  1. Gaussian Process Regression (GPR) — ARD RBF kernel + CI
  2. Random Forest (RF)
  3. Artificial Neural Network (ANN)
  4. Linear Regression (baseline)
  + Sobol Sensitivity Analysis

Instructions:
  1. pip install pandas numpy matplotlib scikit-learn scipy
  2. Place BOOK.csv in the same folder as this script
  3. python wind_turbine_analysis.py
  4. Figures saved in: wind_figures/
     Results saved in: wind_results.csv
     Sobol saved in:   wind_sobol_indices.csv
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_FILE   = 'BOOK.csv'
FIG_DIR     = 'wind_figures'
RANDOM_SEED = 42
TEST_SIZE   = 0.20
GPR_SUBSET  = 500     # GPR training subset (O(n³) constraint)
SOBOL_N     = 1024    # Saltelli base sample → 1024*(2*5+2) = 12288 evaluations
COLOR       = '#1B5E20'  # Dark green for wind

FEATURES = [
    'WindSpeed_ms',
    'WindDir_sin',
    'WindDir_cos',
    'Temperature_C',
    'RotorSpeed_rpm'
]
FEATURE_LABELS = [
    'Wind Speed (m/s)',
    'Wind Dir (sin)',
    'Wind Dir (cos)',
    'Temperature (°C)',
    'Rotor Speed (RPM)'
]
TARGET  = 'Power_kW'
MODELS  = ['Linear Regression', 'Random Forest', 'ANN', 'GPR']
MONTHS  = np.arange(1, 13)
MON_LABS = ['Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']

os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PLOT STYLE — large readable fonts
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         28,
    'axes.titlesize':    32,
    'axes.titleweight':  'bold',
    'axes.labelsize':    28,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   24,
    'ytick.labelsize':   24,
    'legend.fontsize':   24,
    'legend.framealpha': 0.95,
    'axes.linewidth':    2.0,
    'grid.alpha':        0.35,
    'grid.linewidth':    1.2,
    'lines.linewidth':   3.5,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size':  8,
    'ytick.major.size':  8,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#f9f9f9',
    'axes.grid':         True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

def save_fig(fig, name):
    path = f'{FIG_DIR}/{name}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png  ({os.path.getsize(path)/1e6:.1f} MB)")

# ═════════════════════════════════════════════
# PHASE 1 — LOAD AND INSPECT
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 1 — LOADING DATA")
print("="*60)

df = pd.read_csv(DATA_FILE, skiprows=5, header=0)
df.columns = ['DateTime','WindSpeed_ms','NacellePosition_deg',
              'WindDirection_deg','Temperature_C',
              'RotorSpeed_rpm','Power_kW']

print(f"  Raw rows loaded:    {len(df):,}")
print(f"  Date range:         {df['DateTime'].iloc[0]} → {df['DateTime'].iloc[-1]}")

# ═════════════════════════════════════════════
# PHASE 2 — PREPROCESSING
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 2 — PREPROCESSING")
print("="*60)

# 1. Parse datetime
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=False)
df['Month']    = df['DateTime'].dt.month
df['Hour']     = df['DateTime'].dt.hour

# 2. Drop NaN rows
before = len(df)
df.dropna(inplace=True)
print(f"  Dropped NaN rows:       {before - len(df):,}")

# 3. Remove negative power (fault/braking periods)
before = len(df)
df = df[df['Power_kW'] >= 0]
print(f"  Removed negative power: {before - len(df):,}")

# 4. Remove shutdown wind speeds (> 25 m/s)
before = len(df)
df = df[df['WindSpeed_ms'] <= 25.0]
print(f"  Removed high wind rows: {before - len(df):,}")

# 5. Remove power > rated (2100 kW) — sensor spikes
before = len(df)
df = df[df['Power_kW'] <= 2100]
print(f"  Removed power spikes:   {before - len(df):,}")

# 6. Remove rotor speed = 0 with power > 0 (inconsistent)
before = len(df)
df = df[~((df['RotorSpeed_rpm'] < 1) & (df['Power_kW'] > 10))]
print(f"  Removed rotor anomalies:{before - len(df):,}")

print(f"  Clean rows remaining:   {len(df):,}")

# 7. Encode wind direction as sin/cos (cyclic)
df['WindDir_sin'] = np.sin(np.radians(df['WindDirection_deg']))
df['WindDir_cos'] = np.cos(np.radians(df['WindDirection_deg']))

# 8. Sample down for modelling (keep representative subset)
#    Full dataset is ~200k rows — sample 5000 for balance
#    GPR will use 500 of those; RF/ANN use all 5000
print(f"\n  Sampling 5000 points for modelling...")
df_model = df.sample(n=5000, random_state=RANDOM_SEED).reset_index(drop=True)

# 9. Save clean full dataset
df.to_csv('BOOK_clean.csv', index=False)
print(f"  Clean full dataset saved: BOOK_clean.csv ({len(df):,} rows)")

print(f"\n  Feature summary (modelling sample):")
for col in FEATURES + [TARGET]:
    print(f"    {col:<25} min={df_model[col].min():>8.3f}  "
          f"max={df_model[col].max():>8.3f}  "
          f"mean={df_model[col].mean():>8.3f}")

# ═════════════════════════════════════════════
# PHASE 3 — NORMALIZE AND SPLIT
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 3 — NORMALIZE AND SPLIT")
print("="*60)

X = df_model[FEATURES].values
y = df_model[TARGET].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Also keep unscaled y for GPR (normalize_y=True handles internally)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ═════════════════════════════════════════════
# PHASE 4 — TRAIN ALL MODELS
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 4 — MODEL TRAINING")
print("="*60)

cr = {'y_test': y_test, 'X_test': X_test,
      'scaler_X': scaler_X, 'df_model': df_model,
      'X_train': X_train, 'y_train': y_train}

# ── 1. Linear Regression ─────────────────────
print("  [1/4] Linear Regression...", end=' ')
lr = LinearRegression()
lr.fit(X_train, y_train)
yp = lr.predict(X_test)
cr['Linear Regression'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp)
}
print(f"R²={cr['Linear Regression']['R2']:.4f}")

# ── 2. Random Forest ─────────────────────────
print("  [2/4] Random Forest...", end=' ')
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
yp = rf.predict(X_test)
cr['Random Forest'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp),
    'importance': rf.feature_importances_
}
print(f"R²={cr['Random Forest']['R2']:.4f}")

# ── 3. ANN ───────────────────────────────────
print("  [3/4] ANN (64-32)...", end=' ')
ann = MLPRegressor(
    hidden_layer_sizes=(64, 32), activation='relu',
    max_iter=1000, random_state=RANDOM_SEED,
    early_stopping=True, validation_fraction=0.1,
    learning_rate_init=0.001
)
ann.fit(X_train, y_train)
yp = ann.predict(X_test)
cr['ANN'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp)
}
print(f"R²={cr['ANN']['R2']:.4f}")

# ── 4. GPR ───────────────────────────────────
print("  [4/4] GPR (ARD RBF)...", end=' ')
np.random.seed(RANDOM_SEED)
idx    = np.random.choice(len(X_train), GPR_SUBSET, replace=False)
kernel = (ConstantKernel(1.0) *
          RBF(length_scale=[1.0] * len(FEATURES)) +
          WhiteKernel(noise_level=0.1))
gpr = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=5,
    normalize_y=True, alpha=1e-6)
gpr.fit(X_train[idx], y_train[idx])
yp, ys = gpr.predict(X_test, return_std=True)
ls = gpr.kernel_.k1.k2.length_scale
cr['GPR'] = {
    'y_pred': yp, 'y_std': ys,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp),
    'length_scales': ls
}
print(f"R²={cr['GPR']['R2']:.4f}")

# ── RESULTS TABLE ────────────────────────────
print(f"\n  {'Model':<22} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("  " + "-"*52)
for m in MODELS:
    r = cr[m]
    print(f"  {m:<22} {r['R2']:>8.4f} {r['RMSE']:>10.3f} {r['MAE']:>10.3f}")

# Save results CSV
rows = [{'Model': m, 'R2': round(cr[m]['R2'],4),
         'RMSE': round(cr[m]['RMSE'],3),
         'MAE':  round(cr[m]['MAE'],3)} for m in MODELS]
pd.DataFrame(rows).to_csv('wind_results.csv', index=False)
print(f"\n  Results saved: wind_results.csv")

# ═════════════════════════════════════════════
# PHASE 5 — SOBOL SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════
print("\n" + "="*60)
print(f"  PHASE 5 — SOBOL SENSITIVITY (N={SOBOL_N})")
print("="*60)

k = len(FEATURES)
print(f"  Total GPR evaluations: {SOBOL_N*(2*k+2):,}")

sampler = qmc.Sobol(d=2*k, scramble=True, seed=RANDOM_SEED)
raw = sampler.random(SOBOL_N)
A = raw[:, :k]
B = raw[:, k:]

AB = np.array([A.copy() for _ in range(k)])
for i in range(k):
    AB[i][:, i] = B[:, i]

print("  Evaluating GPR on Saltelli samples...", end=' ')
yA  = gpr.predict(A)
yB  = gpr.predict(B)
yAB = np.array([gpr.predict(AB[i]) for i in range(k)])
print("done")

var_total = np.var(np.concatenate([yA, yB]))
S1 = np.zeros(k)
ST = np.zeros(k)
for i in range(k):
    S1[i] = 1 - np.mean((yB - yAB[i])**2) / (2 * var_total)
    ST[i] = np.mean((yA - yAB[i])**2)      / (2 * var_total)
S1 = np.clip(S1, 0, None)

sobol_df = pd.DataFrame({
    'Feature':       FEATURES,
    'Feature_Label': FEATURE_LABELS,
    'S1_FirstOrder': np.round(S1, 4),
    'ST_TotalOrder': np.round(ST, 4),
    'Interaction':   np.round(np.clip(ST - S1, 0, None), 4)
}).sort_values('ST_TotalOrder', ascending=False)

sobol_df.to_csv('wind_sobol_indices.csv', index=False)

print(f"\n  {'Feature':<25} {'S1':>10} {'ST':>10} {'Interaction':>12}")
print("  " + "-"*60)
for _, row in sobol_df.iterrows():
    print(f"  {row['Feature']:<25} {row['S1_FirstOrder']:>10.4f} "
          f"{row['ST_TotalOrder']:>10.4f} {row['Interaction']:>12.4f}")
print(f"\n  Sum S1={S1.sum():.4f}  Sum ST={ST.sum():.4f}")
print(f"  Sobol saved: wind_sobol_indices.csv")

# ── Permutation Importance ───────────────────
print("\n  Computing Permutation Importance...", end=' ')
pi = permutation_importance(gpr, X_test, y_test, n_repeats=15, random_state=RANDOM_SEED)
pi_mean = np.clip(pi.importances_mean, 0, None)
print("done")

# ═════════════════════════════════════════════
# PHASE 6 — FIGURES
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 6 — GENERATING FIGURES")
print("="*60)

yt  = cr['y_test']
lim = [-50, 2150]

# ── FIG 1: Predicted vs Actual (2×2) ─────────
fig, axes = plt.subplots(2, 2, figsize=(22, 20))
fig.suptitle('Predicted vs Actual Power Output\nKelmarsh 1 — Senvion MM92',
             fontsize=36, fontweight='bold', y=1.01)
axes = axes.flatten()
for j, m in enumerate(MODELS):
    ax = axes[j]; yp = cr[m]['y_pred']
    ax.scatter(yt, yp, alpha=0.35, s=12, color=COLOR, edgecolors='none')
    ax.plot(lim, lim, 'k--', lw=2.5, label='1:1 line', zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual Power (kW)')
    ax.set_ylabel('Predicted Power (kW)')
    ax.set_title(m); ax.set_aspect('equal')
    ax.text(0.05, 0.93,
            f'R²   = {cr[m]["R2"]:.4f}\n'
            f'RMSE = {cr[m]["RMSE"]:.2f} kW\n'
            f'MAE  = {cr[m]["MAE"]:.2f} kW',
            transform=ax.transAxes, fontsize=22,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.95))
    ax.legend(fontsize=22)
plt.tight_layout(h_pad=4, w_pad=4)
save_fig(fig, 'Fig1_Predicted_vs_Actual')

# ── FIG 2: Model Comparison Bar Charts ───────
for metric, mlabel in [('R2','R²'), ('RMSE','RMSE (kW)'), ('MAE','MAE (kW)')]:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(f'Model Comparison — {mlabel}\nKelmarsh 1 Wind Turbine', pad=20)
    vals = [cr[m][metric] for m in MODELS]
    x    = np.arange(len(MODELS))
    bars = ax.bar(x, vals, 0.55, color=COLOR, alpha=0.88,
                  edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f'{v:.4f}' if metric=='R2' else f'{v:.2f}',
                ha='center', va='bottom', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylabel(mlabel)
    plt.tight_layout()
    save_fig(fig, f'Fig2_Bar_{metric}')

# ── FIG 3: GPR CI vs Wind Speed ──────────────
yp = cr['GPR']['y_pred']; ys = cr['GPR']['y_std']
X_raw = scaler_X.inverse_transform(X_test)
ws    = X_raw[:, 0]   # WindSpeed_ms is index 0
sidx  = np.argsort(ws)
ws_s, yp_s, ys_s, yt_s = ws[sidx], yp[sidx], ys[sidx], yt[sidx]

fig, ax = plt.subplots(figsize=(16, 9))
ax.scatter(ws_s, yt_s, alpha=0.2, s=12, color='gray', label='Actual', zorder=2)
ax.plot(ws_s, yp_s, color=COLOR, lw=3.5, label='GPR Mean', zorder=4)
ax.fill_between(ws_s, yp_s-1.96*ys_s, yp_s+1.96*ys_s,
                alpha=0.22, color=COLOR, label='±95% CI', zorder=3)
ax.text(0.05, 0.93, f'R² = {cr["GPR"]["R2"]:.4f}',
        transform=ax.transAxes, fontsize=26, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.95))
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (kW)')
ax.set_title('GPR ±95% CI vs Wind Speed — Kelmarsh 1')
ax.legend(loc='upper left')
plt.tight_layout()
save_fig(fig, 'Fig3_GPR_CI_WindSpeed')

# ── FIG 4: GPR CI vs Rotor Speed ─────────────
rs    = X_raw[:, 4]   # RotorSpeed_rpm is index 4
sidx  = np.argsort(rs)
rs_s, yp_s, ys_s, yt_s = rs[sidx], yp[sidx], ys[sidx], yt[sidx]

fig, ax = plt.subplots(figsize=(16, 9))
ax.scatter(rs_s, yt_s, alpha=0.2, s=12, color='gray', label='Actual', zorder=2)
ax.plot(rs_s, yp_s, color=COLOR, lw=3.5, label='GPR Mean', zorder=4)
ax.fill_between(rs_s, yp_s-1.96*ys_s, yp_s+1.96*ys_s,
                alpha=0.22, color=COLOR, label='±95% CI', zorder=3)
ax.text(0.05, 0.93, f'R² = {cr["GPR"]["R2"]:.4f}',
        transform=ax.transAxes, fontsize=26, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.95))
ax.set_xlabel('Rotor Speed (RPM)')
ax.set_ylabel('Power Output (kW)')
ax.set_title('GPR ±95% CI vs Rotor Speed — Kelmarsh 1')
ax.legend(loc='upper left')
plt.tight_layout()
save_fig(fig, 'Fig4_GPR_CI_RotorSpeed')

# ── FIG 5: Power Curve (Wind Speed vs Power) ──
fig, ax = plt.subplots(figsize=(16, 9))
# Bin actual data into wind speed bins
ws_full = df['WindSpeed_ms'].values
pw_full = df['Power_kW'].values
bins    = np.arange(0, 26, 0.5)
bin_idx = np.digitize(ws_full, bins)
ws_mean, pw_mean, pw_std = [], [], []
for b in range(1, len(bins)):
    mask = bin_idx == b
    if mask.sum() > 10:
        ws_mean.append(bins[b-1] + 0.25)
        pw_mean.append(pw_full[mask].mean())
        pw_std.append(pw_full[mask].std())
ws_mean = np.array(ws_mean)
pw_mean = np.array(pw_mean)
pw_std  = np.array(pw_std)

ax.scatter(ws_full[::20], pw_full[::20], alpha=0.08, s=6,
           color='gray', label='Raw SCADA (sampled)', zorder=1)
ax.plot(ws_mean, pw_mean, color=COLOR, lw=3.5,
        label='Binned Mean Power', zorder=4)
ax.fill_between(ws_mean, pw_mean-pw_std, pw_mean+pw_std,
                alpha=0.2, color=COLOR, label='±1 Std Dev', zorder=3)
ax.axhline(2050, color='red', lw=2.5, linestyle='--', label='Rated Power (2050 kW)')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (kW)')
ax.set_title('Measured Power Curve — Kelmarsh 1 (2017–2021)')
ax.set_xlim(0, 25); ax.set_ylim(-50, 2200)
ax.legend(loc='upper left', fontsize=22)
plt.tight_layout()
save_fig(fig, 'Fig5_Power_Curve')

# ── FIG 6: Sobol Indices (S1 + ST grouped) ───
sobol_sorted = sobol_df.sort_values('ST_TotalOrder', ascending=True)
fl   = sobol_sorted['Feature_Label'].tolist()
s1v  = sobol_sorted['S1_FirstOrder'].values
stv  = sobol_sorted['ST_TotalOrder'].values
ypos = np.arange(len(fl))
bh   = 0.35

fig, ax = plt.subplots(figsize=(16, 10))
bars1 = ax.barh(ypos+bh/2, s1v, bh, label='S₁ (First-order)',
                color=COLOR, alpha=0.88, edgecolor='white', linewidth=1.5)
bars2 = ax.barh(ypos-bh/2, stv, bh, label='Sᴛ (Total-order)',
                color=COLOR, alpha=0.40, edgecolor='white',
                linewidth=1.5, hatch='////')
for bar, v in zip(bars1, s1v):
    ax.text(bar.get_width()+0.008, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=20, fontweight='bold')
for bar, v in zip(bars2, stv):
    ax.text(bar.get_width()+0.008, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=20)
ax.set_yticks(ypos)
ax.set_yticklabels(fl, fontsize=24)
ax.set_xlabel('Sobol Sensitivity Index')
ax.set_title('Sobol Sensitivity Analysis — Kelmarsh 1\nFirst-order (S₁) and Total-order (Sᴛ) Indices')
ax.set_xlim(0, max(stv.max(), s1v.max())*1.35)
ax.legend(loc='lower right', fontsize=24)
ax.axvline(0.05, color='red', lw=2, linestyle='--', alpha=0.5)
plt.tight_layout()
save_fig(fig, 'Fig6_Sobol_Indices')

# ── FIG 7: Sobol Interaction ──────────────────
interact = sobol_sorted['Interaction'].values
fig, ax  = plt.subplots(figsize=(14, 9))
bars = ax.barh(fl, interact, color='#E65100', alpha=0.85,
               edgecolor='white', linewidth=1.5, height=0.55)
for bar, v in zip(bars, interact):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=22, fontweight='bold')
ax.set_xlabel('Interaction Effect (Sᴛ − S₁)')
ax.set_title('Feature Interaction Effects — Kelmarsh 1')
ax.set_xlim(0, max(interact.max()*1.35, 0.05))
plt.tight_layout()
save_fig(fig, 'Fig7_Sobol_Interaction')

# ── FIG 8: Sensitivity Comparison (3-panel) ───
ard_ls   = cr['GPR']['length_scales']
ard_sens = 1.0 / np.clip(ard_ls, 0, 20)
ard_pct  = ard_sens / ard_sens.sum() * 100
pi_pct   = np.clip(pi_mean, 0, None)
pi_pct   = pi_pct / pi_pct.sum() * 100 if pi_pct.sum() > 0 else pi_pct

fig, axes = plt.subplots(1, 3, figsize=(28, 10))
fig.suptitle('Sensitivity Analysis Comparison — Kelmarsh 1',
             fontsize=34, fontweight='bold')
titles  = ['ARD Length Scale\n(GPR Kernel)',
           'Permutation Importance\n(R² Drop)',
           'Sobol S₁\n(First-Order Index)']
values  = [ard_pct, pi_pct, S1*100]
xlabels = ['Relative Importance (%)',
           'Relative Importance (%)',
           'First-Order Index × 100']
for ax, title, vals, xlabel in zip(axes, titles, values, xlabels):
    sidx = np.argsort(vals)
    bars = ax.barh([FEATURE_LABELS[j] for j in sidx], vals[sidx],
                   color=COLOR, alpha=0.88, edgecolor='white',
                   linewidth=1.5, height=0.6)
    for bar, v in zip(bars, vals[sidx]):
        ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.set_xlim(0, vals.max()*1.35)
plt.tight_layout(w_pad=3)
save_fig(fig, 'Fig8_Sensitivity_Comparison')

# ── FIG 9: RF Feature Importance ─────────────
imp     = cr['Random Forest']['importance']
imp_pct = imp / imp.sum() * 100
sidx    = np.argsort(imp_pct)

fig, ax = plt.subplots(figsize=(14, 9))
bars = ax.barh([FEATURE_LABELS[j] for j in sidx], imp_pct[sidx],
               color=COLOR, alpha=0.88, edgecolor='white',
               linewidth=1.5, height=0.6)
for bar, v in zip(bars, imp_pct[sidx]):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=22, fontweight='bold')
ax.set_xlabel('Feature Importance (%)')
ax.set_title('Random Forest Feature Importance — Kelmarsh 1')
ax.set_xlim(0, imp_pct.max()*1.3)
plt.tight_layout()
save_fig(fig, 'Fig9_RF_Feature_Importance')

# ── FIG 10: Monthly Power Output Trend ────────
df['Month'] = pd.to_datetime(
    df['DateTime']).dt.month
monthly_mean = df.groupby('Month')['Power_kW'].mean()
monthly_std  = df.groupby('Month')['Power_kW'].std()

fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(MONTHS, monthly_mean.values, color=COLOR, alpha=0.75,
       edgecolor='white', linewidth=1.2, label='Mean Power')
ax.errorbar(MONTHS, monthly_mean.values, yerr=monthly_std.values,
            fmt='none', color='#1A237E', capsize=8, lw=2.5,
            capthick=2.5, label='±1 Std Dev')
ax.set_xticks(MONTHS)
ax.set_xticklabels(MON_LABS)
ax.set_xlabel('Month')
ax.set_ylabel('Mean Power Output (kW)')
ax.set_title('Monthly Average Power Output — Kelmarsh 1 (2017–2021)')
ax.set_ylim(0, 1200)
ax.legend(fontsize=24)
plt.tight_layout()
save_fig(fig, 'Fig10_Monthly_Power_Trend')

# ── FIG 11: Residuals GPR vs ANN ─────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('Prediction Residuals — Kelmarsh 1',
             fontsize=34, fontweight='bold')
for j, m in enumerate(['GPR', 'ANN']):
    ax  = axes[j]
    res = yt - cr[m]['y_pred']
    ax.hist(res, bins=50, color=COLOR, alpha=0.78,
            edgecolor='white', density=True, linewidth=0.8)
    ax.axvline(0, color='black', lw=2.5, linestyle='--', label='Zero')
    ax.axvline(res.mean(), color='red', lw=2.5,
               linestyle='-', label=f'Mean = {res.mean():.2f}')
    ax.text(0.97, 0.97,
            f'Std  = {res.std():.2f} kW\nRMSE = {cr[m]["RMSE"]:.2f} kW',
            transform=ax.transAxes, ha='right', va='top', fontsize=22,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.95))
    ax.set_xlabel('Residual (kW)')
    ax.set_ylabel('Density')
    ax.set_title(m)
    ax.legend()
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig11_Residuals')

# ═════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════
print("\n" + "="*60)
print(f"✓ All figures saved in:  {FIG_DIR}/")
print(f"✓ Model results:         wind_results.csv")
print(f"✓ Sobol indices:         wind_sobol_indices.csv")
print(f"✓ Clean dataset:         BOOK_clean.csv")
print("="*60)
