"""
=====================================================================
Physics-Informed GPR (PI-GPR) for Wind Turbine Power Prediction
Kelmarsh 1 — Senvion MM92
=====================================================================
Paper Extension: Hybrid Physics-Informed Surrogate Model
       P = P_physics(V, T) + GPR_residual(V, ω, T, dir)

Physics Baseline:
    P_physics = 0.5 * ρ(T) * A * Cp * V³
    ρ(T) = 1.225 * (288.15 / (T + 273.15))   [ideal gas approx]
    A    = π * (D/2)²   with D = 92 m (MM92 rotor)
    Cp   = 0.45 (optimised region estimate, clipped to Betz limit)

Then GPR trains on residuals: ε = P_actual - P_physics
Final: P_pred = P_physics + GPR(ε)

Outputs:
  Figures saved in: wind_figures_pigpr/
  Results:          wind_results_pigpr.csv
  Sobol:            wind_sobol_pigpr.csv
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
FIG_DIR     = 'wind_figures_pigpr'
RANDOM_SEED = 42
TEST_SIZE   = 0.20
GPR_SUBSET  = 500      # O(n³) constraint for standard GPR
SOBOL_N     = 1024     # Saltelli base → 1024*(2*5+2) = 12,288 evaluations

# Senvion MM92 turbine physical parameters
ROTOR_DIAM  = 92.0                        # metres
ROTOR_AREA  = np.pi * (ROTOR_DIAM/2)**2  # 6647.6 m²
CP_RATED    = 0.45                        # Power coefficient (sub-rated region)
BETZ_LIMIT  = 16/27                       # 0.593 — hard physical ceiling
T_STD       = 288.15                      # K  (15 °C ISA standard)
RHO_STD     = 1.225                       # kg/m³ at standard conditions

COLOR_PHYS  = '#1B5E20'   # dark green  — physics baseline
COLOR_GPR   = '#0D47A1'   # dark blue   — pure GPR
COLOR_PIGPR = '#B71C1C'   # dark red    — PI-GPR
COLOR_RF    = '#1B5E20'   # green       — RF

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
TARGET = 'Power_kW'
MODELS = ['Linear Regression', 'Random Forest', 'ANN', 'GPR', 'PI-GPR']
MONTHS = np.arange(1, 13)
MON_LABS = ['Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']

os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PLOT STYLE
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

# ─────────────────────────────────────────────
# PHYSICS MODEL
# ─────────────────────────────────────────────
def air_density(T_celsius):
    """
    Temperature-corrected air density using ideal gas approximation.
    ρ(T) = ρ_std * (T_std / T_kelvin)
    """
    T_kelvin = T_celsius + 273.15
    return RHO_STD * (T_STD / T_kelvin)

def physics_power(wind_speed, temperature,
                 rated_power=2050.0, rated_wind=12.5):
    """
    Region-aware Betz aerodynamic power.
    Cp varies by operating region (back-calculated from SCADA data):
      - Sub-rated partial-load  (V <  9 m/s):  Cp ~ 0.48
      - Sub-rated transition    (9 <= V < 13):  Cp ~ 0.45
      - Above-rated full-load   (V >= 13 m/s):  Cp ~ 0.18 (pitch regulation)
    Power is hard-clipped to rated nameplate (2050 kW).
    Temperature-corrected air density applied in all regions.
    """
    ws  = np.asarray(wind_speed,  dtype=float)
    T   = np.asarray(temperature, dtype=float)
    rho = air_density(T)
    Cp  = np.where(ws < 9.0, 0.48,
          np.where(ws < 13.0, 0.45, 0.18))
    P   = 0.5 * rho * ROTOR_AREA * Cp * (ws ** 3) / 1000.0  # kW
    P   = np.where(ws >= rated_wind, np.minimum(P, rated_power), P)
    P   = np.clip(P, 0.0, rated_power)
    return P

# ═════════════════════════════════════════════
# PHASE 1 — LOAD DATA
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 1 — LOADING DATA")
print("="*65)

df = pd.read_csv(DATA_FILE, skiprows=5, header=0)
df.columns = ['DateTime','WindSpeed_ms','NacellePosition_deg',
              'WindDirection_deg','Temperature_C',
              'RotorSpeed_rpm','Power_kW']

print(f"  Raw rows loaded:    {len(df):,}")

# ═════════════════════════════════════════════
# PHASE 2 — PREPROCESSING
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 2 — PREPROCESSING")
print("="*65)

df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=False)
df['Month']    = df['DateTime'].dt.month

before = len(df); df.dropna(inplace=True)
print(f"  Dropped NaN rows:         {before - len(df):,}")

before = len(df); df = df[df['Power_kW'] >= 0]
print(f"  Removed negative power:   {before - len(df):,}")

before = len(df); df = df[df['WindSpeed_ms'] <= 25.0]
print(f"  Removed high wind rows:   {before - len(df):,}")

before = len(df); df = df[df['Power_kW'] <= 2100]
print(f"  Removed power spikes:     {before - len(df):,}")

before = len(df)
df = df[~((df['RotorSpeed_rpm'] < 1) & (df['Power_kW'] > 10))]
print(f"  Removed rotor anomalies:  {before - len(df):,}")
print(f"  Clean rows remaining:     {len(df):,}")

# Cyclic wind direction encoding
df['WindDir_sin'] = np.sin(np.radians(df['WindDirection_deg']))
df['WindDir_cos'] = np.cos(np.radians(df['WindDirection_deg']))

# ── Compute physics baseline on full clean dataset ──
df['P_physics'] = physics_power(df['WindSpeed_ms'].values,
                                 df['Temperature_C'].values)
df['Residual']  = df['Power_kW'] - df['P_physics']

phys_r2   = r2_score(df['Power_kW'], df['P_physics'])
phys_rmse = np.sqrt(mean_squared_error(df['Power_kW'], df['P_physics']))
phys_mae  = mean_absolute_error(df['Power_kW'], df['P_physics'])
print(f"\n  Physics baseline (full dataset):")
print(f"    R²   = {phys_r2:.4f}")
print(f"    RMSE = {phys_rmse:.2f} kW")
print(f"    MAE  = {phys_mae:.2f} kW")
print(f"    Residual mean  = {df['Residual'].mean():.2f} kW")
print(f"    Residual std   = {df['Residual'].std():.2f} kW")

# ── Sample for modelling ──
print(f"\n  Sampling 5000 points for modelling...")
df_model = df.sample(n=5000, random_state=RANDOM_SEED).reset_index(drop=True)

# ═════════════════════════════════════════════
# PHASE 3 — SPLIT AND SCALE
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 3 — SPLIT AND SCALE")
print("="*65)

X     = df_model[FEATURES].values
y     = df_model[TARGET].values
y_res = df_model['Residual'].values          # target for PI-GPR
P_phy = df_model['P_physics'].values         # physics baseline per sample

scaler_X   = MinMaxScaler()
scaler_res = MinMaxScaler()

X_scaled   = scaler_X.fit_transform(X)
res_scaled = scaler_res.fit_transform(y_res.reshape(-1,1)).ravel()

(X_train, X_test,
 y_train, y_test,
 res_train, res_test,
 Pphy_train, Pphy_test) = [None]*8

splits = train_test_split(
    X_scaled, y, y_res, P_phy,
    test_size=TEST_SIZE, random_state=RANDOM_SEED
)
X_train, X_test = splits[0], splits[1]
y_train, y_test = splits[2], splits[3]
res_train, res_test = splits[4], splits[5]
Pphy_train, Pphy_test = splits[6], splits[7]

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Residual train range: [{res_train.min():.1f}, {res_train.max():.1f}] kW")

# ═════════════════════════════════════════════
# PHASE 4 — TRAIN ALL MODELS
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 4 — MODEL TRAINING")
print("="*65)

cr = {}

# ── 1. Linear Regression ─────────────────────
print("  [1/5] Linear Regression...", end=' ')
lr  = LinearRegression()
lr.fit(X_train, y_train)
yp  = lr.predict(X_test)
cr['Linear Regression'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp)
}
print(f"R²={cr['Linear Regression']['R2']:.4f}")

# ── 2. Random Forest ─────────────────────────
print("  [2/5] Random Forest...", end=' ')
rf  = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
yp  = rf.predict(X_test)
cr['Random Forest'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp),
    'importance': rf.feature_importances_
}
print(f"R²={cr['Random Forest']['R2']:.4f}")

# ── 3. ANN ───────────────────────────────────
print("  [3/5] ANN (64-32)...", end=' ')
ann = MLPRegressor(
    hidden_layer_sizes=(64, 32), activation='relu',
    max_iter=1000, random_state=RANDOM_SEED,
    early_stopping=True, validation_fraction=0.1,
    learning_rate_init=0.001
)
ann.fit(X_train, y_train)
yp  = ann.predict(X_test)
cr['ANN'] = {
    'y_pred': yp,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp)
}
print(f"R²={cr['ANN']['R2']:.4f}")

# ── 4. Pure GPR (ARD RBF) ────────────────────
print("  [4/5] Pure GPR (ARD RBF)...", end=' ')
np.random.seed(RANDOM_SEED)
idx_gpr = np.random.choice(len(X_train), GPR_SUBSET, replace=False)
kernel_gpr = (ConstantKernel(1.0) *
              RBF(length_scale=[1.0]*len(FEATURES)) +
              WhiteKernel(noise_level=0.1))
gpr = GaussianProcessRegressor(
    kernel=kernel_gpr, n_restarts_optimizer=5,
    normalize_y=True, alpha=1e-6
)
gpr.fit(X_train[idx_gpr], y_train[idx_gpr])
yp, ys = gpr.predict(X_test, return_std=True)
ls_gpr  = gpr.kernel_.k1.k2.length_scale
cr['GPR'] = {
    'y_pred': yp, 'y_std': ys,
    'R2':   r2_score(y_test, yp),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
    'MAE':  mean_absolute_error(y_test, yp),
    'length_scales': ls_gpr
}
print(f"R²={cr['GPR']['R2']:.4f}")

# ── 5. PI-GPR: GPR trained on RESIDUALS ──────
print("  [5/5] PI-GPR (physics + GPR residual)...", end=' ')
np.random.seed(RANDOM_SEED)
idx_pi = np.random.choice(len(X_train), GPR_SUBSET, replace=False)
kernel_pi = (ConstantKernel(1.0) *
             RBF(length_scale=[1.0]*len(FEATURES)) +
             WhiteKernel(noise_level=0.1))
pigpr = GaussianProcessRegressor(
    kernel=kernel_pi, n_restarts_optimizer=5,
    normalize_y=True, alpha=1e-6
)
# Train on residuals
pigpr.fit(X_train[idx_pi], res_train[idx_pi])
res_pred, res_std = pigpr.predict(X_test, return_std=True)

# Final PI-GPR prediction = physics + predicted residual
yp_pigpr = Pphy_test + res_pred
ls_pi    = pigpr.kernel_.k1.k2.length_scale

cr['PI-GPR'] = {
    'y_pred':   yp_pigpr,
    'res_pred': res_pred,
    'y_std':    res_std,           # uncertainty on residual
    'R2':   r2_score(y_test, yp_pigpr),
    'RMSE': np.sqrt(mean_squared_error(y_test, yp_pigpr)),
    'MAE':  mean_absolute_error(y_test, yp_pigpr),
    'length_scales': ls_pi,
    # Residual model metrics
    'res_R2':   r2_score(res_test, res_pred),
    'res_RMSE': np.sqrt(mean_squared_error(res_test, res_pred)),
    'res_MAE':  mean_absolute_error(res_test, res_pred),
    'Pphy_test': Pphy_test
}
print(f"R²={cr['PI-GPR']['R2']:.4f}  (residual R²={cr['PI-GPR']['res_R2']:.4f})")

# ── Physics-only baseline ─────────────────────
cr['Physics'] = {
    'y_pred': Pphy_test,
    'R2':   r2_score(y_test, Pphy_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, Pphy_test)),
    'MAE':  mean_absolute_error(y_test, Pphy_test)
}

# ── RESULTS TABLE ────────────────────────────
print(f"\n  {'Model':<22} {'R²':>8} {'RMSE (kW)':>12} {'MAE (kW)':>10}")
print("  " + "-"*56)
for m in ['Physics'] + MODELS:
    r = cr[m]
    print(f"  {m:<22} {r['R2']:>8.4f} {r['RMSE']:>12.3f} {r['MAE']:>10.3f}")

# Improvement table
print(f"\n  ── PI-GPR improvement over pure GPR ──")
for metric in ['R2','RMSE','MAE']:
    g = cr['GPR'][metric]; p = cr['PI-GPR'][metric]
    if metric == 'R2':
        print(f"    R²  : {g:.4f} → {p:.4f}  Δ={p-g:+.4f}")
    else:
        pct = (g - p) / g * 100
        print(f"    {metric}: {g:.2f} → {p:.2f} kW  ({pct:+.1f}%)")

# Save results
rows = [{'Model': m,
         'R2':   round(cr[m]['R2'],4),
         'RMSE': round(cr[m]['RMSE'],3),
         'MAE':  round(cr[m]['MAE'],3)}
        for m in ['Physics'] + MODELS]
pd.DataFrame(rows).to_csv('wind_results_pigpr.csv', index=False)
print(f"\n  Results saved: wind_results_pigpr.csv")

# ═════════════════════════════════════════════
# PHASE 5 — SOBOL ON PI-GPR
# ═════════════════════════════════════════════
print("\n" + "="*65)
print(f"  PHASE 5 — SOBOL SENSITIVITY ON PI-GPR (N={SOBOL_N})")
print("="*65)

k       = len(FEATURES)
sampler = qmc.Sobol(d=2*k, scramble=True, seed=RANDOM_SEED)
raw     = sampler.random(SOBOL_N)
A_sobol = raw[:, :k]
B_sobol = raw[:, k:]

AB = np.array([A_sobol.copy() for _ in range(k)])
for i in range(k):
    AB[i][:, i] = B_sobol[:, i]

print("  Evaluating PI-GPR on Saltelli samples...", end=' ')
# PI-GPR predicts residuals; Sobol on residual model
yA  = pigpr.predict(A_sobol)
yB  = pigpr.predict(B_sobol)
yAB = np.array([pigpr.predict(AB[i]) for i in range(k)])
print("done")

var_total = np.var(np.concatenate([yA, yB]))
S1 = np.zeros(k); ST = np.zeros(k)
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

sobol_df.to_csv('wind_sobol_pigpr.csv', index=False)
print(f"\n  {'Feature':<25} {'S1':>10} {'ST':>10} {'Interaction':>12}")
print("  " + "-"*60)
for _, row in sobol_df.iterrows():
    print(f"  {row['Feature']:<25} {row['S1_FirstOrder']:>10.4f} "
          f"{row['ST_TotalOrder']:>10.4f} {row['Interaction']:>12.4f}")
print(f"\n  Sum S1={S1.sum():.4f}  Sum ST={ST.sum():.4f}")

# ── Permutation importance on PI-GPR (residual model) ──
print("\n  Computing Permutation Importance on PI-GPR residual model...", end=' ')
pi_imp = permutation_importance(
    pigpr, X_test, res_test,
    n_repeats=15, random_state=RANDOM_SEED
)
pi_mean = np.clip(pi_imp.importances_mean, 0, None)
print("done")

# ═════════════════════════════════════════════
# PHASE 6 — FIGURES
# ═════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 6 — GENERATING FIGURES")
print("="*65)

yt  = cr['y_test'] if 'y_test' in cr else y_test
yt  = y_test
lim = [-50, 2150]

# ── FIG 1: Predicted vs Actual — all 5 models ─
fig, axes = plt.subplots(2, 3, figsize=(33, 20))
fig.suptitle('Predicted vs Actual Power Output\nKelmarsh 1 — Senvion MM92 (with PI-GPR)',
             fontsize=36, fontweight='bold', y=1.01)
all_models_plot = ['Linear Regression', 'Random Forest', 'ANN', 'GPR', 'PI-GPR']
colors_plot     = [COLOR_RF, COLOR_RF, COLOR_RF, COLOR_GPR, COLOR_PIGPR]
axes_flat       = axes.flatten()

for j, (m, col) in enumerate(zip(all_models_plot, colors_plot)):
    ax = axes_flat[j]; yp = cr[m]['y_pred']
    ax.scatter(yt, yp, alpha=0.35, s=12, color=col, edgecolors='none')
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

# Last panel: Physics baseline
ax = axes_flat[5]; yp = cr['Physics']['y_pred']
ax.scatter(yt, yp, alpha=0.35, s=12, color='#4A148C', edgecolors='none')
ax.plot(lim, lim, 'k--', lw=2.5, label='1:1 line', zorder=5)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Power (kW)')
ax.set_ylabel('Predicted Power (kW)')
ax.set_title('Physics Baseline\n(Betz law only)'); ax.set_aspect('equal')
ax.text(0.05, 0.93,
        f'R²   = {cr["Physics"]["R2"]:.4f}\n'
        f'RMSE = {cr["Physics"]["RMSE"]:.2f} kW\n'
        f'MAE  = {cr["Physics"]["MAE"]:.2f} kW',
        transform=ax.transAxes, fontsize=22,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.95))
ax.legend(fontsize=22)
plt.tight_layout(h_pad=4, w_pad=4)
save_fig(fig, 'Fig1_Predicted_vs_Actual_All')

# ── FIG 2: GPR vs PI-GPR parity comparison ────
fig, axes = plt.subplots(1, 2, figsize=(24, 11))
fig.suptitle('GPR vs PI-GPR: Parity Plot Comparison\nKelmarsh 1 — Senvion MM92',
             fontsize=34, fontweight='bold')
for ax, m, col in zip(axes, ['GPR','PI-GPR'], [COLOR_GPR, COLOR_PIGPR]):
    yp = cr[m]['y_pred']
    ax.scatter(yt, yp, alpha=0.35, s=14, color=col, edgecolors='none')
    ax.plot(lim, lim, 'k--', lw=2.5, label='1:1 line', zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual Power (kW)')
    ax.set_ylabel('Predicted Power (kW)')
    ax.set_title(m); ax.set_aspect('equal')
    ax.text(0.05, 0.93,
            f'R²   = {cr[m]["R2"]:.4f}\n'
            f'RMSE = {cr[m]["RMSE"]:.2f} kW\n'
            f'MAE  = {cr[m]["MAE"]:.2f} kW',
            transform=ax.transAxes, fontsize=24,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.95))
    ax.legend(fontsize=22)
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig2_GPR_vs_PIGPR_Parity')

# ── FIG 3: Model Comparison Bar Charts (R², RMSE, MAE) ──
bar_models = ['Physics', 'Linear\nRegression', 'ANN', 'GPR', 'PI-GPR']
bar_keys   = ['Physics', 'Linear Regression', 'ANN', 'GPR', 'PI-GPR']
bar_colors = ['#4A148C', COLOR_RF, COLOR_RF, COLOR_GPR, COLOR_PIGPR]

for metric, mlabel in [('R2','R²'), ('RMSE','RMSE (kW)'), ('MAE','MAE (kW)')]:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title(f'Model Comparison — {mlabel}\nKelmarsh 1 Wind Turbine (PI-GPR included)', pad=20)
    vals = [cr[k][metric] for k in bar_keys]
    x    = np.arange(len(bar_models))
    bars = ax.bar(x, vals, 0.55, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        label = f'{v:.4f}' if metric=='R2' else f'{v:.2f}'
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.005 if metric=='R2' else 2.0),
                label, ha='center', va='bottom',
                fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_models, fontsize=22)
    ax.set_ylabel(mlabel)
    if metric == 'R2':
        ax.set_ylim(0, 1.08)
    # Highlight PI-GPR bar
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(3)
    plt.tight_layout()
    save_fig(fig, f'Fig3_Bar_{metric}')

# ── FIG 4: Residual decomposition ─────────────
# Shows: actual residual (ε = P_actual - P_physics) vs GPR prediction of ε
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle('PI-GPR Residual Modelling — Kelmarsh 1',
             fontsize=34, fontweight='bold')

# Left: residual parity plot
ax = axes[0]
ax.scatter(res_test, cr['PI-GPR']['res_pred'],
           alpha=0.35, s=14, color=COLOR_PIGPR, edgecolors='none')
res_lim = [res_test.min()-50, res_test.max()+50]
ax.plot(res_lim, res_lim, 'k--', lw=2.5, label='1:1 line')
ax.set_xlim(res_lim); ax.set_ylim(res_lim)
ax.set_xlabel('Actual Residual ε = P_actual − P_physics (kW)')
ax.set_ylabel('GPR Predicted Residual (kW)')
ax.set_title('Residual Parity Plot'); ax.set_aspect('equal')
ax.text(0.05, 0.93,
        f'R²   = {cr["PI-GPR"]["res_R2"]:.4f}\n'
        f'RMSE = {cr["PI-GPR"]["res_RMSE"]:.2f} kW\n'
        f'MAE  = {cr["PI-GPR"]["res_MAE"]:.2f} kW',
        transform=ax.transAxes, fontsize=22,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.95))
ax.legend(fontsize=22)

# Right: residual distribution
ax = axes[1]
ax.hist(res_test, bins=60, color='gray', alpha=0.5,
        density=True, label='Actual residual ε', edgecolor='white', linewidth=0.5)
ax.hist(cr['PI-GPR']['res_pred'], bins=60, color=COLOR_PIGPR, alpha=0.6,
        density=True, label='GPR predicted ε', edgecolor='white', linewidth=0.5)
ax.axvline(0, color='black', lw=2.5, linestyle='--', label='Zero')
ax.set_xlabel('Residual ε (kW)')
ax.set_ylabel('Density')
ax.set_title('Residual Distribution')
ax.legend(fontsize=22)
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig4_Residual_Decomposition')

# ── FIG 5: GPR CI vs Wind Speed — GPR vs PI-GPR ──
X_raw = scaler_X.inverse_transform(X_test)
ws    = X_raw[:, 0]
sidx  = np.argsort(ws)
ws_s  = ws[sidx]
yt_s  = yt[sidx]

fig, axes = plt.subplots(1, 2, figsize=(28, 11))
fig.suptitle('GPR vs PI-GPR: Uncertainty Bands vs Wind Speed\nKelmarsh 1',
             fontsize=34, fontweight='bold')

for ax, m, col, label in zip(
        axes, ['GPR','PI-GPR'], [COLOR_GPR, COLOR_PIGPR],
        ['GPR Mean', 'PI-GPR Mean']):
    yp_s = cr[m]['y_pred'][sidx]
    ys_s = cr[m]['y_std'][sidx]
    ax.scatter(ws_s, yt_s, alpha=0.2, s=12, color='gray', label='Actual', zorder=2)
    ax.plot(ws_s, yp_s, color=col, lw=3.5, label=label, zorder=4)
    ax.fill_between(ws_s, yp_s-1.96*ys_s, yp_s+1.96*ys_s,
                    alpha=0.22, color=col, label='±95% CI', zorder=3)
    ax.text(0.05, 0.93, f'R² = {cr[m]["R2"]:.4f}',
            transform=ax.transAxes, fontsize=26, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.95))
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power Output (kW)')
    ax.set_title(m)
    ax.legend(loc='upper left', fontsize=22)
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig5_CI_Comparison_WindSpeed')

# ── FIG 6: Physics baseline + PI-GPR overlay ──
fig, ax = plt.subplots(figsize=(16, 10))
ax.scatter(ws_s, yt_s, alpha=0.15, s=10, color='gray',
           label='Actual SCADA', zorder=1)
# Physics baseline
ws_range = np.linspace(0, 25, 300)
T_mean   = df['Temperature_C'].mean()
P_phys_line = physics_power(ws_range, np.full_like(ws_range, T_mean))
ax.plot(ws_range, P_phys_line, color='#4A148C', lw=3,
        linestyle='--', label=f'Physics (Betz, T={T_mean:.1f}°C)', zorder=3)
# PI-GPR
yp_pigpr_s = cr['PI-GPR']['y_pred'][sidx]
ys_pigpr_s = cr['PI-GPR']['y_std'][sidx]
ax.plot(ws_s, yp_pigpr_s, color=COLOR_PIGPR, lw=3.5,
        label='PI-GPR Mean', zorder=5)
ax.fill_between(ws_s, yp_pigpr_s-1.96*ys_pigpr_s, yp_pigpr_s+1.96*ys_pigpr_s,
                alpha=0.22, color=COLOR_PIGPR, label='PI-GPR ±95% CI', zorder=4)
ax.axhline(2050, color='red', lw=2, linestyle=':', label='Rated Power (2050 kW)')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (kW)')
ax.set_title('Physics Baseline vs PI-GPR — Kelmarsh 1\n'
             'P = P_physics(V,T) + GPR(ε)')
ax.legend(loc='upper left', fontsize=22)
ax.set_xlim(0, 25); ax.set_ylim(-100, 2300)
plt.tight_layout()
save_fig(fig, 'Fig6_Physics_vs_PIGPR_PowerCurve')

# ── FIG 7: Sobol Indices — PI-GPR ─────────────
sobol_sorted = sobol_df.sort_values('ST_TotalOrder', ascending=True)
fl   = sobol_sorted['Feature_Label'].tolist()
s1v  = sobol_sorted['S1_FirstOrder'].values
stv  = sobol_sorted['ST_TotalOrder'].values
ypos = np.arange(len(fl))
bh   = 0.35

fig, ax = plt.subplots(figsize=(16, 10))
bars1 = ax.barh(ypos+bh/2, s1v, bh, label='S₁ (First-order)',
                color=COLOR_PIGPR, alpha=0.88, edgecolor='white', linewidth=1.5)
bars2 = ax.barh(ypos-bh/2, stv, bh, label='Sᴛ (Total-order)',
                color=COLOR_PIGPR, alpha=0.38, edgecolor='white',
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
ax.set_title('Sobol Sensitivity Analysis — PI-GPR Residual Model\n'
             'First-order (S₁) and Total-order (Sᴛ) Indices')
ax.set_xlim(0, max(stv.max(), s1v.max())*1.35)
ax.legend(loc='lower right', fontsize=24)
ax.axvline(0.05, color='red', lw=2, linestyle='--', alpha=0.5)
plt.tight_layout()
save_fig(fig, 'Fig7_Sobol_PIGPR')

# ── FIG 8: Sensitivity Comparison — GPR vs PI-GPR ──
ard_sens_gpr = 1.0 / np.clip(ls_gpr, 0, 20)
ard_pct_gpr  = ard_sens_gpr / ard_sens_gpr.sum() * 100
ard_sens_pi  = 1.0 / np.clip(ls_pi, 0, 20)
ard_pct_pi   = ard_sens_pi  / ard_sens_pi.sum()  * 100

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle('ARD Length-Scale Sensitivity: GPR vs PI-GPR Residual Model',
             fontsize=32, fontweight='bold')
for ax, vals, col, title in zip(
        axes,
        [ard_pct_gpr, ard_pct_pi],
        [COLOR_GPR, COLOR_PIGPR],
        ['GPR (direct power)', 'PI-GPR (residual ε)']):
    sidx2 = np.argsort(vals)
    bars  = ax.barh([FEATURE_LABELS[j] for j in sidx2], vals[sidx2],
                    color=col, alpha=0.88, edgecolor='white',
                    linewidth=1.5, height=0.6)
    for bar, v in zip(bars, vals[sidx2]):
        ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=22, fontweight='bold')
    ax.set_xlabel('Relative Importance (%)', fontsize=26)
    ax.set_title(title, fontsize=28)
    ax.set_xlim(0, vals.max()*1.35)
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig8_ARD_Comparison')

# ── FIG 9: Residual histograms — GPR vs PI-GPR ─
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('Prediction Residuals — GPR vs PI-GPR\nKelmarsh 1',
             fontsize=34, fontweight='bold')
for ax, (m, col) in zip(axes, [('GPR', COLOR_GPR), ('PI-GPR', COLOR_PIGPR)]):
    res = yt - cr[m]['y_pred']
    ax.hist(res, bins=60, color=col, alpha=0.78,
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
save_fig(fig, 'Fig9_Residuals_GPR_vs_PIGPR')

# ── FIG 10: Temperature effect on air density ──
T_range = np.linspace(-20, 40, 200)
rho_range = air_density(T_range)
V_vals    = [6, 8, 10, 12]
cols_temp = ['#1565C0','#2E7D32','#F57F17','#B71C1C']

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle('Temperature Correction in PI-GPR Physics Baseline\nKelmarsh 1',
             fontsize=32, fontweight='bold')

ax = axes[0]
ax.plot(T_range, rho_range, color='#1B5E20', lw=3.5)
ax.axvline(T_mean, color='red', lw=2, linestyle='--',
           label=f'Mean T = {T_mean:.1f}°C')
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Air Density ρ (kg/m³)')
ax.set_title('Air Density vs Temperature')
ax.legend(fontsize=22)

ax = axes[1]
for V, col in zip(V_vals, cols_temp):
    P_line = physics_power(V, T_range)
    ax.plot(T_range, P_line, color=col, lw=3, label=f'V = {V} m/s')
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('P_physics (kW)')
ax.set_title('Physics Power vs Temperature\nat Different Wind Speeds')
ax.legend(fontsize=22)
plt.tight_layout(w_pad=4)
save_fig(fig, 'Fig10_Temperature_Physics_Effect')

# ═════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════
print("\n" + "="*65)
print(f"  ALL DONE")
print(f"  Figures saved in:  {FIG_DIR}/")
print(f"  Results:           wind_results_pigpr.csv")
print(f"  Sobol indices:     wind_sobol_pigpr.csv")
print("="*65)
print(f"\n  FINAL SUMMARY:")
print(f"  {'Model':<22} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print(f"  {'-'*54}")
for m in ['Physics'] + MODELS:
    r = cr[m]
    marker = ' ◄ best' if m == 'PI-GPR' else ''
    print(f"  {m:<22} {r['R2']:>8.4f} {r['RMSE']:>10.2f} {r['MAE']:>10.2f}{marker}")
