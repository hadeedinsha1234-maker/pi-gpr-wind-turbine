"""
Complete reproduction pipeline for:
"Physics-Informed Gaussian Process Regression for Wind Turbine Power Prediction"
"""

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================
import subprocess
import sys

required = {'numpy', 'pandas', 'scipy', 'sklearn', 'SALib'}
missing = required - {pkg.key for pkg in __import__('pkg_resources').working_set}
if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
np.random.seed(SEED)
SUBSAMPLE_N = 5000
GPR_TRAIN_N = 500
TEST_SIZE = 1000
RATED_POWER = 2050.0
ROTOR_DIAMETER = 92.0
SWEPT_AREA = np.pi * (ROTOR_DIAMETER / 2) ** 2
RHO_STD = 1.225

# =============================================================================
# 1. LOAD DATA
# =============================================================================
FILE_NAME = 'BOOK_clean.csv'
print(f"Loading {FILE_NAME}...")

try:
    df_raw = pd.read_csv(FILE_NAME)
    print(f"Loaded {len(df_raw)} records, {len(df_raw.columns)} columns.")
    print(f"Columns: {list(df_raw.columns)}")
except FileNotFoundError:
    print(f"ERROR: {FILE_NAME} not found in current directory.")
    print(f"Current directory: {__import__('os').getcwd()}")
    sys.exit(1)

# =============================================================================
# 2. AUTO-DETECT COLUMNS
# =============================================================================
# Try to map columns automatically
col_map = {}

def find_col(options, df_cols):
    for opt in options:
        matches = [c for c in df_cols if opt.lower() in c.lower()]
        if matches:
            return matches[0]
    return None

wind_speed_col = find_col(['wind speed', 'windspeed', 'wind_speed', 'vwind', 'v'], df_raw.columns)
wind_dir_col = find_col(['wind direction', 'winddirection', 'wind_dir', 'direction', 'wd'], df_raw.columns)
rotor_speed_col = find_col(['rotor speed', 'rotorspeed', 'rotor_speed', 'rpm', 'omega'], df_raw.columns)
temp_col = find_col(['temperature', 'temp', 'nacelle temp', 'nacelle_temp', 't'], df_raw.columns)
power_col = find_col(['power', 'active power', 'activepower', 'p'], df_raw.columns)

if not all([wind_speed_col, rotor_speed_col, temp_col, wind_dir_col, power_col]):
    print("ERROR: Could not auto-detect all required columns.")
    print(f"  Wind speed: {wind_speed_col}")
    print(f"  Wind dir: {wind_dir_col}")
    print(f"  Rotor speed: {rotor_speed_col}")
    print(f"  Temperature: {temp_col}")
    print(f"  Power: {power_col}")
    print("\nPlease edit the script and set the column names manually.")
    sys.exit(1)

# Rename to standard names
df_raw = df_raw.rename(columns={
    wind_speed_col: 'V',
    wind_dir_col: 'theta_deg',
    rotor_speed_col: 'omega',
    temp_col: 'T',
    power_col: 'P'
})

print(f"\nMapped columns: V={wind_speed_col}, theta={wind_dir_col}, omega={rotor_speed_col}, T={temp_col}, P={power_col}")

# =============================================================================
# 3. PREPROCESSING PIPELINE
# =============================================================================
def preprocess(df):
    df = df.copy()
    required = ['V', 'omega', 'T', 'theta_deg', 'P']
    
    # Step 1: Remove NaN
    df = df.dropna(subset=required)
    print(f"After NaN removal: {len(df)}")
    
    # Step 2: Remove negative power
    df = df[df['P'] >= 0]
    print(f"After negative power filter: {len(df)}")
    
    # Step 3: Cut-out filter
    df = df[df['V'] <= 25.0]
    print(f"After cut-out filter (V<=25): {len(df)}")
    
    # Step 4: Remove power spikes
    df = df[df['P'] <= 2100.0]
    print(f"After power spike filter: {len(df)}")
    
    # Step 5: Remove rotor anomalies
    df = df[~((df['omega'] < 1.0) & (df['P'] > 10.0))]
    print(f"After rotor anomaly filter: {len(df)}")
    
    # Step 6: Cyclic direction encoding
    theta_rad = np.deg2rad(df['theta_deg'])
    df['sin_theta'] = np.sin(theta_rad)
    df['cos_theta'] = np.cos(theta_rad)
    
    return df

df_clean = preprocess(df_raw)
print(f"\nFinal clean records: {len(df_clean)} ({len(df_clean)/len(df_raw)*100:.1f}% of raw)")

# =============================================================================
# 4. STRATIFIED SUBSAMPLING
# =============================================================================
df_clean['V_bin'] = pd.cut(df_clean['V'], bins=np.arange(0, 26, 0.5), labels=False)
df_clean['V_bin'] = df_clean['V_bin'].fillna(0).astype(int)

if len(df_clean) > SUBSAMPLE_N:
    from sklearn.model_selection import train_test_split as tts
    df_sub, _ = tts(
        df_clean, train_size=SUBSAMPLE_N,
        stratify=df_clean['V_bin'], random_state=SEED
    )
else:
    df_sub = df_clean

print(f"\nSubsample: {len(df_sub)} records")

# =============================================================================
# 5. TRAIN/TEST SPLIT
# =============================================================================
features = ['V', 'omega', 'T', 'sin_theta', 'cos_theta']
target = 'P'

X = df_sub[features].values
y = df_sub[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# 6. MIN-MAX SCALING
# =============================================================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 7. PHYSICS BASELINE
# =============================================================================
def air_density(T_celsius):
    return RHO_STD * 288.15 / (T_celsius + 273.15)

def physics_power(V, T, Cp):
    rho = air_density(T)
    P = 0.5 * rho * SWEPT_AREA * Cp * V**3 / 1000.0
    return np.clip(P, 0, RATED_POWER)

def calibrate_cp(V, P, T):
    rho = air_density(T)
    Cp_raw = P * 1000.0 / (0.5 * rho * SWEPT_AREA * V**3 + 1e-6)
    Cp_raw = np.clip(Cp_raw, 0, 16/27)
    return Cp_raw

train_df = pd.DataFrame(X_train, columns=features)
train_df['P'] = y_train
train_df['Cp'] = calibrate_cp(train_df['V'], train_df['P'], train_df['T'])

CP_PARTIAL = train_df[train_df['V'] < 9]['Cp'].median()
CP_TRANS = train_df[(train_df['V'] >= 9) & (train_df['V'] < 13)]['Cp'].median()
CP_FULL = train_df[train_df['V'] >= 13]['Cp'].median()

print(f"\nCalibrated Cp values:")
print(f"  Partial load (V<9):     {CP_PARTIAL:.4f}")
print(f"  Transition (9<=V<13):   {CP_TRANS:.4f}")
print(f"  Full load (V>=13):      {CP_FULL:.4f}")

def physics_model(V, T):
    P = np.zeros_like(V)
    P[V < 9] = physics_power(V[V < 9], T[V < 9], CP_PARTIAL)
    mask = (V >= 9) & (V < 13)
    P[mask] = physics_power(V[mask], T[mask], CP_TRANS)
    P[V >= 13] = physics_power(V[V >= 13], T[V >= 13], CP_FULL)
    return P

y_physics_train = physics_model(X_train[:, 0], X_train[:, 2])
y_physics_test = physics_model(X_test[:, 0], X_test[:, 2])

# =============================================================================
# 8. SURROGATE MODELS
# =============================================================================
print("\n" + "="*70)
print("TRAINING MODELS...")
print("="*70)

# 8.1 Linear Regression
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 8.2 Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 8.3 ANN
print("Training ANN...")
ann = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=SEED
)
ann.fit(X_train_scaled, y_train)
y_pred_ann = ann.predict(X_test_scaled)

# 8.4 GPR (500-point subset)
print("Training GPR...")
gpr_subset_idx = np.random.choice(len(X_train_scaled), size=GPR_TRAIN_N, replace=False)
X_gpr_train = X_train_scaled[gpr_subset_idx]
y_gpr_train = y_train[gpr_subset_idx]

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
    length_scale=[1.0]*5,
    length_scale_bounds=(1e-2, 10.0)
) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=SEED
)
gpr.fit(X_gpr_train, y_gpr_train)
y_pred_gpr, y_std_gpr = gpr.predict(X_test_scaled, return_std=True)

# 8.5 PI-GPR
print("Training PI-GPR...")
epsilon_train = y_train - y_physics_train
epsilon_gpr_train = epsilon_train[gpr_subset_idx]

pi_gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=SEED
)
pi_gpr.fit(X_gpr_train, epsilon_gpr_train)
epsilon_pred, epsilon_std = pi_gpr.predict(X_test_scaled, return_std=True)
y_pred_pi = y_physics_test + epsilon_pred
y_std_pi = epsilon_std

print("\nAll models trained.")

# =============================================================================
# 9. EVALUATION
# =============================================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

models = {
    'Physics Baseline': y_physics_test,
    'Linear Regression': y_pred_lr,
    'Random Forest': y_pred_rf,
    'ANN': y_pred_ann,
    'GPR': y_pred_gpr,
    'PI-GPR': y_pred_pi
}

print("\n" + "="*70)
print("TABLE 1: MODEL PERFORMANCE")
print("="*70)
print(f"{'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("-"*70)
for name, y_p in models.items():
    print(f"{name:<20} {r2(y_test, y_p):>8.4f} {rmse(y_test, y_p):>10.2f} {mae(y_test, y_p):>10.2f}")

# =============================================================================
# 10. DIEBOLD-MARIANO TEST
# =============================================================================
def dm_test(actual, pred1, pred2, h=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    
    n = len(d)
    mean_d = np.mean(d)
    gamma0 = np.mean((d - mean_d)**2)
    
    if n > h:
        gamma = np.mean((d[:-h] - mean_d) * (d[h:] - mean_d))
        var_d = (gamma0 + 2 * gamma) / n
    else:
        var_d = gamma0 / n
    
    if var_d <= 0:
        var_d = gamma0 / n
    
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

dm_stat, dm_p = dm_test(y_test, y_pred_pi, y_pred_gpr, h=1)
print(f"\n{'='*70}")
print("DIEBOLD-MARIANO TEST: PI-GPR vs GPR")
print(f"{'='*70}")
print(f"DM statistic: {dm_stat:.4f}")
print(f"p-value:      {dm_p:.4f}")
if dm_p < 0.05:
    print("Result: Statistically significant difference at 5% level")
else:
    print("Result: NOT statistically significant at 5% level")

# =============================================================================
# 11. SOBOL SENSITIVITY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SOBOL SENSITIVITY ANALYSIS")
print("="*70)

problem = {
    'num_vars': 5,
    'names': ['V', 'omega', 'T', 'sin_theta', 'cos_theta'],
    'bounds': [[0, 1]] * 5
}

N_base = 1024
print(f"Generating Saltelli samples (N={N_base})...")
param_values = saltelli.sample(problem, N_base)
print(f"Sample matrix shape: {param_values.shape}")

# 11.1 GPR Sobol
print("\nEvaluating GPR model on Sobol samples...")
y_sobol_gpr = gpr.predict(param_values)
print("Analyzing GPR Sobol indices...")
Si_gpr = sobol.analyze(problem, y_sobol_gpr, print_to_console=False)

print("\n" + "="*70)
print("TABLE 2: SOBOL INDICES - GPR POWER MODEL")
print("="*70)
print(f"{'Feature':<18} {'S1':>10} {'ST':>10} {'S1_conf':>10} {'ST_conf':>10}")
print("-"*70)
for i, name in enumerate(problem['names']):
    print(f"{name:<18} {Si_gpr['S1'][i]:>10.4f} {Si_gpr['ST'][i]:>10.4f} "
          f"{Si_gpr['S1_conf'][i]:>10.4f} {Si_gpr['ST_conf'][i]:>10.4f}")

# 11.2 PI-GPR Residual Sobol
print("\nEvaluating PI-GPR residual model on Sobol samples...")
y_sobol_residual = pi_gpr.predict(param_values)
print("Analyzing PI-GPR residual Sobol indices...")
Si_pi = sobol.analyze(problem, y_sobol_residual, print_to_console=False)

print("\n" + "="*70)
print("TABLE 3: SOBOL INDICES - PI-GPR RESIDUAL MODEL")
print("="*70)
print(f"{'Feature':<18} {'S1':>10} {'ST':>10} {'S1_conf':>10} {'ST_conf':>10} {'Interact':>10}")
print("-"*70)
for i, name in enumerate(problem['names']):
    interaction = Si_pi['ST'][i] - Si_pi['S1'][i]
    print(f"{name:<18} {Si_pi['S1'][i]:>10.4f} {Si_pi['ST'][i]:>10.4f} "
          f"{Si_pi['S1_conf'][i]:>10.4f} {Si_pi['ST_conf'][i]:>10.4f} {interaction:>10.4f}")

# =============================================================================
# 12. CONSISTENCY CHECKS
# =============================================================================
print("\n" + "="*70)
print("CONSISTENCY CHECKS")
print("="*70)
sum_s1_gpr = np.sum(Si_gpr['S1'])
sum_s1_pi = np.sum(Si_pi['S1'])
print(f"Sum of S1 (GPR):     {sum_s1_gpr:.4f} (should be <= 1.0)")
print(f"Sum of S1 (PI-GPR):  {sum_s1_pi:.4f} (should be <= 1.0)")
print(f"All ST >= S1 (GPR):  {np.all(Si_gpr['ST'] >= Si_gpr['S1'])}")
print(f"All ST >= S1 (PI):   {np.all(Si_pi['ST'] >= Si_pi['S1'])}")

if sum_s1_pi > 1.0:
    print("\nWARNING: Sum of S1 > 1.0 indicates input dependence or numerical noise.")
    print("This is expected if V and omega are strongly correlated.")
    print("Consider using correlation-corrected indices or grouping correlated inputs.")

# =============================================================================
# 13. SAVE RESULTS
# =============================================================================
results = {
    'models': {k: {'rmse': float(rmse(y_test, v)), 
                   'mae': float(mae(y_test, v)), 
                   'r2': float(r2(y_test, v))} 
               for k, v in models.items()},
    'dm_test': {'statistic': float(dm_stat), 'p_value': float(dm_p)},
    'sobol_gpr': {k: v.tolist() if hasattr(v, 'tolist') else v 
                  for k, v in Si_gpr.items()},
    'sobol_pi': {k: v.tolist() if hasattr(v, 'tolist') else v 
                 for k, v in Si_pi.items()},
    'cp_values': {'partial': float(CP_PARTIAL), 
                  'trans': float(CP_TRANS), 
                  'full': float(CP_FULL)},
    'consistency': {
        'sum_s1_gpr': float(sum_s1_gpr),
        'sum_s1_pi': float(sum_s1_pi),
        'st_ge_s1_gpr': bool(np.all(Si_gpr['ST'] >= Si_gpr['S1'])),
        'st_ge_s1_pi': bool(np.all(Si_pi['ST'] >= Si_pi['S1']))
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("DONE! Results saved to analysis_results.json")
print("="*70)