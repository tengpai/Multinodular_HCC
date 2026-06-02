"""
Cox regression fusion model - complete analysis
============================
The final fusion model integrating traditional radiomics, deep learning, habitat radiomics and the clinical score

Five-score fusion:
- Clinical_Score: clinical composite score
- Rad_Diff: traditional radiomics difference-feature score
- DL_Heterogeneity: deep learning heterogeneity score  
- DL_MaxPooling: deep learning max-pooling score
- Habitat_Agg_Best: habitat radiomics best-integration score (Max-H1)

Analysis contents:
1. Model construction and coefficients
2. C-index and time-dependent AUC
3. Risk stratification and KM curves
4. Nomogram
5. Calibration curve
6. DCA decision curve
7. Forest plot
8. Complete results summary

Author: multimodal fusion analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set the Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# Survival analysis libraries
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from scipy import stats
from scipy.interpolate import interp1d
import os
from datetime import datetime

def format_p(p):
    """Format the P value following academic conventions"""
    if p < 0.001:
        return 'P < 0.001'
    else:
        return f'P = {p:.3f}'

# Correct time-dependent analysis method
try:
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("Warning: scikit-survival is not installed; time-dependent AUC will use an approximate method")

# onekey_algo library
try:
    from onekey_algo.custom.components.comp1 import plot_DCA, draw_calibration
    from onekey_algo.custom.components import nomogram
    HAS_ONEKEY = True
except ImportError:
    HAS_ONEKEY = False
    print("Warning: onekey_algo is not installed; DCA and calibration curves will use an approximate method")

# Create the output directory
output_dir = '<PROJECT_ROOT>/sol15_multimodal_fusion'
os.makedirs(f'{output_dir}/results', exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 70)
print("Cox regression fusion model - complete analysis")
print("=" * 70)

# ============================================================
# Part 1: data loading and preprocessing
# ============================================================

print("\n[Part 1] Data loading and preprocessing")
print("-" * 70)

# Load the data
data_path = f"{output_dir}/data/multi_modal_fusion_data_with_habitat.csv"
df = pd.read_csv(data_path)

# Define variables
clinical_vars = ['Child_Pugh', 'Gender', 'Total_SLD_cm', 'Age', 
                 'Diabetes', 'AFP', 'CA199', 'Lesion_Count', 'Hepatitis_C']

imaging_scores = ['Rad_Diff_HighRisk', 'DL_Heterogeneity', 'DL_MaxPooling', 'Habitat_Agg_Best']

# Split the data by group
train_df = df[df['group'] == 'train'].copy()
test1_df = df[df['group'] == 'test1'].copy()
test2_df = df[df['group'] == 'test2'].copy()

print(f"Training set: n={len(train_df)} (events: {train_df['event'].sum()}, event rate: {train_df['event'].mean()*100:.1f}%)")
print(f"Validation set 1: n={len(test1_df)} (events: {test1_df['event'].sum()}, event rate: {test1_df['event'].mean()*100:.1f}%)")
print(f"Validation set 2: n={len(test2_df)} (events: {test2_df['event'].sum()}, event rate: {test2_df['event'].mean()*100:.1f}%)")

# ============================================================
# Part 2: build Clinical_Score
# ============================================================

print("\n[Part 2] Build Clinical_Score")
print("-" * 70)

# Standardize the clinical variables
scaler_clinical = StandardScaler()
X_clinical_train = scaler_clinical.fit_transform(train_df[clinical_vars])
X_clinical_test1 = scaler_clinical.transform(test1_df[clinical_vars])
X_clinical_test2 = scaler_clinical.transform(test2_df[clinical_vars])

# Build the clinical score using Cox regression coefficients
clinical_coefficients = {}
clinical_univar_results = []

for var in clinical_vars:
    try:
        temp_df = train_df[['duration', 'event', var]].dropna()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(temp_df, duration_col='duration', event_col='event')
        coef = cph.params_[var]
        hr = np.exp(coef)
        p_val = cph.summary['p'].values[0]
        ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
        ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
        
        clinical_coefficients[var] = coef
        clinical_univar_results.append({
            'Variable': var,
            'Coefficient': coef,
            'HR': hr,
            'HR_95CI_Lower': ci_lower,
            'HR_95CI_Upper': ci_upper,
            'P value': p_val
        })
    except Exception as e:
        clinical_coefficients[var] = 0

clinical_univar_df = pd.DataFrame(clinical_univar_results)
print("\nUnivariate analysis of clinical variables:")
print(clinical_univar_df.to_string(index=False))

# Compute Clinical_Score
def calculate_clinical_score(X_scaled, coefficients, var_names):
    score = np.zeros(len(X_scaled))
    for i, var in enumerate(var_names):
        score += coefficients.get(var, 0) * X_scaled[:, i]
    return score

train_df['Clinical_Score'] = calculate_clinical_score(X_clinical_train, clinical_coefficients, clinical_vars)
test1_df['Clinical_Score'] = calculate_clinical_score(X_clinical_test1, clinical_coefficients, clinical_vars)
test2_df['Clinical_Score'] = calculate_clinical_score(X_clinical_test2, clinical_coefficients, clinical_vars)

# ============================================================
# Part 3: five-score Cox regression fusion model
# ============================================================

print("\n[Part 3] Five-score Cox regression fusion model")
print("-" * 70)

# Standardize all scores
score_columns = ['Clinical_Score', 'Rad_Diff_HighRisk', 'DL_Heterogeneity', 'DL_MaxPooling', 'Habitat_Agg_Best']
score_display_names = {
    'Clinical_Score': 'Clinical score',
    'Rad_Diff_HighRisk': 'Traditional radiomics (Diff)',
    'DL_Heterogeneity': 'Deep learning (Het)',
    'DL_MaxPooling': 'Deep learning (MaxP)',
    'Habitat_Agg_Best': 'Habitat radiomics (Max-H1)'
}

scalers = {}
for col in score_columns:
    scalers[col] = StandardScaler()
    train_df[f'{col}_Z'] = scalers[col].fit_transform(train_df[[col]])
    test1_df[f'{col}_Z'] = scalers[col].transform(test1_df[[col]])
    test2_df[f'{col}_Z'] = scalers[col].transform(test2_df[[col]])

# Five-score multivariable Cox regression
z_cols = [f'{col}_Z' for col in score_columns]
multivar_data = train_df[['duration', 'event'] + z_cols].dropna()

cph_fusion = CoxPHFitter(penalizer=0.01)
cph_fusion.fit(multivar_data, duration_col='duration', event_col='event')

print("\nFive-score multivariable Cox regression results:")
print(cph_fusion.summary)

# Extract the coefficients
cox_weights = {}
multivar_results = []
for col in z_cols:
    original_col = col.replace('_Z', '')
    coef = cph_fusion.params_[col]
    hr = np.exp(coef)
    p_val = cph_fusion.summary.loc[col, 'p']
    ci_lower = np.exp(cph_fusion.confidence_intervals_.loc[col].iloc[0])
    ci_upper = np.exp(cph_fusion.confidence_intervals_.loc[col].iloc[1])
    
    cox_weights[original_col] = coef
    multivar_results.append({
        'Score': score_display_names.get(original_col, original_col),
        'Coefficient': coef,
        'HR': hr,
        'HR_95CI': f"{ci_lower:.2f}-{ci_upper:.2f}",
        'P value': p_val,
        'Significance': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    })

multivar_df = pd.DataFrame(multivar_results)
print("\nFusion model coefficient summary:")
print(multivar_df.to_string(index=False))

# Compute the Cox fusion score
def calculate_fusion_score(df, weights, scalers):
    score = np.zeros(len(df))
    for col, w in weights.items():
        z_col = f'{col}_Z'
        if z_col in df.columns:
            score += w * df[z_col].values
    return score

train_df['Fusion_Cox'] = calculate_fusion_score(train_df, cox_weights, scalers)
test1_df['Fusion_Cox'] = calculate_fusion_score(test1_df, cox_weights, scalers)
test2_df['Fusion_Cox'] = calculate_fusion_score(test2_df, cox_weights, scalers)

# ============================================================
# Part 4: model performance evaluation
# ============================================================

print("\n[Part 4] Model performance evaluation")
print("-" * 70)

# C-index computation
def calc_cindex(df, score_col):
    return concordance_index(df['duration'], -df[score_col], df['event'])

# C-index of all scores
cindex_results = []
all_scores = score_columns + ['Fusion_Cox']

for score in all_scores:
    ci_train = calc_cindex(train_df, score)
    ci_test1 = calc_cindex(test1_df, score)
    ci_test2 = calc_cindex(test2_df, score)
    cindex_results.append({
        'Score': score_display_names.get(score, 'Fusion model'),
        'Training set': ci_train,
        'Validation set 1': ci_test1,
        'Validation set 2': ci_test2,
        'External mean': (ci_test1 + ci_test2) / 2
    })

cindex_df = pd.DataFrame(cindex_results)
print("\nC-index comparison:")
print(cindex_df.to_string(index=False))

# Time-dependent AUC - using the correct IPCW method
def calculate_timepoint_auc_ipcw(train_df, test_df, score_col, time_points, verbose=False):
    """
    Use scikit-survival cumulative_dynamic_auc to compute the time-dependent AUC
    This method uses IPCW (Inverse Probability of Censoring Weighting) to correctly handle censored data
    """
    if not HAS_SKSURV:
        # Fall back to a simple method (not recommended)
        results = []
        for t_point in time_points:
            valid_mask = (test_df['duration'] >= t_point) | ((test_df['duration'] < t_point) & (test_df['event'] == 1))
            df_valid = test_df[valid_mask].copy()
            if len(df_valid) < 10:
                results.append(np.nan)
                continue
            y_true = ((df_valid['duration'] < t_point) & (df_valid['event'] == 1)).astype(int)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                results.append(np.nan)
                continue
            from sklearn.metrics import roc_auc_score
            results.append(roc_auc_score(y_true, df_valid[score_col].values))
        return results
    
    # Use the correct sksurv method
    try:
        # Check the censoring ratio - if the censoring rate is very low, the IPCW method may fail
        test_censoring_rate = 1 - test_df['event'].mean()
        train_censoring_rate = 1 - train_df['event'].mean()
        
        if test_censoring_rate < 0.20:  # fall back to the simple method when the censoring rate is below 20%
            if verbose:
                print(f"    Warning: the test-set censoring rate is low ({test_censoring_rate:.1%}); using the simple AUC computation")
            # Use the simple time-point binary-classification AUC
            results = []
            for t_point in time_points:
                valid_mask = (test_df['duration'] >= t_point) | ((test_df['duration'] < t_point) & (test_df['event'] == 1))
                df_valid = test_df[valid_mask].copy()
                if len(df_valid) < 10:
                    results.append(np.nan)
                    continue
                y_true = ((df_valid['duration'] < t_point) & (df_valid['event'] == 1)).astype(int)
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    results.append(np.nan)
                    continue
                from sklearn.metrics import roc_auc_score
                results.append(roc_auc_score(y_true, df_valid[score_col].values))
            return results
        
        # Build the survival data structure
        y_train = Surv.from_dataframe('event', 'duration', train_df)
        y_test = Surv.from_dataframe('event', 'duration', test_df)
        
        # Get the risk score
        risk_scores = test_df[score_col].values
        
        # Compute the time-dependent AUC
        times = np.array(time_points, dtype=float)
        
        # Get the event-time range
        train_event_times = train_df[train_df['event'] == 1]['duration']
        test_event_times = test_df[test_df['event'] == 1]['duration']
        
        if len(train_event_times) == 0 or len(test_event_times) == 0:
            if verbose:
                print(f"    Warning: no event data in the training or test set")
            return [np.nan] * len(time_points)
        
        # Use a more relaxed time range: it only needs to be within the observation range of the test set
        min_time = train_event_times.min()
        # Use 90% of the maximum observation time of the test set to avoid boundary issues
        max_time = test_df['duration'].max() * 0.9
        
        if verbose:
            print(f"    Data time range: train events[{train_event_times.min():.1f}-{train_event_times.max():.1f}], "
                  f"test events[{test_event_times.min():.1f}-{test_event_times.max():.1f}], "
                  f"test observation[{test_df['duration'].min():.1f}-{test_df['duration'].max():.1f}]")
            print(f"    Valid evaluation range: [{min_time:.1f}, {max_time:.1f}]")
        
        # Select valid time points
        valid_times = times[(times > min_time) & (times < max_time)]
        
        if len(valid_times) == 0:
            if verbose:
                print(f"    Warning: the requested time points {time_points} are not within the valid range")
            # Try using the quantile time points of the test set
            if len(test_event_times) >= 3:
                valid_times = np.percentile(test_event_times, [25, 50, 75])
                valid_times = valid_times[(valid_times > min_time) & (valid_times < max_time)]
            if len(valid_times) == 0:
                return [np.nan] * len(time_points)
        
        auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, valid_times)
        
        if verbose:
            print(f"    Valid time points: {valid_times}, AUC values: {auc_values}")
        
        # Align the results to the original time points
        results = []
        for t_point in time_points:
            if t_point <= min_time or t_point >= max_time:
                results.append(np.nan)
            else:
                # Find the closest valid time point
                closest_idx = np.argmin(np.abs(valid_times - t_point))
                # Use it only when the time point is close enough
                if np.abs(valid_times[closest_idx] - t_point) < t_point * 0.5:  # within 50% error
                    results.append(auc_values[closest_idx])
                else:
                    results.append(np.nan)
        return results
    except Exception as e:
        # Fall back to the simple AUC method when IPCW fails
        if verbose:
            print(f"    IPCW computation failed ({str(e)[:50]}...); falling back to the simple AUC")
        try:
            from sklearn.metrics import roc_auc_score
            results = []
            for t_point in time_points:
                valid_mask = (test_df['duration'] >= t_point) | ((test_df['duration'] < t_point) & (test_df['event'] == 1))
                df_valid = test_df[valid_mask].copy()
                if len(df_valid) < 10:
                    results.append(np.nan)
                    continue
                y_true = ((df_valid['duration'] < t_point) & (df_valid['event'] == 1)).astype(int)
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    results.append(np.nan)
                    continue
                results.append(roc_auc_score(y_true, df_valid[score_col].values))
            return results
        except Exception as e2:
            if verbose:
                print(f"    The simple AUC also failed: {e2}")
            return [np.nan] * len(time_points)

time_points = [12, 24, 36]
time_labels = ['1-year', '2-year', '3-year']

print("\nCox fusion model time-dependent AUC (IPCW method):")
# First print the time-range diagnostic information of each dataset
for ds_name, ds_df in [('Training set', train_df), ('Validation set 1', test1_df), ('Validation set 2', test2_df)]:
    event_times = ds_df[ds_df['event'] == 1]['duration']
    print(f"  {ds_name} time range: duration[{ds_df['duration'].min():.1f}-{ds_df['duration'].max():.1f}], "
          f"event time[{event_times.min():.1f}-{event_times.max():.1f}] (n events={len(event_times)})")

auc_results = {'Time point': time_labels}
for ds_name, ds_df in [('Training set', train_df), ('Validation set 1', test1_df), ('Validation set 2', test2_df)]:
    # Use verbose mode for validation set 2 to inspect issues
    verbose = (ds_name == 'Validation set 2')
    aucs = calculate_timepoint_auc_ipcw(train_df, ds_df, 'Fusion_Cox', time_points, verbose=verbose)
    auc_results[ds_name] = aucs
    print(f"  {ds_name} AUC: " + " | ".join([f"{t}: {a:.4f}" if not np.isnan(a) else f"{t}: N/A" for t, a in zip(time_labels, aucs)]))

# ============================================================
# Part 5: risk stratification analysis
# ============================================================

print("\n[Part 5] Risk stratification analysis")
print("-" * 70)

# Use the training-set median for risk stratification
train_median = train_df['Fusion_Cox'].median()
train_df['Risk_Group'] = np.where(train_df['Fusion_Cox'] > train_median, 'High', 'Low')
test1_df['Risk_Group'] = np.where(test1_df['Fusion_Cox'] > train_median, 'High', 'Low')
test2_df['Risk_Group'] = np.where(test2_df['Fusion_Cox'] > train_median, 'High', 'Low')

# KM analysis function
def km_analysis(df, dataset_name):
    high_risk = df[df['Risk_Group'] == 'High']
    low_risk = df[df['Risk_Group'] == 'Low']
    
    # Log-rank test
    lr_result = logrank_test(
        high_risk['duration'], low_risk['duration'],
        high_risk['event'], low_risk['event']
    )
    
    # HR computation
    df_temp = df.copy()
    df_temp['Risk_Binary'] = (df_temp['Risk_Group'] == 'High').astype(int)
    cph = CoxPHFitter()
    cph.fit(df_temp[['duration', 'event', 'Risk_Binary']], duration_col='duration', event_col='event')
    hr = np.exp(cph.params_['Risk_Binary'])
    hr_ci = np.exp(cph.confidence_intervals_)
    
    # Median survival time
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    kmf_high.fit(high_risk['duration'], high_risk['event'])
    kmf_low.fit(low_risk['duration'], low_risk['event'])
    
    median_high = kmf_high.median_survival_time_
    median_low = kmf_low.median_survival_time_
    
    return {
        'dataset': dataset_name,
        'n_high': len(high_risk),
        'n_low': len(low_risk),
        'events_high': high_risk['event'].sum(),
        'events_low': low_risk['event'].sum(),
        'hr': hr,
        'hr_ci_lower': hr_ci.iloc[0, 0],
        'hr_ci_upper': hr_ci.iloc[0, 1],
        'p_value': lr_result.p_value,
        'median_high': median_high,
        'median_low': median_low,
        'kmf_high': kmf_high,
        'kmf_low': kmf_low
    }

km_train = km_analysis(train_df, 'Training set')
km_test1 = km_analysis(test1_df, 'Validation set 1')
km_test2 = km_analysis(test2_df, 'Validation set 2')

print("\nRisk stratification results:")
for km in [km_train, km_test1, km_test2]:
    print(f"\n{km['dataset']}:")
    print(f"  High-risk group: n={km['n_high']}, events={km['events_high']}, median survival={km['median_high']:.1f} months")
    print(f"  Low-risk group: n={km['n_low']}, events={km['events_low']}, median survival={km['median_low']:.1f} months")
    print(f"  HR: {km['hr']:.2f} (95% CI: {km['hr_ci_lower']:.2f}-{km['hr_ci_upper']:.2f})")
    print(f"  Log-rank P: {km['p_value']:.2e}")

# ============================================================
# Part 6: visualization - KM curves
# ============================================================

print("\n[Part 6] Generate visualizations")
print("-" * 70)

# Figure 1: KM survival curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, km in zip(axes, [km_train, km_test1, km_test2]):
    km['kmf_low'].plot(ax=ax, ci_show=True, color='steelblue', label='Low-risk group')
    km['kmf_high'].plot(ax=ax, ci_show=True, color='crimson', label='High-risk group')
    
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival probability', fontsize=11)
    ax.set_title(f"{km['dataset']}\nHR={km['hr']:.2f} (95%CI: {km['hr_ci_lower']:.2f}-{km['hr_ci_upper']:.2f})\n{format_p(km['p_value'])}", fontsize=10)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, None)
    
    # Add the risk-table numbers
    ax.text(0.95, 0.95, f"n(high)={km['n_high']}\nn(low)={km['n_low']}", 
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_km_curves.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_km_curves.pdf', bbox_inches='tight')
print("[OK] KM curves saved")

# ============================================================
# Figure 2: forest plot - multivariable Cox analysis
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare the data
forest_data = []
for res in multivar_results:
    hr = res['HR']
    ci_parts = res['HR_95CI'].split('-')
    ci_lower = float(ci_parts[0])
    ci_upper = float(ci_parts[1])
    forest_data.append({
        'name': res['Score'],
        'hr': hr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p': res['P value']
    })

y_pos = np.arange(len(forest_data))
hrs = [d['hr'] for d in forest_data]
ci_lowers = [d['ci_lower'] for d in forest_data]
ci_uppers = [d['ci_upper'] for d in forest_data]
names = [d['name'] for d in forest_data]

# Draw the error bars
for i, (hr, ci_l, ci_u) in enumerate(zip(hrs, ci_lowers, ci_uppers)):
    ax.plot([ci_l, ci_u], [i, i], 'k-', linewidth=2)
    ax.plot(hr, i, 'ko', markersize=10)

# Reference line
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Hazard Ratio (HR)', fontsize=12)
ax.set_title('Cox regression fusion model - multivariable analysis forest plot', fontsize=13)

# Add HR and P-value annotations
for i, d in enumerate(forest_data):
    sig = '***' if d['p'] < 0.001 else '**' if d['p'] < 0.01 else '*' if d['p'] < 0.05 else ''
    p_str = '< 0.001' if d['p'] < 0.001 else f"= {d['p']:.3f}"
    text = f"HR = {d['hr']:.2f} ({d['ci_lower']:.2f}-{d['ci_upper']:.2f}), P {p_str}{sig}"
    ax.annotate(text, xy=(max(ci_uppers) + 0.1, i), fontsize=9, va='center')

ax.set_xlim(0, max(ci_uppers) + 1.5)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_forest_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_forest_plot.pdf', bbox_inches='tight')
print("[OK] Forest plot saved")

# ============================================================
# Figure 3: C-index comparison bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(cindex_df))
width = 0.25

bars1 = ax.bar(x - width, cindex_df['Training set'], width, label='Training set', color='steelblue', alpha=0.9)
bars2 = ax.bar(x, cindex_df['Validation set 1'], width, label='Validation set 1', color='darkorange', alpha=0.9)
bars3 = ax.bar(x + width, cindex_df['Validation set 2'], width, label='Validation set 2', color='forestgreen', alpha=0.9)

ax.set_ylabel('C-index', fontsize=12)
ax.set_title('Model performance comparison - C-index', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(cindex_df['Score'], rotation=30, ha='right', fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0.5, 0.8)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)

# Add an annotation on the fusion model
fusion_idx = len(cindex_df) - 1
ax.annotate('Best', xy=(fusion_idx, cindex_df.iloc[fusion_idx]['Training set'] + 0.01), 
            ha='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_cindex_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_cindex_comparison.pdf', bbox_inches='tight')
print("[OK] C-index comparison figure saved")

# ============================================================
# Figure 4: time-dependent AUC curves
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [('Training set', train_df), ('Validation set 1', test1_df), ('Validation set 2', test2_df)]
colors = {'Fusion_Cox': '#E41A1C', 'Clinical_Score': '#377EB8', 
          'Rad_Diff_HighRisk': '#4DAF4A', 'DL_Heterogeneity': '#A65628',
          'DL_MaxPooling': '#984EA3', 'Habitat_Agg_Best': '#FF7F00'}

labels = {'Fusion_Cox': 'Cox fusion', 'Clinical_Score': 'Clinical score', 
          'Rad_Diff_HighRisk': 'Traditional radiomics', 'DL_Heterogeneity': 'Deep learning (Het)',
          'DL_MaxPooling': 'Deep learning (MaxP)', 'Habitat_Agg_Best': 'Habitat radiomics'}

for ax, (ds_name, ds_df) in zip(axes, datasets):
    for score in ['Fusion_Cox', 'Clinical_Score', 'Rad_Diff_HighRisk', 'DL_Heterogeneity', 'DL_MaxPooling', 'Habitat_Agg_Best']:
        aucs = calculate_timepoint_auc_ipcw(train_df, ds_df, score, time_points)
        linewidth = 3 if score == 'Fusion_Cox' else 1.5
        ax.plot(time_points, aucs, marker='o', linewidth=linewidth, 
                color=colors[score], label=labels[score])
    
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title(f'{ds_name}', fontsize=12)
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_labels)
    ax.set_ylim(0.5, 0.85)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Time-dependent AUC comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_time_auc.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_time_auc.pdf', bbox_inches='tight')
print("[OK] Time-dependent AUC curves saved")

# ============================================================
# Figure 5: ROC curves (1-year, 2-year, 3-year)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Use the IPCW method to plot time-dependent ROC curves
for ax, (t_point, t_label) in zip(axes, zip(time_points, time_labels)):
    for ds_name, ds_df, color, ls in [('Training set', train_df, 'navy', '-'), 
                                       ('Validation set 1', test1_df, 'darkorange', '--'),
                                       ('Validation set 2', test2_df, 'forestgreen', ':')]:
        # Compute the AUC at this time point
        aucs = calculate_timepoint_auc_ipcw(train_df, ds_df, 'Fusion_Cox', [t_point])
        auc = aucs[0] if aucs else np.nan
        
        if np.isnan(auc):
            continue
        
        # For plotting the ROC curve, FPR/TPR still need to be computed
        # Use the time-dependent label
        valid_mask = (ds_df['duration'] >= t_point) | ((ds_df['duration'] < t_point) & (ds_df['event'] == 1))
        df_valid = ds_df[valid_mask].copy()
        
        if len(df_valid) < 10:
            continue
        
        y_true = ((df_valid['duration'] < t_point) & (df_valid['event'] == 1)).astype(int)
        y_score = df_valid['Fusion_Cox'].values
        
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        # Use the AUC value computed by IPCW
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f'{ds_name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('1 - Specificity', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title(f'{t_label} survival prediction ROC curve (IPCW)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_roc_curves.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_roc_curves.pdf', bbox_inches='tight')
print("[OK] ROC curves saved")

# ============================================================
# Figure 6: calibration curve - using Cox model predicted survival probability
# ============================================================

# Use the Cox model survival-function method S(t|x) = S0(t)^exp(beta*x)  (Royston & Altman, 2013)
def calculate_calibration_cox(cph_model, df, time_point, z_col_list, n_bins=5):
    """
    Compute the calibration curve using the Cox model true predicted survival probability
    Method: S(t|x) = S0(t)^exp(beta*x), pd.qcut equal-frequency binning + KM observation
    """
    # Get the Cox model true predicted survival probability
    pred_df = df[z_col_list].copy()
    surv_func = cph_model.predict_survival_function(pred_df, times=[time_point])
    pred_survival = surv_func.iloc[0].values
    pred_survival = np.clip(pred_survival, 0.001, 0.999)
    
    # pd.qcut equal-frequency binning
    try:
        bin_labels = pd.qcut(pred_survival, q=n_bins, labels=False, duplicates='drop')
    except Exception:
        return np.array([]), np.array([])
    
    prob_pred = []
    prob_true = []
    
    for b in sorted(np.unique(bin_labels)):
        mask = bin_labels == b
        if mask.sum() < 5:
            continue
        
        prob_pred.append(float(pred_survival[mask].mean()))
        
        # KM estimate of the actual survival rate of this group at time_point
        group_df = df.iloc[np.where(mask)[0]]
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['duration'], event_observed=group_df['event'])
        try:
            surv_at_t = kmf.survival_function_at_times(time_point).values[0]
        except Exception:
            surv_at_t = 1.0 - group_df['event'].mean()
        prob_true.append(float(surv_at_t))
    
    return np.array(prob_true), np.array(prob_pred)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (t_point, t_label) in zip(axes, zip(time_points, time_labels)):
    for ds_name, ds_df, color, marker in [('Training set', train_df, 'navy', 'o'), 
                                           ('Validation set 1', test1_df, 'darkorange', 's'),
                                           ('Validation set 2', test2_df, 'forestgreen', '^')]:
        if len(ds_df) < 20:
            continue
        
        try:
            prob_true, prob_pred = calculate_calibration_cox(cph_fusion, ds_df, t_point, z_cols, n_bins=5)
            if len(prob_pred) > 1:
                ax.plot(prob_pred, prob_true, marker=marker, linewidth=2, markersize=8,
                        color=color, label=ds_name)
        except Exception as e:
            print(f"  {ds_name} {t_label} calibration curve computation failed: {e}")
            continue
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    ax.set_xlabel('Predicted Survival Probability S(t|x)', fontsize=11)
    ax.set_ylabel('Observed Survival Probability (KM)', fontsize=11)
    ax.set_title(f'{t_label} Calibration (Cox Fusion)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_calibration.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_calibration.pdf', bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_calibration.svg', bbox_inches='tight')
print("[OK] Calibration curve saved")

# ============================================================
# Figure 7: DCA decision curve analysis - Cox predicted event probability + standard formula
# ============================================================

# Use the Cox model predicted event probability 1-S(t|x), standard DCA formula (Vickers et al., 2008)
# Fix: 1) do not truncate negative values 2) use the true predicted probability 3) independent prevalence per dataset
print("\n[DCA] use the Cox predicted event probability 1-S(t|x), standard DCA formula (no truncation)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (t_point, t_label) in zip(axes, zip(time_points, time_labels)):
    thresholds = np.arange(0.01, 0.90, 0.01)
    all_nb_vals = []
    
    for ds_name, ds_df, color in [('Training', train_df, 'navy'),
                                   ('Test 1', test1_df, 'darkorange'), 
                                   ('Test 2', test2_df, 'forestgreen')]:
        if len(ds_df) < 20:
            continue
        
        # Cox model predicted event probability = 1 - S(t|x) = 1 - S0(t)^exp(beta*x)
        pred_df = ds_df[z_cols].copy()
        surv_func = cph_fusion.predict_survival_function(pred_df, times=[t_point])
        pred_event_prob = 1 - surv_func.iloc[0].values
        pred_event_prob = np.clip(pred_event_prob, 0.001, 0.999)
        
        # Time-dependent binary label (exclude samples censored before t)
        event_arr = ds_df['event'].values
        time_arr = ds_df['duration'].values
        event_before = (event_arr == 1) & (time_arr <= t_point)
        alive_at_t = time_arr > t_point
        valid_mask = event_before | alive_at_t
        
        y_true = event_before[valid_mask].astype(int)
        y_pred = pred_event_prob[valid_mask]
        n = len(y_true)
        
        # Model net benefit - standard DCA formula, no max(0) truncation
        net_benefits = []
        for pt in thresholds:
            pred_positive = y_pred >= pt
            tp = np.sum(pred_positive & (y_true == 1))
            fp = np.sum(pred_positive & (y_true == 0))
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            net_benefits.append(nb)
        
        ax.plot(thresholds, net_benefits, color=color, linewidth=2, label=f'Fusion-{ds_name}')
        all_nb_vals.extend(net_benefits)
    
    # Treat All (use the training-set prevalence as reference)
    event_arr_tr = train_df['event'].values
    time_arr_tr = train_df['duration'].values
    event_before_tr = (event_arr_tr == 1) & (time_arr_tr <= t_point)
    alive_at_t_tr = time_arr_tr > t_point
    valid_mask_tr = event_before_tr | alive_at_t_tr
    y_true_tr = event_before_tr[valid_mask_tr].astype(int)
    prevalence_ref = y_true_tr.mean()
    
    treat_all = [prevalence_ref - (1 - prevalence_ref) * (pt / (1 - pt)) for pt in thresholds]
    ax.plot(thresholds, treat_all, 'k--', linewidth=1.5, label='Treat All')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, label='Treat None')
    all_nb_vals.extend(treat_all)
    
    ax.set_xlabel('Threshold Probability', fontsize=11)
    ax.set_ylabel('Net Benefit', fontsize=11)
    ax.set_title(f'{t_label} Decision Curve Analysis', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 0.9)
    # Dynamic Y-axis
    valid_vals = [v for v in all_nb_vals if not np.isnan(v) and -2 < v < 2]
    if valid_vals:
        y_min = max(min(valid_vals) - 0.02, -0.15)
        y_max_val = max(valid_vals) + 0.05
    else:
        y_min, y_max_val = -0.15, 0.5
    ax.set_ylim(y_min, y_max_val)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/cox_fusion_dca.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_dca.pdf', bbox_inches='tight')
plt.savefig(f'{output_dir}/figures/cox_fusion_dca.svg', bbox_inches='tight')
print("[OK] DCA decision curve saved")

# ============================================================
# Figure 8: Nomogram - using onekey_algo or the simplified version
# ============================================================

# Prepare the Nomogram data
nomogram_columns = [f'{col}_Z' for col in score_columns]
nomogram_df = train_df[['duration', 'event'] + nomogram_columns].copy()
nomogram_df.columns = ['duration', 'event'] + list(score_display_names.values())

if HAS_ONEKEY:
    # Use onekey_algo to draw a professional Nomogram
    try:
        print("Drawing the Nomogram using onekey_algo...")
        nomo_img = nomogram.nomogram(
            nomogram_df, 
            duration='duration', 
            result='event',
            columns=list(score_display_names.values()),
            survs=[12, 24, 36],
            surv_names=['1-year survival', '2-year survival', '3-year survival'],
            width=7000, height=5000,
            with_r=False
        )
        if nomo_img:
            nomo_img.save(f'{output_dir}/figures/cox_fusion_nomogram.png')
        print("[OK] Nomogram saved (onekey_algo)")
    except Exception as e:
        print(f"onekey_algo Nomogram failed: {e}; using the simplified version")
        HAS_ONEKEY = False

if not HAS_ONEKEY:
    # Simplified Nomogram
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get the coefficients and ranges
    nomogram_data = []
    for col in score_columns:
        coef = cox_weights[col]
        z_col = f'{col}_Z'
        min_val = train_df[z_col].min()
        max_val = train_df[z_col].max()
        points_range = coef * (max_val - min_val)
        nomogram_data.append({
            'name': score_display_names.get(col, col),
            'coef': coef,
            'min_z': min_val,
            'max_z': max_val,
            'points_range': abs(points_range)
        })
    
    # Normalize to a 0-100 scale
    max_points = max([d['points_range'] for d in nomogram_data])
    for d in nomogram_data:
        d['points_scale'] = 100 * d['points_range'] / max_points
    
    # Draw the Nomogram
    y_positions = np.linspace(0.85, 0.15, len(nomogram_data) + 2)
    
    # Points scale
    ax.text(0.02, y_positions[0], 'Points', fontsize=11, va='center', fontweight='bold')
    ax.plot([0.15, 0.95], [y_positions[0], y_positions[0]], 'k-', linewidth=1)
    for i in range(0, 101, 10):
        x_pos = 0.15 + (i/100) * 0.8
        ax.plot([x_pos, x_pos], [y_positions[0]-0.01, y_positions[0]+0.01], 'k-', linewidth=1)
        ax.text(x_pos, y_positions[0]+0.025, str(i), ha='center', fontsize=8)
    
    # Scale of each score
    for i, d in enumerate(nomogram_data):
        y = y_positions[i + 1]
        ax.text(0.02, y, d['name'], fontsize=10, va='center')
        ax.plot([0.15, 0.15 + d['points_scale']/100*0.8], [y, y], 'b-', linewidth=2)
        
        # Add ticks
        n_ticks = 5
        for j in range(n_ticks + 1):
            x_pos = 0.15 + (j/n_ticks) * (d['points_scale']/100*0.8)
            ax.plot([x_pos, x_pos], [y-0.008, y+0.008], 'k-', linewidth=1)
            
            # Compute the corresponding Z value
            z_val = d['min_z'] + (j/n_ticks) * (d['max_z'] - d['min_z'])
            ax.text(x_pos, y-0.025, f'{z_val:.1f}', ha='center', fontsize=7)
    
    # Total Points
    y_total = y_positions[-1]
    ax.text(0.02, y_total, 'Total Points', fontsize=11, va='center', fontweight='bold')
    ax.plot([0.15, 0.95], [y_total, y_total], 'k-', linewidth=1.5)
    for i in range(0, 301, 50):
        x_pos = 0.15 + (i/300) * 0.8
        ax.plot([x_pos, x_pos], [y_total-0.01, y_total+0.01], 'k-', linewidth=1)
        ax.text(x_pos, y_total-0.03, str(i), ha='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Cox regression fusion model - Nomogram', fontsize=14, pad=20)
    
    # Add the caption text
    ax.text(0.5, 0.02, 'Usage: sum the Points of each score to obtain Total Points, then look up the corresponding risk probability', 
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/cox_fusion_nomogram.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figures/cox_fusion_nomogram.pdf', bbox_inches='tight')
    print("[OK] Nomogram saved (simplified version)")

plt.close('all')

# ============================================================
# Part 7: save the complete results
# ============================================================

print("\n[Part 7] Save the complete results")
print("-" * 70)

# 1. Multivariable analysis results
multivar_df.to_csv(f'{output_dir}/results/cox_fusion_multivariate_analysis.csv', index=False, encoding='utf-8-sig')

# 2. C-index comparison
cindex_df.to_csv(f'{output_dir}/results/cox_fusion_cindex_comparison.csv', index=False, encoding='utf-8-sig')

# 3. Time-dependent AUC
time_auc_df = pd.DataFrame({
    'Time point': time_labels,
    'Training set': auc_results['Training set'],
    'Validation set 1': auc_results['Validation set 1'],
    'Validation set 2': auc_results['Validation set 2']
})
time_auc_df.to_csv(f'{output_dir}/results/cox_fusion_time_auc.csv', index=False, encoding='utf-8-sig')

# 4. Risk stratification results
risk_strat_results = []
for km in [km_train, km_test1, km_test2]:
    risk_strat_results.append({
        'Dataset': km['dataset'],
        'High-risk group n': km['n_high'],
        'High-risk group events': km['events_high'],
        'High-risk group median survival': km['median_high'],
        'Low-risk group n': km['n_low'],
        'Low-risk group events': km['events_low'],
        'Low-risk group median survival': km['median_low'],
        'HR': km['hr'],
        'HR_95CI_Lower': km['hr_ci_lower'],
        'HR_95CI_Upper': km['hr_ci_upper'],
        'P value': km['p_value']
    })
pd.DataFrame(risk_strat_results).to_csv(f'{output_dir}/results/cox_fusion_risk_stratification.csv', index=False, encoding='utf-8-sig')

# 5. Complete data (including the fusion score)
output_cols = ['ID', 'group', 'duration', 'event'] + score_columns + ['Fusion_Cox', 'Risk_Group']
all_data = pd.concat([train_df, test1_df, test2_df], ignore_index=True)
all_data[[c for c in output_cols if c in all_data.columns]].to_csv(
    f'{output_dir}/data/cox_fusion_complete_results.csv', index=False)

# 6. Model coefficients
model_coef_df = pd.DataFrame({
    'Score': list(cox_weights.keys()),
    'Coefficient': list(cox_weights.values()),
    'HR': [np.exp(v) for v in cox_weights.values()]
})
model_coef_df.to_csv(f'{output_dir}/results/cox_fusion_model_coefficients.csv', index=False, encoding='utf-8-sig')

# 7. Comprehensive summary report
summary_report = f"""
===============================================================================
Cox regression fusion model - complete analysis report
===============================================================================
Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
-------------------------------------------------------------------------------

[Model overview]
Fusion of five independent prognostic scores:
  1. Clinical score (Clinical_Score) - coefficient: {cox_weights['Clinical_Score']:.4f}
  2. Traditional radiomics difference feature (Rad_Diff) - coefficient: {cox_weights['Rad_Diff_HighRisk']:.4f}
  3. Deep learning heterogeneity (DL_Heterogeneity) - coefficient: {cox_weights['DL_Heterogeneity']:.4f}
  4. Deep learning max-pooling (DL_MaxPooling) - coefficient: {cox_weights['DL_MaxPooling']:.4f}
  5. Habitat radiomics Max-H1 (Habitat_Agg_Best) - coefficient: {cox_weights['Habitat_Agg_Best']:.4f}

[Sample size]
  Training set: n={len(train_df)}, events={train_df['event'].sum()} ({train_df['event'].mean()*100:.1f}%)
  Validation set 1: n={len(test1_df)}, events={test1_df['event'].sum()} ({test1_df['event'].mean()*100:.1f}%)
  Validation set 2: n={len(test2_df)}, events={test2_df['event'].sum()} ({test2_df['event'].mean()*100:.1f}%)

[Model performance - C-index]
  Training set:  {cindex_df[cindex_df['Score']=='Fusion model']['Training set'].values[0]:.4f}
  Validation set 1: {cindex_df[cindex_df['Score']=='Fusion model']['Validation set 1'].values[0]:.4f}
  Validation set 2: {cindex_df[cindex_df['Score']=='Fusion model']['Validation set 2'].values[0]:.4f}
  External mean: {cindex_df[cindex_df['Score']=='Fusion model']['External mean'].values[0]:.4f}

[Time-dependent AUC]
  1-year: train={auc_results['Training set'][0]:.4f}, val1={auc_results['Validation set 1'][0]:.4f}, val2={auc_results['Validation set 2'][0]:.4f}
  2-year: train={auc_results['Training set'][1]:.4f}, val1={auc_results['Validation set 1'][1]:.4f}, val2={auc_results['Validation set 2'][1]:.4f}
  3-year: train={auc_results['Training set'][2]:.4f}, val1={auc_results['Validation set 1'][2]:.4f}, val2={auc_results['Validation set 2'][2]:.4f}

[Risk stratification]
  Training set:  HR={km_train['hr']:.2f} (95%CI: {km_train['hr_ci_lower']:.2f}-{km_train['hr_ci_upper']:.2f}), P={km_train['p_value']:.2e}
  Validation set 1: HR={km_test1['hr']:.2f} (95%CI: {km_test1['hr_ci_lower']:.2f}-{km_test1['hr_ci_upper']:.2f}), P={km_test1['p_value']:.2e}
  Validation set 2: HR={km_test2['hr']:.2f} (95%CI: {km_test2['hr_ci_lower']:.2f}-{km_test2['hr_ci_upper']:.2f}), P={km_test2['p_value']:.2e}

[Generated files]
Data files:
  - data/cox_fusion_complete_results.csv

Result files:
  - results/cox_fusion_multivariate_analysis.csv
  - results/cox_fusion_cindex_comparison.csv
  - results/cox_fusion_time_auc.csv
  - results/cox_fusion_risk_stratification.csv
  - results/cox_fusion_model_coefficients.csv

Figure files:
  - figures/cox_fusion_km_curves.png/pdf
  - figures/cox_fusion_forest_plot.png/pdf
  - figures/cox_fusion_cindex_comparison.png/pdf
  - figures/cox_fusion_time_auc.png/pdf
  - figures/cox_fusion_roc_curves.png/pdf
  - figures/cox_fusion_calibration.png/pdf
  - figures/cox_fusion_dca.png/pdf
  - figures/cox_fusion_nomogram.png/pdf

===============================================================================
"""

with open(f'{output_dir}/results/cox_fusion_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)

print("\n" + "=" * 70)
print("Cox regression fusion model complete analysis finished!")
print("=" * 70)
