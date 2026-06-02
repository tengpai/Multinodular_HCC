"""
Subgroup analysis extension: 11-subgroup HR analysis for the training set + test set 1 + test set 2
All stratification cutoffs and the Fusion binary cutoff are fixed to the training-set median (pure external-validation logic)

Output:
  - results/subgroup_all_cohorts.csv
  - figures/subgroup_forest_all_cohorts.png/pdf/svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Data loading
# ============================================================================
results_path = '<PROJECT_ROOT>/sol15_multimodal_fusion/data/cox_fusion_complete_results.csv'
df = pd.read_csv(results_path)

train_df = df[df['group'] == 'train'].copy()
test1_df = df[df['group'] == 'test1'].copy()
test2_df = df[df['group'] == 'test2'].copy()

print(f"Training set: n={len(train_df)}, events={int(train_df['event'].sum())}")
print(f"Test set 1: n={len(test1_df)}, events={int(test1_df['event'].sum())}")
print(f"Test set 2: n={len(test2_df)}, events={int(test2_df['event'].sum())}")

# ============================================================================
# Compute all median cutoffs based on the training set (fixed)
# ============================================================================
score_cols = ['Clinical_Score', 'Rad_Diff_HighRisk', 'DL_Heterogeneity', 'DL_MaxPooling', 'Habitat_Agg_Best']
score_labels = {
    'Clinical_Score': 'Clinical score',
    'Rad_Diff_HighRisk': 'Traditional radiomics (Diff) score',
    'DL_Heterogeneity': 'DL heterogeneity score',
    'DL_MaxPooling': 'DL extreme-value score',
    'Habitat_Agg_Best': 'Habitat score'
}

# Training-set median (used for stratification of all cohorts)
train_medians = {col: train_df[col].median() for col in score_cols}
fusion_median = train_df['Fusion_Cox'].median()

print(f"\nTraining-set Fusion_Cox median (fixed cutoff): {fusion_median:.4f}")
for col, med in train_medians.items():
    print(f"  {score_labels[col]} median: {med:.4f}")


# ============================================================================
# Define the subgroup analysis function
# ============================================================================
def run_subgroup_analysis(data, cohort_name, train_medians, fusion_median):
    """Perform 11-subgroup HR analysis for a given cohort"""
    analysis_df = data.copy()
    
    # Use the training-set Fusion median for the high/low-risk binary split
    analysis_df['Fusion_HighRisk'] = (analysis_df['Fusion_Cox'] >= fusion_median).astype(int)
    
    # Use the training-set median of each score for subgroup stratification
    analysis_df['Clinical_High'] = (analysis_df['Clinical_Score'] >= train_medians['Clinical_Score']).astype(int)
    analysis_df['RadDiff_High'] = (analysis_df['Rad_Diff_HighRisk'] >= train_medians['Rad_Diff_HighRisk']).astype(int)
    analysis_df['DLHet_High'] = (analysis_df['DL_Heterogeneity'] >= train_medians['DL_Heterogeneity']).astype(int)
    analysis_df['DLMaxP_High'] = (analysis_df['DL_MaxPooling'] >= train_medians['DL_MaxPooling']).astype(int)
    analysis_df['Habitat_High'] = (analysis_df['Habitat_Agg_Best'] >= train_medians['Habitat_Agg_Best']).astype(int)
    
    # Define the 11 subgroups
    subgroups = [
        ('Overall population',                    analysis_df),
        ('Clinical score low',                  analysis_df[analysis_df['Clinical_High'] == 0]),
        ('Clinical score high',                  analysis_df[analysis_df['Clinical_High'] == 1]),
        ('Traditional radiomics (Diff) score low',        analysis_df[analysis_df['RadDiff_High'] == 0]),
        ('Traditional radiomics (Diff) score high',        analysis_df[analysis_df['RadDiff_High'] == 1]),
        ('DL heterogeneity score low',              analysis_df[analysis_df['DLHet_High'] == 0]),
        ('DL heterogeneity score high',              analysis_df[analysis_df['DLHet_High'] == 1]),
        ('DL extreme-value score low',              analysis_df[analysis_df['DLMaxP_High'] == 0]),
        ('DL extreme-value score high',              analysis_df[analysis_df['DLMaxP_High'] == 1]),
        ('Habitat score low',                  analysis_df[analysis_df['Habitat_High'] == 0]),
        ('Habitat score high',                  analysis_df[analysis_df['Habitat_High'] == 1]),
    ]
    
    results = []
    for name, sub_data in subgroups:
        n = len(sub_data)
        events = int(sub_data['event'].sum())
        n_high = int(sub_data['Fusion_HighRisk'].sum())
        n_low = n - n_high
        
        # Check whether both the high- and low-risk groups have enough samples and events
        events_high = int(sub_data[sub_data['Fusion_HighRisk']==1]['event'].sum())
        events_low = int(sub_data[sub_data['Fusion_HighRisk']==0]['event'].sum())
        
        if n_high < 5 or n_low < 5 or events_high < 3 or events_low < 3:
            print(f"  [{cohort_name}] {name}: skipped (n_high={n_high}, n_low={n_low}, events_high={events_high}, events_low={events_low})")
            results.append({
                'Cohort': cohort_name, 'Subgroup': name, 'n': n, 'Events': events,
                'n_high': n_high, 'n_low': n_low,
                'HR': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'P value': np.nan
            })
            continue
        
        try:
            cph = CoxPHFitter()
            cph.fit(sub_data[['duration', 'event', 'Fusion_HighRisk']],
                    duration_col='duration', event_col='event')
            hr = cph.hazard_ratios_['Fusion_HighRisk']
            ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
            ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
            pvalue = cph.summary['p']['Fusion_HighRisk']
            
            sig = '***' if pvalue < 0.001 else ('**' if pvalue < 0.01 else ('*' if pvalue < 0.05 else 'ns'))
            print(f"  [{cohort_name}] {name}: HR={hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), P={pvalue:.4f} {sig}  [n={n}, events={events}, high/low={n_high}/{n_low}]")
            
            results.append({
                'Cohort': cohort_name, 'Subgroup': name, 'n': n, 'Events': events,
                'n_high': n_high, 'n_low': n_low,
                'HR': hr, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper, 'P value': pvalue
            })
        except Exception as e:
            print(f"  [{cohort_name}] {name}: analysis failed - {e}")
            results.append({
                'Cohort': cohort_name, 'Subgroup': name, 'n': n, 'Events': events,
                'n_high': n_high, 'n_low': n_low,
                'HR': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'P value': np.nan
            })
    
    return results


# ============================================================================
# Run the subgroup analysis for the three cohorts
# ============================================================================
all_results = []

print("\n" + "="*70)
print("Training set subgroup analysis")
print("="*70)
all_results.extend(run_subgroup_analysis(train_df, 'Training set', train_medians, fusion_median))

print("\n" + "="*70)
print("Test set 1 subgroup analysis")
print("="*70)
all_results.extend(run_subgroup_analysis(test1_df, 'Test set 1', train_medians, fusion_median))

print("\n" + "="*70)
print("Test set 2 subgroup analysis")
print("="*70)
all_results.extend(run_subgroup_analysis(test2_df, 'Test set 2', train_medians, fusion_median))

# Save the results
results_df = pd.DataFrame(all_results)
results_df.to_csv('results/subgroup_all_cohorts.csv', index=False, encoding='utf-8-sig')
print(f"\nResults saved: results/subgroup_all_cohorts.csv ({len(results_df)} rows)")

# Print the summary table
print("\n" + "="*70)
print("Summary table (for the paper)")
print("="*70)
for _, row in results_df.iterrows():
    if pd.isna(row['HR']):
        hr_str = "N/A"
        ci_str = "N/A"
        p_str = "N/A"
    else:
        hr_str = f"{row['HR']:.2f}"
        ci_str = f"{row['CI_Lower']:.2f}–{row['CI_Upper']:.2f}"
        if row['P value'] < 0.001:
            p_str = "<0.001"
        else:
            p_str = f"{row['P value']:.3f}"
    print(f"{row['Cohort']:8s} | {row['Subgroup']:20s} | n={row['n']:4d} | events={row['Events']:4d} | HR={hr_str:8s} | 95%CI={ci_str:14s} | P={p_str:8s}")


# ============================================================================
# Forest plot: single column merged, three rows under each subgroup (Train/Test1/Test2)
# ============================================================================
print("\n" + "="*70)
print("Plotting the forest plot")
print("="*70)

subgroup_order = [
    'Overall population',
    'Clinical score low', 'Clinical score high',
    'Traditional radiomics (Diff) score low', 'Traditional radiomics (Diff) score high',
    'DL heterogeneity score low', 'DL heterogeneity score high',
    'DL extreme-value score low', 'DL extreme-value score high',
    'Habitat score low', 'Habitat score high',
]

cohort_order = ['Training set', 'Test set 1', 'Test set 2']
cohort_colors = {'Training set': '#2c3e50', 'Test set 1': '#e74c3c', 'Test set 2': '#3498db'}
cohort_markers = {'Training set': 's', 'Test set 1': 'D', 'Test set 2': 'o'}

# Color by subgroup dimension
dimension_colors = {
    'Overall population': '#555555',
    'Clinical score low': '#3498db', 'Clinical score high': '#3498db',
    'Traditional radiomics (Diff) score low': '#e67e22', 'Traditional radiomics (Diff) score high': '#e67e22',
    'DL heterogeneity score low': '#e74c3c', 'DL heterogeneity score high': '#e74c3c',
    'DL extreme-value score low': '#9b59b6', 'DL extreme-value score high': '#9b59b6',
    'Habitat score low': '#27ae60', 'Habitat score high': '#27ae60',
}

# Build the plotting data: arranged from bottom to top
plot_items = []  # (y_pos, label, hr, ci_lower, ci_upper, pvalue, cohort_name, subgroup_name, is_label_row)

y = 0
y_positions = []
for sg in reversed(subgroup_order):
    for cohort in reversed(cohort_order):
        row = results_df[(results_df['Subgroup'] == sg) & (results_df['Cohort'] == cohort)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        plot_items.append({
            'y': y,
            'subgroup': sg,
            'cohort': cohort,
            'n': row['n'],
            'events': row['Events'],
            'hr': row['HR'],
            'ci_lower': row['CI_Lower'],
            'ci_upper': row['CI_Upper'],
            'pvalue': row['P value'],
        })
        y += 1
    y += 0.6  # spacing between subgroups

fig, ax = plt.subplots(figsize=(18, 22))

for item in plot_items:
    hr = item['hr']
    if pd.isna(hr):
        continue
    
    color = cohort_colors[item['cohort']]
    marker = cohort_markers[item['cohort']]
    ms = 9 if item['subgroup'] != 'Overall population' else 11
    
    ax.errorbar(hr, item['y'],
                xerr=[[hr - item['ci_lower']], [item['ci_upper'] - hr]],
                fmt=marker, color=color, markersize=ms,
                capsize=4, capthick=1.5, linewidth=1.5, zorder=3)

# HR=1 reference line
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

# Build the y-axis labels + HR annotations
y_ticks = []
y_labels = []
max_ci_upper = max([item['ci_upper'] for item in plot_items if not pd.isna(item['ci_upper'])], default=5)
text_x = min(max_ci_upper + 0.5, 8.0)

for item in plot_items:
    y_ticks.append(item['y'])
    
    cohort_short = {'Training set': 'Train', 'Test set 1': 'Test1', 'Test set 2': 'Test2'}[item['cohort']]
    label = f"  {cohort_short} (n={item['n']}, e={item['events']})"
    # If this is the last row of the subgroup (train), add the subgroup name
    y_labels.append(label)
    
    # Add the HR text annotation
    if not pd.isna(item['hr']):
        if item['pvalue'] < 0.001:
            p_str = "P<0.001"
        elif item['pvalue'] < 0.01:
            p_str = f"P={item['pvalue']:.3f}"
        elif item['pvalue'] < 0.05:
            p_str = f"P={item['pvalue']:.3f}"
        else:
            p_str = f"P={item['pvalue']:.3f}"
        
        sig_mark = '***' if item['pvalue'] < 0.001 else ('**' if item['pvalue'] < 0.01 else ('*' if item['pvalue'] < 0.05 else ''))
        text = f"HR={item['hr']:.2f} ({item['ci_lower']:.2f}–{item['ci_upper']:.2f}) {p_str} {sig_mark}"
        ax.annotate(text, xy=(text_x, item['y']), fontsize=8, va='center', fontfamily='monospace')

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=9)

# Add subgroup titles (bold annotation at the top row of each subgroup)
current_sg = None
for item in reversed(plot_items):
    if item['subgroup'] != current_sg:
        current_sg = item['subgroup']
        # Find all y coordinates of this subgroup
        sg_items = [it for it in plot_items if it['subgroup'] == current_sg]
        max_y = max(it['y'] for it in sg_items)
        # Annotate the subgroup name on the top row
        dim_color = dimension_colors.get(current_sg, 'black')
        ax.annotate(current_sg, xy=(-0.02, max_y + 0.35), fontsize=10, fontweight='bold',
                    color=dim_color, va='bottom', ha='right',
                    xycoords=('axes fraction', 'data'))

# Add separator lines between subgroups
y_prev = None
for i, item in enumerate(plot_items):
    if i > 0 and item['subgroup'] != plot_items[i-1]['subgroup']:
        mid_y = (item['y'] + plot_items[i-1]['y']) / 2
        ax.axhline(y=mid_y, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xlabel('Hazard Ratio (High-risk vs Low-risk by Fusion Score)', fontsize=13)
ax.set_title('Forest plot of fusion model subgroup analysis - training set and dual external validation sets\nForest Plot of Fusion Model Subgroup Analysis (Training + Test1 + Test2)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, text_x + 6)
ax.grid(axis='x', alpha=0.2)

# Legend - cohort colors
legend_handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50', markersize=10, label='Training set (n=282)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markersize=10, label='Test set 1 (n=113)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Test set 2 (n=491)'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/subgroup_forest_all_cohorts.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/subgroup_forest_all_cohorts.pdf', bbox_inches='tight')
plt.savefig('figures/subgroup_forest_all_cohorts.svg', format='svg', bbox_inches='tight')
plt.close()
print("Forest plot saved → figures/subgroup_forest_all_cohorts.*")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*70)
print("Cross-cohort HR consistency summary")
print("="*70)

for sg in subgroup_order:
    sg_data = results_df[results_df['Subgroup'] == sg]
    hrs = sg_data.set_index('Cohort')['HR']
    ps = sg_data.set_index('Cohort')['P value']
    print(f"\n{sg}:")
    for cohort in cohort_order:
        if cohort in hrs.index and not pd.isna(hrs[cohort]):
            p = ps[cohort]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {cohort}: HR={hrs[cohort]:.2f}, P={p:.4f} {sig}")
        else:
            print(f"  {cohort}: N/A")

# Statistical significance summary
print("\n\nSignificance summary:")
for cohort in cohort_order:
    cohort_data = results_df[results_df['Cohort'] == cohort]
    valid = cohort_data.dropna(subset=['P value'])
    sig_count = (valid['P value'] < 0.05).sum()
    total = len(valid)
    print(f"  {cohort}: {sig_count}/{total} subgroups reached P<0.05")

# Direction consistency of HR>1
print("\nDirection consistency (HR>1):")
for cohort in cohort_order:
    cohort_data = results_df[results_df['Cohort'] == cohort]
    valid = cohort_data.dropna(subset=['HR'])
    hr_gt1 = (valid['HR'] > 1).sum()
    total = len(valid)
    print(f"  {cohort}: {hr_gt1}/{total} subgroups with HR>1")
