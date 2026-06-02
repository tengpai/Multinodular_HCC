"""
Time-dependent IDI and NRI computation
Use the correct survival-analysis method, referring to Uno's method

Time-dependent IDI: at a specific time point t, compare the discrimination improvement of two models
Time-dependent NRI: at a specific time point t, compare the reclassification improvement of two models
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
data_path = '<PROJECT_ROOT>/sol15_multimodal_fusion/data/cox_fusion_complete_results.csv'
df = pd.read_csv(data_path)

# Separate the datasets
train_df = df[df['group'] == 'train'].copy()
test1_df = df[df['group'] == 'test1'].copy()
test2_df = df[df['group'] == 'test2'].copy()

print(f"Training set: n={len(train_df)}, events={train_df['event'].sum()}")
print(f"Validation set 1: n={len(test1_df)}, events={test1_df['event'].sum()}")
print(f"Validation set 2: n={len(test2_df)}, events={test2_df['event'].sum()}")

score_cols = ['Clinical_Score', 'Rad_Diff_HighRisk', 'DL_Heterogeneity', 'DL_MaxPooling', 'Habitat_Agg_Best']
score_labels = ['Clinical score', 'Traditional radiomics (Diff)', 'Deep learning (Het)', 'Deep learning (MaxP)', 'Habitat radiomics (Max-H1)']


def calculate_survival_probability(data, score_col, time_point, duration_col='duration', event_col='event'):
    """
    Use the Cox model to compute the survival probability at a specific time point
    """
    cph = CoxPHFitter()
    df_fit = data[[duration_col, event_col, score_col]].dropna().copy()
    cph.fit(df_fit, duration_col=duration_col, event_col=event_col)
    
    # Get the baseline survival function
    baseline_survival = cph.baseline_survival_
    
    # Get the cumulative risk probability of each individual at time_point (1 - survival probability)
    partial_hazard = cph.predict_partial_hazard(df_fit[[score_col]])
    
    # Find the time closest to time_point
    times = baseline_survival.index.values
    idx = np.abs(times - time_point).argmin()
    baseline_surv_at_t = baseline_survival.iloc[idx, 0]
    
    # Individualized survival probability: S(t|X) = S_0(t)^exp(beta*X)
    surv_prob = baseline_surv_at_t ** partial_hazard.values.flatten()
    
    # Cumulative incidence = 1 - survival probability
    cum_incidence = 1 - surv_prob
    
    return cum_incidence, df_fit


def time_dependent_idi_nri(data, ref_score, new_score, time_point, duration_col='duration', event_col='event'):
    """
    Compute Time-dependent IDI and NRI
    
    Based on the methods of Pencina et al. (2008) and Uno et al. (2011)
    
    Time-dependent IDI:
    IDI(t) = [mean(P_new(t)|event at t) - mean(P_new(t)|no event at t)] 
           - [mean(P_ref(t)|event at t) - mean(P_ref(t)|no event at t)]
    
    where "event at t" means an event occurred before time t
    
    Time-dependent NRI:
    Censoring needs to be considered, using IPCW (Inverse Probability of Censoring Weighting)
    """
    df_tmp = data[[duration_col, event_col, ref_score, new_score]].dropna().copy()
    
    # Compute the predicted probability of the two models at time_point
    pred_ref, _ = calculate_survival_probability(df_tmp, ref_score, time_point, duration_col, event_col)
    pred_new, _ = calculate_survival_probability(df_tmp, new_score, time_point, duration_col, event_col)
    
    # Define the event group and the no-event group at time_point
    # Event group: event occurred before time_point (duration <= time_point & event == 1)
    # No-event group: still alive at time_point (duration > time_point or (duration <= time_point & event == 0 but censored))
    
    events_by_t = (df_tmp[duration_col] <= time_point) & (df_tmp[event_col] == 1)
    survivors_at_t = df_tmp[duration_col] > time_point
    
    # For censored data, if the censoring time <= time_point, we do not know their true status, so they need to be excluded or IPCW used
    censored_by_t = (df_tmp[duration_col] <= time_point) & (df_tmp[event_col] == 0)
    
    # Simplified handling: only use individuals whose status we know exactly
    # Event group + survival group (excluding individuals censored before time_point)
    known_status = events_by_t | survivors_at_t
    
    if events_by_t.sum() == 0 or survivors_at_t.sum() == 0:
        return {
            'time_point': time_point,
            'IDI': np.nan,
            'NRI_continuous': np.nan,
            'n_events_by_t': events_by_t.sum(),
            'n_survivors_at_t': survivors_at_t.sum(),
            'n_excluded_censored': censored_by_t.sum()
        }
    
    # Compute IDI
    # Discrimination slope = mean(P|event) - mean(P|no event)
    disc_ref = pred_ref[events_by_t].mean() - pred_ref[survivors_at_t].mean()
    disc_new = pred_new[events_by_t].mean() - pred_new[survivors_at_t].mean()
    idi = disc_new - disc_ref
    
    # Standard error of IDI (Bootstrap or analytic method) - here a simplified analytic method is used
    n_events = events_by_t.sum()
    n_nonevents = survivors_at_t.sum()
    
    var_ref_events = pred_ref[events_by_t].var() / n_events
    var_ref_nonevents = pred_ref[survivors_at_t].var() / n_nonevents
    var_new_events = pred_new[events_by_t].var() / n_events
    var_new_nonevents = pred_new[survivors_at_t].var() / n_nonevents
    
    # Approximate IDI standard error
    se_idi = np.sqrt(var_ref_events + var_ref_nonevents + var_new_events + var_new_nonevents)
    z_idi = idi / se_idi if se_idi > 0 else 0
    p_idi = 2 * (1 - stats.norm.cdf(abs(z_idi)))
    
    # Compute the continuous NRI
    # NRI_event = P(P_new > P_ref | event) - P(P_new < P_ref | event)
    # NRI_nonevent = P(P_new < P_ref | no event) - P(P_new > P_ref | no event)
    
    diff_events = pred_new[events_by_t] - pred_ref[events_by_t]
    diff_nonevents = pred_new[survivors_at_t] - pred_ref[survivors_at_t]
    
    # Event group: we want the new model to predict higher (positive reclassification)
    nri_events = ((diff_events > 0).mean() - (diff_events < 0).mean())
    
    # No-event group: we want the new model to predict lower (negative reclassification)
    nri_nonevents = ((diff_nonevents < 0).mean() - (diff_nonevents > 0).mean())
    
    nri_continuous = nri_events + nri_nonevents
    
    # NRI standard error (simplified)
    se_nri_events = np.sqrt(((diff_events > 0).var() + (diff_events < 0).var()) / n_events)
    se_nri_nonevents = np.sqrt(((diff_nonevents < 0).var() + (diff_nonevents > 0).var()) / n_nonevents)
    se_nri = np.sqrt(se_nri_events**2 + se_nri_nonevents**2)
    z_nri = nri_continuous / se_nri if se_nri > 0 else 0
    p_nri = 2 * (1 - stats.norm.cdf(abs(z_nri)))
    
    return {
        'time_point': time_point,
        'n_events_by_t': int(n_events),
        'n_survivors_at_t': int(n_nonevents),
        'n_excluded_censored': int(censored_by_t.sum()),
        'Disc_ref': disc_ref,
        'Disc_new': disc_new,
        'IDI': idi,
        'IDI_SE': se_idi,
        'IDI_Z': z_idi,
        'IDI_P': p_idi,
        'NRI_events': nri_events,
        'NRI_nonevents': nri_nonevents,
        'NRI_continuous': nri_continuous,
        'NRI_SE': se_nri,
        'NRI_Z': z_nri,
        'NRI_P': p_nri
    }


# Compute IDI and NRI at different time points
print("\n" + "="*70)
print("Time-dependent IDI/NRI (Fusion model vs each single score)")
print("="*70)

time_points = [12, 24, 36]  # 1-year, 2-year, 3-year

all_results = []

for dataset_name, dataset in [('Training set', train_df), ('Validation set 1', test1_df), ('Validation set 2', test2_df)]:
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    for t in time_points:
        print(f"\n--- Time point: {t} months ---")
        
        for score, label in zip(score_cols, score_labels):
            try:
                result = time_dependent_idi_nri(dataset, score, 'Fusion_Cox', t)
                result['Dataset'] = dataset_name
                result['Reference model'] = label
                all_results.append(result)
                
                # Only print the detailed results of the training set at 3 years
                if dataset_name == 'Training set' and t == 36:
                    print(f"  {label}:")
                    print(f"    IDI = {result['IDI']:.4f} (SE={result['IDI_SE']:.4f}, P={result['IDI_P']:.4f})")
                    print(f"    NRI = {result['NRI_continuous']:.4f} (P={result['NRI_P']:.4f})")
            except Exception as e:
                print(f"  {label}: computation failed - {e}")

# Organize the results
results_df = pd.DataFrame(all_results)
results_df = results_df[['Dataset', 'Reference model', 'time_point', 'n_events_by_t', 'n_survivors_at_t',
                          'Disc_ref', 'Disc_new', 'IDI', 'IDI_SE', 'IDI_P',
                          'NRI_events', 'NRI_nonevents', 'NRI_continuous', 'NRI_P']]

# Save the complete results
results_df.to_csv('results/time_dependent_idi_nri_correct.csv', index=False)

# Print the summary table (3-year time point)
print("\n" + "="*70)
print("3-year time point IDI/NRI summary table (Fusion model vs each single score)")
print("="*70)

summary_3y = results_df[results_df['time_point'] == 36].copy()
summary_3y['IDI_str'] = summary_3y.apply(
    lambda x: f"{x['IDI']:.4f} {'*' if x['IDI_P'] < 0.05 else ''}", axis=1
)
summary_3y['NRI_str'] = summary_3y.apply(
    lambda x: f"{x['NRI_continuous']:.4f} {'*' if x['NRI_P'] < 0.05 else ''}", axis=1
)

print(summary_3y[['Dataset', 'Reference model', 'IDI_str', 'NRI_str']].to_string(index=False))

# Compute the averages
print("\n" + "="*70)
print("Summary statistics")
print("="*70)

for dataset_name in ['Training set', 'Validation set 1', 'Validation set 2']:
    subset = summary_3y[summary_3y['Dataset'] == dataset_name]
    print(f"\n{dataset_name} (3-year time point):")
    print(f"  Mean IDI: {subset['IDI'].mean():.4f}")
    print(f"  Mean NRI: {subset['NRI_continuous'].mean():.4f}")
    print(f"  IDI significant (P<0.05): {(subset['IDI_P'] < 0.05).sum()}/{len(subset)}")
    print(f"  NRI significant (P<0.05): {(subset['NRI_P'] < 0.05).sum()}/{len(subset)}")

# Final conclusion
print("\n" + "="*70)
print("Final conclusion")
print("="*70)

train_3y = summary_3y[summary_3y['Dataset'] == 'Training set']
mean_idi = train_3y['IDI'].mean()
mean_nri = train_3y['NRI_continuous'].mean()
n_sig_idi = (train_3y['IDI_P'] < 0.05).sum()
n_sig_nri = (train_3y['NRI_P'] < 0.05).sum()

if mean_idi > 0 and n_sig_idi >= 3:
    conclusion_idi = "The fusion model significantly improves individual risk discrimination (positive IDI improvement)"
elif mean_idi > 0:
    conclusion_idi = "The fusion model shows a small IDI improvement, but most are not significant"
elif mean_idi < 0 and n_sig_idi >= 3:
    conclusion_idi = "The fusion model IDI decreases significantly (single scores discriminate better)"
else:
    conclusion_idi = "The fusion model IDI shows no significant improvement"

print(f"\nTraining set 3-year time point:")
print(f"  Mean IDI = {mean_idi:.4f}, significant {n_sig_idi}/5")
print(f"  Mean NRI = {mean_nri:.4f}, significant {n_sig_nri}/5")
print(f"\nConclusion: {conclusion_idi}")

# External validation
ext_3y = summary_3y[summary_3y['Dataset'].isin(['Validation set 1', 'Validation set 2'])]
ext_mean_idi = ext_3y['IDI'].mean()
ext_mean_nri = ext_3y['NRI_continuous'].mean()
print(f"\nExternal validation 3-year time point:")
print(f"  Mean IDI = {ext_mean_idi:.4f}")
print(f"  Mean NRI = {ext_mean_nri:.4f}")

print("\nResults saved to: results/time_dependent_idi_nri_correct.csv")
