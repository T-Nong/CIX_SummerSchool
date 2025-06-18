import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from CIX_course_Fitting_ToM_functions import estimate_dirichlet_rfx

# ------------------------------------------------------------
# Model Recovery Analysis

def organize_elbos_by_simmodel(fit_result):
    """
    Returns:
        sim_model_names: sorted list of simulated models
        fit_model_names: sorted list of fitted models
        sim_to_elbo: dict of simulated_model -> matrix [N x K] of ELBOs
    """
    # Nested structure: sim_model → fit_model → list of elbos
    nested = defaultdict(lambda: defaultdict(dict))

    for entry in fit_result:
        sim = entry["simulated_model"]
        fit = entry["fitted_model"]
        pid = entry["participant_nb"]
        elbo = entry["model_fit_result"]["elbo"]
        nested[sim][fit][pid] = elbo

    sim_model_names = sorted(nested.keys())
    fit_model_names = sorted({m for fit_dict in nested.values() for m in fit_dict})

    # Build ELBO matrices for each simulated model
    sim_to_elbo = {}

    for sim in sim_model_names:
        fit_dict = nested[sim]
        participants = sorted({pid for model in fit_dict for pid in fit_dict[model]})
        matrix = np.zeros((len(participants), len(fit_model_names)))

        for j, fit in enumerate(fit_model_names):
            for i, pid in enumerate(participants):
                matrix[i, j] = fit_dict.get(fit, {}).get(pid, 0.0)

        sim_to_elbo[sim] = matrix

    return sim_model_names, fit_model_names, sim_to_elbo

def model_recovery_confusion(sim_model_names, fit_model_names, sim_to_elbo):
    """
    Computes confusion matrix: rows=simulated, cols=fitted, values=exceedance probs
    """
    conf_matrix = np.zeros((len(sim_model_names), len(fit_model_names)))

    for i, sim_model in enumerate(sim_model_names):
        elbo_matrix = sim_to_elbo[sim_model]
        alpha, xp = estimate_dirichlet_rfx(elbo_matrix)
        # Normalize alpha to get model frequencies
        conf_matrix[i] = alpha / np.sum(alpha) 
        # Alternatively, you can use exceedance probabilities directly
        # conf_matrix[i] = xp  # Use exceedance probability as measure


    return conf_matrix

def plot_confusion_matrix(conf_matrix, sim_model_names, fit_model_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=fit_model_names, yticklabels=sim_model_names)
    plt.xlabel("Recovered (fitted) model")
    plt.ylabel("True (simulated) model")
    plt.title("Model Recovery Confusion Matrix (Exceedance Probability)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Parameter recovery 

def organize_recovery_data(results, full_sims_hist):
    # 1. Create a list of recovered data
    recovery_data = []

    # 2. Match results with simulation history
    for res in results:
        sim_id = res['participant_nb']
        fitted = res['model_fit_result']['params']

        # Find matching simulation in full_sims_hist
        matching_sim = next((s for s in full_sims_hist if s['sim_id'] == sim_id and s['model']==res['simulated_model']), None)
        if matching_sim is None:
            continue

        # Extract true parameters
        true_phi = matching_sim['agent1'].phi
        true_theta = matching_sim['agent1'].theta

        # Extract recovered parameters (handle missing gracefully)
        rec_phi = fitted.get('phi', None)
        rec_theta = fitted.get('theta', None)

        # Append to recovery data
        recovery_data.append({
            'sim_id': sim_id,
            'true_phi': true_phi,
            'true_theta': true_theta,
            'rec_phi': rec_phi,
            'rec_theta': rec_theta,
            'sim_model': res['simulated_model'],
            'fitted_model': res['fitted_model']
        })

    # 3. Convert to DataFrame
    df = pd.DataFrame(recovery_data)
    df = df[df['sim_model'] == df['fitted_model']]
    
    return df

def plot_recovery(df, true_col, rec_col, title):

    nb_params = df[true_col].iloc[0].size  # Number of parameters to plot
    print(f"Number of parameters to plot: {nb_params}")

    if nb_params == 0:
        print("No parameters to plot.")
        return
    
    for i in range(nb_params):    
        temp = []
        print(f"Plotting parameter {i+1} for {title}")
        
        for j in range(len(df)):
            # Extract true and recovered parameters for the current row
            t_c = df[true_col].iloc[j]
            r_c = df[rec_col].iloc[j]

            # Append to temporary list
            temp.append({'true_param': t_c[i], 'fitted_param': r_c[i] if r_c.ndim>0 else float(r_c)})        
            # print(f"True: {t_c[i]}, Recovered: {r_c[i]}")

        # Create DataFrame for plotting
        # print(temp)
        df_temp = pd.DataFrame(temp)        

        # Scatter plot
        sns.scatterplot(x='true_param', y='fitted_param', data=df_temp)
        plt.plot([df_temp['true_param'].min(), df_temp['true_param'].max()],
                [df_temp['fitted_param'].min(), df_temp['fitted_param'].max()], 'r--')
        plt.xlabel("Simulated")
        plt.ylabel("Recovered")
        plt.title(title+f" parameter {i+1}")
        plt.grid(True)
        plt.show()