import numpy as np
import torch
import torch
import pyro
import pyro.distributions as dist
from scipy.special import digamma
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO
from collections import defaultdict
from CIX_course_Modelling_ToM_functions import *


# ----------------------------------------------------------
# Definition of Pyro models for each agent

def q_learning_pyro_model(actions, opponent_actions, rewards, in_dict = None):
    """
    Pyro model for Q-learning agent.
    actions: Tensor of actions taken by the agent.
    opponent_actions: Tensor of actions taken by the opponent (not used in this model).
    rewards: Tensor of rewards received by the agent.
    in_dict: Additional information for the model (e.g., game, player).

    Returns: A Pyro model function that can be used with SVI for fitting.
    """
    nb_trials = len(actions)
    def model():
        alpha = pyro.sample("alpha", dist.Normal(0.0, 3.0))
        beta = pyro.sample("beta", dist.Normal(0.0, 3.0))
        bias = pyro.sample("bias", dist.Normal(0.0, 3.0))
        Q = torch.zeros(2)
        theta = [alpha]
        phi = [beta, bias]

        for t in range(nb_trials):
            a = actions[t]
            r = rewards[t]
            probs = g_Qlearning(Q.detach().numpy(), torch.stack(phi).detach().numpy(), None, None)
            pyro.sample(f"action_{t}", dist.Categorical(torch.tensor([1-probs, probs])), obs=a)
            Qnp = f_Qlearning(Q.detach().numpy(), torch.stack(theta).detach().numpy(), [np.asarray(a), np.asarray(r)], None)
            Q = torch.tensor(Qnp, dtype=torch.float)

    return model

def fictitious_learner_pyro_model(actions, opponent_actions, rewards, in_dict = None):
    """
    Pyro model for Fictitious Learner agent.
    actions: Tensor of actions taken by the agent.
    opponent_actions: Tensor of actions taken by the opponent (not used in this model).
    rewards: Tensor of rewards received by the agent.
    in_dict: Additional information for the model (e.g., game, player).

    Returns: A Pyro model function that can be used with SVI for fitting.
    """
    nb_trials = len(actions)
    def model():
        alpha = pyro.sample("alpha", dist.Normal(0.0, 3.0))
        beta = pyro.sample("beta", dist.Normal(0.0, 3.0))
        bias = pyro.sample("bias", dist.Normal(0.0, 3.0))
        Q = torch.zeros(2)
        theta = [alpha]
        phi = [beta, bias]

        for t in range(nb_trials):
            a = actions[t]
            r = rewards[t]
            probs = g_fictitious_learner(Q.detach().numpy(), torch.stack(phi).detach().numpy(), None, None)
            pyro.sample(f"action_{t}", dist.Categorical(torch.tensor([1-probs, probs])), obs=a)
            Qnp = f_fictitious_learner(Q.detach().numpy(), torch.stack(theta).detach().numpy(), [np.asarray(a), np.asarray(r)], None)
            Q = torch.tensor(Qnp, dtype=torch.float)

    return model

def influence_learning_pyro_model(actions, opponent_actions, rewards, in_dict):
    """
    Pyro model for Influence Learning agent.
    actions: Tensor of actions taken by the agent.
    opponent_actions: Tensor of actions taken by the opponent.
    rewards: Tensor of rewards received by the agent (not used in this model).
    in_dict: Additional information for the model (e.g., game, player).

    Returns: A Pyro model function that can be used with SVI for fitting.
    """

    nb_trials = len(actions)
    def model():
        eta = pyro.sample("eta", dist.Normal(0.0, 3.))
        lambd = pyro.sample("lambd", dist.Normal(0.0, 3.))
        beta_opp = pyro.sample("beta_opp", dist.Normal(0.0, 3.))
        bias = pyro.sample("bias", dist.Normal(0.0,3.))
        beta = pyro.sample("beta", dist.Normal(0.0, 3.))

        x_init_val = sigmoid(torch.tensor(0.5), inverse=True)  # initial belief log-odds P(o=1)
        x = torch.tensor([x_init_val], dtype=torch.float)

        theta = [eta, lambd, beta_opp]
        phi = [beta, bias]
        # in_dict = {'game': game, 'player': player_role}

        for t in range(nb_trials):
            a = actions[t]
            o = opponent_actions[t]
            probs = g_influence_learner(x.detach().numpy(), torch.stack(phi).detach().numpy(), None, in_dict)
            pyro.sample(f"action_{t}", dist.Categorical(torch.tensor([1-probs, probs])), obs=a)
            xnp = f_influence_learner(x.detach().numpy(), torch.stack(theta).detach().numpy(), [np.asarray(o), np.asarray(a)], in_dict)
            x = torch.tensor(xnp, dtype=torch.float)

    return model

def MIIL_pyro_model(actions, opponent_actions, rewards, in_dict):
    """
    Pyro model for Mixed-Intentions Influence Learner (MIIL).
    actions: Tensor of actions taken by the agent.
    opponent_actions: Tensor of actions taken by the opponent.
    rewards: Tensor of rewards received by the agent (not used in this model).
    in_dict: Additional information for the model (e.g., game, player).

    Returns: A Pyro model function that can be used with SVI for fitting.
    """

    nb_trials = len(actions)
    def model():
        eta = pyro.sample("eta", dist.Normal(0.0, 3.0))       # agent's prediction error weight
        lambd = pyro.sample("lambd", dist.Normal(0.0, 3.0))   # other's prediction error weight
        beta_opp = pyro.sample("beta_opp", dist.Normal(0.0, 3.0))     # opponent's softmax temp
        invprec = pyro.sample("invprec", dist.Normal(0.0, 3.0))  # game belief precision
        bias = pyro.sample("bias", dist.Normal(0.0, 3.0))
        beta1 = pyro.sample("beta1", dist.Normal(0.0, 3.0))
        beta2 = pyro.sample("beta2", dist.Normal(0.0, 3.0))

        # initial log-odds beliefs: [P(o=1|g1), P(o=1|g2), P(game1)]
        # Ensure x is a tensor with the expected initial shape, e.g., [3]
        x_init_val = sigmoid(0.5,inverse=True)
        x = torch.tensor([x_init_val] * 3, dtype=torch.float)

        theta = [eta, lambd, beta_opp, invprec, bias]
        phi = [beta1, beta2]
        # in_dict = {'game1': payoff_game1, 'game2': payoff_game2, 'player': player_role}

        for t in range(nb_trials):
            a = actions[t]
            o = opponent_actions[t]
            probs_a1 = g_MIIL(x.detach().numpy(), torch.stack(phi).detach().numpy(), None, in_dict)
            probs_a1 = float(np.clip(probs_a1, 1e-6, 1 - 1e-6).item()) # Clip to avoid 0 or 1 exactly

            # Create the probabilities tensor for the two actions [P(a=0), P(a=1)]
            action_probs = torch.tensor([1 - probs_a1, probs_a1])

            pyro.sample(f"action_{t}", dist.Categorical(action_probs), obs=a)
            xnp = f_MIIL(x.detach().numpy(), torch.stack(theta).detach().numpy(), [np.asarray(o), np.asarray(a)], in_dict)
            x = torch.tensor(xnp, dtype=torch.float)

    return model

# ----------------------------------------------------------
# Definition of Pyro guides for each agent

def guide_Qlearn():
    ''' Guide for Q-learning agent.'''
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    bias_loc = pyro.param("bias_loc", torch.tensor(0.0))
    pyro.sample("alpha", dist.Normal(alpha_loc,2.0))
    pyro.sample("beta", dist.Normal(beta_loc,2.0))
    pyro.sample("bias", dist.Normal(bias_loc,2.0))


def guide_FPlayer():
    ''' Guide for Fictitious Learner agent.'''
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    bias_loc = pyro.param("bias_loc", torch.tensor(0.0))
    pyro.sample("alpha", dist.Normal(alpha_loc,2.0))
    pyro.sample("beta", dist.Normal(beta_loc,2.0))
    pyro.sample("bias", dist.Normal(bias_loc,2.0))


def guide_influence_learning():
    ''' Guide for Influence Learning agent.'''
    eta_loc = pyro.param("eta_loc", torch.tensor(0.0))
    pyro.sample("eta", dist.Normal(eta_loc,2.0))
    lambd_loc = pyro.param("lambd_loc", torch.tensor(0.0))
    pyro.sample("lambd", dist.Normal(lambd_loc,2.0))
    beta_opp_loc = pyro.param("beta_opp_loc", torch.tensor(0.0))
    pyro.sample("beta_opp", dist.Normal(beta_opp_loc,2.0))
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    pyro.sample("beta", dist.Normal(beta_loc,2.0))
    bias_loc = pyro.param("bias_loc", torch.tensor(0.0))
    pyro.sample("bias", dist.Normal(bias_loc,2.0))


def guide_MIIL():
    ''' Guide for Mixed-Intentions Influence Learner (MIIL) agent.'''
    eta_loc = pyro.param("eta_loc", torch.tensor(0.0))
    pyro.sample("eta", dist.Normal(eta_loc,2.0))
    lambd_loc = pyro.param("lambd_loc", torch.tensor(0.0))
    pyro.sample("lambd", dist.Normal(lambd_loc,2.0))
    beta_opp_loc = pyro.param("beta_opp_loc", torch.tensor(0.0))
    pyro.sample("beta_opp", dist.Normal(beta_opp_loc,2.0))
    invprec_loc = pyro.param("invprec_loc", torch.tensor(0.0))
    pyro.sample("invprec", dist.Normal(invprec_loc,2.0))
    bias_loc = pyro.param("bias_loc", torch.tensor(0.0))
    pyro.sample("bias", dist.Normal(bias_loc,2.0))
    beta1_loc = pyro.param("beta1_loc", torch.tensor(0.0))
    pyro.sample("beta1", dist.Normal(beta1_loc,2.0))
    beta2_loc = pyro.param("beta2_loc", torch.tensor(0.0))
    pyro.sample("beta2", dist.Normal(beta2_loc,2.0))

# ------------------------------------------------------------
# Function to fit a Pyro model using SVI

def fit_model(data, model_fn, in_dict, guide_fn = None, n_steps=1000, tolerance=1e-4, verbose=True, patience=100):
    """
    Fits a Pyro model using Stochastic Variational Inference (SVI).
    data: DataFrame with 'agent1_action', 'agent2_action', 'agent1_reward' columns.
    model_fn: Function to create the Pyro model.
    in_dict: Additional information for the model (e.g., game, player).
    guide_fn: Optional guide function for the model. If None, an AutoGuide will be created.
    n_steps: Number of SVI steps to perform.
    tolerance: Tolerance for early stopping based on loss improvement.
    verbose: Whether to print progress information.
    patience: Number of steps without improvement before stopping.
    Returns: Dictionary with fitted model parameters, ELBO, and action probabilities.
    """

    a = torch.tensor(data['agent1_action'].values, dtype=torch.long)
    o = torch.tensor(data['agent2_action'].values, dtype=torch.long)
    r = torch.tensor(data['agent1_reward'].values, dtype=torch.float)

    # Reset param store for each fitting
    pyro.clear_param_store()

    model = model_fn(a, o, r, in_dict)
    # optimizer = Adam({"lr": 0.01})
    optimizer = ClippedAdam({"lr": 1e-3, "clip_norm": 10.0})

    if guide_fn is None:
        # Automatically create a guide function if not provided
        # guide_fn = pyro.infer.autoguide.AutoLaplaceApproximation(model)
        guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
        custom_guide = False
    else:
        # Use the provided guide function
        guide = guide_fn
        custom_guide = True
    

    svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

    losses = []
    best_loss = float("inf")
    steps_since_improvement = 0

    for step in range(n_steps):
        loss = svi.step()
        losses.append(loss)

        # Improvement check
        if loss < best_loss - tolerance:
            best_loss = loss
            # Extract parameters
            best_params = {name: pyro.param(name).clone().detach()
                        for name in pyro.get_param_store().keys()}
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        # Verbose logging
        if verbose and step % 100 == 0:
            print(f"[{step}] ELBO: {-loss:.2f}, Best: {-best_loss:.2f}")

        # Early stopping condition
        if steps_since_improvement >= patience:
            if verbose:
                print(f"Early stopping at step {step}: no improvement in {patience} steps.")
            break

    # Re-arrange best_params into theta and phi
    print("Best parameters found:"+ str(best_params))

    if custom_guide:
        if model_fn == q_learning_pyro_model: 
            f = f_Qlearning
            g = g_Qlearning
            dat = [a,r]
            phi = np.array([best_params["beta_loc"].item(), best_params["bias_loc"].item()])
            theta = np.array([best_params["alpha_loc"].item()])            
        elif model_fn == fictitious_learner_pyro_model:
            f = f_fictitious_learner
            g = g_fictitious_learner
            dat = [a,r]
            phi = np.array([best_params["beta_loc"].item(), best_params["bias_loc"].item()])
            theta = np.array([best_params["alpha_loc"].item()])
        elif model_fn == influence_learning_pyro_model:
            f = f_influence_learner
            g = g_influence_learner
            dat = [o,a]
            phi = np.array([best_params["beta_loc"].item(), best_params["bias_loc"].item()])
            theta = np.array([best_params["eta_loc"].item(), best_params["lambd_loc"].item(), best_params["beta_opp_loc"].item()])
        elif model_fn == MIIL_pyro_model:
            f = f_MIIL
            g = g_MIIL
            dat = [o,a]
            phi = np.array([best_params["beta1_loc"].item(), best_params["beta2_loc"].item()])
            theta = np.array([best_params["eta_loc"].item(), best_params["lambd_loc"].item(), best_params["beta_opp_loc"].item(), 
                            best_params["invprec_loc"].item(), best_params["bias_loc"].item()])            
    else:
        if model_fn == q_learning_pyro_model: 
            f = f_Qlearning
            g = g_Qlearning
            dat = [a,r]
            phi = np.array([best_params["AutoMultivariateNormal.loc"][1].numpy(), best_params["AutoMultivariateNormal.loc"][2].numpy()])
            theta = np.array([best_params["AutoMultivariateNormal.loc"][0].numpy()])            
        elif model_fn == fictitious_learner_pyro_model:
            f = f_fictitious_learner
            g = g_fictitious_learner
            dat = [a,r]
            phi = np.array([best_params["AutoMultivariateNormal.loc"][1].numpy(), best_params["AutoMultivariateNormal.loc"][2].numpy()])
            theta = np.array([best_params["AutoMultivariateNormal.loc"][0].numpy()])
        elif model_fn == influence_learning_pyro_model:
            f = f_influence_learner
            g = g_influence_learner
            dat = [o,a]
            phi = np.array([best_params["AutoMultivariateNormal.loc"][3].numpy(), best_params["AutoMultivariateNormal.loc"][4].numpy()])
            theta = np.array([best_params["AutoMultivariateNormal.loc"][0].numpy(), best_params["AutoMultivariateNormal.loc"][1].numpy(), best_params["AutoMultivariateNormal.loc"][2].numpy()])
        elif model_fn == MIIL_pyro_model:
            f = f_MIIL
            g = g_MIIL
            dat = [o,a]
            phi = np.array([best_params["AutoMultivariateNormal.loc"][5].numpy(), best_params["AutoMultivariateNormal.loc"][6].numpy()])
            theta = np.array([best_params["AutoMultivariateNormal.loc"][0].numpy(), best_params["AutoMultivariateNormal.loc"][1].numpy(), best_params["AutoMultivariateNormal.loc"][2].numpy(),
                                best_params["AutoMultivariateNormal.loc"][3].numpy(), best_params["AutoMultivariateNormal.loc"][4].numpy()])
            
    best_params = {
                    "phi": phi,
                    "theta": theta
                }
    
    # Simulate predictions step-by-step
    probs = predict_model_probs(data=dat, f=f, g=g, params=best_params, in_dict=in_dict)
    
    return {
        "params": best_params,
        "elbo": best_loss,
        "action_probs": probs#,
        # "guide": guide,
        # "losses": losses
    }

def predict_model_probs(data, f, g, params, in_dict):
    """
    Predicts action probabilities for a given model and parameters.
    data: DataFrame with 'agent1_action', 'agent2_action', 'agent1_reward' columns.
    model_fn: Function to create the model.
    params: Dictionary with 'phi' and 'theta' parameters.
    in_dict: Additional information for the model (e.g., game, player).
    Returns: Numpy array of predicted action probabilities.
    """
    # Extract parameters
    phi = torch.tensor(params["phi"], dtype=torch.float)
    theta = torch.tensor(params["theta"], dtype=torch.float)

    # Simulate predictions
    probs = []
    x = torch.zeros(in_dict["dim_x"])  # Initialize hidden states

    for t in range(data[0].shape[0]):
        probs_t = g(x.detach().numpy(), phi.detach().numpy(), None, in_dict) #probability of action 0
        # print("Probs_t: ", probs_t)
        probs.append(probs_t)        
        u = [data[0][t].numpy(), data[1][t].numpy()] 
        xnp = f(x.detach().numpy(), theta.detach().numpy(), u, in_dict)
        x = torch.tensor(xnp, dtype=torch.float)

    # Flatten the list
    flattened_probs = [float(item) if isinstance(item, np.ndarray) else item for item in probs]
    # print("Flattened probs: ", flattened_probs)

    return np.array(flattened_probs, dtype=float)

def compute_goodness_of_fit(res, data):
    """
    Computes goodness of fit metrics for the fitted model.
    res: Dictionary with model fit results, including 'action_probs' and 'params'.  
    data: DataFrame with 'agent1_action', 'agent2_action', 'agent1_reward' columns.
    Returns: Dictionary with accuracy, balanced accuracy, log likelihood, AIC, and BIC.
    """

    action_probs = res['action_probs']
    actions = np.array(data['agent1_action']).T

    # Model accuracy
    accuracy = np.mean(np.round(action_probs) == actions)

    # Model balanced accuracy
    balanced_accuracy = (np.mean(np.round(action_probs[actions == 0]) == 0) + np.mean(np.round(action_probs[actions == 1]) == 1)) / 2
    
    # # Model R^2: computed as 1 - (SS_res / SS_tot), where SS_res is the sum of squares of residuals and SS_tot is the total sum of squares (not suitable for binary probabilities!)
    # r_squared = 1 - np.sum((actions - action_probs) ** 2) / np.sum((actions - np.mean(actions)) ** 2)

    # Model log likelihood
    action_probs = np.clip(action_probs, 1e-8, 1 - 1e-8) # Avoid log(0) issues
    log_likelihood = np.sum(np.log(action_probs) * actions + np.log(1 - action_probs) * (1 - actions))

    # Model AIC
    n_params = len(res['params']['phi']) + len(res['params']['theta'])
    aic = 2 * n_params - 2 * log_likelihood

    # Model BIC
    n_trials = len(actions)
    bic = n_params * np.log(n_trials) - 2 * log_likelihood

    return {
            "accuracy": accuracy,   
            "balanced_accuracy": balanced_accuracy,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic
            }

def plot_fit_results(res, sim_hist, fitted_model_name, n_trials):
    '''Plots the fit results of the model against the simulated history.'''

    plt.figure(figsize=(12, 6))
    plt.plot(res['action_probs'], label='Model action predictions', color='blue')
    plt.scatter(sim_hist['trial_nb'], sim_hist['agent1_action'], label='Agent 1 actions', color='orange', alpha=0.5)
    plt.scatter(sim_hist['trial_nb'], sim_hist['agent2_action'], label='Agent 2 actions', color='green', alpha=0.5, marker='x')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random action baseline')
    plt.xticks(ticks=range(0, n_trials, 20), rotation=45)
    plt.title(f'Action Predictions vs Actual Actions for {fitted_model_name}')
    plt.xlabel('Trial Number')
    plt.ylabel('Action (0 or 1)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------
# Compare model fits of the simulated agent and plot results

# Convert Free Energies to Per-Participant Posterior Model Probabilities
def build_elbo_matrix(fit_result):
    """
    Builds ELBO matrix: participants × fitted models
    fit_result: List of dictionaries with keys 'fitted_model', 'participant_nb', and 'model_fit_result'.
    Returns: ELBO matrix (numpy array) and model names (list).
    """
    elbos_by_model = defaultdict(dict)  # model → {participant_nb: elbo}

    # Gather ELBOs
    for entry in fit_result:
        model = entry["fitted_model"]
        participant_nb = entry["participant_nb"]
        elbo = entry["model_fit_result"]["elbo"]
        elbos_by_model[model][participant_nb] = elbo

    # All model names and participant indices
    model_names = sorted(elbos_by_model.keys())
    # Unique participant numbers
    sim_nbs = sorted({entry["participant_nb"] for entry in fit_result})
    # Number of participants and models
    n_participants = len(sim_nbs)
    n_models = len(model_names)

    # Initialize ELBO matrix
    elbo_matrix = np.zeros((n_participants, n_models))

    # Fill the ELBO matrix
    # Each row corresponds to a participant, each column to a model
    for m_idx, model in enumerate(model_names):
        for s_idx, participant_nb in enumerate(sim_nbs):
            elbo_matrix[s_idx, m_idx] = elbos_by_model[model][participant_nb]

    return elbo_matrix, model_names

# Estimate Dirichlet Parameters (Frequency Inference via VB)
def estimate_dirichlet_rfx(probs, max_iter=100, tol=1e-4):
    """
    RFX VB estimation of model frequencies.
    probs: NxK matrix of participant-level model posterior probabilities.
    Returns: Dirichlet alphas and exceedance probabilities.
    """
    N, K = probs.shape
    alpha = np.ones(K)

    # Compute model posterior per subject: softmax over negative ELBOs
    log_evs = -probs
    log_evs -= log_evs.max(axis=1, keepdims=True)
    p_mnk = np.exp(log_evs + 1e-08)  # Avoid log(0)
    p_mnk /= p_mnk.sum(axis=1, keepdims=True)

    # Initialize Dirichlet parameters (alphas)
    for _ in range(max_iter):
        # Update Dirichlet parameters using VB update rule
        # E[log r_k] = digamma(alpha_k) - digamma(sum(alpha))
        alpha_prev = alpha.copy()
        E_log_r = digamma(alpha) - digamma(np.sum(alpha))

        # Update alpha using the expected log probabilities
        # r_nk = p_mnk * exp(E[log r_k])
        r_nk = p_mnk * np.exp(E_log_r)

        # Normalize r_nk to get the expected model frequencies
        r_nk /= r_nk.sum(axis=1, keepdims=True)

        # Update alpha parameters
        # alpha_k = sum(r_nk_n) for each model k
        alpha = r_nk.sum(axis=0)

        # Ensure alphas are positive
        alpha[alpha <= 0] = 1e-08  # Avoid numerical issues

        # Check convergence
        # If the change in alpha is less than the tolerance, stop
        if np.linalg.norm(alpha - alpha_prev) < tol:
            break

    # Exceedance probabilities via Dirichlet sampling
    exceedance_probs = compute_exceedance_prob(alpha)
    return alpha, exceedance_probs

def compute_exceedance_prob(alpha, n_samples=1_000_000):
    """
    Exceedance probability: P(model_k > all others)
    alpha: Dirichlet parameters (K-dimensional array).
    n_samples: Number of samples to draw from the Dirichlet distribution.
    Returns: Exceedance probabilities for each model (K-dimensional array).
    """
    # Sample from Dirichlet distribution
    # This gives us a large number of samples from the Dirichlet distribution
    samples = np.random.dirichlet(alpha, size=n_samples)
    # Find the index of the maximum value in each sample
    # This gives us the model that was chosen in each sample
    best = np.argmax(samples, axis=1)
    return np.bincount(best, minlength=len(alpha)) / n_samples

# To Visualize model frequencies (alpha) and exceedance probabilities
def bootstrap_alphas(probs, n_boot=100):
    """
    Bootstraps alpha estimates from resampled participant probabilities.
    Returns:
        mean_alphas: K-dimensional array
        ci_lower, ci_upper: 95% confidence intervals (K,)
    """
    N = probs.shape[0]
    K = probs.shape[1]
    alpha_samples = np.zeros((n_boot, K))
    
    # Resample N participants with replacement
    for i in range(n_boot):
        # Resample participant probabilities with replacement
        # This is equivalent to resampling rows of probs
        resampled = probs[np.random.choice(N, N, replace=True)]
        # Estimate Dirichlet parameters from resampled probabilities
        alpha_i, _ = estimate_dirichlet_rfx(resampled)
        # Store the alpha parameters
        alpha_samples[i] = alpha_i

    # Compute mean and 95% CI from bootstrap samples
    mean_alpha = np.mean(alpha_samples, axis=0)
    ci_lower = np.percentile(alpha_samples, 2.5, axis=0)
    ci_upper = np.percentile(alpha_samples, 97.5, axis=0)
    return mean_alpha, ci_lower, ci_upper

def plot_rfx_model_comparison_with_ci(model_names, alphas, exceedance_probs, ci_lower, ci_upper):
    """
    Plot RFX model comparison with confidence intervals on alphas.
    model_names: List of model names.
    alphas: Estimated model frequencies (K-dimensional array).
    exceedance_probs: Exceedance probabilities (K-dimensional array).
    ci_lower, ci_upper: 95% confidence intervals for alphas (K-dimensional arrays).
    returns: None
    """
    K = len(model_names)
    colors = sns.color_palette("Set2", K)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # α plot with CI
    lower_errors = np.maximum(0, alphas - ci_lower) # Ensure non-negative error bar length
    upper_errors = np.maximum(0, ci_upper - alphas) # Ensure non-negative error bar length

    
    axes[0].bar(model_names, alphas, color=colors, yerr=[lower_errors, upper_errors],
                capsize=5, ecolor='black')
    axes[0].set_title("Estimated Model Frequencies")
    axes[0].set_ylabel("Frequency")
    axes[0].set_ylim(0, max(ci_upper) * 1.2)

    # Exceedance probabilities
    axes[1].bar(model_names, exceedance_probs, color=colors)
    axes[1].set_title("Exceedance Probabilities")
    axes[1].set_ylabel("P(model > all others)")
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()