import numpy as np

# Utility functions
def sigmoid(x, inverse=False, slope=1, center=0):
    if inverse:
        x = np.clip(x, 1e-8, 1 - 1e-8)  # avoid log(0)
        return np.log(x / (1 - x))
    return 1 / (1 + np.exp(-slope * (x - center)))

def is_weird(X):
    """
    Checks if X contains NaNs, Infs, or complex numbers.

    Parameters:
    - X: numeric, list/tuple of numerics, or dict of numerics

    Returns:
    - bool: True if "weird", False if all fine, np.nan if non-numeric
    """

    if isinstance(X, (int, float, np.number, np.ndarray)):
        X = np.array(X)
        return np.any(np.isnan(X) | np.isinf(X) | ~np.isreal(X))

    elif isinstance(X, (list, tuple)):
        return any(is_weird(x) for x in X)

    elif isinstance(X, dict):
        return any(is_weird(v) for v in X.values())

    else:
        return np.nan  # Non-numeric and unsupported

def fplayer(P, beta, player, game):
    """
    Computes the decision variable DV from the payoff table and belief about the opponent's move.

    Arguments:
    - P: probability that the opponent plays action 1 (i.e., o=1)
    - beta: behavioral temperature
    - player: player index (1 or 2)
    - game: 2x2x2 payoff matrix (game[a, b, i] = payoff to player i for actions a, b)

    Returns:
    - DV: decision variable, indicating incentive to choose action 1
    """
    if player == 2:
        payoff = game[:, :, 1]
        DV = P * (payoff[0, 0] - payoff[0, 1]) + (1 - P) * (payoff[1, 0] - payoff[1, 1])
    elif player == 1:
        payoff = game[:, :, 0]
        DV = P * (payoff[0, 0] - payoff[1, 0]) + (1 - P) * (payoff[0, 1] - payoff[1, 1])
    else:
        raise ValueError("player must be 1 or 2")

    return - DV / beta

# ----------------------------------------------------------
# Definition of Evolution and Decision functions
def f_Qlearning(x, P, u, in_dict=None):
    """
    Reinforcement Learning (Q-learning) for 2-armed bandit.

    Parameters:
    - x: array-like of shape (2,), current Q-values
    - P: array-like, [invsigmoid(alpha)]
    - u: list or array-like [action, reward], where:
        - action: 0 or 1
        - reward: feedback received
    - in_dict: unused (included for compatibility)

    Returns:
    - fx: updated Q-values (array of shape (2,))
    """
    x = np.asarray(x)
    alpha = sigmoid(P[0])  # learning rate
    a = int(u[0]) #previous choice
    r = u[1]  # prev reward

    fx = np.copy(x)
    fx[a] = x[a] + alpha * (r - x[a])  # update chosen action value

    return fx


def g_Qlearning(x, P, u=None, in_dict=None):
    """
    Softmax decision rule for Q-learning (2-armed bandit).

    Parameters:
    - x : Q-values, array-like of shape (2,)
    - P : list or array, where
        P[0] = log inverse temperature (beta),
        P[1] = optional bias term (optional)
    - u : unused
    - in_dict : unused

    Returns:
    - gx : float, P(a=1|x)
    """
    x = np.asarray(x)
    beta = np.exp(P[0])  # inverse temperature

    dQ = x[0] - x[1]  # difference in Q-values

    if len(P) > 1:
        z = beta * dQ + P[1]
    else:
        z = beta * dQ

    gx = sigmoid(z)

    return gx

def f_fictitious_learner(x, P, u, in_dict):
    """
    Fictitious learning model (single game)

    Parameters:
    - x: array-like of shape (2,), current Q-values
    - P: array-like, [invsigmoid(alpha)]
    - u: list or array-like [action, reward], where:
        - action: 0 or 1
        - reward: feedback received
    - in_dict: unused (included for compatibility)

    Returns:
    - fx: updated Q-values (array of shape (2,))
    """
    if is_weird(u):
        return x

    x = np.asarray(x)
    alpha = sigmoid(P[0])  # learning rate
    a = int(u[0]) #previous choice
    r = u[1]  # prev reward

    fx = np.copy(x)
    fx[a] = x[a] + alpha * (r - x[a])  # update chosen action value
    fx[1-a] = x[1-a] + alpha * (1-r - x[1-a])  # update other action value

    return fx


def g_fictitious_learner(x, P, u, in_):
    """
    Fictitious learning model: observation function.
    Same as softmax decision rule for Q-learning (2-armed bandit).

    Parameters:
    - x : Q-values, array-like of shape (2,)
    - P : list or array, where
        P[0] = log inverse temperature (beta),
        P[1] = optional bias term (optional)
    - u : unused
    - in_dict : unused

    Returns:
    - gx : float, P(a=1|x)
    """
    x = np.asarray(x)
    beta = np.exp(P[0])  # inverse temperature

    dQ = x[0] - x[1]  # difference in Q-values

    if len(P) > 1:
        z = beta * dQ + P[1]
    else:
        z = beta * dQ

    gx = sigmoid(z)

    return gx

def f_influence_learner(x, P, u, in_dict):
    """
    Hampton's influence learning model (single game)

    Parameters:
    - x: [log-odds of P(o=1)]
    - P: [invsigmoid eta, invsigmoid lambda, log(beta)]
    - u: [o, a] where o = opponent's last move, a = agent's last move
    - in_dict: dictionary with keys ['game', 'player']

    Returns:
    - fx: updated hidden state (log-odds of updated P(o=1))
    """
    if is_weird(u):
        return x

    # Ensure x is treated as a 1D array if it's a scalar
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)

    o = u[0]  # opponent's last choice
    a = u[1]  # agent's last choice
    p0 = sigmoid(x[0])  # prior P(o=1)

    eta = sigmoid(P[0])  # learning rate for prediction error 1
    lamb = sigmoid(P[1])  # learning rate for prediction error 2
    beta = np.exp(P[2])  # opponent's temperature

    game = in_dict['game']
    player = in_dict['player']

    if player == 2:
        payoff = game[:, :, 0]
        payoff = np.asarray(payoff)
        k1 = payoff[0, 0] - payoff[1, 0] - payoff[0, 1] + payoff[1, 1]
        k2 = payoff[1, 0] - payoff[1, 1]
    elif player == 1:
        payoff = game[:, :, 1]
        payoff = np.asarray(payoff)
        k1 = payoff[0, 0] - payoff[1, 0] - payoff[0, 1] + payoff[1, 1]
        k2 = payoff[0, 1] - payoff[1, 1]
    else:
        raise ValueError("Player index must be 1 or 2.")

    # Second-order opponent's belief
    q0 = (beta * x[0] - k2) / k1

    PE1 = o - p0
    PE2 = a - q0

    # Influence learning update
    p = p0 + eta * PE1 + lamb * k1 * p0 * (1 - p0) * PE2
    p = np.clip(p, 0, 1)  # bound between 0 and 1

    fx = np.array([sigmoid(p, inverse=True)])  # convert back to log-odds

    return fx

def g_influence_learner(x, P, u, in_):
    """
    Hampton's influence learning model: observation function.

    Arguments:
    - x: hidden states (x[0] = log-odds of P(o=1))
    - P: parameters (P[0] = log temperature, P[1] = bias)
    - u: unused
    - in_: dictionary with keys:
        - 'game': 2x2x2 payoff table
        - 'player': agent's index (0 or 1)

    Returns:
    - gx: probability that the agent picks the first option (a=1)
    """
    # Ensure x is treated as a 1D array if it's a scalar
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)

    game = in_['game']
    player = in_['player']
    Po = sigmoid(x[0])  # P(o=1)
    temperature = np.exp(P[0])
    DV = fplayer(Po, temperature, player, game)
    gx = sigmoid(DV + P[1])  # P(a=1) with bias
    return gx

def f_MIIL(x, P, u, in_):
    """
    Evolution function for MIIL (Mixed-Intentions Influence Learning) model (Coordination Game vs Hide and Seek).

    Args:
    - x: hidden states (shape: [3])
    - P: parameters [eta_inv, lambda_inv, log_beta, log_inv_precision, bias]
    - u: [opponent's move, agent's move]
    - in_: dictionary with 'game1', 'game2', 'player'

    Returns:
    - fx: updated hidden states (array with shape [9])
    """
    if is_weird(u):
        return x  # Return unchanged state on undefined input

    # Ensure x is treated as a 1D array if it's a scalar
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)

    o = u[0]  # Opponent's last move
    a = u[1]  # Agent's last move

    # Prior beliefs
    p0_game1 = sigmoid(x[0])
    p0_game2 = sigmoid(x[1])
    p_game = sigmoid(x[2])

    # Parameters
    eta = sigmoid(P[0])
    lambda_ = sigmoid(P[1])
    beta = np.exp(P[2])

    # Extract game data
    game1 = in_['game1']
    game2 = in_['game2']
    player = in_['player']
    opp = 2 if player == 1 else 1

    if player == 2:
        payoff1 = game1[:, :, 0]
        payoff2 = game2[:, :, 0]
        k1_game1 = payoff1[0, 0] - payoff1[1, 0] - payoff1[0, 1] + payoff1[1, 1]
        k2_game1 = payoff1[1, 0] - payoff1[1, 1]
        k1_game2 = payoff2[0, 0] - payoff2[1, 0] - payoff2[0, 1] + payoff2[1, 1]
        k2_game2 = payoff2[1, 0] - payoff2[1, 1]
    elif player == 1:
        payoff1 = game1[:, :, 1]
        payoff2 = game2[:, :, 1]
        k1_game1 = payoff1[0, 0] - payoff1[1, 0] - payoff1[0, 1] + payoff1[1, 1]
        k2_game1 = payoff1[0, 1] - payoff1[1, 1]
        k1_game2 = payoff2[0, 0] - payoff2[1, 0] - payoff2[0, 1] + payoff2[1, 1]
        k2_game2 = payoff2[0, 1] - payoff2[1, 1]
    else:
        raise ValueError("Invalid player index. Must be 1 or 2.")

    # First-order prediction errors
    PE1_game1 = o - p0_game1
    PE1_game2 = o - p0_game2
    PE1 = o - (p0_game1 * p_game + (1 - p_game) * p0_game2)

    # Opponent's belief about agent's move
    q0_game1 = (beta * x[0] - k2_game1) / k1_game1
    q0_game2 = (beta * x[1] - k2_game2) / k1_game2

    PE2_game1 = a - q0_game1
    PE2_game2 = a - q0_game2
    PE2 = a - (q0_game1 * p_game + (1 - p_game) * q0_game2)
    PE2_game2_opp = o - p0_game2

    # Influence learning rule
    p_game1 = p0_game1 + eta * PE1_game1 + lambda_ * k1_game1 * p0_game1 * (1 - p0_game1) * PE2_game1
    p_game2 = p0_game2 + eta * PE1_game2 + lambda_ * k1_game2 * p0_game2 * (1 - p0_game2) * PE2_game2

    # Clip to valid probability range
    p_game1 = np.clip(p_game1, 0, 1)
    p_game2 = np.clip(p_game2, 0, 1)

    delta_p = lambda_ * k1_game2 * q0_game2 * (1 - q0_game2) * PE2_game2_opp

    fx = np.zeros((3, 1))

    # Inverse sigmoid to return in log-odds space
    fx[0] = sigmoid(p_game1, inverse=True)
    fx[1] = sigmoid(p_game2, inverse=True)

    # Softmax-style update for game mode
    slope = np.exp(P[3])
    center = P[4]
    p_game_mode = sigmoid(np.abs(fx[0]) - np.abs(fx[1]), slope=slope, center=center)
    fx[2] = sigmoid(p_game_mode, inverse=True)

    # Save prediction errors for fMRI correlates
    # fx[3] = PE1_game1
    # fx[4] = PE1_game2
    # fx[5] = PE2_game1
    # fx[6] = PE2_game2
    # fx[7] = PE1
    # fx[8] = delta_p

    return fx


def g_MIIL(x, P, u, in_dict):
    """
    MIIL's 2-games influence learning model: observation function

    Parameters:
    - x: [log-odds P(o=1|game1), log-odds P(o=1|game2), log-odds P(game1)]
    - P: [log temperature game1, log temperature game2, bias]
    - u: unused input
    - in_dict: dict with keys ['game1', 'game2', 'player']

    Returns:
    - gx: probability agent chooses first option (P(a=1))
    """
    # Ensure x is treated as a 1D array if it's a scalar
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)

    game1 = in_dict['game1']
    game2 = in_dict['game2']
    player = in_dict['player']

    # Convert log-odds to probabilities
    Po_game1 = sigmoid(x[0])
    Po_game2 = sigmoid(x[1])
    Po_game = sigmoid(x[2])

    # Compute decision variable for each game
    DV_game1 = fplayer(Po_game1, np.exp(P[0]), player, game1)
    DV_game2 = fplayer(Po_game2, np.exp(P[1]), player, game2)

    # Combined decision variable based on game mixture
    DV = Po_game * DV_game1 + (1 - Po_game) * DV_game2

    # Probability of action = 0
    gx = sigmoid(DV)

    return gx

def f_Mixed_AA(x, P, u, in_dict):
    """
    Mixed AA model: evolution function
    """
    if is_weird(u):
        return x

    own_action_n_2 = u[0] #own_action at t-2
    other_action_n_2 = u[1] #reward at t-2
    own_action_n_1 = u[2] #own_action at t-1
    other_action_n_1 = u[3] #reward at t-1
    other_action_n = u[4] #own_action at t

    hist_key = str(own_action_n_2) + str(other_action_n_2) + str(own_action_n_1) + str(other_action_n_1)
    if ('occ',hist_key) not in x:
        x[("occ",hist_key)] += 1

    if other_action_n == 0:
        x[("chose_0",hist_key)] += 1

    return x

def g_Mixed_AA(x, P, u, in_dict):
    """
    Mixed AA model: observation function
    """
    if is_weird(u):
        return x

    own_action_n_2 = u[0] #own_action at t-2
    other_action_n_2 = u[1] #reward at t-2
    own_action_n_1 = u[2] #own_action at t-1
    other_action_n_1 = u[3] #reward at t-1
    other_action_n = u[4] #own_action at t

    game = in_dict['game']
    player = in_dict['player']

    hist_key = str(own_action_n_2) + str(other_action_n_2) + str(own_action_n_1) + str(other_action_n_1)
    if game[0,0,1]==1:#if CG game
        gx = x[("chose_0",hist_key)]/x[("occ",hist_key)]
    else:#if HaS
        gx = 1 - x[("chose_0",hist_key)]/x[("occ",hist_key)]
    return gx

# ----------------------------------------------------------
# Definition of Agents classes
class Agent:
    def __init__(self, f_func, g_func, x_init, phi, theta, game, player_id, name="Agent"):
        # Evolution function
        self.f_func = f_func
        # Decision function
        self.g_func = g_func
        # Init hidden states
        self.x_init = np.array(x_init)        
        # Decision function parameters
        self.phi = np.array(phi)
        # Evolution function parameters
        self.theta = np.array(theta)
        # Game payoff matrix
        self.game = game
        # Player's role
        self.player = player_id  # 1 or 2
        # Player's name
        self.name = name
        # Hidden states
        self.x = np.array(x_init)        
        # History of the agents' actions and rewards
        self.history = []

    def choose_action(self,t):
        in_struct = {'game': self.game, 'player': self.player}
        result = self.g_func(self.x, self.phi, None, in_struct)

        # Extract the probability. If the function returns a tuple, take the first element.
        # if isinstance(result, tuple):
        #     gx = result[0]
        # else:
        #     gx = result
        gx = result

        return int(np.random.rand() > gx)

    def update(self, other_action, own_action,t):
        if self.player == 1:
            reward = self.game[own_action, other_action, self.player - 1]
        else:
            reward = self.game[other_action, own_action, self.player - 1]
        u = [other_action, own_action]
        in_struct = {'game': self.game, 'player': self.player}
        self.x = self.f_func(self.x, self.theta, u, in_struct)
        self.history.append((own_action, other_action, reward))

class MixedAgent:
    def __init__(self, f_func, g_func, x_init, phi, theta, game1, game2, game1_trials, game2_trials, player_id, name="Agent"):
        # Evolution function
        self.f_func = f_func
        # Decision function
        self.g_func = g_func
        # Hidden states
        self.x_init = x_init if isinstance(x_init, dict) else np.array(x_init)
        # Decision function parameters
        self.phi = np.array(phi)
        # Evolution function parameters
        self.theta = np.array(theta)
        # Game payoff matrices
        self.game1 = game1
        self.game2 = game2
        # Trials ranges in each game
        self.game1_trials = game1_trials
        self.game2_trials = game2_trials
        # Player's role
        self.player = player_id  # 1 or 2
        # Player's name
        self.name = name
        # Hidden states
        self.x = x_init if isinstance(x_init, dict) else np.array(x_init)
        # History of the agents' actions and rewards
        self.history = []

    def choose_action(self,t):
        # Determine which game to use based on the current trial
        if self.game1_trials[t]:
            game = self.game1
        else:
            game = self.game2

        # Prepare the game structure in-struct for the evolution function
        if self.f_func == f_MIIL:
            in_struct = {'game1': self.game1, 'game2': self.game2, 'player': self.player}
        else:
            in_struct = {'game': game, 'player': self.player}

        if t <= 2:
            gx = 0.5
        else:
            other_action = 0 # dummy other_action at t; useless anyways
            own_action_n_1 = self.history[t-1][0] #own_action at t-1
            other_action_n_1 = self.history[t-1][1] #other_action at t-1
            own_action_n_2 = self.history[t-2][0] #own_action at t-2
            other_action_n_2 = self.history[t-2][1] #other_action at t-2
            u = [own_action_n_2, other_action_n_2, own_action_n_1, other_action_n_1, other_action]

            result = self.g_func(self.x, self.phi, u, in_struct)

            # # Extract the probability. If the function returns a tuple, take the first element.
            # if isinstance(result, tuple):
            #     gx = result[0]
            # else:
            #     gx = result
            gx = result

        return int(np.random.rand() > gx)

    def update(self, other_action, own_action,t):
        # Determine which game to use based on the current trial
        if self.game1_trials[t]:
            game = self.game1
        else:
            game = self.game2

        # Calculate the reward based on the player's role and actions
        # Note: game[a, b, i] = payoff to player i for actions a, b
        if self.player == 1:
            reward = game[own_action, other_action, self.player - 1]
        else:
            reward = game[other_action, own_action, self.player - 1]

        # Prepare the input u for the evolution function
        if self.f_func == f_Qlearning or self.f_func == f_fictitious_learner:
            u = [own_action, reward]
        elif self.f_func == f_Mixed_AA:
            if t < 2:
                u = [np.nan,np.nan,np.nan,np.nan,np.nan]
            else:
                own_action_n_1 = self.history[t-1][0] #own_action at t-1
                other_action_n_1 = self.history[t-1][1] #other_action at t-1
                own_action_n_2 = self.history[t-2][0] #own_action at t-2
                other_action_n_2 = self.history[t-2][1] #other_action at t-2
                u = [own_action_n_2, other_action_n_2, own_action_n_1, other_action_n_1, other_action]
        else:
            u = [other_action, own_action]

        # Prepare the game structure in-struct for the evolution function
        if self.f_func == f_MIIL:
            in_struct = {'game1': self.game1, 'game2': self.game2, 'player': self.player}
        else:
            in_struct = {'game': game, 'player': self.player}

        # Update the hidden states using the evolution function
        self.x = self.f_func(self.x, self.theta, u, in_struct)
        self.history.append((own_action, other_action, reward))
