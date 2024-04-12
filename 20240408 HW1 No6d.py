import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import RL_HW1_utils

from RL_HW1_utils import get_argmax, bandit

SEED = 50
np.random.seed(SEED)


def update_policy(H: np.ndarray) -> np.ndarray:
    return np.exp(H) / np.exp(H).sum()


def update_H(H: np.ndarray,
             policy: np.ndarray,
             alpha: float,
             A: int,
             curr_reward: float,
             avg_reward: float) -> np.ndarray:
    selec = np.zeros(len(H), dtype=np.float32)
    selec[A] = 1.0
    H = H + alpha * (curr_reward - avg_reward) * (selec - policy)
    return H


# running the k-armed bandit algorithm
def run_bandit(K: int,
               q_star: np.ndarray,
               rewards: np.ndarray,
               optim_acts_ratio: np.ndarray,
               alpha: float,
               baseline: bool,
               num_steps: int = 1000) -> None:
    H = np.zeros(K, dtype=np.float32)  # initialize preference
    policy = np.ones(K, dtype=np.float32) / K
    ttl_reward = 0
    ttl_optim_acts = 0

    for i in range(num_steps):

        A = np.random.choice(np.arange(K), p=policy)
        reward, is_optim = bandit(q_star, A)
        avg_reward = 0

        if baseline:
            # Get average reward unitl timestep=i
            avg_reward = ttl_reward / i if i > 0 else reward

        # Update preference and policy
        H = update_H(H, policy, alpha, A, reward, avg_reward)
        policy = update_policy(H)

        ttl_reward += reward
        ttl_optim_acts += is_optim
        rewards[i] = reward
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)


if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10  # Number of arms
    alphas = [0.1, 0.4]
    baselines = [False, True]
    hyper_params = list(itertools.product(baselines, alphas))

    num_steps = 1000
    total_rounds = 2000

    rewards = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))
    q_star = np.random.normal(loc=4.0, scale=1.0, size=K)

    print(hyper_params)
    for i, (is_baseline, alpha) in enumerate(hyper_params):
        for curr_round in range(total_rounds):
            run_bandit(K,
                       q_star,
                       rewards[i, curr_round],
                       optim_acts_ratio[i, curr_round],
                       alpha,
                       is_baseline,
                       num_steps)

    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    for val in optim_acts_ratio:
        plt.plot(val)
    plt.title("RL HW1 No 6d\nGradient Bandit Algorithms")
    plt.show()

    record = {
        'hyper_params': hyper_params,
        'optim_acts_ratio': optim_acts_ratio
    }

    # with open('./history/sga_record.pkl', 'wb') as f:
    #     pickle.dump(record, f)