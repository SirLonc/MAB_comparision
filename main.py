import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# -------------------------------
# Настройка среды MAB
# -------------------------------
class Bandit:
    def __init__(self, true_means):
        self.true_means = np.array(true_means)
        self.n_arms = len(true_means)

    def pull(self, arm):
        return np.random.normal(self.true_means[arm], 1.0)

# -------------------------------
# Стратегии
# -------------------------------
def epsilon_greedy(bandit, episodes=1000, epsilon=0.1):
    n_arms = bandit.n_arms
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    rewards = []

    for t in range(episodes):
        if np.random.rand() < epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(Q)

        r = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (r - Q[arm]) / N[arm]
        rewards.append(r)
    return np.array(rewards)

def ucb(bandit, episodes=1000, c=2):
    n_arms = bandit.n_arms
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    rewards = []

    for t in range(1, episodes + 1):
        ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-5))
        arm = np.argmax(ucb_values)

        r = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (r - Q[arm]) / N[arm]
        rewards.append(r)
    return np.array(rewards)

def softmax(bandit, episodes=1000, tau=0.1):
    n_arms = bandit.n_arms
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    rewards = []

    for t in range(episodes):
        exp_Q = np.exp(Q / tau)
        probs = exp_Q / np.sum(exp_Q)
        arm = np.random.choice(n_arms, p=probs)

        r = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (r - Q[arm]) / N[arm]
        rewards.append(r)
    return np.array(rewards)

def thompson_sampling(bandit, episodes=1000):
    n_arms = bandit.n_arms
    mu_hat = np.zeros(n_arms)
    lambda_hat = np.ones(n_arms)
    rewards = []

    for t in range(episodes):
        sampled = np.random.normal(mu_hat, 1/np.sqrt(lambda_hat))
        arm = np.argmax(sampled)

        r = bandit.pull(arm)
        lambda_hat[arm] += 1
        mu_hat[arm] += (r - mu_hat[arm]) / lambda_hat[arm]
        rewards.append(r)
    return np.array(rewards)

def cheater(bandit, episodes=1000):
    best_arm = np.argmax(bandit.true_means)
    rewards = [bandit.pull(best_arm) for _ in range(episodes)]
    return np.array(rewards)

# -------------------------------
# Функция для доверительного интервала
# -------------------------------
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    mean = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

# -------------------------------
# Симуляция и графики
# -------------------------------
if __name__ == "__main__":
    true_means = np.random.normal(scale=5.0, size=15)
    bandit = Bandit(true_means)
    episodes = 100000  
    strategies = {}

    for name, func in tqdm([
        ("Epsilon-Greedy", epsilon_greedy),
        ("UCB", ucb),
        ("Softmax", softmax),
        ("Thompson Sampling", thompson_sampling),
        ("Cheater", cheater)
    ], desc="Simulating strategies"):
        rewards = func(bandit, episodes)
        strategies[name] = rewards

    # График кумулятивной награды
    plt.figure(figsize=(12, 6))
    for name, rewards in strategies.items():
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, label=name)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("MAB Strategies: Cumulative Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Статистические оценки
    print("\nСредняя награда ± 95% доверительный интервал:")
    for name, rewards in strategies.items():
        mean, lower, upper = mean_confidence_interval(rewards)
        print(f"{name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    # p-value: сравнение с Cheater
    cheater_rewards = strategies["Cheater"]
    print("\nP-values по t-тесту против Cheater:")
    for name, rewards in strategies.items():
        if name == "Cheater":
            continue
        t_stat, p_val = stats.ttest_ind(rewards, cheater_rewards)
        print(f"{name} vs Cheater: p-value = {p_val:.4e}")
