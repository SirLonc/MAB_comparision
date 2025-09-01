import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym
from scipy import stats

# -------------------------------
# Среда MAB для Gymnasium
# -------------------------------
class MABEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, true_means, episode_length=1):
        super().__init__()
        self.true_means = np.array(true_means)
        self.n_arms = len(true_means)
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.episode_length = episode_length
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        reward = np.random.normal(self.true_means[action], 1.0)
        terminated = self.current_step >= self.episode_length
        truncated = False
        info = {"arm_selected": action}
        return np.array([0.0], dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        pass

# -------------------------------
# Стратегии
# -------------------------------
def epsilon_greedy(env, episodes=1000, epsilon=0.1):
    Q = np.zeros(env.n_arms)
    N = np.zeros(env.n_arms)
    rewards = []

    for _ in range(episodes):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        _, reward, _, _, _ = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)
        env.reset()
    return np.array(rewards)

def ucb(env, episodes=1000, c=2):
    Q = np.zeros(env.n_arms)
    N = np.zeros(env.n_arms)
    rewards = []

    for t in range(1, episodes+1):
        ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-5))
        action = np.argmax(ucb_values)

        _, reward, _, _, _ = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)
        env.reset()
    return np.array(rewards)

def thompson_sampling(env, episodes=1000):
    mu_hat = np.zeros(env.n_arms)
    lambda_hat = np.ones(env.n_arms)
    rewards = []

    for _ in range(episodes):
        sampled = np.random.normal(mu_hat, 1/np.sqrt(lambda_hat))
        action = np.argmax(sampled)

        _, reward, _, _, _ = env.step(action)
        lambda_hat[action] += 1
        mu_hat[action] += (reward - mu_hat[action]) / lambda_hat[action]
        rewards.append(reward)
        env.reset()
    return np.array(rewards)

def cheater(env, episodes=1000):
    best_arm = np.argmax(env.true_means)
    rewards = []
    for _ in range(episodes):
        _, reward, _, _, _ = env.step(best_arm)
        rewards.append(reward)
        env.reset()
    return np.array(rewards)

def ppo_agent(env, total_timesteps=10000):
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    for _ in tqdm(range(total_timesteps // 1024), desc="Training PPO"):
        model.learn(total_timesteps=1024, reset_num_timesteps=False)

    rewards = []
    for _ in range(total_timesteps // env.episode_length):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
    return np.array(rewards)

# -------------------------------
# Доверительный интервал
# -------------------------------
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

# -------------------------------
# Симуляция
# -------------------------------
if __name__ == "__main__":
    true_means = [1.0, 1.5, 2.0, 1.2, 2.0, 12.0, 6.0, 4.0]
    env = MABEnv(true_means, episode_length=1)
    episodes = 50000
    strategies = {}

    for name, func in [
        ("Epsilon-Greedy", epsilon_greedy),
        ("UCB", ucb),
        ("Thompson Sampling", thompson_sampling),
        ("PPO", ppo_agent),
        ("Cheater", cheater)
    ]:
        rewards = func(env, episodes)
        strategies[name] = rewards

    # Кумулятивная награда
    plt.figure(figsize=(12,6))
    for name, rewards in strategies.items():
        plt.plot(np.cumsum(rewards), label=name)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("MAB Strategies Comparison")
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
