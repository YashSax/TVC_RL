from rocketgym.environment import Environment
from baseline import baseline_policy
from utils import run_episode
from tqdm import tqdm

total_cum_reward = 0
num_episodes = 100
for _ in tqdm(range(num_episodes)):
    cum_reward = run_episode(baseline_policy, render=False)
    total_cum_reward += cum_reward
print("Average Cumulative Reward:", total_cum_reward / num_episodes)
