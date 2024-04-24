from rocketgym_local.environment import Environment
from baseline import baseline_policy
from utils import *
from tqdm import tqdm

cum_rewards = []
num_episodes = 100
for _ in tqdm(range(num_episodes)):
    cum_reward = run_episode(baseline_policy, render=False)
    cum_rewards.append(cum_reward)

print("Average Cumulative Reward:", sum(cum_rewards) / num_episodes)
print("Best performance:", max(cum_rewards))
print("Worst performance:", min(cum_rewards))