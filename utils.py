from rocketgym_local.environment import Environment
from utils import *
from tqdm import tqdm

ACTION_LEFT = 0
ACTION_MID = 1
ACTION_RIGHT = 2
ACTION_NONE = 3

def run_episode(policy, render=True):
    env = Environment()
    env.curriculum.start_height = 5
    env.curriculum.enable_random_starting_rotation()

    observation = env.reset()
    done = False
    cum_reward = num_timesteps = 0
    while not done:
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        num_timesteps += 1
        cum_reward += reward
        if render:
            env.render()

    return cum_reward

def evaluate_policy(policy, num_episodes=100):
    cum_rewards = []
    for _ in tqdm(range(num_episodes)):
        cum_reward = run_episode(policy, render=False)
        cum_rewards.append(cum_reward)

    print("Average Cumulative Reward:", sum(cum_rewards) / num_episodes)
    print("Best performance:", max(cum_rewards))
    print("Worst performance:", min(cum_rewards))
