from rocketgym_local.environment import Environment
from utils import *
from tqdm import tqdm
import math
import torch

ACTION_LEFT = 0
ACTION_MID = 1
ACTION_RIGHT = 2
ACTION_NONE = 3

def run_episode(policy, render=True, tensor=False):
    env = Environment()
    env.curriculum.start_height = 5
    env.curriculum.enable_random_starting_rotation()

    observation = env.reset()
    done = False
    cum_reward = num_timesteps = 0
    while not done:
        if tensor:
            observation = torch.Tensor(observation)
        action = policy(observation)
        if tensor:
            action = action[0].sample().item()
        observation, reward, done, info = env.step(action)
        num_timesteps += 1
        cum_reward += reward
        if render:
            env.render()

    return cum_reward, crashed(observation, env)


LANDING_VEL_THRESH = 5
LANDING_ANG_VEL_THRESH = 5
LANDING_ANG_THRESH = 20 * math.pi / 180
MAX_TIME_IN_AIR = 4
def crashed(final_observation, env):
    _, vel_y, vel_x, ang_vel, angle = final_observation
    pos_x, _ = env.get_rocket_screen_position()

    vel_violation = (vel_y**2 + vel_x**2)**0.5  > LANDING_VEL_THRESH
    ang_vel_violation = ang_vel > LANDING_ANG_VEL_THRESH
    angle_violation = angle > LANDING_ANG_THRESH
    fuel_violation = env.timestep > MAX_TIME_IN_AIR
    out_of_bounds_violation = pos_x < 0 or pos_x > env.canvas_shape[1]

    if any([
        vel_violation,
        ang_vel_violation,
        angle_violation,
        fuel_violation,
        out_of_bounds_violation
    ]):
        return True
    return False
    

def evaluate_policy(policy, num_episodes=100, tensor=False):
    cum_rewards = []
    total_crashes = 0
    for _ in tqdm(range(num_episodes)):
        cum_reward, crashed = run_episode(policy, render=False, tensor=tensor)
        cum_rewards.append(cum_reward)
        total_crashes += crashed

    print("Average Cumulative Reward:", sum(cum_rewards) / num_episodes)
    print("Best performance:", max(cum_rewards))
    print("Worst performance:", min(cum_rewards))
    print("Crash percentage:", total_crashes / num_episodes * 100)