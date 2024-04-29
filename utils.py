from rocketgym_local.environment import Environment
from rocketgym_local.constants import *
from utils import *
from tqdm import tqdm
import math
import torch
import random
from copy import deepcopy

ACTION_LEFT = 0
ACTION_MID = 1
ACTION_RIGHT = 2
ACTION_NONE = 3

action_dict = {
    ACTION_LEFT : "left",
    ACTION_MID : "mid",
    ACTION_RIGHT : "right",
    ACTION_NONE : "none"
}

def run_episode(policy, render=True, tensor=False, safeRL=False):
    env = Environment()
    danger_persistence = Persistence(20)

    observation = env.reset()
    done = False
    cum_reward = num_timesteps = 0
    while not done:
        if tensor:
            observation = torch.Tensor(observation)
        action = policy(observation)
        if tensor:
            action = action[0].sample().item()
        if safeRL:
            if is_dangerous(env):
                _, action = action_with_highest_acc(env)
        
        observation, reward, done, info = env.step(action)
        num_timesteps += 1
        cum_reward += reward
        if render:
            env.render()

    return cum_reward, crashed(observation, env)

LANDING_VEL_THRESH = 5
LANDING_ANG_VEL_THRESH = 2
LANDING_ANG_THRESH = 15 * math.pi / 180
MAX_TIME_IN_AIR = 4
def crashed(final_observation, env, verbose=False):
    _, vel_y, vel_x, ang_vel, angle = final_observation
    pos_x = env.rocket.position_x

    vel_violation = (vel_y**2 + vel_x**2)**0.5  > LANDING_VEL_THRESH
    ang_vel_violation = abs(ang_vel) > LANDING_ANG_VEL_THRESH
    angle_violation = abs(angle) > LANDING_ANG_THRESH
    fuel_violation = env.timestep > MAX_TIME_IN_AIR
    out_of_bounds_violation = pos_x < -10 or pos_x > 10

    violations = [vel_violation, ang_vel_violation, angle_violation, fuel_violation, out_of_bounds_violation]
    if verbose:
        names = ["vel", "ang_vel", "angle", "fuel", "out of bounds"]

        for v, n in zip(violations, names):
            if v:
                print("Violation occured:", n)

    return any(violations)
    

def evaluate_policy(policy, num_episodes=300, tensor=False, safeRL=False):
    cum_rewards = []
    total_crashes = 0
    for _ in tqdm(range(num_episodes)):
        cum_reward, crashed = run_episode(policy, render=False, tensor=tensor, safeRL=safeRL)
        cum_rewards.append(cum_reward)
        total_crashes += crashed

    print("Average Cumulative Reward:", sum(cum_rewards) / num_episodes)
    print("Best performance:", max(cum_rewards))
    print("Worst performance:", min(cum_rewards))
    print("Crash percentage:", total_crashes / num_episodes * 100)


def is_dangerous(curr_env):
    ''' Returns True if the current state is adjacent to an unsafe state. '''
    for action in [ACTION_LEFT, ACTION_MID, ACTION_RIGHT, ACTION_NONE]:
        env = deepcopy(curr_env)
        env.step(action)
        safe = is_safe(env)
        if not safe:
            return True
    return False

def action_with_highest_acc(curr_env):
    # Find the action that maximizes the y acceleration
    best_acc = -1e99
    best_action = ACTION_NONE
    for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_MID]:
        env = deepcopy(curr_env)
        env.step(action)
        if env.rocket.acceleration_y > best_acc:
            best_acc = env.rocket.acceleration_y
            best_action = action
    return best_acc, best_action

def is_safe(env):
    best_acc, _ = action_with_highest_acc(env)
    
    # env.rocket.position_y = env.rocket.velocity_y * lt + 1 / 2 * best_acc * lt ^ 2
    # lt <- time it will take to hit the ground if you continue at acceleration = best_acc

    min_height_time = -env.rocket.velocity_y / best_acc
    min_height = env.rocket.position_y + env.rocket.velocity_y * min_height_time + 1 / 2 * best_acc * min_height_time ** 2
    # print(min_height_time, best_acc, env.rocket.velocity_y)

    if min_height > 0:
        return True
    
    landing_velocity = (env.rocket.velocity_y ** 2 - 2 * best_acc * env.rocket.position_y) ** 0.5

    return landing_velocity <= LANDING_VEL_THRESH

class Persistence():
    def __init__(self, n, on_val=True, off_val=False):
        self.n = n
        self.time_since_last_on = 1e9
        self.on_val = on_val
        self.off_val = off_val

        self.val = off_val
    
    def update(self):
        self.time_since_last_on += 1
        if self.time_since_last_on > self.n:
            self.val = self.off_val

    def on(self):
        self.val = self.on_val
        self.time_since_last_on = 0
    
    def is_on(self):
        return self.val