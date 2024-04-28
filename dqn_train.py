import rocketgym_local.constants
from rocketgym_local.constants import *
from rocketgym_local.dashboard import Dashboard
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import json
import os.path

from rocketgym_local.environment import Environment
from dqn_network import Agent


def train(curriculum, softmax, save_progress, model=None, directory="models", epochs=50, steps_per_epoch=4000, max_ep_len=1000):
    dash = Dashboard()

    # Setting up the environment
    env = Environment()
    # env.curriculum.start_height = 5
    # env.curriculum.enable_random_starting_rotation()

    if softmax:
        exploration = Exploration.SOFTMAX
        exploration_dec = TEMP_DECREASE
        exploration_min = TEMP_MIN
        exploration_start = 1
    else:
        exploration = Exploration.EPSILON_GREEDY
        exploration_dec = EPS_DECREASE
        exploration_min = EPS_MIN
        exploration_start = 0.5

    algorithm = "deepQ"

    if model is None:
        agent = Agent(gamma=0.99, epsilon=exploration_start, lr=0.001,
                      input_dims=[5], batch_size=64, n_actions=4, exploration_dec=exploration_dec, exploration_min=exploration_min, exploration=exploration)
    else:
        agent = Agent(gamma=0.99, epsilon=0, lr=0.001,
                      input_dims=[5], batch_size=64, n_actions=4, exploration_dec=exploration_dec, exploration_min=exploration_min, exploration=exploration)
        agent.q_eval.load_state_dict(torch.load(model))

        # env.curriculum.set_random_height(1, 10)
        # env.curriculum.enable_increasing_height()

    scores = []
    velocities = []
    angles = []

    episode = 0
    for epoch in range(epochs):
        steps_this_epoch = 0
        while steps_this_epoch < steps_per_epoch:
            steps_this_episode = 0
            episode += 1

            score = 0
            done = False

            if curriculum and i == 200:
                env.curriculum.set_random_height(1, 1)
                env.curriculum.enable_increasing_height()

            observation = env.reset()

            while not done:
                action = agent.choose_action(observation)
                new_observation, reward, done, info = env.step(action)
                score += reward
                steps_this_episode += 1

                agent.store_transition(observation, action,
                                    reward, new_observation, done)
                agent.learn()

                observation = new_observation

                if model is not None or steps_this_episode >= max_ep_len:
                    # env.render()
                    break
            
            steps_this_epoch += steps_this_episode

            if save_progress and episode % 100 == 0:
                # dash.plot_log(env.rocket.flight_log, episode=episode)
                torch.save(agent.q_eval.state_dict(),
                        os.path.join(directory, f"model_{episode}"))

            scores.append(score)

            avg_score = np.mean(scores[-100:])
            velocity = env.rocket.flight_log.velocity_y[-1]

            if velocity < 0:
                velocities.append(velocity)
                angles.append(math.degrees(
                    env.rocket.get_unsigned_angle_with_y_axis()))

            avg_vel = np.mean(velocities[-100:])
            avg_ang = np.mean(angles[-100:])
            if episode % 100 == 0:
                print(
                    f"Episode: {episode}\n\tEpsilon: {agent.epsilon}\n\tScore: {score:.2f}\n\tAverage score: {avg_score:.4f}\n\tAverage velocity: {avg_vel:.2f}\n\tAverage angle: {avg_ang:.2f}")
        
        with open(os.path.join(directory, "reward.json"), "w") as f:
            json.dump(scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Rocket Landing - Reinforcemeng Learning')

    parser.add_argument('--curriculum', action='store_true',
                        help="Use Curriculum Learning")
    parser.add_argument('--softmax', action='store_true',
                        help="Use Softmax exploration instead of eps-greedy")
    parser.add_argument('--save', action='store_true',
                        help="Save flight logs and models every 100 episodes")
    parser.add_argument('-model',
                        help="Path to the model to load. Overrides the curriculum and exploration settings. Renders the scene from the start.")
    parser.add_argument('-dir',
                        help="Path to save the results.")
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-steps_per_epoch", type=int)
    parser.add_argument("-max_ep_len", type=int)

    args = parser.parse_args()

    train(args.curriculum, args.softmax, args.save, args.model, args.dir, args.epochs, args.steps_per_epoch, args.max_ep_len)