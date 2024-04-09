from rocketgym.environment import Environment

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