from rocketgym.environment import Environment
from baseline import baseline_policy

env = Environment()
env.curriculum.start_height = 5
env.curriculum.enable_random_starting_rotation()

observation = env.reset()
done = False
cum_reward = num_timesteps = 0
while not done:
    action = baseline_policy(observation)
    observation, reward, done, info = env.step(action)
    num_timesteps += 1
    cum_reward += reward
    env.render()

print("Cumulative reward:", cum_reward)