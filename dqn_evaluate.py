from utils import *
from dqn_network import Agent
import torch

MODEL_DIR = "./models/model_1900"
agent = Agent(gamma=0.99, epsilon=0, lr=0.001,
                      input_dims=[5], batch_size=64, n_actions=4)
agent.q_eval.load_state_dict(torch.load(MODEL_DIR))

run_episode(policy=lambda observation: agent.choose_action(observation), render=True)
