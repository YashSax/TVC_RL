from utils import *
import torch
from spinup import ppo_pytorch as ppo
from rocketgym_local.environment import Environment

def spinpg():
    env = Environment()
    ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='./base_ppo_results', exp_name='ppo')
    ppo(lambda: env, steps_per_epoch=1000, epochs=100, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs)

def test_model():
    model = torch.load("./base_ppo_results/pyt_save/model.pt").pi
    model.eval()
    evaluate_policy(model, tensor=True)

test_model()