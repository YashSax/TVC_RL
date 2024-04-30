from utils import *
import torch
from spinup import ppo_pytorch as ppo
from rocketgym_local.environment import Environment
import os

dir="./reward_15_ppo"

def spinpg():
    env = Environment()
    ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir=dir, exp_name='ppo')
    ppo(lambda: env, steps_per_epoch=4000, epochs=50, gamma=0.9999, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, safe_rl=None) 

def test_model():
    model = torch.load(os.path.join(dir, "pyt_save/model.pt")).pi
    model.eval()
    # run_episode(model, tensor=True)
    evaluate_policy(model, tensor=True)

test_model()
# spinpg()