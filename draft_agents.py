import os 
import glob 
import gym 
import trading_env 
import pandas as pd 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 
from argparse import ArgumentParser

if __name__ == '__main__': 

    parser = ArgumentParser()
    parser.add_argument('--enjoy', action = 'store_true')
    args = parser.parse_args()

    if not os.path.exists('draft_runs'): 
        os.mkdir('draft_runs')
        os.mkdir('draft_runs/draft_agents')
        os.mkdir('draft_runs/draft_logs')

    model = PPO('MlpPolicy', make_vec_env('Trading-v0', 8), 
            verbose = 1, device = torch.device('cpu'), 
            tensorboard_log = './draft_runs/draft_logs')
    if args.enjoy: 
        model.load('draft_runs/draft_agents/agent_pth')
    else: 
        model.learn(total_timesteps = 20e5, 
                    tb_log_name = 'draft_agent_0')
        model.save('draft_runs/draft_agents/agent_sb'.format('dqn' if args.dqn else 'ppo'))

    eval_ep = 10
    env = gym.make('Trading-v0')
    for ep in range(eval_ep): 
        s = env.reset()
        done = False 
        while not done: 
            action = model.predict(s)
            ns, r, done, info = env.step(action)
            env.render()
            s = ns 
