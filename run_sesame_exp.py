import numpy as np 
import pandas as pd
import torch 
import stable_baselines3 as sb3 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os 
import gym 
import trading_env 


if __name__ == "__main__":

    env_id = 'Trading-v5'

    model = PPO('MlpPolicy', make_vec_env(env_id, 8), 
            verbose = 1, device = torch.device('cpu'), 
            tensorboard_log = './sesame_runs/')

    
    model.learn(total_timesteps = 100, 
                tb_log_name = 'sesame_0') 

    model.save('./sesame_trained/sesame_0')