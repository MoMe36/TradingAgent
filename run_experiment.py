import numpy as np 
import pandas as pd
import torch 
import stable_baselines3 as sb3 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os 
import gym 
import trading_env 
from argparse import ArgumentParser 
import glob 
from experiment_utils import activation_fns, envs


# ================================================================================================================================
# https://ai.plainenglish.io/combining-technical-indicators-with-deep-learning-for-stock-trading-aebf155fe22f
# https://towardsdatascience.com/implementation-of-technical-indicators-into-a-machine-learning-framework-for-quantitative-trading-44a05be8e06
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996
# ================================================================================================================================

def parse_params(params): 
    return {l[0]:l[1] for l in params} 

def run_exp(exp_params):
    activation_fn = activation_fns[exp_params['act']]
    layers = [int(nb) for nb in exp_params['layers'].split(',')]
    nb_threads = int(exp_params['nb_threads'])
    freq_save = int(exp_params['save_every'])
    env_id = 'Trading-{}'.format(envs[exp_params['env']])
    train_steps = int(exp_params['train_steps']) * 10e4

    tmp_env = gym.make(env_id)
    tmp_env.reset()
    env_data = tmp_env.get_env_specs()

    trained_agents = glob.glob('trained_models/{}/*'.format(env_data['folder_name'] if 'folder_name' in env_data.keys() else env_data['env_name']))
    run_idx = len([agent for agent in trained_agents if 'agent' in agent]) 

    run_name = 'agent_{:03d}'.format(run_idx)
    
    model = PPO('MlpPolicy', make_vec_env(env_id, nb_threads), 
                verbose = 1, device = torch.device('cpu'), 
                tensorboard_log = './runs/{}/'.format(env_data['folder_name'] if 'folder_name' in env_data.keys() else env_data['env_name']))

    model.learn(total_timesteps = train_steps, 
                tb_log_name = run_name) 

    env_data = model.env.envs[0].get_env_specs()
    env_data['run_name'] = run_name

    env_folder = 'trained_models/{}'.format(env_data['folder_name'] if 'folder_name' in env_data.keys() else env_data['env_name'])
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    if not os.path.exists('{}/{}'.format(env_folder, run_name)): 
        os.mkdir('{}/{}'.format(env_folder, run_name))
        model.save('{}/{}/{}'.format(env_folder, run_name, run_name))

    recap = pd.Series(env_data.values(), index = env_data.keys())
    recap.to_csv('{}/{}/recap.csv'.format(env_folder, run_name), index = True)


if __name__ == "__main__": 

    parser = ArgumentParser()

    parser.add_argument('--env', default = 'BC_N')
    parser.add_argument('--act', default = 'relu')
    parser.add_argument('--layers', default = '256,256')
    parser.add_argument('--save_every', default = '200000')
    parser.add_argument('--nb_threads', default = '8')
    parser.add_argument('--train_steps', default = '50')

    args = parser.parse_args()

    run_exp(parse_params(args._get_kwargs()))

