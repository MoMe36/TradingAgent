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


# ==== ==== ==== ==== ==== ==== ==== 
# Fonctionne avec DuelingDQN pour un départ au même index avec des sommes différentes
# Test avec différents index mais même somme: Ne fonctionne pas 
# Test avec reward: exp(net_worth - (high + low)*0.5) à partir du même endroit avec la même somme: (self.reward_scaler.transform([[self.net_worth]])[0,0] - self.reward_scaler.transform([[self.prev_net_worth]])[0,0]) * 10.
# Le test au-dessus fonctionne mais échoue pour plusieurs positions 
# Test avec récompense: delta baseline + shorter episodes 
# Test avec récompense: delta baseline + shorter episodes + random idx
# Test avec récompense: delta baseline + shorter episodes + random inital balance + fixed idx
#       SEMBLE FONCTIONNER MAIS NE FAIT RIEN DÈS QUE BALANCE EN DESSOUS DU COURS ?!
# Test avec fixed starting idx + randomized open + 150 ts + 3 lookback + reward = dNW
    # FONCTIONNE même avec des initial balance < à Open 
# ==== ==== ==== ==== ==== ==== ====  

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
        print('Loading')
        model = PPO.load('draft_runs/draft_agents/agent_sb')
    else: 
        model.learn(total_timesteps = 10e5, 
                    tb_log_name = 'draft_agent_0')
        model.save('draft_runs/draft_agents/agent_sb')

    eval_ep = 10
    env = gym.make('Trading-v0')
    for ep in range(eval_ep): 
        s = env.reset()
        done = False 
        while not done: 
            action = model.predict(s, 
                                  deterministic = True)[0]
            ns, r, done, info = env.step(action)
            env.render()
            s = ns 
