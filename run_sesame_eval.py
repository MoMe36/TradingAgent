import numpy as np 
import pandas as pd
import torch 
import stable_baselines3 as sb3 
from stable_baselines3 import PPO
import os 
import gym 
import trading_env 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')
matplotlib.use('TKAgg')

class Policy: 
    def __init__(self, **kwargs): 
        return 
    def act(self, state): 
        return np.random.uniform(0.,1.)

class PPO_Trading(Policy):
    def __init__(self, **kwargs): 
        self.load(kwargs['path_to_trained'])
    def load(self, path_to_trained): 
        self.model = PPO.load(path_to_trained)
    def act(self, state): 
        action = self.model.predict(state, deterministic = True)[0]
        return action[0]

class SellPolicy(Policy): 
    def act(self, state): 
        return -1. 

class EnzoPolicy(Policy): 
    def __init__(self): 
        scaler = MinMaxScaler().fit(np.array([0.,23.]).reshape(-1,1))
        self.store_hours = list(scaler.transform(np.array([21, 22, 23,
                                                              0, 1, 2, 3,
                                                              4, 5, 6, 7, 
                                                              12, 13, 14, 
                                                              15, 16]).reshape(-1,1)).flatten())
    def act(self, state): 
        hour = state[53]
        if hour in list(self.store_hours): 
            # input('store')
            return 1.
        else: 
            # input('sell')
            return -1.
class Run: 
    def __init__(self, pol): 
        self.pol = pol 
        self.rewards = []
        self.actions = []
        self.stock = []
        self.prod = []

    def run(self, env, idx): 

        s = env.reset(idx)
        done = False 
        rewards = []
        actions = []
        stock = []
        prod = []
        while not done: 
            action = self.pol.act(s)
            s, r, done, info = env.step(action)
            rewards.append(r*1000.)
            actions.append(action)
            stock.append(info['stock'])
            prod.append(info['prod'])
        self.rewards.append(rewards)
        self.actions.append(actions)
        self.stock.append(stock)
        self.prod.append(prod)


if __name__ == "__main__":

    env_id = 'Trading-v5'

    env = gym.make(env_id)
    env.ep_length = 150
    enzo_pol = Run(EnzoPolicy())
    model = Run(PPO_Trading(path_to_trained = './sesame_trained/sesame_0'))
    sell_policy  = Run(SellPolicy())
    random_policy = Run(Policy())
    
    nb_eps = 10
    rewards = []
    actions = []
    start_idx = np.random.randint(100, 6000, size = (nb_eps, ))

    for idx in start_idx: 
        for xp in [enzo_pol, model, sell_policy, random_policy]: 
            xp.run(env, idx)


    for i in range(nb_eps): 
        f, axes = plt.subplots(4,1, figsize = (15,20))
        axes = axes.flatten()
        axes[0].plot(np.cumsum(enzo_pol.rewards[i]), label = 'Enzo')
        axes[0].plot(np.cumsum(model.rewards[i]), label = 'PPO')
        axes[0].plot(np.cumsum(sell_policy.rewards[i]), label = 'Sell')
        axes[0].plot(np.cumsum(random_policy.rewards[i]), label = 'Random')
        axes[0].set_title('Cumulative reward: {}'.format(start_idx[i]), weight = 'bold')
        axes[0].legend()

        axes[1].plot(enzo_pol.stock[i], label = 'Enzo')
        axes[1].plot(model.stock[i], label = 'PPO')
        axes[1].plot(sell_policy.stock[i], label = 'Sell')
        axes[1].plot(random_policy.stock[i], label = 'Random')
        axes[1].set_title('Stock: {}'.format(start_idx[i]), weight = 'bold')
        axes[1].legend()

        axes[2].plot(np.cumsum(enzo_pol.prod[i]), label = 'Enzo')
        axes[2].plot(np.cumsum(model.prod[i]), label = 'PPO')
        axes[2].plot(np.cumsum(sell_policy.prod[i]), label = 'Sell')
        axes[2].plot(np.cumsum(random_policy.prod[i]), label = 'Random')
        axes[2].set_title('Prod: {}'.format(start_idx[i]), weight = 'bold')
        axes[2].legend()

        axes[3].hist(model.actions[i], label = 'Ep: {}'.format(i))
        axes[3].legend()

        plt.show()
        plt.close()


