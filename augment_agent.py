import numpy as np 
import gym 
import trading_env 
import torch 
import pandas as pd 
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from argparse import ArgumentParser 

class FiMetricsRecorder(BaseCallback): 
    def __init__(self, val_env = None, verbose = 0): 
        super().__init__(verbose)
        self.val_env = val_env 

    def _on_step(self): 
        e0 = self.model.get_env().envs[0]
        info = e0.info
        if len(info) > 0: 
            for k in e0.metrics_names: 
                # print(k.upper() + ' : ' + str(info[k]))
                self.logger.record('FiMetrics/{}'.format(k.upper()), info[k])

            done = False 
            s = self.val_env.reset()
            while not done: 
                action = self.model.predict(s, deterministic = True)[0]
                s, r, done, info_val = self.val_env.step(action)
            for k in info_val.keys(): 
                self.logger.record('Val/FiMetrics/{}'.format(k.upper()), info_val[k])

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--e', action = 'store_true')
    args = parser.parse_args()

    train = not args.e 

    if train: 
        env = make_vec_env('Trading-v4', n_envs = 8)
        
        for e in env.envs: 
            e.set_data('btc_train.csv')
            e.to_test()


        val_env = gym.make('Trading-v4')
        val_env.set_data('btc_test.csv')

        model = PPO('MlpPolicy', env = env, verbose = 2, 
            tensorboard_log = 'augment_runs/')
        cb = FiMetricsRecorder(val_env = val_env, verbose = 0)
        model.learn(2000000, callback = cb, tb_log_name = 'test_noaugment')
        model.save('augment_models/test_noaugment')
    else: 
        model = PPO.load('augment_models/test_noaugment')
        env = gym.make('Trading-v4')
        env.to_test()
        # env.set_data('btc_test.csv')
        while True: 
            done = False 
            reward = 0. 
            s = env.reset()
            while not done:
                action = model.predict(s, deterministic = True)[0]
                s, r, done, info = env.step(action)
                reward += r 
            # print(pd.DataFrame.from_dict(info, orient = 'index'))
            env.render() 
