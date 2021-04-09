import gym 
import trading_env
import numpy as np 
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os 
import pandas as pd 
from argparse import ArgumentParser
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm
import torch  
import glob 
plt.style.use('ggplot')


if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--r', action = 'store_true')
    parser.add_argument('--rand', action = 'store_true')
    parser.add_argument('--d', action = 'store_true')
    parser.add_argument('--name', default = "agent_ppo")
    parser.add_argument('--dqn', action = 'store_true')


    args = parser.parse_args()

    if not os.path.exists("ppo_trading_sb.zip") or args.train:

        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[1024, 1024], vf=[1024, 1024])])

        if(args.dqn): 
            args.name = 'DQN_' + args.name
            model = DQN('MlpPolicy', gym.make('Trading-v2'), 
            verbose = 1, device = torch.device('cpu'), 
            tensorboard_log = './runs/')
        else: 
            model = PPO('MlpPolicy', make_vec_env('Trading-v2', 8), 
                verbose = 1, device = torch.device('cpu'), 
                tensorboard_log = './runs/')
        
        model.learn(total_timesteps = 20e6, 
                    tb_log_name = args.name, 
                    callback = CheckpointCallback(save_freq = 10000, save_path = "./trained_models", 
                                                  name_prefix = args.name))
        model.save('{}_trading_sb'.format('dqn' if args.dqn else 'ppo'))
    else: 
        print('Loading agent')
        if(args.dqn):
            model = DQN.load('dqn_trading_sb') 
        else: 
            model = PPO.load('ppo_trading_sb')
    # model = PPO('MlpPolicy', env, verbose = 1)


    eval_eps = 100
    pbar = tqdm(total = eval_eps)
    env = gym.make('Trading-v0')
    rewards = []
    baseline_diff = []
    for ep in range(eval_eps): 
        done = False 
        ep_reward = 0
        s = env.reset()
        while not done: 
            if args.rand: 
                action = env.get_random_action()
            else: 
                action = model.predict(s, deterministic = args.d)[0]
            ns, r, done, info = env.step(action)
            s = ns 
            if args.r: 
                env.render()
            ep_reward += r
        baseline_diff.append(env.get_baseline_diff())
        rewards.append(ep_reward)
        pbar.update(1)
    pbar.close()


    agent_name = 'random' if args.rand else 'agent_{}'.format('deterministic' if args.d else 'stochastic')
    rewards = pd.DataFrame(rewards, columns = [agent_name + '_' + args.name]) 
    rewards.to_csv('reward_{}_{}.csv'.format(agent_name, args.name), index = False)

    baseline_diff = pd.DataFrame(baseline_diff, columns = [agent_name + '_' + args.name]) 
    baseline_diff.to_csv('baseline_diff_{}_{}.csv'.format(agent_name, args.name), index = False)

    reward_files = glob.glob('reward*.csv')
    df = pd.concat([pd.read_csv(f) for f in reward_files], axis = 1)
    plt.figure(figsize = (12,8))
    sns.histplot(df, kde = True)
    plt.savefig('trading_reward_dist.png')

    baseline_files = glob.glob('baseline*.csv')
    df = pd.concat([pd.read_csv(f) for f in baseline_files], axis = 1)

    plt.figure(figsize = (12,8))
    sns.histplot(df, kde = True)
    plt.savefig('trading_baseline_dist.png')
    # for col in df.columns: 
        # sns.histplot(df[col])
    # sns.histplot(rewards, kde = True)
    # plt.savefig('./trading_{}.png'.format('random' if args.rand else 'agent'))

