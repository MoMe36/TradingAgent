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


def save_runs(data, filename, col):
    data_df = pd.DataFrame(data, columns = [col])
    data_df.to_csv(filename + '.csv', index = False) 
    # rewards = pd.DataFrame(rewards, columns = [agent_name + '_' + args.name]) 
    # rewards.to_csv('reward_{}_{}.csv'.format(agent_name, args.name), index = False)

def make_plot(filenames, name): 
    df = pd.concat([pd.read_csv(f) for f in sorted(filenames)], axis = 1)
    plt.figure(figsize = (12,8))
    sns.histplot(df, kde = False,
                 multiple = "stack", 
                 shrink = 0.8)
    plt.savefig('{}.png'.format(name))

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--r', action = 'store_true')
    parser.add_argument('--rand', action = 'store_true')
    parser.add_argument('--d', action = 'store_true')
    parser.add_argument('--name', default = "agent_ppo")
    parser.add_argument('--dqn', action = 'store_true')
    parser.add_argument('--ver', default = '2') # THIS IS ENVIRONMENT VERSION 
    parser.add_argument('--short', action = 'store_true')

    args = parser.parse_args()

    env_ver = int(args.ver)

    if not os.path.exists("ppo_trading_sb.zip") or args.train:

        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])

        if(args.dqn): 
            args.name = 'DQN_' + args.name
            model = DQN('MlpPolicy', gym.make('Trading-v{}'.format(env_ver)), 
            verbose = 1, device = torch.device('cpu'), 
            tensorboard_log = './runs/')
        else: 
            model = PPO('MlpPolicy', make_vec_env('Trading-v{}'.format(env_ver), 8), 
                verbose = 1, device = torch.device('cpu'), 
                tensorboard_log = './runs/')
        run_name = '{}_{}_{}_{}'.format(model.env.envs[0].get_env_name(), 'dqn' if args.dqn else 'ppo', 
                                        model.env.envs[0].reward_name(), args.name)
        model.learn(total_timesteps = 50e5 if args.short else 10e6, 
                    tb_log_name = run_name, 
                    callback = CheckpointCallback(save_freq = 200000, save_path = "./trained_models", 
                                                  name_prefix = run_name))
        model.save('{}_trading_sb'.format(run_name))
    else: 
        
        if not args.rand:
            print('Loading agent')
            if(args.dqn):
                model = DQN.load('dqn_trading_sb') 
            else: 
                agents = sorted(glob.glob('*.zip'))
                for i,a in enumerate(agents): 
                    print('{} - {}'.format(i,a.replace('.zip', '')))
                selected_agent = input('Select agent: \t')
                model = PPO.load(agents[int(selected_agent)].replace('.zip', ''))
                if(args.name == "agent_ppo"): 
                    args.name = agents[int(selected_agent)].replace('.zip', '')
        else: 
            print('Random agent')
    # model = PPO('MlpPolicy', env, verbose = 1)


    eval_eps = 100
    pbar = tqdm(total = eval_eps)
    env = gym.make('Trading-v{}'.format(env_ver))
    rewards = []
    baseline_diff = []
    net_worth_delta = []
    win_ratio = []
    avg_win = []
    avg_loss = []
    profit_factor = []
    for ep in range(eval_eps): 
        done = False 
        ep_reward = 0
        s = env.reset()
        net_worth_delta.append(env.net_worth)
        ep_step_counter = 0
        ep_wins = 0 
        ep_wins_val = []
        ep_loss_val = []
        ep_money_made = 0.
        ep_money_lost = 0.
        while not done: 
            if args.rand: 
                action = env.get_random_action()
            else: 
                action = model.predict(s, deterministic = args.d)[0]

            net_before_action = env.net_worth

            ns, r, done, info = env.step(action)

            net_after_action = env.net_worth
            if(net_after_action > net_before_action): 
                ep_wins += 1
                ep_wins_val.append(net_after_action - net_before_action)
                ep_money_made += ep_wins_val[-1]
            else: 
                ep_loss_val.append(np.abs(net_after_action - net_before_action))
                ep_money_lost += ep_loss_val[-1]
            ep_step_counter += 1 

            s = ns 
            if args.r: 
                env.render()
            ep_reward += r

        profit_factor.append(ep_money_made / ep_money_lost)
        avg_win.append(np.array(ep_wins_val).mean())
        avg_loss.append(np.array(ep_loss_val).mean())
        win_ratio.append(float(ep_wins)/float(ep_step_counter))
        net_worth_delta[-1] = (env.net_worth - net_worth_delta[-1])
        baseline_diff.append(env.get_baseline_diff())
        rewards.append(ep_reward)
        pbar.update(1)
    pbar.close()


    agent_name = 'random' if args.rand else 'agent_{}'.format('deterministic' if args.d else 'stochastic')
    run_name = '{}_{}'.format(agent_name,args.name)
    
    pbar = tqdm(total = 7)

    save_runs(rewards, 'run_recap_reward_{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*reward*.csv'), 'hist_reward_dist')
    pbar.update(1)
    save_runs(baseline_diff, 'run_recap_baseline_delta_{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*baseline*.csv'), 'hist_baseline_delta')
    pbar.update(1)
    save_runs(net_worth_delta, 'run_recap_net_worth_delta_{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*net_worth_delta*.csv'), 'hist_net_worth_delta')
    pbar.update(1)
    save_runs(win_ratio, 'run_recap_win_ratio{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*win_ratio*.csv'), 'hist_win_ratio')
    pbar.update(1)
    save_runs(avg_win, 'run_recap_avg_win_mean{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*avg_win*.csv'), 'hist_avg_win')
    pbar.update(1)
    save_runs(avg_loss, 'run_recap_avg_loss_mean{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*avg_loss*.csv'), 'hist_avg_loss')
    pbar.update(1)
    save_runs(profit_factor, 'run_recap_profit_factor{}.csv'.format(run_name), run_name)
    make_plot(glob.glob('*profit_factor*.csv'), 'hist_profit_factor')
    pbar.update(1)
    pbar.close()