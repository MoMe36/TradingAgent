import numpy as np 
import gym 
import trading_env 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from stable_baselines3 import PPO 
import os 
import glob 
from argparse import ArgumentParser 
from experiment_utils import envs
from tqdm import tqdm 
plt.style.use('ggplot')

fix_name = lambda fn,fn_info : 'agent_' + fn.split(fn_info)[0].split('/')[-1][:-1]

def parse_recap(recap_path): 

    data = pd.read_csv(os.path.join(recap_path, 'recap.csv'))
    data.columns = ['specs', 'value']
    data = data.set_index('specs', drop = True)
    return data.value.to_dict()

def save_stat(data, filename, pbar = None):
    data_df = pd.Series(data)
    data_df.to_csv(filename, index = False) 
    
    if pbar is not None: 
        pbar.update(1)

def make_plot(filenames, info, folder_path): 
    
    dfs = []
    for file in filenames: 
        dfs.append(pd.read_csv(file))
        dfs[-1].columns = [fix_name(file, info)]

    dfs = pd.concat(dfs, axis = 1)
    v_dfs = pd.DataFrame(None, columns = [info, 'name'])
    for col in dfs.columns: 
        n = np.empty((dfs.shape[0],1), dtype = 'object')
        n[:,] = col
        current_data = np.hstack([dfs[col].values.reshape(-1,1), n])
        current_df = pd.DataFrame(current_data, columns = [info, 'name'])
        v_dfs = pd.concat([v_dfs,current_df])
    
    plt.figure(figsize = (24,16))
    ax = sns.histplot(v_dfs, x = info, multiple = 'dodge', hue = 'name')

    plt.title(info.upper(), weight = 'bold', fontsize = 30)
    plt.ylabel('Count', weight = 'bold', fontsize = 20)
    plt.xlabel(info, weight = 'bold', fontsize = 20)
    
    # INCREASES LEGEND SIZE
    plt.setp(ax.get_legend().get_texts(), fontsize=22)

    # INCREASES LABELS SIZE
    for t_x, t_y in zip(ax.get_xticklabels(), ax.get_yticklabels()): 
        t_x.set_fontsize(22) 
        t_y.set_fontsize(22) 

    plt.savefig(os.path.join(folder_path, info.upper() + '.png'))



if __name__ == '__main__': 

    parser = ArgumentParser()
    parser.add_argument('--render', action = 'store_true')
    parser.add_argument('--random_agent', action = 'store_true')
    parser.add_argument('--deterministic', action= 'store_true')
    parser.add_argument('--eval_ep', default= '200')
    parser.add_argument('--agent_id', default = '0')
    parser.add_argument('--env_id', default = 'BC_N')
    parser.add_argument('--do_plot', action = 'store_true')
    parser.add_argument('--ep_ts', default = '0')
    parser.add_argument('--set_stock', default = '')
    args = parser.parse_args()





    deterministic = args.deterministic

    if not args.random_agent: 
        directory = 'trained_models/{}/agent_{:03d}'.format(args.env_id, int(args.agent_id))
        run_data = parse_recap(directory) 
        model = PPO.load(os.path.join(directory, run_data['run_name']))
        env_nb = envs[run_data['env_name']]
    else: 
        env_nb = envs[args.env_id]
    env = gym.make('Trading-{}'.format(env_nb))
    if int(args.ep_ts) != 0: 
        env.ep_timesteps = int(args.ep_ts)

    if args.set_stock != '': 
        env.set_data(args.set_stock)

    rewards = []
    baseline_diff = []
    net_worth_delta = []
    win_ratio = []
    avg_win = []
    avg_loss = []
    profit_factor = []

    pbar = tqdm(total = int(args.eval_ep))

    for ep in range(int(args.eval_ep)): 
        s = env.reset()
        done = False 
        ep_reward = 0
        net_worth_delta.append(env.trader.net_worth)
        ep_step_counter = 0
        ep_wins = 0 
        ep_wins_val = []
        ep_loss_val = []
        ep_money_made = 0.
        ep_money_lost = 0.
        while not done: 
            if args.random_agent: 
                action = env.get_random_action()
            else: 
                action = model.predict(s, deterministic = args.deterministic)[0]

            net_before_action = env.trader.net_worth
            # print('{}\nAction:{}'.format(pd.DataFrame(s.reshape(10,10), columns = 'Open,High,Low,Close,Volume,Balance,NetWorth,Held,Sold,Bought'.split(',')), action))
            # print(env.get_formatted_obs())
            # input()
            ns, r, done, info = env.step(action)
            net_after_action = env.trader.net_worth
            if(net_after_action > net_before_action): 
                ep_wins += 1
                ep_wins_val.append(net_after_action - net_before_action)
                ep_money_made += ep_wins_val[-1]
            else: 
                ep_loss_val.append(np.abs(net_after_action - net_before_action))
                ep_money_lost += ep_loss_val[-1]
            ep_step_counter += 1 

            s = ns 
            if args.render: 
                env.render()
            ep_reward += r

        profit_factor.append(ep_money_made / ep_money_lost)
        avg_win.append(np.array(ep_wins_val).mean())
        avg_loss.append(np.array(ep_loss_val).mean())
        win_ratio.append(float(ep_wins)/float(ep_step_counter))
        net_worth_delta[-1] = (env.trader.net_worth - net_worth_delta[-1])
        baseline_diff.append(env.get_baseline_diff())
        rewards.append(ep_reward)
        pbar.update(1)
    pbar.close()

    agent_name = 'rand' if args.random_agent else '{:04d}{}{}'.format(int(args.agent_id), 
                                                                    'd' if args.deterministic else '',
                                                                     args.set_stock)
    base_recap_path = 'trained_models/{}/stats/'.format(args.env_id)
    base_plot_path = 'trained_models/{}/plots/'.format(args.env_id)
    if not os.path.exists(base_recap_path): 
        os.mkdir(base_recap_path)
    if not os.path.exists(base_plot_path): 
        os.mkdir(base_plot_path)
    
    to_save = [[rewards, os.path.join(base_recap_path, '{}_rewards.csv'.format(agent_name))], 
               [baseline_diff, os.path.join(base_recap_path, '{}_baseline_diff.csv'.format(agent_name))], 
               [net_worth_delta, os.path.join(base_recap_path, '{}_net_worth_delta.csv'.format(agent_name))], 
               [win_ratio, os.path.join(base_recap_path, '{}_win_ratio.csv'.format(agent_name))], 
               [avg_win, os.path.join(base_recap_path, '{}_avg_win.csv'.format(agent_name))], 
               [avg_loss, os.path.join(base_recap_path, '{}_avg_loss.csv'.format(agent_name))], 
               [profit_factor, os.path.join(base_recap_path, '{}_profit_factor.csv'.format(agent_name))]
               ]

    pbar = tqdm(total = len(to_save))
    for s_d in to_save: 
        save_stat(*s_d, pbar)
    pbar.close()


    if args.do_plot: 
        print('Starting plots')
        all_data = sorted(glob.glob(os.path.join(base_recap_path, '*.csv')))
        for info_type in ['rewards', 'baseline_diff', 'net_worth_delta', 'win_ratio', 'avg_win', 'avg_loss', 'profit_factor']: 
            selected = [f for f in all_data if info_type in f]
            make_plot(selected,info_type,base_plot_path)
