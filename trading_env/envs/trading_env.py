#### https://pylessons.com/RL-BTC-BOT-backbone/

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import pygame as pg 
import time
from collections import deque 
import gym 
from gym import spaces
import os 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

<<<<<<< HEAD
np.set_printoptions(precision = 3)
get_extreme_prices = lambda x : x[['High', 'Low']].values.flatten().reshape(-1,1)
get_volumes = lambda x : x[['Volume']].values.flatten().reshape(-1,1)

=======
>>>>>>> parent of b420023... commit before refactoring archi
class TradingEnv(gym.Env): 
    metadata = {'render.modes':['human']}
    def __init__(self, initial_balance = 8000, lookback_window = 30, episode_length = 300): 

<<<<<<< HEAD
    def get_data_path(self): 
        return 'price.csv'

    def get_initial_balance(self): 
        return self.reset_balance() 

    def get_episode_length(self): 
        return 300 

    def reset_balance(self): 
        # return self.get_initial_balance() * np.random.uniform(0.8,1.3) if self.randomize_initial_balance else self.get_initial_balance()
        return self.data.Open[self.current_index] * np.random.uniform(0.8,1.3)

    def get_data(self): 
=======
        # ENV PARAMETERS
>>>>>>> parent of b420023... commit before refactoring archi

        current_path = os.path.realpath(__file__).split('/')[:-1]
        path = os.path.join(*current_path)
        path = os.path.join('/', path, self.get_data_path())


        self.data = pd.read_csv(path)
<<<<<<< HEAD
        self.data = self.data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna().reset_index(drop = True)
    
    def get_env_name(self): 
        return "BC_I"

    def get_lookback_window(self): 
        return 10

    def prepare_normalization(self): 
        price_values = get_extreme_prices(self.data)
        volume = get_volumes(self.data)
        # self.scaler = StandardScaler().fit(price_values)
        self.scaler = MinMaxScaler().fit(price_values)
        self.volume_scaler = MinMaxScaler().fit(volume)

        # print(self.scaler.mean_, self.scaler.var_**0.5)
        # input()

    def get_env_specs(self): 
        return {**{'env_name': self.get_env_name(), 
                'reward_strategy': self.get_reward_name(), 
                'lookback': str(self.get_lookback_window()), 
                'actions': 'discrete' if isinstance(self.action_space, spaces.Discrete) else 'continuous',
                'episode_length': self.episode_length, 
                'initial_balance': self.initial_balance, 
                'randomize_balance': self.randomize_initial_balance}, **self.get_additional_env_infos()}
    
    def get_additional_env_infos(self): 
        return {'normalized': 'true',
                'normalizing_strategy': 'MinMax',  
                'using_volumes':'true', 
                'volume_norm': 'MinMax', 
                'initial_balance_strategy':'0.8-1.3'}

    def __init__(self): 

        # ENV PARAMETERS

        self.get_data() 

        self.prepare_normalization()

        self.randomize_initial_balance = True 
        self.total_steps = self.data.shape[0]
        self.lookback_window = self.get_lookback_window()
        self.episode_length = self.get_episode_length()
        # self.initial_balance = self.get_initial_balance()
        self.current_index = self.lookback_window + 1 
=======
        self.data = self.data.dropna().reset_index(drop = True)
        self.total_steps = self.data.shape[0]
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.current_index = lookback_window + 1 
>>>>>>> parent of b420023... commit before refactoring archi
        self.current_step = 0
        self.market_history = deque(maxlen = self.lookback_window)
        self.orders_history = deque(maxlen = self.lookback_window)

        self.initialize_state()

        self.current_episode = 0

        # VIZ PARAMETERS

        self.render_ready = False
<<<<<<< HEAD
        self.render_size = np.array([1200, 1000])
=======
        self.render_size = np.array([1000, 800])
        # self.render_window_samples = 4
>>>>>>> parent of b420023... commit before refactoring archi
        self.render_window_samples = 120
        self.candle_start_height = 0.2
        self.nb_y_label = 4 
        self.graph_height_ratio = 0.8
        self.draw_order_history = []


        # GYM PARAMETERS
        self.observation_space = spaces.Box(low = -10e5, high = 10e5, shape = self.get_state_size())
        self.action_space = spaces.Discrete(2)

    def get_state_size(self): 
        return self.reset().shape

    def get_random_action(self): 
        return self.action_space.sample()

    def init_render(self):

        pg.init()
        self.screen = pg.display.set_mode(self.render_size)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont('lemonmilk', 20)

    def initialize_state(self): 

        self.initial_balance = self.reset_balance()
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.balance
        self.stock_held = 0. 
        self.stock_sold = 0. 
        self.stock_bought = 0. 

        self.episode_reward = 0.

    def step(self, action):

        done = False 

        self.current_index += 1
        self.current_step += 1 

        if(isinstance(action, np.ndarray)): 
            action = np.argmax(action.flatten())
        elif (isinstance(action, list)):
            action = np.argmax(np.array(action).flatten())

        new_price_data = self.data.drop('Date', axis = 1).iloc[self.current_index,: ]
        current_price = np.random.uniform(new_price_data.Open, new_price_data.Close)

        # print('\n'*10)
        # print('Index: {}'.format(self.current_index))
        # print('Current data:\n{}'.format(new_price_data))
        self.baseline_value = self.baseline_action_held * current_price

        if action == 0 and self.balance > 0: 
            self.stock_bought = self.balance / current_price
            self.balance = 0. 
            self.stock_held += self.stock_bought
        
        elif action == 1 and self.stock_held > 0: 
            self.stock_sold = self.stock_held
            self.balance += self.stock_held * current_price
            self.stock_held = 0 

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price
        self.orders_history.append([self.balance, self.net_worth, self.stock_held, self.stock_bought, self.stock_sold])
        self.market_history.append(list(new_price_data.values.flatten()))

        # print('Marker history:\n{}'.format(np.array(self.market_history)))
        # print('Orders history:\n{}'.format(np.array(self.orders_history)))
        reward = self.compute_reward()
        
        self.episode_reward += reward
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True

        if self.current_step == self.episode_length: 
            done = True 
            # input('Ep reward: {:.3f}'.format(self.episode_reward))
    

        return self.get_obs(), reward, done, {} 

    def compute_reward(self):
        return self.net_worth - self.prev_net_worth
        # return self.get_baseline_diff()
        # reward = np.exp(0.001 * (self.net_worth - self.initial_balance)) -1
        # reward = -np.exp(0.1 * (self.net_worth - 10000))

        # reward = 0.
        # if self.current_step == self.episode_length: 
        #     reward = 0.1 * (self.net_worth - 1.5 * self.data.drop('Date', axis = 1).iloc[self.current_index,-2])
        return reward 

<<<<<<< HEAD
    def get_reward_name(self):
        return "dNW"

=======
>>>>>>> parent of b420023... commit before refactoring archi
    def get_baseline_diff(self): 
        return self.net_worth - self.baseline_value

    def get_obs(self): 

        agent_actions = np.array(self.orders_history)
        market_history = np.array(self.market_history)
        state = np.hstack([market_history, agent_actions])

        # print('State 0-4:\n{}'.format(state[:,:4]))
        # print('State 4:\n{}'.format(state[:,4].reshape(-1,1)))
        # print('State 5:-3:\n{}'.format(state[:,5:-3]))
        # print('='*5 + 'Normalizing' + "="*5)
        for i in range(4): 
            # print('Before: \t {}'.format(state[:,i].flatten()))
            state[:,i] = self.scaler.transform(state[:,i].reshape(-1,1)).flatten()
            # print('After: \t {}'.format(state[:,i].flatten()))
        for i in range(5,7): 
            # print('Before: \t {}'.format(state[:,i].flatten()))
            state[:,i] = self.scaler.transform(state[:,i].reshape(-1,1)).flatten()
            # print('After: \t {}'.format(state[:,i].flatten()))

        # state[:,:4] = (state[:,:4] - self.scaler.mean_) / self.scaler.var_**0.5 
        state[:,4] = self.volume_scaler.transform(state[:,4].reshape(-1,1)).flatten()
        # state[:,5:-3] = (state[:,5:-3] - self.scaler.mean_) / self.scaler.var_**0.5 
        # print('State 0-4:\n{}'.format(state[:,:4]))
        # print('State 4:\n{}'.format(state[:,4].reshape(-1,1)))
        # print('State 5:-3:\n{}'.format(state[:,5:-3]))


        # print('State:\n{}'.format(state))
        # input()
        self.draw_order_history.append(self.net_worth)

        return state.flatten()

    def reset(self, manual_index = None): 

        self.current_index = manual_index + self.lookback_window + 1 if manual_index is not None else np.random.randint(self.lookback_window + 1, self.total_steps - (self.episode_length + 1))
        self.current_step = 0
        
        self.initialize_state()
        self.baseline_action_held = self.balance / self.data.Open[self.current_index]
        self.baseline_value = self.baseline_action_held * self.data.High[self.current_index] 
        
        # STATE IS BALANCE, NET WORTH, HELD, SOLD, BOUGHT 
        self.orders_history = np.zeros((self.lookback_window,5))
        self.orders_history[:, 0] = self.balance # balance
        self.orders_history[:, 1] = self.balance # net_worth


        self.orders_history = deque([list(oh) for oh in self.orders_history], maxlen = self.lookback_window)
        self.market_history = self.data.drop('Date', axis=  1).iloc[self.current_index - self.lookback_window: self.current_index].values
        self.market_history = deque([list(mh) for mh in self.market_history], maxlen = self.lookback_window)


        obs = self.get_obs()

        self.draw_order_history = [self.balance]

        return obs 
    def render(self): 

        if not self.render_ready: 
            self.init_render()
            self.render_ready = True
        self.render_env()

    def render_env(self): 
        self.clock.tick(30.)
        self.screen.fill((100,100,100))
        self.draw()
        pg.display.flip()


    def draw_bg(self, alpha_screen, current_data):

        x_label_pos = self.render_size[0] * 0.02
        y_label_inc = self.render_size[1] * (1. - self.candle_start_height) / self.nb_y_label
        y_label_pos = self.render_size[1] * (1. - self.candle_start_height)
        y_labels = np.linspace(current_data.min(), current_data.max(), self.nb_y_label)
        x_labels = np.linspace(0, self.render_size[1], self.nb_y_label)
        for i in range(self.nb_y_label): 
            # DRAWING GRID 
            pg.draw.line(alpha_screen, (220,220,220,50), 
                np.array([0, y_label_pos - 5]).astype(int), np.array([self.render_size[0], y_label_pos - 5]).astype(int) )
            pg.draw.line(alpha_screen, (220,220,220,50), 
                np.array([x_labels[i], 0]).astype(int), np.array([x_labels[i], self.render_size[1]]).astype(int) )

            # WRITING LABELS
            label = self.font.render('{:.1f}'.format(y_labels[i]), 50, (250,250,250))
            self.screen.blit(label, np.array([x_label_pos, y_label_pos - 30]).astype(int))
            y_label_pos -= y_label_inc

    def draw_candles(self, alpha_screen, current_data, scaling_data, y_magn, rect_width, colors): 

        color = colors[0]
        x_pos = self.render_size[0]* 0.5 - rect_width * current_data.shape[0] * 0.5
        y_pos = current_data[:,1] + current_data[:,2]
        
        x_pos_vol = [x_pos]
        
        for i in range(current_data.shape[0]): 

            rect_height = np.abs(current_data[i,0] - current_data[i,-1]) * y_magn
            rect_center = ((current_data[i,0] + current_data[i,1]) * 0.5 - scaling_data.min()) * y_magn 

            shape = np.array([x_pos, 
                             self.render_size[1] * (1. - self.candle_start_height) - rect_center, 
                             rect_width * 0.9,
                             0.5 * rect_height], dtype = int)
            

            line_height = np.abs(current_data[i,1] - current_data[i,2]) * y_magn
          

            line_up = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center + 0.5 * line_height]).astype(int)
            line_down = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center - 0.5 * line_height]).astype(int)
            
            line_height = line_height.astype(int)
            
            # if i > 0: 
            # color = colors[1] if (current_data[i,0] + current_data[i,1]) * 0.5 > (current_data[i-1,0] + current_data[i-1,1])*0.5 else colors[0] 
            color = colors[1] if current_data[i,0] < current_data[i,-1] else colors[0]

            pg.draw.rect(self.screen, color, list(shape))
            pg.draw.line(alpha_screen, (250,250,250,80), line_up, line_down, width = 2)

            x_pos += rect_width
            x_pos_vol.append(x_pos)

        self.screen.blit(alpha_screen,(0,0))

        return x_pos_vol

    def draw_agent_balance(self,scaling_data, y_magn, rect_width, colors):

        center_pos_x = self.render_size[0] * 0.5
        
        if(len(self.draw_order_history) > 1): 
            for i in reversed(range(1, len(self.draw_order_history))):

                height_current = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i] - scaling_data.min()) * y_magn
                height_previous = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i-1] - scaling_data.min()) * y_magn


                pg.draw.line(self.screen, colors[-1], 
                             np.array([center_pos_x, height_current]).astype(int), 
                             np.array([center_pos_x - rect_width, height_previous]).astype(int), width = 8)
                center_pos_x -= rect_width

    def draw_volumes(self, volumes, volume_magn, x_pos_vol, colors): 

        volumes = np.hstack([np.array(x_pos_vol[1:]).reshape(-1,1), volumes.reshape(-1,1)])
        volumes[:,1] = self.render_size[1] - volumes[:,1] * volume_magn
        volumes = np.vstack([np.array([volumes[0,0], self.render_size[1]]).reshape(1,-1), 
                             volumes,
                             np.array([volumes[-1,0], self.render_size[1]]).reshape(1,-1)])
        pg.draw.line(self.screen, (0,0,0), np.array([0, self.render_size[1] *  self.graph_height_ratio - 5]).astype(int), np.array([self.render_size[0], self.render_size[1] * self.graph_height_ratio - 5]).astype(int), width = 6)
        pg.draw.polygon(self.screen, ((150,30,30)), volumes.astype(int), width = 0)

    def draw(self): 

        alpha_screen = self.screen.convert_alpha()
        alpha_screen.fill([0,0,0,0])

        colors = [(220,0,0), (0,220,0), (0,80,220)]
        color = colors[0]


        data_idx = [np.clip(self.current_index - int(self.render_window_samples * 0.5), 0, self.data.shape[0]), 
                    np.clip(self.current_index + int(self.render_window_samples * 0.5), 0, self.data.shape[0])]


        if(data_idx[0] == data_idx[1]):
            return 

        data = self.data.drop(['Date', 'Volume'] ,axis = 1).values[data_idx[0]:data_idx[1], :]
        agent_hist = np.array(self.orders_history)[:,1]
        scaling_data = np.hstack([data.flatten(), agent_hist.flatten()])
        

        self.draw_bg(alpha_screen, scaling_data)


        y_magn = (self.render_size[1] * self.graph_height_ratio) / (scaling_data.max() - scaling_data.min())
        volumes = self.data['Volume'].values[data_idx[0]:data_idx[1]]
        volume_magn = (self.render_size[1] * (1. - self.graph_height_ratio)) / (self.data.Volume.max() - self.data.Volume.min())

        rect_width = 2 * (self.render_size[0]/data.shape[0])

        x_pos_vol = self.draw_candles(alpha_screen, data, scaling_data, y_magn, rect_width, colors)

        self.draw_agent_balance(scaling_data, y_magn, rect_width, colors)
        
        self.draw_volumes(volumes, volume_magn, x_pos_vol, colors)
        
        viz_data = "Steps:                       {}/{}\nBaseline diff:         {:.2f}\nNet_Worth:            {:.2f}\nP_Net_Worth:        {:.2f}\nEp_Reward:             {:.2f}\nReward:                   {:.2f}\nBalance:                 {:.2f}\nHeld:                        {:.3f}\nBought:                  {:.3f}\nSold:                        {:.3f}".format(self.current_step,
                                                                                                                                       self.episode_length,
                                                                                                                                       self.get_baseline_diff(), 
                                                                                                                                       self.net_worth, 
                                                                                                                                       self.prev_net_worth, 
                                                                                                                                       self.episode_reward, 
                                                                                                                                       self.compute_reward(), 
                                                                                                                                       self.balance,
                                                                                                                                       self.stock_held,
                                                                                                                                       self.stock_bought, 
                                                                                                                                       self.stock_sold)
        for i,vz in enumerate(viz_data.split('\n')): 
            label = self.font.render(vz, 50, (250,250,250))
            self.screen.blit(label, np.array([self.render_size[0] * 0.7, self.render_size[1] * (0.5 + i * 0.05)]).astype(int))


class TradingCARG(TradingEnv): 

    def get_episode_length(self): 
        return 100

    def get_env_name(self): 
        return "BC_CARG"
    def get_reward_name(self): 
        return "CARG_finish"
    def compute_reward(self): 
        if self.current_step < self.episode_length: 
            if self.net_worth < 0.5 * self.initial_balance: 
                return -10.
            else: 
                return 0.
        else: 
            reward = (self.net_worth/ self.initial_balance) -1.
            return reward

class TradingAction(TradingEnv): 

    def get_episode_length(self): 
        return 200
    def get_lookback_window(self):
        return 5

    def get_env_name(self): 
        return "BC_Ac"
    def get_reward_name(self): 
        return "nb_action_minus_one"
    def compute_reward(self): 
        return np.exp(self.stock_held - 1.)


class McDonaldEnv(TradingEnv): 

    def get_data_path(self): 
        return 'MCD.csv'

    def get_lookback_window(self): 
        return 5

    def get_env_name(self): 
        return "MCD"

    def get_additional_env_infos(self): 
        return {'initialization': 'random_uniform(50.,100.)'}



class NormalizedEnv(TradingEnv): 
<<<<<<< HEAD

    def get_env_name(self): 
        return "BC_N"

    def get_additional_env_infos(self): 
        return {'normalized': 'true', 
                'using_volumes':'false'}

    def get_lookback_window(self): 
        return 10 

=======
>>>>>>> parent of b420023... commit before refactoring archi
    def __init__(self): 
        super().__init__() 
        
        for col in ['Open', 'High', 'Low', 'Close']: 
            self.data[col] = MinMaxScaler().fit_transform(self.data[col].values.reshape(-1,1))
        self.initial_balance = 0.6
        self.initialize_state()

<<<<<<< HEAD
    def get_obs(self): 

        agent_actions = np.array(self.orders_history)
        market_history = np.array(self.market_history)

        # REMOVING VOLUME
        market_history = market_history[:,:-1]
        state = np.hstack([market_history, agent_actions])

        self.draw_order_history.append(self.net_worth)

        return state.flatten()

class AugmentedEnv(NormalizedEnv): 

    def get_lookback_window(self): 
        return 50 

    def get_env_name(self): 
        return "BC_A"

    def get_additional_env_infos(self): 
        return {'normalized': 'true', 
                'using_volumes':'true', 
                'using_streaks': 'true', 
                'using_mfa_mfe': 'true'}

    def initialize_state(self): 
        super().initialize_state()
        self.winning_streak = 0 
        self.losing_streak = 0.
        self.mfa = 0. 
        self.mfe = 0. 

    def get_obs(self): 
        obs = super().get_obs()

        if(self.net_worth > self.prev_net_worth): 
            self.winning_streak += 1 
            self.losing_streak = 0 
        else: 
            self.winning_streak = 0 
            self.losing_streak += 1

        self.mfa = np.array(self.market_history).min() - np.array(self.market_history)[0,0]
        self.mfe = np.array(self.market_history).max() - np.array(self.market_history)[0,0]

        streaks = np.array([self.winning_streak, self.losing_streak]) * 0.1
        mf = np.array([self.mfa, self.mfe])


        # obs = np.hstack([obs, streaks, mf])
        obs = np.hstack([obs, streaks])

        return obs 



class AppleEnv(TradingEnv):
    def get_env_name(self): 
        return "APL_N" 

    def get_data(self): 
        current_path = os.path.realpath(__file__).split('/')[:-1]
        path = os.path.join(*current_path)
        path = os.path.join('/', path, 'aapl.csv')

        self.data = pd.read_csv(path)
        self.data = self.data.dropna().reset_index(drop = True)
    def __init__(self, initial_balance = 30.): 
        super().__init__()

        scalers = []
        for col in ['Open', 'High', 'Low', 'Close']: 
            scaler = MinMaxScaler()
            self.data[col] = scaler.fit_transform(self.data[col].values.reshape(-1,1))
            scalers.append(scaler)

        self.initial_balance = scalers[0].transform([[initial_balance]])[0,0]
        self.initialize_state()




# SUBCLASS FOR A UNIQUE INITIALIZATION TO CHECK AGENT 
class TradingEnvFix(TradingEnv): 
    def get_env_name(self): 
        return "BC_F"
=======
# SUBCLASS FOR A UNIQUE INITIALIZATION TO CHECK AGENT 
class TradingEnvFix(TradingEnv): 
>>>>>>> parent of b420023... commit before refactoring archi
    def __init__(self): 
        super().__init__()
        self.randomize_initial_balance = False

    def reset(self, manual_index = 0): 
        return super().reset(manual_index = manual_index)
   

if __name__ == "__main__": 

<<<<<<< HEAD
    # env = TradingEnv()
    # env = AppleEnv()
    # env = AugmentedEnv()
    env = McDonaldEnv()
=======
    env = TradingEnv()
>>>>>>> parent of b420023... commit before refactoring archi
    
    rewards = []
    for ep in range(10): 
        done = False 
        s = env.reset()
        ep_reward = 0. 
        counter = 0
        print('State size: {}'.format(s.shape))
        while not done: 

            ns, r, done, _ = env.step(np.random.randint(3))
            ep_reward += r
            time.sleep(0.001)
            env.render()
            counter += 1

        rewards.append(ep_reward)
    print(pd.Series(rewards).describe())