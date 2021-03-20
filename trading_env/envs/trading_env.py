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
from sklearn.preprocessing import MinMaxScaler

class TradingEnv(gym.Env): 
    metadata = {'render.modes':['human']}
    def __init__(self, initial_balance = 8000, lookback_window = 30, episode_length = 300): 

        # ENV PARAMETERS

        current_path = os.path.realpath(__file__).split('/')[:-1]
        path = os.path.join(*current_path)
        path = os.path.join('/', path, 'price.csv')


        self.data = pd.read_csv(path)
        self.data = self.data.dropna().reset_index(drop = True)
        self.total_steps = self.data.shape[0]
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.current_index = lookback_window + 1 
        self.current_step = 0
        self.market_history = deque(maxlen = self.lookback_window)
        self.orders_history = deque(maxlen = self.lookback_window)

        self.initialize_state()

        self.current_episode = 0

        # VIZ PARAMETERS

        self.render_ready = False
        self.render_size = np.array([1000, 800])
        # self.render_window_samples = 4
        self.render_window_samples = 120
        self.candle_start_height = 0.2
        self.nb_y_label = 4 
        self.graph_height_ratio = 0.8
        self.draw_order_history = []


        # GYM PARAMETERS
        self.observation_space = spaces.Box(low = -10e5, high = 10e5, shape = self.get_state_size())
        # self.action_space = spaces.Discrete(3)
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

        self.balance = self.initial_balance * np.random.uniform(0.8,1.2)
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

        self.baseline_value = new_price_data[0] * self.initial_balance
        # if action == 0: 
        #     pass 
        if action == 0 and self.balance > 0: 
            self.stock_bought = self.balance / current_price
            self.balance -= self.stock_bought * current_price
            self.stock_held += self.stock_bought
        
        elif action == 1 and self.stock_held > 0: 
            self.stock_sold = self.stock_held
            self.balance += self.stock_held * current_price
            self.stock_held = 0 

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.stock_held, self.stock_bought, self.stock_sold])
        self.market_history.append(list(new_price_data.values.flatten()))

        reward = self.compute_reward()
        
        self.episode_reward += reward
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True

        if self.current_step >= self.episode_length: 
            done = True 
    

        return self.get_obs(), reward, done, {} 

    def compute_reward(self):
        # return self.get_baseline_diff()
        return self.net_worth - self.prev_net_worth
        # reward = np.exp(0.001 * (self.net_worth - self.initial_balance)) -1
        # reward = -np.exp(0.1 * (self.net_worth - 10000))

        # reward = 0.
        # if self.current_step == self.episode_length: 
        #     reward = 0.1 * (self.net_worth - 1.5 * self.data.drop('Date', axis = 1).iloc[self.current_index,-2])
        return reward 

    def get_baseline_diff(self): 
        return self.net_worth - self.baseline_value

    def get_obs(self): 

        agent_actions = np.array(self.orders_history)
        market_history = np.array(self.market_history)
        state = np.hstack([market_history, agent_actions])

        self.draw_order_history.append(self.net_worth)

        return state.flatten()

    def reset(self, manual_index = None): 

        self.current_index = manual_index + self.lookback_window + 1 if manual_index is not None else np.random.randint(self.lookback_window + 1, self.total_steps - (self.episode_length + 1))
        self.current_step = 0
        
        self.initialize_state()
        self.baseline_value = self.balance
        
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

    def draw(self): 

        data_idx = [np.clip(self.current_index - int(self.render_window_samples * 0.5), 0, self.data.shape[0]), 
                    np.clip(self.current_index + int(self.render_window_samples * 0.5), 0, self.data.shape[0])]


        if(data_idx[0] == data_idx[1]):
            return 

        data = self.data.drop(['Date', 'Volume'] ,axis = 1).values[data_idx[0]:data_idx[1], :]
        agent_hist = np.array(self.orders_history)[:,1]
        scaling_data = np.hstack([data.flatten(), agent_hist.flatten()])
        
        y_magn = (self.render_size[1] * self.graph_height_ratio) / (scaling_data.max() - scaling_data.min())
        
        volumes = self.data['Volume'].values[data_idx[0]:data_idx[1]]
        volume_magn = (self.render_size[1] * (1. - self.graph_height_ratio)) / (self.data.Volume.max() - self.data.Volume.min())


        alpha_screen = self.screen.convert_alpha()
        alpha_screen.fill([0,0,0,0])

        x_label_pos = self.render_size[0] * 0.02
        y_label_inc = self.render_size[1] * (1. - self.candle_start_height) / self.nb_y_label
        y_label_pos = self.render_size[1] * (1. - self.candle_start_height)
        y_labels = np.linspace(scaling_data.min(), scaling_data.max(), self.nb_y_label)
        x_labels = np.linspace(0, self.render_size[1], self.nb_y_label)
        for i in range(self.nb_y_label): 
            # DRAWING GRID 
            pg.draw.line(alpha_screen, (220,220,220,50), 
                np.array([0, y_label_pos - 5]).astype(int), np.array([self.render_size[0], y_label_pos - 5]).astype(int) )
            pg.draw.line(alpha_screen, (220,220,220,50), 
                np.array([x_labels[i], 0]).astype(int), np.array([x_labels[i], self.render_size[1]]).astype(int) )

            # self.screen.blit(alpha_screen,(0,0))            
            # WRITING LABELS
            label = self.font.render('{:.1f}'.format(y_labels[i]), 50, (250,250,250))
            self.screen.blit(label, np.array([x_label_pos, y_label_pos - 30]).astype(int))
            y_label_pos -= y_label_inc


        rect_width = 2 * (self.render_size[0]/data.shape[0])

        x_pos = self.render_size[0]* 0.5 - rect_width * data.shape[0] * 0.5
        y_pos = data[:,1] + data[:,2]
        
        x_pos_vol = [x_pos]

        colors = [(220,0,0), (0,220,0), (0,80,220)]
        color = colors[0]
        for i in range(data.shape[0]): 

            rect_height = np.abs(data[i,0] - data[i,-1]) * y_magn
            rect_center = ((data[i,0] + data[i,1]) * 0.5 - scaling_data.min()) * y_magn 

            shape = np.array([x_pos, 
                             self.render_size[1] * (1. - self.candle_start_height) - rect_center, 
                             rect_width * 0.9,
                             0.5 * rect_height], dtype = int)
            

            line_height = np.abs(data[i,1] - data[i,2]) * y_magn
            # print('Height: {:.0f}\nCenter: {:.0f}\nCenter proj: {:.0f}\nUp: {:.0f}\nLow: {:.0f}\nOHLC:{}'.format(line_height, 
            #                                                                     rect_center,
            #                                                                     self.render_size[1] * (1. - self.candle_start_height) - rect_center,
            #                                                                     self.render_size[1] * (1. - self.candle_start_height) -  rect_center + 0.5 * line_height,
            #                                                                     self.render_size[1] * (1. - self.candle_start_height) -  rect_center - 0.5 * line_height, 
            #                                                                     data[i,:]))
           


            line_up = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center + 0.5 * line_height]).astype(int)
            line_down = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center - 0.5 * line_height]).astype(int)
            
            line_height = line_height.astype(int)
            
            if i > 0: 
                color = colors[1] if (data[i,0] + data[i,1]) * 0.5 > (data[i-1,0] + data[i-1,1])*0.5 else colors[0] 

            pg.draw.rect(self.screen, color, list(shape))
            pg.draw.line(alpha_screen, (250,250,250,80), line_up, line_down, width = 2)

            x_pos += rect_width
            x_pos_vol.append(x_pos)


        # DRAWING AGENT'S BALANCE

        center_pos_x = self.render_size[0] * 0.5 
        # for i in reversed(range(1, len(self.orders_history))):

        #     height_current = self.render_size[1] * (1. - self.candle_start_height) - (self.orders_history[i][1] - scaling_data.min()) * y_magn
        #     height_previous = self.render_size[1] * (1. - self.candle_start_height) - (self.orders_history[i-1][1] - scaling_data.min()) * y_magn


        #     pg.draw.line(self.screen, colors[-1], 
        #                  np.array([center_pos_x, height_current]).astype(int), 
        #                  np.array([center_pos_x - rect_width, height_previous]).astype(int), width = 3)
        #     center_pos_x -= rect_width
        self.screen.blit(alpha_screen,(0,0))

        if(len(self.draw_order_history) > 1): 
            for i in reversed(range(1, len(self.draw_order_history))):

                height_current = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i] - scaling_data.min()) * y_magn
                height_previous = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i-1] - scaling_data.min()) * y_magn


                pg.draw.line(self.screen, colors[-1], 
                             np.array([center_pos_x, height_current]).astype(int), 
                             np.array([center_pos_x - rect_width, height_previous]).astype(int), width = 8)
                center_pos_x -= rect_width

        
        # DRAWING VOLUME AS A POLYGON 
        volumes = np.hstack([np.array(x_pos_vol[1:]).reshape(-1,1), volumes.reshape(-1,1)])
        volumes[:,1] = self.render_size[1] - volumes[:,1] * volume_magn
        volumes = np.vstack([np.array([volumes[0,0], self.render_size[1]]).reshape(1,-1), 
                             volumes,
                             np.array([volumes[-1,0], self.render_size[1]]).reshape(1,-1)])
        pg.draw.line(self.screen, (0,0,0), np.array([0, self.render_size[1] *  self.graph_height_ratio - 5]).astype(int), np.array([self.render_size[0], self.render_size[1] * self.graph_height_ratio - 5]).astype(int), width = 6)
        pg.draw.polygon(self.screen, ((150,30,30)), volumes.astype(int), width = 0)
        
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




class NormalizedEnv(TradingEnv): 
    def __init__(self): 
        super().__init__() 
        
        for col in ['Open', 'High', 'Low', 'Close']: 
            self.data[col] = MinMaxScaler().fit_transform(self.data[col].values.reshape(-1,1))
        self.initial_balance = 0.6
        self.initialize_state()

# SUBCLASS FOR A UNIQUE INITIALIZATION TO CHECK AGENT 
class TradingEnvFix(TradingEnv): 
    def __init__(self): 
        super().__init__()

    def reset(self, manual_index = 0): 
        return super().reset(manual_index = manual_index)
    
    def initialize_state(self): 

        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.balance
        self.stock_held = 0. 
        self.stock_sold = 0. 
        self.stock_bought = 0. 

        self.episode_reward = 0.


if __name__ == "__main__": 

    env = TradingEnv()
    
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