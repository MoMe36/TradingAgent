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


def get_data(filename):

    current_path = os.path.realpath(__file__).split('/')[:-1]
    path = os.path.join(*current_path)
    path = os.path.join('/', path, filename)
    data = pd.read_csv(path)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna().reset_index(drop = True)


def get_scaler(data, min_max = True): 
    if min_max: 
        price_scaler = MinMaxScaler()
        volume_scaler = MinMaxScaler()
    else: 
        price_scaler = StandardScaler()
        volume_scaler = StandardScaler()
    prices = data[['High', 'Low']].values.flatten().reshape(-1,1)
    volumes = data['Volume'].values.flatten().reshape(-1,1)
    price_scaler.fit(np.vstack([prices, np.array([[0.]])]))
    volume_scaler.fit(np.vstack([volumes, np.array([[0.]])]))

    return price_scaler, volume_scaler



class TradingEnv(gym.Env): 
    metadata = {'render.modes':['human']}


    def __init__(self, filename = 'price.csv', 
                       lookback_window = 3, 
                       ep_timesteps = 150): 
        super().__init__()


        self.show_random_trader = True


        self.data = get_data(filename)
        self.lookback_window = lookback_window
        self.ep_timesteps = ep_timesteps

        self.price_scaler, self.volume_scaler = get_scaler(self.data)
        self.reward_scaler = MinMaxScaler()
        self.reward_scaler.fit(self.data[['High', 'Low']].values.flatten().reshape(-1,1))
        
        self.market_history = deque(maxlen = lookback_window)
        self.orders_history = deque(maxlen = lookback_window) 

        self.reset()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -10., high = 10., shape = self.get_obs().shape)

        self.prepare_render()

    def get_env_specs(self): 

        specs = {'env_name':'Old'} 
        return specs 

    def get_random_action(self): 
        return self.action_space.sample()

    def prepare_render(self): 

        # VISUALIZATION 
        self.render_ready = False
        self.render_size = np.array([1200, 1000])
        self.render_window_samples = 4
        self.render_window_samples = 120
        self.candle_start_height = 0.2
        self.nb_y_label = 4 
        self.graph_height_ratio = 0.8
        self.draw_order_history = []


    def step(self, action): 


        current_price = np.random.uniform(self.data.Low[self.current_index], self.data.High[self.current_index])
        if self.current_ts == 0 :
            self.baseline_hold = self.balance / current_price
        self.baseline_value = self.baseline_hold * current_price 
        
        if action == 0: # BUY
            if self.balance > 0.: 
                self.stock_bought = self.balance / current_price 
                self.stock_held = self.stock_bought
                self.balance = 0.
                self.nb_ep_orders += 1
            else: 
                self.stock_bought = 0

            self.stock_sold = 0.
        else: # SELL
            if self.stock_held > 0: 
                self.stock_sold = self.stock_held
                self.balance = self.stock_sold * current_price
                self.stock_held = 0. 
                self.nb_ep_orders += 1
            else: 
                self.stock_sold = 0. 
            self.stock_bought = 0.

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.draw_order_history.append(self.net_worth)


        # ============================================================================
        # ============================================================================
        # ============================================================================
        #                   RANDOM TRADER ACTION 
        random_trader_action = self.action_space.sample()
        if random_trader_action == 0: # BUY
            if self.random_trader_balance > 0.: 
                self.random_trader_stock_bought = self.random_trader_balance / current_price 
                self.random_trader_stock_held = self.random_trader_stock_bought
                self.random_trader_balance = 0.
            else: 
                self.random_trader_stock_bought = 0

            self.random_trader_stock_sold = 0.
        else: # SELL
            if self.random_trader_stock_held > 0: 
                self.random_trader_stock_sold = self.random_trader_stock_held
                self.random_trader_balance = self.random_trader_stock_sold * current_price
                self.random_trader_stock_held = 0. 
            else: 
                self.random_trader_stock_sold = 0. 
            self.random_trader_stock_bought = 0.

        self.random_trader_net_worth = self.random_trader_balance + self.random_trader_stock_held * current_price
        self.random_trader_draw_order_history.append(self.random_trader_net_worth)

        # ============================================================================
        # ============================================================================
        # ============================================================================

        self.current_index += 1 
        self.current_ts += 1

        idx = self.current_index


        self.orders_history.append([self.balance, self.net_worth, self.stock_held, self.stock_sold, self.stock_bought])
        self.market_history.append([self.data.Open[idx], self.data.High[idx], self.data.Low[idx], self.data.Close[idx], self.data.Volume[idx]])
        
        done = False 
        if self.current_ts == self.ep_timesteps: 
            done = True
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True

        reward = self.compute_reward()
        self.ep_reward += reward

        return self.get_obs(), reward, done, {}

    def compute_reward(self): 
        # r = self.net_worth - 0.5 * (self.data.High[self.current_index] + self.data.Low[self.current_index])
        # if r < 0: 
        #     return r * 0.001
        # else: 
        #     return 0.01 * r
        # return self.reward_scaler.transform([[self.net_worth]])[0,0] - self.reward_scaler.transform([[self.baseline_value]])[0,0]
        return (self.reward_scaler.transform([[self.net_worth]])[0,0] - self.reward_scaler.transform([[self.prev_net_worth]])[0,0]) * 10.
        # return self.reward_scaler.transform([[self.net_worth - self.prev_net_worth]])[0,0]
    def get_obs(self): 

        market = pd.DataFrame(np.array(self.market_history), columns = 'Open,High,Low,Close,Volume'.split(','))
        orders = pd.DataFrame(np.array(self.orders_history), columns = 'Balance,NW,H,S,B'.split(','))
        obs = pd.concat([market, orders], axis = 1)
        
        for col in 'Open,High,Low,Close,Balance,NW'.split(','): 
            obs[col] = self.price_scaler.transform(obs[col].values.reshape(-1,1))
        obs['Volume'] = self.volume_scaler.transform(obs['Volume'].values.reshape(-1,1))
        return obs.values.flatten() 

    def reset(self): 

        self.current_index = 20#np.random.randint(self.lookback_window + 1, self.data.shape[0] - (self.ep_timesteps + 1))
        self.current_ts = 0 
        self.draw_order_history = []
        self.ep_reward = 0.
        self.nb_ep_orders = 0 
        
        # print(self.current_index)
        self.initial_balance = self.data.Open[self.current_index] * np.random.uniform(0.8,1.2)
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.net_worth
        self.stock_held = 0. 
        self.stock_bought = 0. 
        self.stock_sold = 0.

        self.baseline_value = 0. 
        self.baseline_hold = 0. 

        self.random_trader_balance = self.balance
        self.random_trader_net_worth = self.balance
        self.random_trader_stock_held = 0.
        self.random_trader_stock_sold = 0.
        self.random_trader_stock_bought = 0.
        self.random_trader_draw_order_history = []

        for i in range(self.current_index - self.lookback_window + 1, self.current_index + 1): 
            self.orders_history.append([self.balance, self.net_worth, self.stock_held, self.stock_sold, self.stock_bought])
            self.market_history.append([self.data.Open[i], self.data.High[i], self.data.Low[i], self.data.Close[i], self.data.Volume[i]])

        return self.get_obs()

    def render(self): 
        if not self.render_ready: 
            self.init_render()
            self.render_ready = True
        self.render_env()

    def init_render(self):

        pg.init()
        self.screen = pg.display.set_mode(self.render_size)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont('lemonmilk', 20)

    def render_env(self): 
        self.clock.tick(30.)
        self.screen.fill((100,100,100))
        self.draw()
        pg.display.flip()

    def draw_grid(self, alpha_screen, scaling_data): 

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
    def draw_candles(self, alpha_screen,data, scaling_data, y_magn, colors): 

        rect_width = 2 * (self.render_size[0]/data.shape[0])

        x_pos = self.render_size[0]* 0.5 - rect_width * data.shape[0] * 0.5
        y_pos = data[:,1] + data[:,2]
        
        x_pos_vol = [x_pos]

        
        color = colors[0]
        for i in range(data.shape[0]): 

            rect_height = np.abs(data[i,0] - data[i,-1]) * y_magn
            rect_center = ((data[i,0] + data[i,1]) * 0.5 - scaling_data.min()) * y_magn 

            shape = np.array([x_pos, 
                             self.render_size[1] * (1. - self.candle_start_height) - rect_center, 
                             rect_width * 0.9,
                             0.5 * rect_height], dtype = int)
            

            line_height = np.abs(data[i,1] - data[i,2]) * y_magn
         
            line_up = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center + 0.5 * line_height]).astype(int)
            line_down = np.array([x_pos + rect_width * 0.5 - 2, self.render_size[1] * (1. - self.candle_start_height) -  rect_center - 0.5 * line_height]).astype(int)
            
            line_height = line_height.astype(int)
            
            if i > 0: 
                color = colors[1] if (data[i,0] + data[i,1]) * 0.5 > (data[i-1,0] + data[i-1,1])*0.5 else colors[0] 

            pg.draw.rect(self.screen, color, list(shape))
            pg.draw.line(alpha_screen, (250,250,250,80), line_up, line_down, width = 2)

            x_pos += rect_width
            x_pos_vol.append(x_pos)

        return rect_width, x_pos_vol

    def draw_agent(self, alpha_screen, scaling_data, y_magn, rect_width, colors): 

        center_pos_x = self.render_size[0] * 0.5 
        self.screen.blit(alpha_screen,(0,0))

        if(len(self.draw_order_history) > 1): 
            for i in reversed(range(1, len(self.draw_order_history))):

                height_current = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i] - scaling_data.min()) * y_magn
                height_previous = self.render_size[1] * (1. - self.candle_start_height) - (self.draw_order_history[i-1] - scaling_data.min()) * y_magn


                pg.draw.line(self.screen, colors[-1], 
                             np.array([center_pos_x, height_current]).astype(int), 
                             np.array([center_pos_x - rect_width, height_previous]).astype(int), width = 8)

                if self.show_random_trader: 
                    random_trader_height_current = self.render_size[1] * (1. - self.candle_start_height) - (self.random_trader_draw_order_history[i] - scaling_data.min()) * y_magn
                    random_trader_height_previous = self.render_size[1] * (1. - self.candle_start_height) - (self.random_trader_draw_order_history[i-1] - scaling_data.min()) * y_magn
                    pg.draw.line(self.screen, colors[0], 
                             np.array([center_pos_x, random_trader_height_current]).astype(int), 
                             np.array([center_pos_x - rect_width, random_trader_height_previous]).astype(int), width = 1)
                    if i == len(self.draw_order_history) -1:

                        line_col = [239, 192, 0] if height_current < random_trader_height_current else (0,0,0) 
                        pg.draw.line(self.screen, line_col, 
                                np.array([center_pos_x, random_trader_height_current]).astype(int), 
                                np.array([center_pos_x, height_current]).astype(int))
                        for h in [height_current, random_trader_height_current]: 
                            pg.draw.line(self.screen, line_col, 
                                np.array([center_pos_x - 5, h - 5]).astype(int), 
                                np.array([center_pos_x + 5, h + 5]).astype(int))
                        info = self.font.render('{:.1f}'.format(self.draw_order_history[-1] - self.random_trader_draw_order_history[-1]), 50, line_col)
                        self.screen.blit(info, np.array([center_pos_x + 20, 0.5 * (height_current + random_trader_height_current)]).astype(int))


                center_pos_x -= rect_width
    
    def draw_volumes(self, alpha_screen, volumes, volume_magn, x_pos_vol): 
        volumes = np.hstack([np.array(x_pos_vol[1:]).reshape(-1,1), volumes.reshape(-1,1)])
        volumes[:,1] = self.render_size[1] - volumes[:,1] * volume_magn
        volumes = np.vstack([np.array([volumes[0,0], self.render_size[1]]).reshape(1,-1), 
                             volumes,
                             np.array([volumes[-1,0], self.render_size[1]]).reshape(1,-1)])
        pg.draw.line(self.screen, (0,0,0), np.array([0, self.render_size[1] *  self.graph_height_ratio - 5]).astype(int), np.array([self.render_size[0], self.render_size[1] * self.graph_height_ratio - 5]).astype(int), width = 6)
        pg.draw.polygon(self.screen, ((150,30,30)), volumes.astype(int), width = 0)
    def draw_infos(self): 
        viz_data = "Steps:                       {}/{}\nNet_Worth:            {:.2f}\nBaseline:                 {:.2f}\nBaselineDelta:            {:.2f}\nEp_Reward:             {:.2f}\nReward:                   {:.2f}\nBalance:                 {:.2f}\nHeld:                        {:.3f}\nBought:                  {:.3f}\nSold:                        {:.3f}\nOrders:                     {}".format(self.current_ts,
                                                                                                                                       self.ep_timesteps,
                                                                                                                                       self.net_worth, 
                                                                                                                                       self.baseline_value,
                                                                                                                                       self.net_worth - self.baseline_value,  
                                                                                                                                       self.ep_reward,
                                                                                                                                       self.compute_reward(), 
                                                                                                                                       self.balance,
                                                                                                                                       self.stock_held,
                                                                                                                                       self.stock_bought, 
                                                                                                                                       self.stock_sold, 
                                                                                                                                       self.nb_ep_orders)
        for i,vz in enumerate(viz_data.split('\n')): 
            label = self.font.render(vz, 50, (250,250,250))
            self.screen.blit(label, np.array([self.render_size[0] * 0.7, self.render_size[1] * (0.45 + i * 0.05)]).astype(int))

    def draw(self): 
        colors = [(220,0,0), (0,220,0), (0,80,220)]
        data_idx = [np.clip(self.current_index - int(self.render_window_samples * 0.5), 0, self.data.shape[0]), 
                    np.clip(self.current_index + int(self.render_window_samples * 0.5), 0, self.data.shape[0])]


        if(data_idx[0] == data_idx[1]):
            return 

        data = self.data.drop(['Date', 'Volume'] ,axis = 1).values[data_idx[0]:data_idx[1], :]
        agent_hist = np.array(self.orders_history)[:,1]
        scaling_data = np.hstack([data.flatten()*1.2, data.flatten()*0.8, 
                       np.array(self.draw_order_history).flatten() * 1.05, np.array(self.draw_order_history).flatten() * 0.95])
        
        y_magn = (self.render_size[1] * self.graph_height_ratio) / (scaling_data.max() - scaling_data.min())
        
        volumes = self.data['Volume'].values[data_idx[0]:data_idx[1]]
        volume_magn = (self.render_size[1] * (1. - self.graph_height_ratio)) / (self.data.Volume.max() - self.data.Volume.min())


        alpha_screen = self.screen.convert_alpha()
        self.draw_grid(alpha_screen, scaling_data)
        rect_width, x_pos_vol = self.draw_candles(alpha_screen, data, scaling_data, y_magn, colors)
        self.draw_agent(alpha_screen, scaling_data, y_magn, rect_width, colors)
        self.draw_volumes(alpha_screen, volumes, volume_magn, x_pos_vol)
        self.draw_infos()        
        
        

class TradingEnv_State(TradingEnv): 

    def get_env_specs(self): 

        specs = {'env_name':'BC_S', 
                'reward_strategy':'deltaNW', 
                'lookback_window': self.lookback_window, 
                'ep_timesteps': self.ep_timesteps, 
                'state': 'b_n,nw_n,stocks,close_n,mom,bb,vol_n', 
                'init_idx': 'random_idx', 
                'init_b': '0.8-1.2_close'} 
        return specs 


    def __init__(self, filename = 'price.csv', 
                        lookback_window = 10, 
                        ep_timesteps = 150): 

        self.show_random_trader = True
        self.data = get_data(filename)
        self.lookback_window = lookback_window
        self.ep_timesteps = ep_timesteps 


        self.market_history = deque(maxlen = self.lookback_window)
        self.orders_history = deque(maxlen = self.lookback_window)

        self.reset()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -10., high = 10., shape = self.get_obs().shape)

        self.prepare_render()

    def reset(self): 

        self.current_index = np.random.randint(self.lookback_window + 3, self.data.shape[0] - (self.ep_timesteps + 1))
        self.current_ts = 0 
        self.draw_order_history = []
        self.ep_reward = 0.
        self.nb_ep_orders = 0 
        
        # print(self.current_index)
        sma = self.data.Close[self.current_index - self.lookback_window : self.current_index].mean()
        vol_sma = self.data.Volume[self.current_index - self.lookback_window : self.current_index].mean()

        # print(self.lookback_window)
        # print(sma, vol_sma)
        # print(self.data.Volume[self.current_index - self.lookback_window+1 : self.current_index +1])

        self.initial_balance = self.data.Close[self.current_index] * np.random.uniform(0.8,1.2)
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.net_worth
        self.stock_held = 0. 
        self.stock_bought = 0. 
        self.stock_sold = 0.

        self.baseline_value = 0. 
        self.baseline_hold = 0.

        self.random_trader_balance = self.balance
        self.random_trader_net_worth = self.balance
        self.random_trader_stock_held = 0.
        self.random_trader_stock_sold = 0.
        self.random_trader_stock_bought = 0.
        self.random_trader_draw_order_history = [] 

        # print('Starting idx: {}'.format(self.current_index))
        # print('Data:\n{}'.format(self.data.iloc[self.current_index-(self.lookback_window + 1): self.current_index + self.lookback_window]))
        self.debug_idx = deque(maxlen = self.lookback_window)

        for i in range(self.current_index - self.lookback_window, self.current_index): 
            self.orders_history.append([self.balance / sma, self.net_worth / sma, 1 if self.stock_held > 0 else 0 , 1 if self.stock_sold > 0 else 0 , 1 if self.stock_bought > 0 else 0])
            self.market_history.append([(self.data.Close[i]/sma) - 1., 
                                         self.data.Close[i]/self.data.Close[i-self.lookback_window] -1.,
                                         (self.data.Close[i] - sma)/(self.data.Close[i - self.lookback_window: i].std() * 2.),
                                         (self.data.Volume[i] / vol_sma) - 1.])
            self.debug_idx.append(i)
        # print('Added idx: {}'.format(self.debug_idx))
        # input()
        return self.get_obs()


    def get_obs(self): 

        orders = pd.DataFrame(np.array(self.orders_history), columns = 'balance,nw,Held,Sold,Bought'.upper().split(','))
        market = pd.DataFrame(np.array(self.market_history), columns = 'close_n,mom,bb,vol_n'.upper().split(','))
        state = pd.concat([market, orders], axis = 1)

        return state.values.flatten()

    def get_baseline_diff(self): 
        return self.net_worth - self.baseline_value 

    def step(self, action): 

        current_price = self.data.Close[self.current_index+1]#np.random.uniform(self.data.Low[self.current_index], self.data.High[self.current_index])
        # print('Current price: {} IDX: {}'.format(current_price, self.current_index + 1))
        if self.current_ts == 0 :
            self.baseline_hold = self.balance / current_price
        self.baseline_value = self.baseline_hold * current_price 
        
        if action == 0: # BUY
            if self.balance > 0.: 
                self.stock_bought = self.balance / current_price 
                self.stock_held = self.stock_bought
                self.balance = 0.
                self.nb_ep_orders += 1
            else: 
                self.stock_bought = 0

            self.stock_sold = 0.
        else: # SELL
            if self.stock_held > 0: 
                self.stock_sold = self.stock_held
                self.balance = self.stock_sold * current_price
                self.stock_held = 0. 
                self.nb_ep_orders += 1
            else: 
                self.stock_sold = 0. 
            self.stock_bought = 0.

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.draw_order_history.append(self.net_worth)

        # ============================================================================
        # ============================================================================
        # ============================================================================
        #                   RANDOM TRADER ACTION 

        random_trader_action = self.action_space.sample()
        if random_trader_action == 0: # BUY
            if self.random_trader_balance > 0.: 
                self.random_trader_stock_bought = self.random_trader_balance / current_price 
                self.random_trader_stock_held = self.random_trader_stock_bought
                self.random_trader_balance = 0.
            else: 
                self.random_trader_stock_bought = 0

            self.random_trader_stock_sold = 0.
        else: # SELL
            if self.random_trader_stock_held > 0: 
                self.random_trader_stock_sold = self.random_trader_stock_held
                self.random_trader_balance = self.random_trader_stock_sold * current_price
                self.random_trader_stock_held = 0. 
            else: 
                self.random_trader_stock_sold = 0. 
            self.random_trader_stock_bought = 0.

        self.random_trader_net_worth = self.random_trader_balance + self.random_trader_stock_held * current_price
        self.random_trader_draw_order_history.append(self.random_trader_net_worth)

        # ============================================================================
        # ============================================================================
        # ============================================================================



        sma = self.data.Close[self.current_index - self.lookback_window : self.current_index].mean()
        vol_sma = self.data.Volume[self.current_index - self.lookback_window : self.current_index].mean()

        self.orders_history.append([self.balance / sma, self.net_worth / sma, self.stock_held , self.stock_sold ,self.stock_bought])
        self.market_history.append([(self.data.Close[self.current_index]/sma) - 1., 
                                     self.data.Close[self.current_index]/self.data.Close[self.current_index-self.lookback_window] -1.,
                                     (self.data.Close[self.current_index] - sma)/(self.data.Close[self.current_index - self.lookback_window: self.current_index].std() * 2.),
                                     (self.data.Volume[self.current_index] / vol_sma) - 1.])
        # self.debug_idx.append(self.current_index)
        # print('Visible IDX: {}'.format(self.debug_idx))
        # input()
        self.current_index += 1 
        self.current_ts += 1

        idx = self.current_index

        
        done = False 
        if self.current_ts == self.ep_timesteps: 
            done = True
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True

        reward = self.compute_reward()
        self.ep_reward += reward

        return self.get_obs(), reward, done, {}

    def compute_reward(self): 

        return (self.net_worth - self.prev_net_worth)*0.1



class TradingEnv_State2(TradingEnv_State): 

    def get_env_specs(self): 

        specs = {'folder_name':'BC_S', 
                'env_name':'BC_S2', 
                'reward_strategy':'deltaNW', 
                'lookback_window': self.lookback_window, 
                'ep_timesteps': self.ep_timesteps, 
                'state': 'b_n,nw_n,stocks,open_n,high_n,low_n,close_n,mom,bb,vol_n', 
                'init_idx': 'random_idx', 
                'init_b': '0.8-1.2_close'} 
        return specs 


    def reset(self): 

        self.current_index = np.random.randint(self.lookback_window * 3 + 1, self.data.shape[0] - (self.ep_timesteps + 2))
        self.current_ts = 0 
        self.draw_order_history = []
        self.ep_reward = 0.
        self.nb_ep_orders = 0 
        
        # print(self.current_index)
        sma = self.data.Close[self.current_index - self.lookback_window : self.current_index].mean()
        vol_sma = self.data.Volume[self.current_index - self.lookback_window : self.current_index].mean()

        # print(self.lookback_window)
        # print(sma, vol_sma)
        # print(self.data.Volume[self.current_index - self.lookback_window+1 : self.current_index +1])

        self.initial_balance = self.data.Close[self.current_index] * np.random.uniform(0.8,1.2)
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.net_worth
        self.stock_held = 0. 
        self.stock_bought = 0. 
        self.stock_sold = 0.

        self.baseline_value = 0. 
        self.baseline_hold = 0.

        self.random_trader_balance = self.balance
        self.random_trader_net_worth = self.balance
        self.random_trader_stock_held = 0.
        self.random_trader_stock_sold = 0.
        self.random_trader_stock_bought = 0.
        self.random_trader_draw_order_history = [] 

        # print('Starting idx: {}'.format(self.current_index))
        # print('Data:\n{}'.format(self.data.iloc[self.current_index-(self.lookback_window + 1): self.current_index + self.lookback_window]))
        self.debug_idx = deque(maxlen = self.lookback_window)

        for i in range(self.current_index - self.lookback_window, self.current_index): 
            self.orders_history.append([self.balance / sma, self.net_worth / sma, 1 if self.stock_held > 0 else 0 , 1 if self.stock_sold > 0 else 0 , 1 if self.stock_bought > 0 else 0])
            self.market_history.append([(self.data.Open[i]/sma) - 1., (self.data.High[i]/sma) - 1., (self.data.Low[i]/sma) - 1.,  
                                         (self.data.Close[i]/sma) - 1., 
                                         self.data.Close[i]/self.data.Close[i-self.lookback_window] -1.,
                                         (self.data.Close[i] - sma)/(self.data.Close[i - self.lookback_window: i].std() * 2.),
                                         (self.data.Volume[i] / vol_sma) - 1.])
            self.debug_idx.append(i)
        # print('Added idx: {}'.format(self.debug_idx))
        # input()
        return self.get_obs()

    def get_formatted_obs(self): 
        orders = pd.DataFrame(np.array(self.orders_history), columns = 'balance,nw,Held,Sold,Bought'.upper().split(','))
        market = pd.DataFrame(np.array(self.market_history), columns = 'open_n,high_n,low_n,close_n,mom,bb,vol_n'.upper().split(','))
        state = pd.concat([market, orders], axis = 1)
        return state

    def get_obs(self): 

       return self.get_formatted_obs().values.flatten()

    def step(self, action): 

        current_price = self.data.Close[self.current_index+1]#np.random.uniform(self.data.Low[self.current_index], self.data.High[self.current_index])
        # print('Current price: {} IDX: {}'.format(current_price, self.current_index + 1))
        if self.current_ts == 0 :
            self.baseline_hold = self.balance / current_price
        self.baseline_value = self.baseline_hold * current_price 
        
        if action == 0: # BUY
            if self.balance > 0.: 
                self.stock_bought = self.balance / current_price 
                self.stock_held = self.stock_bought
                self.balance = 0.
                self.nb_ep_orders += 1
            else: 
                self.stock_bought = 0

            self.stock_sold = 0.
        else: # SELL
            if self.stock_held > 0: 
                self.stock_sold = self.stock_held
                self.balance = self.stock_sold * current_price
                self.stock_held = 0. 
                self.nb_ep_orders += 1
            else: 
                self.stock_sold = 0. 
            self.stock_bought = 0.

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.draw_order_history.append(self.net_worth)

        # ============================================================================
        # ============================================================================
        # ============================================================================
        #                   RANDOM TRADER ACTION 

        random_trader_action = self.action_space.sample()
        if random_trader_action == 0: # BUY
            if self.random_trader_balance > 0.: 
                self.random_trader_stock_bought = self.random_trader_balance / current_price 
                self.random_trader_stock_held = self.random_trader_stock_bought
                self.random_trader_balance = 0.
            else: 
                self.random_trader_stock_bought = 0

            self.random_trader_stock_sold = 0.
        else: # SELL
            if self.random_trader_stock_held > 0: 
                self.random_trader_stock_sold = self.random_trader_stock_held
                self.random_trader_balance = self.random_trader_stock_sold * current_price
                self.random_trader_stock_held = 0. 
            else: 
                self.random_trader_stock_sold = 0. 
            self.random_trader_stock_bought = 0.

        self.random_trader_net_worth = self.random_trader_balance + self.random_trader_stock_held * current_price
        self.random_trader_draw_order_history.append(self.random_trader_net_worth)

        # ============================================================================
        # ============================================================================
        # ============================================================================



        sma = self.data.Close[self.current_index - self.lookback_window : self.current_index].mean()
        vol_sma = self.data.Volume[self.current_index - self.lookback_window : self.current_index].mean()

        self.orders_history.append([self.balance / sma, self.net_worth / sma, self.stock_held , self.stock_sold ,self.stock_bought])
        self.market_history.append([(self.data.Open[self.current_index]/sma) - 1., (self.data.High[self.current_index]/sma) - 1., 
                                    (self.data.Low[self.current_index]/sma) - 1., (self.data.Close[self.current_index]/sma) - 1., 
                                     self.data.Close[self.current_index]/self.data.Close[self.current_index-self.lookback_window] -1.,
                                     (self.data.Close[self.current_index] - sma)/(self.data.Close[self.current_index - self.lookback_window: self.current_index].std() * 2.),
                                     (self.data.Volume[self.current_index] / vol_sma) - 1.])
        # self.debug_idx.append(self.current_index)
        # print('Visible IDX: {}'.format(self.debug_idx))
        # input()
        self.current_index += 1 
        self.current_ts += 1

        idx = self.current_index

        
        done = False 
        if self.current_ts == self.ep_timesteps: 
            done = True
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True

        reward = self.compute_reward()
        self.ep_reward += reward

        return self.get_obs(), reward, done, {}

if __name__ == '__main__': 
    env = TradingEnv_State2()
    s = env.reset()

    print('State shape: {}\nState:{}'.format(s.shape, env.get_formatted_obs()))

    done = False
    ep_rewards = []
    for i in range(10): 
        ep_reward = 0
        done = False 
        s = env.reset()
        while not done: 
            action = env.action_space.sample()

            ns, r, done, info = env.step(action)
            ep_reward += r
            env.render()
        ep_rewards.append(ep_reward)
        print(np.mean(ep_rewards), np.std(ep_rewards))

