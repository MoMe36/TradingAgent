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
import env_utils
import ta 

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

class Trader: 
    def __init__(self, initial_balance): 
        self.reset(initial_balance)

    def reset(self, initial_balance):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = self.balance
        self.prev_net_worth = self.balance 
        self.stock_held = 0.
        self.stock_bought = 0.
        self.stock_sold = 0.

        self.net_worth_history = [self.initial_balance]

    def get_state(self, current_price = None):
        if current_price != None: 
            return [self.balance/current_price -1., self.net_worth/current_price -1.,  self.stock_held, self.stock_sold, self.stock_bought]
        else: 
            return [self.balance, self.net_worth, self.stock_held, self.stock_sold, self.stock_bought]

    def update(self, action, current_price): 
        successful_order = 0

        if action == 0: # BUY
            if self.balance > 0.: 
                self.stock_bought = self.balance / current_price 
                self.stock_held = self.stock_bought
                self.balance = 0.
                successful_order = 1
            else: 
                self.stock_bought = 0

            self.stock_sold = 0.
        else: # SELL
            if self.stock_held > 0: 
                self.stock_sold = self.stock_held
                self.balance = self.stock_sold * current_price
                self.stock_held = 0. 
                successful_order = 1
            else: 
                self.stock_sold = 0. 
            self.stock_bought = 0.

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.stock_held * current_price

        self.net_worth_history.append(self.net_worth)

        return self.get_state(), successful_order


class TradingEnv(gym.Env): 
    metadata = {'render.modes':['human']}


    def __init__(self, filename = 'price.csv', 
                       lookback_window = 5, 
                       ep_timesteps = 150): 
        super().__init__()


        self.show_random_trader = True


        self.data = get_data(filename)
        self.lookback_window = lookback_window
        self.ep_timesteps = ep_timesteps

        self.initalize_env()

    def set_data(self, data_path): 

        print('Fetching {}'.format(data_path))
        self.data = get_data(data_path)
        self.initalize_env()

    def set_lbw(self, lbw): 

        self.lookback_window = lbw
        self.initalize_env()

    def augment_data(self): 
        
        self.emas = [5,25,100]
        self.data['rsi'] = ta.momentum.RSIIndicator(close = self.data.Close, fillna = True).rsi()
        self.data['stoch'] = ta.momentum.StochasticOscillator(high =  self.data.High, 
                                             low = self.data.Low,
                                             close = self.data.Close, fillna = True).stoch()
        self.data['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(close = self.data.Close, 
                                                              low = self.data.Low, 
                                                              high = self.data.High, 
                                                              volume = self.data.Volume, 
                                                              fillna = True).chaikin_money_flow()
        for window in self.emas: 
            self.data['ema{}'.format(window)] = ta.trend.EMAIndicator(close = self.data.Close, 
                                                      window = window, 
                                                      fillna = True).ema_indicator()
        bb = ta.volatility.BollingerBands(close = self.data.Close,
                                       fillna = True)
        self.data['bb'] = bb.bollinger_pband()#(df.Close - bb.bollinger_mavg())/(2. * bb.bollinger_wband())
        # self.data['close_n']
        for ema in self.emas: 
            self.data['close_n{}'.format(ema)] = self.data.Close / self.data['ema{}'.format(ema)]
        self.data['rsi_n'] = self.data['rsi'] * 0.01
        self.data['stoch_n'] = self.data['stoch'] * 0.01

    def initalize_env(self): 

        self.augment_data()

        self.orders_history = deque(maxlen = self.lookback_window) 

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
        
        self.baseline_trader.update(0, current_price) # Always buys (= hold)
        trader_state, made_order = self.trader.update(action,current_price)
        self.random_trader.update(self.get_random_action(),current_price)

        self.orders_history.append(self.trader.get_state(current_price))
        self.nb_ep_orders += made_order

        self.current_index += 1 
        self.current_ts += 1

        idx = self.current_index
        
        done = False 
        if self.current_ts == self.ep_timesteps: 
            done = True
        if self.trader.net_worth < 0.5 * self.trader.initial_balance: 
            done = True

        reward = self.compute_reward()
        self.ep_reward += reward

        return self.get_obs(), reward, done, {}

    def compute_reward(self): 
        return self.trader.net_worth - self.trader.prev_net_worth

    def get_formatted_obs(self): 

        # market = pd.DataFrame(np.array(self.market_history), columns = 'Open,High,Low,Close,Volume'.split(','))
        orders = pd.DataFrame(np.array(self.orders_history), columns = 'Balance,NW,H,S,B'.split(','))
        feature_cols = [close for close in self.data.columns if close.startswith('close_n')]
        feature_cols += ['rsi_n', 'stoch_n','bb']
        obs = self.data[feature_cols].iloc[self.current_index - self.lookback_window:self.current_index,: ].reset_index(drop = True)
        obs = pd.concat([obs, orders], axis = 1)

        return obs

    def get_obs(self): 
        # obs = self.get_formatted_obs()
        # order_obs = self.trader.get_new_state()
        # return np.hstack([obs.values.flatten(), order_obs])
        return self.get_formatted_obs().values.flatten() 

    def reset(self): 

        self.current_index = np.random.randint(self.lookback_window + 2 + np.max(self.emas), self.data.shape[0] - (self.ep_timesteps + 1))
        self.current_ts = 0 
        self.ep_reward = 0.
        self.nb_ep_orders = 0 

        self.trader = Trader(self.data.Open[self.current_index] * np.random.uniform(0.8,1.2))
        self.random_trader = Trader(self.trader.initial_balance)
        self.baseline_trader = Trader(self.trader.initial_balance)
        

        # for i in range(self.current_index - self.lookback_window, self.current_index): 
        #     self.market_history.append(self.construct_market_state(i))
        for i in range(self.lookback_window):
            self.orders_history.append(self.trader.get_state(self.data.Open[self.current_index]))
        return self.get_obs()

    # def construct_orders_state(self, idx):
    #     return self.trader.get_state() #[self.balance, self.net_worth, self.stock_held, self.stock_sold, self.stock_bought]

    # def construct_market_state(self,idx): 
        

    

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

    def draw(self): 
        colors = [(220,0,0), (0,220,0), (0,80,220)]
        data_idx = [np.clip(self.current_index - int(self.render_window_samples * 0.5), 0, self.data.shape[0]), 
                    np.clip(self.current_index + int(self.render_window_samples * 0.5), 0, self.data.shape[0])]


        if(data_idx[0] == data_idx[1]):
            return 

        # data = self.data.drop(['Date', 'Volume'] ,axis = 1).values[data_idx[0]:data_idx[1], :]
        data = self.data[['High', 'Low', 'Open', 'Close']].iloc[data_idx[0]:data_idx[1],:].reset_index(drop = True)
        data_bounds = np.array([data.max() * 1.02, data.min()*0.98]).flatten()
        scaling_data = np.hstack([data_bounds, 
                       np.array(self.trader.net_worth_history).flatten() * 1.02, np.array(self.trader.net_worth_history).flatten() * 0.98])
        
        y_magn = (self.render_size[1] * self.graph_height_ratio) / (scaling_data.max() - scaling_data.min())
        
        volumes = self.data['Volume'].values[data_idx[0]:data_idx[1]]
        volume_magn = (self.render_size[1] * (1. - self.graph_height_ratio)) / (self.data.Volume.max() - self.data.Volume.min())

        trader_dict = {'trader':self.trader,
                       'baseline': self.baseline_trader, 
                       'random': self.random_trader}

        trader_color_dict = {'trader':(104,225,202),
                       'baseline': (241,144,202), 
                       'random': (185,33,227)}

        alpha_screen = self.screen.convert_alpha()
        env_utils.draw_grid(self.screen, self.render_size, self.candle_start_height, self.nb_y_label, alpha_screen, scaling_data, self.font)
        rect_width, x_pos_vol = env_utils.draw_candles(self.screen, self.render_size, self.candle_start_height, alpha_screen, data, scaling_data, y_magn, colors)
        env_utils.draw_agent(self.screen, self.render_size, self.candle_start_height, alpha_screen, scaling_data, y_magn, 
                             trader_dict, trader_color_dict, rect_width, colors, self.font)
        env_utils.draw_volumes(self.screen, self.render_size, self.graph_height_ratio, 
                            alpha_screen, volumes, volume_magn, x_pos_vol)
        env_utils.draw_infos(self) 
        
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
    # env = TradingEnv_State2()
    env = TradingEnv()
    # env.set_data('aapl.csv')
    # env.set_lbw(20)
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
            print('{}\n\n'.format(env.get_formatted_obs()))
            ns, r, done, info = env.step(action)
            # print(ns[-1])
            ep_reward += r
            env.render()
        ep_rewards.append(ep_reward)
        print(np.mean(ep_rewards), np.std(ep_rewards))

