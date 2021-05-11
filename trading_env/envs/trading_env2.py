import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import ta 
import os 
import glob 
import gym 
from sklearn.decomposition import PCA
plt.style.use('ggplot')

def get_data(filename):

    current_path = os.path.realpath(__file__).split('/')[:-1]
    path = os.path.join(*current_path)
    path = os.path.join('/', path, filename)
    data = pd.read_csv(path)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna().reset_index(drop = True)




class TradingEnvF(gym.Env): 

    def __init__(self): 

        self.ep_length = 200
        self.max_sma = 100 
        self.obs_window = 4
        self.construct_window = self.ep_length + self.max_sma
        self.cum_windows = [10,20,30,50,60]

        self.set_data('btc.csv')

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(shape = self.reset().shape, low = -50., high = 50.)

    def set_data(self, filename): 
        self.data = get_data(filename) 
        self.pca = PCA(n_components = 40)


        close = np.zeros((self.data.shape[0] - self.construct_window,self.construct_window))
        volume = np.zeros((self.data.shape[0] - self.construct_window,self.construct_window))
        for i in range(self.data.shape[0] - self.construct_window): 
            close[i] = self.data.Close[i:i + self.construct_window].values
            volume[i] = self.data.Volume[i:i + self.construct_window].values
        self.pca_data = np.hstack([close, volume])
        self.pca_data_transformed = self.pca.fit_transform(self.pca_data)


    def generate_data(self): 

        alpha = np.random.uniform(0.,1.)
        interp_extremes = np.random.randint(0, self.pca_data.shape[0], size = (2))
        t0 = self.pca_data_transformed[interp_extremes[0]]
        t1 = self.pca_data_transformed[interp_extremes[1]]
        t0_r = self.pca.inverse_transform(t0)
        t1_r = self.pca.inverse_transform(t1)
        lerped = (alpha * t0_r + (1. - alpha) * t1_r).flatten()

        close = lerped[:int(lerped.shape[0]/2)]
        volume = lerped[int(lerped.shape[0]/2):]


        stds_close = []
        stds_vol = []
        for idx in interp_extremes: 
            stds_close.append(np.std(self.pca_data[idx,:int(self.pca_data.shape[1]/2)])**0.5)
            stds_vol.append(np.std(self.pca_data[idx,int(self.pca_data.shape[1]/2):])**0.5)
        
        std_close = stds_close[0] * alpha + (1.- alpha) * stds_close[1]
        std_vol = stds_vol[0] * alpha + (1.- alpha) * stds_vol[1]

        std_lerped = np.mean(np.sqrt(np.std(np.vstack([t0.reshape(1,-1), t1.reshape(1,-1)]), axis = 1)))
        lerped_r = alpha * t0_r + (1. - alpha) * t1_r

        noisy_close = close + self.get_brownian_noise(std_close)
        noisy_vol = (volume + self.get_brownian_noise(std_vol)).clip(0., np.inf)

        episode_data = pd.DataFrame(np.hstack([noisy_close.reshape(-1,1), noisy_vol.reshape(-1,1)]), columns = ['Close', 'Volume'])

        return episode_data

    def get_brownian_noise(self, std): 
        cumulative_window = np.random.choice(self.cum_windows, p = np.ones(len(self.cum_windows)) / float(len(self.cum_windows)))
        returns = np.random.normal(0., std * np.random.uniform(0.8,1.5), 
                                  size=(self.construct_window))
        returns_clamped = np.concatenate([
            np.cumsum(c) for c in np.split(returns, returns.shape[0]//cumulative_window)])
        return returns_clamped

    def augment_data(self, episode_data): 
        episode_data['rsi'] = ta.momentum.RSIIndicator(close = episode_data.Close).rsi() * 0.01
        episode_data['bb'] = ta.volatility.BollingerBands(close = episode_data.Close).bollinger_pband()
        episode_data['vpt'] = ta.volume.VolumePriceTrendIndicator(close = episode_data.Close, 
                                        volume = episode_data.Volume).volume_price_trend()
        episode_data['sma5'] = ta.trend.SMAIndicator(window = 5, close = episode_data.Close).sma_indicator()
        episode_data['sma20'] = ta.trend.SMAIndicator(window = 20, close = episode_data.Close).sma_indicator()
        episode_data['sma50'] = ta.trend.SMAIndicator(window = 50, close = episode_data.Close).sma_indicator()
        episode_data['close5'] = episode_data.Close / episode_data.sma5
        episode_data['close20'] = episode_data.Close / episode_data.sma20
        episode_data['close50'] = episode_data.Close / episode_data.sma50
        episode_data['sma_fast'] = episode_data.sma5 / episode_data.sma20
        episode_data['sma_slow'] = episode_data.sma20 / episode_data.sma50

        return episode_data

    def step(self, action): 

        self.episode_data['close'].append(self.current_data.Close[self.ep_idx])
        self.episode_data['volume'].append(self.current_data.Volume[self.ep_idx])
        self.episode_data['balance'].append(self.balance)
        self.episode_data['stocks'].append(self.stocks)
        self.episode_data['net_worth'].append(self.net_worth)
        
        done = False
        reward = 0 
        order = 0

        next_price = self.current_data.Close[self.ep_idx + 1]
        if action == 0: # BUY
            if self.balance > 0: 
                self.stocks = self.balance / next_price
                self.balance = 0
                order = 1
        else: # SELL 
            if self.stocks > 0: 
                self.balance = next_price * self.stocks
                self.stocks = 0 
                order = -1



        self.net_worth = self.balance + self.stocks * next_price
        reward = self.net_worth - self.prev_net_worth 
        self.prev_net_worth = self.net_worth


        self.episode_data['orders'].append(order)
        self.episode_data['rewards'].append(reward)

        self.ep_idx += 1 
        if self.ep_idx >= self.ep_length + self.max_sma -1 : 
            done = True 
        if self.net_worth < 0.5 * self.initial_balance: 
            done = True 

        return self.get_obs(), reward, done, {}

    def get_obs(self): 
        
        obs = self.current_data.loc[self.ep_idx - self.obs_window: self.ep_idx,self.features].values.flatten()
        obs = np.hstack([obs, np.array([self.balance, self.stocks])])

        return obs 
    def reset(self): 

        self.episode_data = {'close':[], 
                             'volume':[],
                             'balance':[],
                             'stocks':[],  
                             'net_worth':[],
                             'orders':[],
                             'rewards':[]}

        episode_data = self.generate_data()
        self.current_data = self.augment_data(episode_data)
        self.features = self.current_data.drop(['Close', 'Volume', 'sma5', 'sma20', 'sma50'], axis = 1).columns
        self.ep_idx = self.max_sma - 1
        self.balance = self.current_data.Close[self.ep_idx] * np.random.uniform(0.8,1.2)
        self.initial_balance = self.balance
        self.stocks = 0
        self.net_worth = self.balance
        self.prev_net_worth = self.net_worth

        obs = self.get_obs()  
        
        return obs


    def render(self): 

        f, axes = plt.subplots(2,1, figsize = (15,9))
        axes = axes.flatten()

        axes[0].plot(self.episode_data['close'], label = 'Close', linewidth = 3)
        axes[0].plot(self.episode_data['net_worth'], label = 'NetWorth')
        for i,o in enumerate(self.episode_data['orders']): 
            if o == 1: 
                axes[0].scatter(i,self.episode_data['net_worth'][i], color = 'green', marker = '^')
            elif o == -1: 
                axes[0].scatter(i,self.episode_data['net_worth'][i], color = 'red', marker = 'v')
        axes[0].legend()
        stocks = pd.DataFrame(self.episode_data['stocks'], columns = ['stock_count'])
        stocks = stocks[stocks['stock_count'] > 0]
        axes[1].plot(stocks.index, stocks)
        axes[1].scatter(stocks.index, stocks)


        # axes[1].scatter(np.arange(len(self.episode_data['stocks'])), self.episode_data['stocks'], label = 'Stocks')
        plt.pause(0.1)
        input('Press enter to quit viz')

if __name__ == '__main__': 

    env = TradingEnvF()
    env.set_data('btc.csv')
    done = False 
    s = env.reset()
    while not done:   
        action = 1. if np.random.uniform(0.,1.) > 0.9 else 0. #env.action_space.sample()
        ns, r, done, info = env.step(action)

    env.render()

