import gym 
import numpy as np 
import pandas as pd 
import os 
from gym import spaces 
import ta 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 

def ta_complete(df): 

    df['sma_price'] = ta.trend.sma_indicator(df['price_mw'])
    df['price_n'] = df['price_mw'] / df.sma_price -1. 
    df['prod_n'] = df['prod_mw']/df['prod_mw'].mean() -1
    df['day_n'] = MinMaxScaler().fit_transform(df[['day']])
    df['hour_n'] = MinMaxScaler().fit_transform(df[['input_hour']])

    # print(df.input_hour.max(), df.input_hour.min())
    # input(df.hour_n.describe())
        
    df = df.dropna().reset_index(drop = True)
    # print(df.head())
    return df 

class SEnv(gym.Env): 
    metadata = {'render.modes':['human']}
    
    def __init__(self, filename = None, 
                 look_back = 5, ep_length = 240, efficiency = 0.75, 
                 max_output = 0.25, max_hourly_stock = 5., 
                 max_stock = 100.): 
        super().__init__()


        if filename == None: 
            filename =  './merged_data.csv'
        data = pd.read_csv(filename)
        # print(data.columns)
        self.to_drop = ['input_hour', 'price_mw', 'prod_mw', 'sma_price', 'day']
        self.hist_cols = ['price_n', 'prod_n']
        self.data = ta_complete(data)
        self.look_back = look_back
        self.ep_length = ep_length
        self.current_ts = 0 
        self.start_idx = 0 
        self.current_stock = 0.
        self.max_stock = max_stock
        self.max_hourly_stock = max_hourly_stock
        self.max_output = max_output
        self.stock = 0. 
        self.efficiency = efficiency

        self.action_space = spaces.Box(low = -1., high = 1. , shape = (1,))
        self.observation_space = spaces.Box(low = -20., high = 20., shape = (65,))

    def reset(self, idx = None): 

        self.current_ts = 0
        if idx == None: 
            self.stock = np.random.uniform(0.,1.) * self.max_stock
            self.start_idx = np.random.randint(self.look_back +1, self.data.shape[0] - (self.ep_length+1))
        else: 
            self.stock = 0
            self.start_idx = idx

        return self.get_obs()

    def get_obs(self):

        """
        L'état est composé de: 
    
        * L'heure et le jour 
        * Les prédictions des modèles pour le prix et la météo, incluses dans le dataset 
        * Un historique de l'évolution des prix et de la prod (horizon de longueur self.look_back)
        * La quantité d'énergie stockée normalisée
        * 66 features  

        """

        current_idx = self.start_idx + self.current_ts

        d = self.data.drop(self.to_drop + self.hist_cols, axis = 1)
        s_data = list(d.iloc[current_idx,:].values.flatten())
        s_hist = []
        for c in self.hist_cols: 
            s_hist += list(self.data[c].values.flatten()[current_idx-self.look_back: current_idx])

        s_elec = [self.current_stock / self.max_stock]
        
        s = s_data + s_hist + s_elec
        
        return s 

    def step(self, action): 

        self.current_ts += 1 

        action = (np.clip(action, 0.,1.) - 0.5)*2.

        next_prod = self.data['prod_mw'][self.current_ts]
        next_price = self.data['price_mw'][self.current_ts]

        """
        Si action > 0: on charge 
        Dans ce cas, on stocke une partie de l'energie produite, le reste est vendu 
        Stocke est adapté dans les bornes 

        ============

        Si action < 0: on décharge 
        Dans ce cas, on vend une fraction de la quantité max vendable par timestep + toute la prod 
        La fraction de la quantité est affectée par le rendement.
        On prend aussi garde à conserver stock dans les bornes 


        ============

        La récompense c'est le cash obtenu par la vente 
        L'épisode se termine lorsqu'on a fait le max de timesteps 

        """

        if action >= 0.: # LOAD
            order_sell = (1. - action) * next_prod # On a ordre de vendre une partie (1-action) de la prod 
            # order_stock = np.min([action * next_prod, self.max_hourly_stock])
            order_stock = action * next_prod


            diff_max = self.max_hourly_stock - order_stock
            diff_sell = 0. if diff_max > 0. else np.abs(diff_max)

            order_stock -= diff_sell

            stock_max = self.max_stock - (self.stock + order_stock) # est-ce qu'on dépasse le max ? 
            diff_stock = 0. if stock_max > 0. else np.abs(stock_max)

            order_stock -= diff_stock

            total_sell = order_sell + diff_sell + diff_stock

            cash_from_sale = next_price * total_sell
            self.stock = np.min([self.stock + order_stock, self.max_stock])
            # cash_from_sale = (1. - action) * next_price * next_prod 
            # self.stock= np.min([self.stock + np.min([action * next_prod, self.max_hourly_stock]), self.max_stock])

        else: 
            withdrawn_energy = np.min([np.abs(action) * self.max_output * self.max_stock, self.stock])
            total_sell = next_prod + withdrawn_energy * self.efficiency 
            cash_from_sale = next_price * total_sell
            self.stock = np.clip(self.stock - withdrawn_energy, 0., self.max_stock)


        done = True if self.current_ts == self.ep_length else False 
        r = cash_from_sale
        # return self.get_obs(), 0.001 * r, done, {'stock': self.stock, 'prod': total_sell}
        return self.get_obs(), r - next_prod * next_price, done, {'stock': self.stock, 'prod': total_sell}




if __name__ == "__main__": 

    env = SEnv('../../merged_data.csv')
    s = env.reset() 

    i = 0 
    done = False 
    rewards = []
    while not done: 
        action = np.random.uniform(-1.,1.)
        ns, r, done, _ = env.step(action)
        i += 1 
        rewards.append(r)
    print(i)
    print(np.array(rewards).reshape(-1,1), np.sum(rewards), np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards))
