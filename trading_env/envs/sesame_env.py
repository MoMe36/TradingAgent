import gym 
import numpy as np 
import pandas as pd 
import os 
from gym import spaces 
import ta 
from sklearn.preprocessing import MinMaxScaler 


def ta_complete(df): 

    df['sma_price'] = ta.trend.sma_indicator(df['price_mw'])
    df['price_n'] = df['price_mw'] / df.sma_price -1. 
    df['prod_n'] = df['prod_mw']/df['prod_mw'].mean() -1
    df['day_n'] = MinMaxScaler().fit_transform(df[['day']])
    df['hour_n'] = MinMaxScaler().fit_transform(df[['input_hour']])
        
    df = df.dropna().reset_index(drop = True)
    # print(df.head())
    return df 

class SEnv(gym.Env): 
    metadata = {'render.modes':['human']}
    
    def __init__(self, filename = None, 
                 look_back = 5, ep_length = 240, efficiency = 0.7, max_output = 0.25): 
        super().__init__()


        if filename == None: 
            filename =  './merged_data.csv'
        data = pd.read_csv(filename)
        # print(data.columns)
        self.to_drop = ['input_hour', 'price_mw', 'prod_mw', 'sma_price']
        self.hist_cols = ['price_n', 'prod_n']
        self.data = ta_complete(data)
        self.look_back = look_back
        self.ep_length = ep_length
        self.current_ts = 0 
        self.start_idx = 0 
        self.current_stock = 0.
        self.max_stock = 30.
        self.max_output = max_output
        self.stock = 0. 
        self.efficiency = efficiency

        self.action_space = spaces.Box(low = -1., high = 1. , shape = (1,))
        self.observation_space = spaces.Box(low = -20., high = 20., shape = (66,))

    def reset(self): 

        self.current_ts = 0
        self.stock = 0. 
        self.start_idx = np.random.randint(self.look_back +1, self.data.shape[0] - (self.ep_length+1))

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
            cash_from_sale = (1. - action) * next_price * next_prod 
            self.stock = np.min([self.max_stock + action * self.efficiency * next_prod, self.max_stock])
        else: 
            withdrawn_energy = np.min([np.abs(action) * self.max_output * self.max_stock, self.stock])
            energy_to_sell = next_prod + withdrawn_energy * self.efficiency 
            cash_from_sale = next_price * energy_to_sell
            self.stock = np.max([self.stock - withdrawn_energy, 0.])


        done = True if self.current_ts == self.ep_length else False 
        r = cash_from_sale
        return self.get_obs(), r, done, {}




if __name__ == "__main__": 

    env = SEnv('../../merged_data.csv')
    s = env.reset() 

    i = 0 
    done = False 
    while not done: 
        action = np.random.uniform(-1.,1.)
        ns, r, done, _ = env.step(action)
        i += 1 
    print(i)
