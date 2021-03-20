import gym 
import trading_env 
import pandas as pd 
import numpy as np 
import time 

if __name__ == "__main__": 

    env = gym.make('Trading-v0')
    
    rewards = []
    for ep in range(10): 
        done = False 
        s = env.reset()
        ep_reward = 0. 
        counter = 0
        while not done: 

            ns, r, done, _ = env.step(np.random.randint(3))
            ep_reward += r
            time.sleep(0.001)
            env.render()
            counter += 1

        rewards.append(ep_reward)
    print(pd.Series(rewards).describe())