import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import pandas as pd 
plt.style.use('ggplot')
matplotlib.use('TKAgg')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 
import glob
import seaborn as sns 

if __name__ == "__main__": 

    files = sorted(glob.glob('*.csv'))
    files = [f for f in files if not 'run' in f]
    
    # f, axes = plt.subplots(1,2, figsize = (8,8))
    # axes = axes.flatten()

    eps = []
    cols = []
    for f in files: 
        if 'cash' in f:
            df = pd.read_csv(f)
            eps.append(df.sum(0).values.reshape(-1,1))
            cols.append(f.split('_')[0])
            sns.kdeplot(df.sum(0).values.flatten())
            # plt.hist(df.sum(0).values.flatten(), label = cols[-1], bins = 50)

    eps = np.hstack(eps)
    # print(eps.shape)
    # plt.hist(eps)
    plt.legend()
    plt.show()

    # print(np.vstack(eps).T)
    # sns.histplot(pd.DataFrame(np.vstack(eps).T, columns = cols))
    


    # for value in values: 
    #     plt.plot(value, alpha = 0.2)

    # plt.figure(figsize =(10,5))
    # vals = np.hstack([v.reshape(-1,1) for v in values])
    # c= plt.plot(vals.max(axis = 1), alpha = 0.8)
    # plt.plot(vals.min(axis = 1), color = c[0].get_color(), alpha = 0.8)
    # plt.plot(0.5 * (vals.min(axis = 1) + vals.max(axis = 1)), color = 'k', label = 'mean')
    # plt.fill_between(np.arange(vals.shape[0]), vals.max(axis = 1), vals.min(axis = 1), color = c[0].get_color())
    # plt.title("Reward per episode distribution during training", weight = 'bold')
    # plt.xlabel('Timesteps', weight = 'bold')
    # plt.ylabel('Episode reward', weight = 'bold')
    # plt.legend()
    # plt.savefig('./rewards.png')
    plt.show()