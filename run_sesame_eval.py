import numpy as np 
import pandas as pd
import torch 
import stable_baselines3 as sb3 
from stable_baselines3 import PPO
import os 
import gym 
import trading_env 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import pygad
from scipy.optimize import minimize, Bounds 
import seaborn as sns 
plt.style.use('ggplot')
matplotlib.use('TKAgg')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 

current_env = gym.make('Trading-v5')
current_env.ep_length = 100

def scipy_fitness(x): 
    r = Run(GAPolicy(action_list = x))
    out = r.run(current_env, 150)
    print(out)
    return -out

def fitness_func(solution, solution_idx): 
    r = Run(GAPolicy(action_list = solution))
    return r.run(current_env, 150)
def find_optimal_policy(use_scipy = True): 

    if use_scipy: 
        x0 = np.random.uniform(-0.1,0.1, size =(current_env.ep_length,))
        res = minimize(scipy_fitness, x0, bounds = Bounds(-1.,1.), options = {'maxiter': 20})
        sol = res.x
    else: 
        num_generations = 300
        num_parents_mating = 4

        sol_per_pop = 50
        num_genes = 100

        init_range_low = -1.
        init_range_high = 1.

        parent_selection_type = "sss"
        keep_parents = 1

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 10

        ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)

        ga_instance.run()
        sol, _ ,__ = ga_instance.best_solution()
    np.savetxt('./best_sol.txt', sol.flatten())
    print(sol)
    input('saved')


class Policy: 
    def __init__(self, **kwargs): 
        return 
    def act(self, state): 
        return np.random.uniform(-1.,1.)

class GAPolicy(Policy): 
    def __init__(self, **kwargs):
        self.action_list = kwargs['action_list']
        self.idx = 0
    def act(self, state): 
        action = self.action_list[self.idx]
        self.idx = (self.idx + 1 )%len(self.action_list)
        return action 

class PPO_Trading(Policy):
    def __init__(self, **kwargs): 
        self.load(kwargs['path_to_trained'])
    def load(self, path_to_trained): 
        self.model = PPO.load(path_to_trained)
    def act(self, state): 
        action = self.model.predict(state, deterministic = True)[0]

        return action[0] if isinstance(action, list) else action

class SellPolicy(Policy): 
    def act(self, state): 
        return -1. 



class EnzoPolicy(Policy): 
    def __init__(self): 
        scaler = MinMaxScaler().fit(np.array([0.,23.]).reshape(-1,1))
        self.store_hours = list(scaler.transform(np.array([21, 22, 23,
                                                          0, 1, 2, 3,
                                                          4, 5, 6, 7, 
                                                          12, 13, 14, 
                                                          15, 16]).reshape(-1,1)).flatten())
    def act(self, state): 
        hour = state[53]
        if hour in list(self.store_hours): 
            # input('store')
            return 1.
        else: 
            # input('sell')
            return -1.
class Run: 
    def __init__(self, pol): 
        self.pol = pol 
        self.rewards = []
        self.actions = []
        self.stock = []
        self.prod = []
        self.energy_withdrawn = []
        self.penality = []
        self.cash_from_sale = []
        self.price = []

    def run(self, env, idx): 

        s = env.reset(idx)
        done = False 
        rewards = []
        actions = []
        stock = []
        prod = []
        energy_withdrawn = []
        penality = []
        cash_from_sale = []
        price = []
        while not done: 
            action = self.pol.act(s)
            s, r, done, info = env.step(action)
            # print('Stock: {:.1f} - Action: {:.1f} - Cash :{:.1f} - Total sell : {:.2f} - Prod: {:.2f}'.format(info['stock'],
            #                                                                                                  action,
            #                                                                                                  info['cash_from_sale'], 
            #                                                                                                  info['total_sell'], 
            #                                                                                                  info['prod']))
            # print('='*10)
            # print("action: {:.1f} ".format(action).upper())
            # for k in info.keys(): 
            #     print('{}: {:.2f}'.format(k.upper(), info[k]))
            # print('='*10)
            # print('\n')
            # input()
            rewards.append(r)
            actions.append(action)
            stock.append(info['stock'])
            prod.append(info['prod'])
            energy_withdrawn.append(info['withdrawn_energy'])
            penality.append(info['penality'])
            cash_from_sale.append(info['cash_from_sale'])
            price.append(info['price'])
        self.rewards.append(np.array(rewards).reshape(-1,1))
        self.actions.append(np.array(actions).reshape(-1,1))
        self.stock.append(np.array(stock).reshape(-1,1))
        self.prod.append(np.array(prod).reshape(-1,1))
        self.energy_withdrawn.append(np.array(energy_withdrawn).reshape(-1,1))
        self.penality.append(np.array(penality).reshape(-1,1))
        self.cash_from_sale.append(np.array(cash_from_sale).reshape(-1,1))
        self.price.append(np.array(price).reshape(-1,1))

        return np.sum(rewards)

    def to_csv(self): 
        n = type(self.pol).__name__
    
        pd.DataFrame(np.hstack(self.cash_from_sale), columns = ['cash_{}'.format(i) for i in range(len(self.cash_from_sale))]).to_csv('./plots/{}_{}.csv'.format(n, 'cash_from_sale'), index = False)
        
        # input(pd.read_csv('./plots/{}_{}.csv'.format(n, 'cash_from_sale')).head())
        # if n == "PPO_Trading": 
        #     pd.DataFrame(np.hstack([np.array(self.prod), np.array(self.price)]), columns = ['prod', 'price']).to_csv('./plots/{}.csv'.format('prod_price'))


if __name__ == "__main__":

    # find_optimal_policy(False)

    env_id = 'Trading-v5'

    env = gym.make(env_id)
    env_2 = gym.make(env_id.replace('5','6'))
    env.ep_length = 200
    env_2.ep_length = env.ep_length
    optim_pol = Run(GAPolicy(action_list = np.loadtxt('./best_sol.txt')))
    enzo_pol = Run(EnzoPolicy())
    model = Run(PPO_Trading(path_to_trained = './sesame_trained/sesame_12'))
    model2 = Run(PPO_Trading(path_to_trained = './sesame_trained/sesame_10'))
    # model.run(env, 150)
    sell_policy  = Run(SellPolicy())
    random_policy = Run(Policy())
    
    nb_eps = 20

    rewards = []
    actions = []
    start_idx = np.random.randint(100, 6000, size = (nb_eps, ))

    for idx in start_idx: 
        for xp in [enzo_pol, model, sell_policy, random_policy, optim_pol, model2]: 
            if xp == model2: 
                xp.run(env_2,idx)
            else: 
                xp.run(env, idx)
    # for xp in [enzo_pol, sell_policy, random_policy, optim_pol, model2]: 
    #     xp.to_csv()

    ppod_diff_recap = []
    manual_diff_recap = []
    plt.figure(figsize = (10,7))
    for i in range(nb_eps): 
        cash_from_sale2 = np.hstack(model2.cash_from_sale[i]).flatten()
        manual_cash = np.array(enzo_pol.cash_from_sale[i]).flatten()
        prod = np.array(model.prod[i]).flatten()
        price = np.array(model.price[i]).flatten()
        ppod_diff = cash_from_sale2 - prod * price
        manual_diff = manual_cash - prod * price
        ppod_diff_recap.append(np.sum(ppod_diff.flatten()))
        manual_diff_recap.append(np.sum(manual_diff.flatten()))

    data = np.vstack([np.array(ppod_diff_recap), np.array(manual_diff_recap)])
    cols = ['ep_{}'.format(i) for i in range(data.shape[1])]
    data = data.T
    data = pd.DataFrame(data, columns = ['PPO', 'Manual'])

    sns.kdeplot(data = data, fill=True, common_norm=False, palette="crest",
                alpha=.7, linewidth=0) 
    plt.title('Comparison of profits distributions after 240 timesteps for various policies', weight = 'bold')
    plt.xlabel('Profits', weight = 'bold')
    plt.ylabel('Density', weight = 'bold')
    plt.savefig('./plots/profits.png')
    print(data.describe())
    data['ppo_sup'] = data.PPO.apply(lambda x : 1. if x > 0 else 0.)
    data['manual_sup'] = data.Manual.apply(lambda x : 1. if x > 0 else 0.)
    print(data.ppo_sup.value_counts())
    print(data.manual_sup.value_counts())
    plt.show()
    input('done')



    for i in range(nb_eps): 
        f, axes = plt.subplots(3,1, figsize = (15,20))
        axes = axes.flatten()
        cash_from_sale = np.hstack(model.cash_from_sale[i]).flatten()
        cash_from_sale2 = np.hstack(model2.cash_from_sale[i]).flatten()
        cash_ga = optim_pol.cash_from_sale[i].flatten()
        manual_cash = np.array(enzo_pol.cash_from_sale[i]).flatten()
        prod = np.array(model.prod[i]).flatten()
        price = np.array(model.price[i]).flatten()
        ga_diff = cash_ga - prod * price
        ppo_diff = cash_from_sale - prod * price
        ppod_diff = cash_from_sale2 - prod * price
        manual_diff = manual_cash - prod * price
        rewards = np.array(model.rewards[i])

        axes[0].plot(prod * price, label = 'Direct sell')
        axes[0].plot(cash_from_sale, label = 'PPO')
        axes[0].plot(cash_from_sale2, label = 'PPOD')
        axes[0].plot(cash_ga, label = 'GA')
        axes[0].plot(manual_cash, label = 'Manual')
        axes[0].legend()
        axes[0].set_title("Direct vs Stock",weight = 'bold')
        axes[1].bar(np.arange(env.ep_length), ppo_diff, label = 'PPO')
        axes[1].bar(np.arange(env.ep_length), ga_diff, label = 'GA', alpha = 0.8)
        axes[1].bar(np.arange(env.ep_length), ppod_diff, label = 'PPOD', alpha = 0.6)
        axes[1].bar(np.arange(env.ep_length), manual_diff, label = 'Manual', alpha = 0.6)

        axes[1].legend()
        axes[1].set_title("PPO Continuous: {:.2f}, PPO Discrete: {:.2f} - GA: {:.2f} - Manual: {:.2f}".format(np.sum(ppo_diff),np.sum(ppod_diff), np.sum(ga_diff), np.sum(manual_diff)), weight = 'bold')

        
        
        axes[2].bar(np.arange(env.ep_length), np.cumsum(ppo_diff), label = 'PPO')
        axes[2].bar(np.arange(env.ep_length), np.cumsum(ga_diff), label = 'GA', alpha =0.8)
        axes[2].bar(np.arange(env.ep_length), np.cumsum(ppod_diff), label = 'PPOD', alpha =0.5)
        axes[2].bar(np.arange(env.ep_length), np.cumsum(manual_diff), label = 'Manual', alpha =0.3)
        axes[2].legend()
        axes[2].set_title("Reward cumsum: PPO {:.2f} Manual: {:.2f}".format(np.cumsum(ppod_diff)[-1], np.sum(manual_diff)), weight = 'bold')
        # plt.suptitle('Reward - diff: {}'.format(np.sum(cash_from_sale - prod * price - rewards)))


        plt.show()
        plt.close()


#     for i in range(nb_eps): 
#         f, axes = plt.subplots(4,1, figsize = (15,20))
#         axes = axes.flatten()
#         axes[0].plot(np.cumsum(enzo_pol.cash_from_sale[i]), label = 'Enzo')
#         axes[0].plot(np.cumsum(model.cash_from_sale[i]), label = 'PPO')
#         axes[0].plot(np.cumsum(model2.cash_from_sale[i]), label = 'PPO_Discrete')
#         axes[0].plot(np.cumsum(sell_policy.cash_from_sale[i]), label = 'Sell')
#         axes[0].plot(np.cumsum(random_policy.cash_from_sale[i]), label = 'Random')
#         axes[0].plot(np.cumsum(optim_pol.cash_from_sale[i]), label = 'GA')
#         axes[0].set_title('Cumulative reward: {}'.format(start_idx[i]), weight = 'bold')
#         axes[0].legend()

#         axes[1].plot(enzo_pol.stock[i], label = 'Enzo')
#         axes[1].plot(model.stock[i], label = 'PPO')
#         axes[1].plot(model2.stock[i], label = 'PPO_Discrete')
#         axes[1].plot(sell_policy.stock[i], label = 'Sell')
#         axes[1].plot(random_policy.stock[i], label = 'Random')
#         axes[1].plot(optim_pol.stock[i], label = 'GA')
#         axes[1].set_title('Stock: {}'.format(start_idx[i]), weight = 'bold')
#         axes[1].legend()

#         # axes[2].plot(np.cumsum(enzo_pol.prod[i]), label = 'Enzo')
#         # axes[2].plot(np.cumsum(model.prod[i]), label = 'PPO')
#         # axes[2].plot(np.cumsum(sell_policy.prod[i]), label = 'Sell')
#         # axes[2].plot(np.cumsum(random_policy.prod[i]), label = 'Random')
#         # axes[2].set_title('Prod: {}'.format(start_idx[i]), weight = 'bold')
#         # axes[2].legend()

#         axes[2].hist(model.actions[i], label = 'Ep: {}'.format(i), bins = 20, range =(-1.,1.))
#         axes[2].legend()

#         axes[3].hist(model.energy_withdrawn[i], label = 'Ep: {}'.format(i), bins = 20, range =(-1.,1.))
#         axes[3].set_title("Energy withdrawn", weight = 'bold')

#         # axes[5].bar(np.arange(len(model.penality[i])), model.penality[i])
#         # axes[5].set_title("Penalities", weight = 'bold')
# # 
#         # f.tight_layout()

#         # axes[6].hist(model.energy_withdrawn[i], label = 'Ep: {}'.format(i), bins = 20, range =(-1.,1.))
#         # axes[6].set_title("Energy withdrawn", weight = 'bold')

#         plt.show()
#         plt.close()


