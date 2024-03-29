import numpy as np 
import torch 


activation_fns = {'relu' :torch.nn.ReLU,
                  'tanh': torch.nn.Tanh}

envs = {'BC_I': 'v0',
        'BC_S': 'v1',
        'BC_S2':'v2',
        'MCD': 'v3'}