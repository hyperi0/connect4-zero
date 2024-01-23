
from torch import tensor
from collections import OrderedDict

import math
from connectx import *
import numpy as np

class MCTS():
    def __init__(self, config, policy, c_puct):
        self.config = config
        self.policy = policy
        self.c_puct = c_puct
        self.visited = []
        self.P = {}
        self.N = {}
        self.Q = {}

    def u_value(self, s, a):
        normalized_counts = math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
        return self.Q[s][a] + self.c_puct * self.P[s][a] * normalized_counts
    
    def search(self, s):
        if is_terminal_grid(s, self.config):
            return -score_game(s, self.config)
        
        # first visit: initialize to neural net's predicted action probs and value
        if s not in self.visited:
            self.visited.append(s)
            pi, v = self.policy.predict(s)
            legal_mask = legal_moves_mask(s, self.config)
            self.P[s] = [p * mask for p, mask in zip(pi, legal_mask)]
            self.N[s] = [0 for _ in range(self.config.columns)]
            self.Q[s] = [0 for _ in range(self.config.columns)]
            return -v
        
        # choose action with best upper confidence bound
        max_u, best_a = -np.inf, -1
        for a in legal_moves(s, self.config):
            u = self.u_value(s, a)
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        # calculate state value recursively
        next_s = drop_piece(s, a, 1, self.config)
        next_s = reverse_grid(next_s)
        v = self.search(next_s)

        # update node values from recursive search results
        self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + v) / (1 + self.N[s][a])
        self.N[s][a] += 1
        return -v
    
    def pi(self, s):
        total_counts = sum(self.N[s])
        return [self.N[s][a] / total_counts for a in range(self.config.columns)]
    
    def best_action(self, s):
        return np.argmax(self.N[s])
    
    def stochastic_action(self, s):
        return np.random.choice(self.config.columns, p=self.pi(s))

import numpy as np
from collections import deque
from connectx import *
from mcts import MCTS
from nnet_torch import Policy
from tqdm import trange
import time

class ConnectXAgent():
    def __init__(
            self,
            config,
            n_sims_train=10,
            c_puct=1,
            device='cuda',
    ):
        self.config = config
        self.n_sims_train = n_sims_train
        self.c_puct = c_puct
        self.policy = Policy(device)
        self.tree = MCTS(self.config, self.policy, self.c_puct)
        self.examples = []

    def train(self, n_iters=10, n_eps=100, max_memory=1000):
        examples = deque(maxlen=max_memory)
        for i in range(n_iters):
            with trange(n_eps, unit='eps') as pbar:
                pbar.set_description(f'Iteration {i} of {n_iters}')
                for e in pbar:
                    examples.extend(self.execute_episode())
            self.policy.train(examples)
        
    def execute_episode(self):
        examples = []
        s = empty_grid(self.config)
        tree = MCTS(self.config, self.policy, self.c_puct)
        mark = 1

        while True:
            for _ in range(self.n_sims_train):
                tree.search(s)
            action_probs = tree.pi(s)
            examples.append([s, action_probs])
            a = np.random.choice(len(action_probs), p=action_probs)
            s = drop_piece(s, a, 1, self.config)
            # backup scores on game end
            if is_terminal_grid(s, self.config):
                reward = score_game(s, self.config)
                for ex in examples:
                    ex.append(reward * mark)
                    mark *= -1
                self.examples = examples
                return examples
            else: # swap board perspective
                s = reverse_grid(s)
                mark *= -1

    def choose_move(self, board, mark, max_time=1.5, deterministic=True):
        board = [-1 if token == 2 else token for token in board]
        board = np.reshape(board, (self.config.rows, self.config.columns))
        if mark == 2:
            board = -board
        s = tuple(map(tuple, board))
        start_time = time.time()
        while time.time() - start_time < max_time:
            self.tree.search(s)
        if deterministic:
            return self.tree.best_action(s)
        else:
            return self.tree.stochastic_action(s)
        
    def move_probabilities_values(self, board, mark, n_sims=10):
        board = [-1 if token == 2 else token for token in board]
        board = np.reshape(board, (self.config.rows, self.config.columns))
        if mark == 2:
            board = -board
        s = tuple(map(tuple, board))
        for _ in range(n_sims):
            self.tree.search(s)
        info = {
            "P": self.tree.P[s],
            "Q": self.tree.Q[s],
            "N": self.tree.N[s],
            "pi": self.tree.pi(s)
        }
        return info

    def load_policy(self, state_dict):
        self.policy.nnet.load_state_dict(state_dict)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
    
class PolicyNet(nn.Module):
    def __init__(self, rows=6, columns=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc = nn.Linear(32 * (columns - 4) * (rows - 4), 64)
        self.pi_fc = nn.Linear(64, columns)
        self.v = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        pi = F.log_softmax(self.pi_fc(x), dim=1)
        v = torch.tanh(self.v(x))
        return pi, v
    
class ExperienceDataset(Dataset):
    def __init__(self, examples, device='cpu'):
        s, pi, v = zip(*examples)
        self.s = torch.tensor(s, dtype=torch.float).unsqueeze(1).to(device) # single channel input for convnet
        self.pi = torch.tensor(pi, dtype=torch.float).to(device)
        self.v = torch.tensor(v, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, i):
        return {'s': self.s[i], 'pi': self.pi[i], 'v': self.v[i]}

class Policy():
    def __init__(self, device='cpu'):
        self.device = device
        self.nnet = PolicyNet().to(device)
    
    def loss_fn_pi(self, pi_target, pi_pred):
        return -torch.sum(pi_target * pi_pred) / pi_target.size()[0]
    
    def train(self, examples, batch_size=64, epochs=5, lr=1e-2):
        train_dataset = ExperienceDataset(examples, device=self.device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        loss_fn_v = torch.nn.MSELoss()
        for i in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                s, pi, v = batch['s'], batch['pi'], batch['v']
                pi_pred, v_pred = self.nnet(s)
                loss = self.loss_fn_pi(pi, pi_pred) + loss_fn_v(v_pred.squeeze(), v)
                loss.backward()
                optimizer.step()

    def predict(self, s):
        input = torch.tensor(s, dtype=torch.float).reshape((1,1,6,7)).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(input)
            return torch.exp(pi)[0].tolist(), v[0].item()

import random
import numpy as np

### Generate new agent function using model
def build_agent(model):
    def _function(obs, config):
        # Use the best model to select a column
        col, _ = model.predict(np.array(obs['board']).reshape(1, config.rows, config.columns))
        # Check if selected column is valid
        is_valid = (obs['board'][int(col)] == 0)
        # If not valid, select random move. 
        if is_valid:
            return int(col)
        else:
            return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

    return _function

# Get new board with given piece dropped
def drop_piece(grid, col, mark, config):
    next_grid = np.asarray(grid)
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return tuple(map(tuple, next_grid))

def legal_moves(grid, config):
    return [col for col in range(config.columns) if grid[0][col] == 0]

def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(-1) == config.inarow

def is_terminal_grid(grid, config):
    return check_winner(grid, config) != 0

def check_window_winner(window, config):
    if window.count(1) == config.inarow:
        return 1
    if window.count(-1) == config.inarow:
        return -1
    else:
        raise Exception("Winner not found")
    
def check_winner(grid, config):
    np_grid = np.asarray(grid)
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(np_grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # Check for draw 
    if list(np_grid[0, :]).count(0) == 0:
        return 3
    
    return 0

def score_game(grid, config):
    np_grid = np.asarray(grid)
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(np_grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # Check for draw 
    if list(np_grid[0, :]).count(0) == 0:
        return 0
    
    return 0

def empty_grid(config):
    return tuple(map(tuple, np.zeros((config.rows, config.columns), dtype=int)))

def legal_moves_mask(grid, config):
    moves = legal_moves(grid, config)
    mask = [1 if i in moves else 0 for i in range(config.columns)]
    return mask

def reverse_grid(grid):
    return tuple(tuple(-piece for piece in row) for row in grid)

state_dict = OrderedDict([('conv1.weight', tensor([[[[-1.6688e-01, -2.7829e-01,  1.8371e-01],
          [ 3.1041e-01,  8.7124e-02,  1.3298e-01],
          [ 6.4617e-02,  8.1658e-02, -2.2440e-01]]],


        [[[-7.5015e-02, -8.3385e-02,  1.5270e-01],
          [-3.8883e-01,  1.1110e-01, -2.0112e-01],
          [-2.4376e-01, -2.0044e-01, -1.0296e-01]]],


        [[[-3.4700e-01, -1.1523e-01, -4.6128e-02],
          [ 2.0625e-01,  1.8828e-01, -3.8106e-01],
          [ 3.1566e-04,  8.4213e-02, -3.7357e-02]]],


        [[[ 1.4870e-01,  6.9143e-02,  2.3696e-01],
          [-1.1273e-01, -2.2863e-01, -2.4046e-02],
          [-2.5167e-01, -3.7740e-01, -1.3708e-01]]],


        [[[ 5.7782e-02,  1.7664e-02,  1.9084e-01],
          [ 2.0442e-01, -1.1656e-01,  2.0907e-01],
          [ 1.0629e-01, -2.8460e-01,  1.9668e-01]]],


        [[[ 7.5975e-02,  2.7900e-01,  3.5451e-01],
          [-2.9543e-01, -1.9273e-01,  6.2651e-02],
          [-1.5291e-01,  1.5749e-01,  2.8444e-01]]],


        [[[-2.6064e-01, -1.3592e-01, -6.1021e-02],
          [ 1.2362e-01,  1.5104e-02, -1.1616e-02],
          [ 1.0785e-01, -2.9917e-02, -3.9060e-01]]],


        [[[-2.9239e-01,  2.7685e-01, -2.2914e-01],
          [-6.9445e-02,  2.2757e-01, -3.0882e-01],
          [-2.5945e-01, -3.0677e-01, -3.5958e-01]]],


        [[[ 2.6069e-01, -2.2579e-01, -3.1423e-01],
          [-1.0717e-01,  3.1079e-01, -1.5124e-01],
          [-1.7589e-01,  1.0791e-01, -3.0195e-01]]],


        [[[-8.1809e-02, -2.9326e-01, -1.9472e-01],
          [ 5.5152e-03, -5.2491e-03,  9.3603e-02],
          [ 2.1319e-01,  4.2851e-02, -3.0364e-01]]],


        [[[ 2.0448e-04,  2.9916e-01, -9.4009e-02],
          [ 6.7259e-02, -1.4598e-01, -4.7946e-02],
          [ 2.3746e-01,  1.5485e-01,  6.1860e-03]]],


        [[[-2.9478e-01,  2.3739e-01,  3.1811e-01],
          [-3.1801e-01, -2.6739e-01, -2.9408e-01],
          [-8.9321e-02,  2.3796e-01,  8.2713e-02]]],


        [[[-2.1643e-01, -1.7326e-01, -5.8219e-02],
          [-1.0453e-01,  3.2737e-01, -2.1100e-01],
          [ 2.7159e-01, -3.0002e-01, -4.1931e-02]]],


        [[[-3.3038e-01, -2.1225e-01, -1.0734e-01],
          [-4.7545e-02, -2.3534e-01, -2.2185e-01],
          [ 2.8326e-01,  1.4831e-01,  3.3077e-01]]],


        [[[ 2.4142e-01, -1.6173e-02,  2.6237e-01],
          [ 1.4854e-01,  5.2987e-02,  1.6492e-01],
          [-3.1219e-01,  8.1088e-02,  3.2328e-01]]],


        [[[ 2.5231e-01,  7.8709e-03,  1.5805e-03],
          [-1.4629e-01,  1.8783e-01, -2.0686e-01],
          [-2.5998e-01,  3.9508e-01,  4.6150e-02]]],


        [[[-2.8118e-01, -3.2052e-01, -5.6527e-02],
          [-1.3191e-01, -3.1030e-01, -2.6705e-01],
          [ 1.4293e-01,  3.1624e-01,  3.0805e-01]]],


        [[[-1.3442e-01, -8.7406e-02,  8.9781e-03],
          [ 1.8806e-02, -8.6340e-02, -3.6641e-01],
          [ 2.4193e-01,  2.4046e-01, -3.5978e-01]]],


        [[[-2.5171e-01, -2.9836e-01,  1.2042e-01],
          [-8.0644e-02,  1.7547e-01,  2.9891e-02],
          [ 1.3219e-01, -3.6585e-01,  3.2288e-01]]],


        [[[-3.5415e-01,  6.3747e-02,  2.5029e-01],
          [ 1.4269e-01, -7.3615e-03, -2.5853e-01],
          [ 2.3782e-01,  2.0944e-01,  2.0010e-01]]],


        [[[ 8.0113e-02,  2.6966e-01,  8.0416e-02],
          [-1.2361e-01,  6.6346e-02,  7.2232e-02],
          [ 3.2928e-01,  2.1081e-01,  1.8434e-01]]],


        [[[-2.5593e-01,  3.0712e-01,  1.0653e-01],
          [ 2.9742e-01, -3.4879e-01,  3.0827e-01],
          [-3.1552e-02,  1.3271e-01,  2.9560e-01]]],


        [[[ 2.3943e-01, -7.5375e-02, -1.5610e-01],
          [ 3.8928e-01, -8.0115e-02, -4.1765e-02],
          [ 1.5354e-01,  6.6933e-02, -2.6472e-02]]],


        [[[ 2.2635e-01,  1.2778e-01,  1.5772e-02],
          [-9.4131e-02,  6.9669e-02,  2.0658e-01],
          [ 3.7320e-01,  1.0479e-01,  1.8324e-01]]],


        [[[-6.8034e-02,  2.4883e-01, -6.2145e-02],
          [-1.2771e-01,  2.4200e-01, -2.2832e-01],
          [ 3.7957e-01,  5.7893e-02, -1.2161e-01]]],


        [[[-9.4633e-02,  1.9290e-01, -1.3509e-01],
          [ 3.4079e-01, -1.2623e-01,  1.2583e-02],
          [-2.6734e-01, -1.7231e-01,  3.4417e-01]]],


        [[[-2.1853e-01,  3.2460e-01,  3.3166e-02],
          [-1.2003e-01, -3.3760e-01,  6.5162e-02],
          [ 2.6825e-01,  1.4892e-02,  1.6907e-01]]],


        [[[-2.7898e-01, -2.3237e-02,  3.7865e-01],
          [ 3.1392e-01, -3.1746e-01,  3.0160e-01],
          [ 3.0519e-01, -8.8431e-02,  2.9922e-02]]],


        [[[ 9.7321e-02, -1.1526e-01,  2.0824e-01],
          [-1.5732e-01, -2.6126e-01, -1.4181e-01],
          [ 1.3670e-01,  1.6336e-01,  2.4497e-01]]],


        [[[-1.1748e-01, -4.6061e-02,  1.9256e-01],
          [ 3.0804e-01,  1.6896e-01,  1.2176e-01],
          [-9.4467e-02,  9.9395e-02,  2.1678e-01]]],


        [[[ 2.6883e-01, -3.0472e-01,  9.9240e-02],
          [-1.7883e-01, -3.4113e-01, -5.2743e-03],
          [ 1.4561e-01,  1.8019e-01,  1.8664e-01]]],


        [[[ 2.7183e-01, -1.0577e-01,  1.8967e-01],
          [-1.2081e-02,  3.3169e-01, -5.5392e-02],
          [-8.5495e-02,  2.8549e-01, -1.8502e-01]]]], device='cuda:0')), ('conv1.bias', tensor([-0.2159,  0.1642, -0.2221, -0.1680,  0.2660, -0.0931, -0.0301,  0.0147,
        -0.2186,  0.0050, -0.0208,  0.0527, -0.1017, -0.1474,  0.2003,  0.1985,
        -0.0353, -0.2189,  0.1848,  0.0009, -0.2054, -0.0341, -0.2092,  0.2458,
        -0.0809,  0.1018,  0.0478, -0.0510,  0.0508,  0.2371,  0.2050, -0.0599],
       device='cuda:0')), ('conv2.weight', tensor([[[[-4.9726e-02,  1.4035e-02,  1.1783e-02],
          [-4.3484e-02, -3.2050e-02, -1.7138e-02],
          [-2.6976e-03,  2.2965e-02,  3.4009e-02]],

         [[ 3.1681e-02, -3.1375e-02,  4.9750e-02],
          [-3.5319e-02,  2.4782e-02, -2.8038e-02],
          [ 1.5773e-02,  5.2346e-02, -2.0031e-02]],

         [[-4.9085e-02,  1.2972e-02,  1.8548e-02],
          [-5.6224e-02, -4.9181e-02, -3.8335e-02],
          [-3.7156e-02,  6.5654e-03,  5.4540e-02]],

         [[-4.9718e-02,  2.9057e-02, -2.7071e-02],
          [ 5.0499e-04,  2.9323e-02,  4.8271e-03],
          [-4.2187e-02, -4.7516e-02, -5.7701e-02]],

         [[-8.2275e-03, -4.3635e-02, -2.1015e-02],
          [ 1.0397e-02, -2.0034e-02,  1.4002e-02],
          [-5.0402e-02, -9.6229e-03,  9.8732e-04]],

         [[-1.1542e-02,  3.4075e-02,  2.9670e-03],
          [ 3.1821e-02, -3.3183e-03, -3.1709e-02],
          [-2.3088e-02, -5.4864e-02,  4.3637e-02]],

         [[-1.2490e-02, -3.3436e-02,  8.2578e-03],
          [ 5.5726e-02, -2.2751e-02,  7.1930e-03],
          [-2.8192e-02,  9.8828e-03, -2.2778e-02]],

         [[-2.3795e-02,  4.9540e-02, -1.8193e-02],
          [-5.5535e-02,  5.0987e-02, -3.6722e-02],
          [-5.1166e-02, -3.0349e-02, -5.2074e-02]],

         [[-3.2328e-02, -1.5839e-02, -5.8888e-02],
          [-4.3711e-02,  3.4585e-02, -3.6303e-02],
          [ 1.5189e-02,  4.4519e-03, -5.7590e-02]],

         [[ 2.8803e-02, -9.6532e-03,  2.4666e-02],
          [-3.2041e-03, -2.3640e-02,  2.2849e-02],
          [-3.5390e-03,  5.6900e-02, -2.9213e-03]],

         [[-3.3444e-02,  1.1310e-02,  1.1041e-02],
          [ 5.8251e-02,  4.0527e-03,  1.5128e-02],
          [-2.0895e-02, -1.3077e-02, -4.9100e-02]],

         [[-2.1344e-02,  5.4216e-02, -2.8204e-02],
          [ 8.7543e-03,  5.2169e-02, -1.8805e-02],
          [-1.4686e-02, -1.9250e-02,  5.4566e-02]],

         [[-2.7181e-02,  4.3195e-03, -3.8190e-02],
          [-2.6352e-02,  5.6525e-03, -2.3527e-02],
          [-4.9010e-02,  3.4062e-02, -7.9202e-03]],

         [[-3.0991e-02, -1.3364e-02,  4.6850e-02],
          [ 5.7573e-02, -1.4778e-02,  5.4982e-02],
          [ 8.4038e-04,  5.7154e-02,  4.1445e-03]],

         [[ 1.2295e-03,  1.2471e-02,  1.1935e-02],
          [ 2.3307e-02,  1.1767e-02, -4.6683e-02],
          [-2.3290e-02, -1.7393e-02, -2.0954e-02]],

         [[ 1.6256e-02, -3.0083e-02,  9.1604e-03],
          [-5.1769e-02,  2.8816e-04, -5.7699e-02],
          [ 1.0282e-02, -2.6570e-02, -4.3569e-02]],

         [[-3.9067e-02, -9.5151e-03, -2.4314e-02],
          [-1.1003e-02,  2.8928e-02, -2.2397e-02],
          [-2.6492e-02, -3.2240e-02, -3.3895e-02]],

         [[ 2.7388e-02, -3.8943e-02, -3.1669e-02],
          [-2.5372e-02, -1.2206e-02, -1.2458e-02],
          [-2.7337e-02,  4.9518e-02, -3.3061e-02]],

         [[-2.7852e-02, -3.2274e-02, -2.4058e-02],
          [-5.6558e-02,  3.7129e-02,  4.9902e-03],
          [ 3.1626e-02,  8.6693e-03, -1.5555e-02]],

         [[-3.8774e-02, -2.2177e-03, -4.3799e-02],
          [-4.5361e-03,  2.3353e-02, -1.4998e-02],
          [-4.0275e-02,  2.0841e-02, -2.5546e-02]],

         [[-3.2436e-02,  7.3064e-04,  4.8089e-02],
          [-2.0403e-02,  5.3720e-02,  5.0457e-03],
          [-1.6575e-02, -4.1736e-03, -2.3268e-03]],

         [[-3.4843e-02, -3.8757e-02, -5.4195e-02],
          [ 4.7189e-02,  5.6678e-03,  5.3934e-02],
          [ 2.3392e-02, -3.9752e-02, -4.1773e-02]],

         [[ 5.5161e-02,  2.0864e-02, -2.6254e-02],
          [-1.1974e-02, -4.9629e-02, -1.8668e-02],
          [ 1.7638e-02, -5.2115e-02, -2.8758e-03]],

         [[-7.6567e-03, -4.8539e-02, -4.8364e-02],
          [ 2.8963e-02,  2.3195e-02, -3.5384e-02],
          [-5.7822e-02, -7.2767e-03, -3.3886e-02]],

         [[ 5.2410e-02, -5.7158e-03, -3.0626e-02],
          [ 4.1961e-02, -3.3456e-02, -7.4594e-03],
          [ 4.1571e-02,  5.4673e-02,  4.4440e-02]],

         [[ 3.2700e-02,  9.3443e-03,  2.8303e-03],
          [ 3.7368e-03, -3.3900e-02,  2.1067e-02],
          [ 5.4686e-02,  6.1188e-04, -1.7544e-02]],

         [[ 1.0915e-02,  5.0240e-02,  5.4272e-02],
          [-5.2618e-02,  1.2846e-02, -8.9759e-04],
          [-3.5478e-02, -1.7646e-02,  5.6642e-02]],

         [[ 8.7872e-03, -1.2334e-02,  4.6480e-02],
          [-9.7828e-03, -4.7962e-02, -3.2665e-02],
          [-2.0469e-02, -4.0128e-02,  1.4166e-02]],

         [[ 2.6194e-02, -3.6449e-02,  3.9728e-02],
          [ 3.9427e-02,  8.8658e-03,  3.2758e-02],
          [ 5.3261e-02,  8.2362e-03, -1.2925e-02]],

         [[ 4.4203e-02, -2.2900e-02, -5.7689e-02],
          [-3.8959e-02,  4.0875e-02, -3.4762e-02],
          [ 4.4613e-03,  5.3644e-02,  2.0431e-02]],

         [[-4.9567e-02, -3.7279e-02, -1.7899e-02],
          [-1.4039e-02,  1.0311e-02, -1.6143e-02],
          [ 2.7564e-02,  2.4709e-02, -3.6667e-02]],

         [[-4.4055e-02, -2.7872e-02,  1.1989e-02],
          [ 5.7841e-03, -3.4845e-02,  1.3310e-02],
          [-4.7348e-02,  5.0222e-02,  5.1338e-02]]],


        [[[-2.7818e-02,  2.8300e-02, -6.8990e-02],
          [ 1.8429e-02,  2.3933e-02, -1.5237e-02],
          [ 1.0956e-01,  6.2979e-02, -2.6757e-02]],

         [[ 7.4986e-02, -1.8531e-02, -1.1929e-02],
          [ 6.2264e-03, -5.1886e-02, -7.2712e-02],
          [ 1.7909e-02, -4.8887e-02,  4.5656e-02]],

         [[-2.3097e-03,  3.6816e-02, -9.6652e-03],
          [-1.1836e-02, -5.4733e-02, -1.3578e-02],
          [-6.4012e-02,  4.3518e-02,  1.1169e-01]],

         [[ 9.3885e-05,  2.9030e-02, -4.4646e-03],
          [-3.0858e-02, -1.0208e-01, -3.3332e-02],
          [-1.6313e-02, -1.0834e-01,  2.9566e-03]],

         [[ 1.0457e-02, -1.2866e-02,  2.4994e-02],
          [ 3.5378e-02,  8.3861e-03, -2.0790e-02],
          [-4.8899e-02,  8.2721e-02, -6.9468e-02]],

         [[-1.6654e-02,  4.6027e-02,  8.3200e-02],
          [ 6.5233e-02, -6.2745e-02, -5.8471e-02],
          [ 1.6724e-02, -4.2690e-02, -5.9327e-02]],

         [[-8.4832e-03, -6.0828e-02, -4.3356e-02],
          [-2.3651e-02,  1.8521e-02,  1.0522e-01],
          [ 1.6567e-03,  1.6177e-02,  1.4039e-02]],

         [[-7.4342e-03, -3.2893e-02, -5.9182e-02],
          [-2.4352e-02, -8.2291e-02, -4.9493e-02],
          [-7.1368e-02,  1.3183e-02,  3.5902e-02]],

         [[-5.0486e-02, -4.6802e-02, -1.0105e-01],
          [-1.1433e-02,  3.2610e-02,  1.7247e-02],
          [ 2.8978e-03, -5.2942e-02, -3.2161e-02]],

         [[-7.2851e-02, -6.5914e-02, -3.4819e-02],
          [-5.4568e-02,  1.0028e-01,  5.9407e-02],
          [-5.4219e-03,  4.5077e-02,  2.2588e-02]],

         [[ 1.0613e-02, -4.3285e-02, -3.5316e-02],
          [ 4.4387e-02,  4.3119e-02, -5.6573e-03],
          [-6.6300e-02,  3.0737e-02,  2.3357e-02]],

         [[-5.1102e-02,  5.6605e-03,  2.2041e-02],
          [ 8.1421e-02,  1.5057e-02, -2.0535e-03],
          [-2.0738e-02, -8.3124e-03,  2.9277e-02]],

         [[ 9.6402e-02, -5.1765e-03,  1.5779e-02],
          [-1.1889e-01, -2.2835e-03,  3.3303e-02],
          [ 7.9610e-02,  4.0938e-02, -8.0422e-02]],

         [[-3.6408e-02, -1.7446e-02, -4.5909e-02],
          [ 6.7532e-02,  9.2171e-02,  1.5945e-02],
          [-6.5366e-02,  2.1863e-02, -3.1559e-02]],

         [[ 2.0571e-02,  8.2435e-02, -4.3189e-02],
          [ 8.0295e-02,  2.7931e-02,  1.2060e-02],
          [-3.5301e-02, -1.2246e-02, -6.5237e-02]],

         [[-5.0708e-02,  7.4245e-02,  7.5958e-03],
          [ 6.8449e-02, -4.1922e-04, -5.1284e-02],
          [ 4.1550e-02, -1.9731e-03, -2.4886e-02]],

         [[-9.9305e-02, -6.5480e-03, -3.9356e-02],
          [ 9.0692e-02,  3.1859e-02, -7.4017e-02],
          [-1.4736e-02, -5.8806e-02, -8.0367e-02]],

         [[ 4.8938e-02, -3.4560e-02,  3.9152e-02],
          [-7.0931e-03, -7.9859e-04,  1.9154e-03],
          [-4.1752e-03, -8.0842e-04,  8.8507e-03]],

         [[ 9.0915e-03,  5.1776e-03, -8.4884e-03],
          [-5.4518e-02,  6.3283e-02,  7.3300e-02],
          [ 4.9079e-02,  2.9246e-02, -5.0159e-03]],

         [[-1.4450e-02,  3.7326e-02, -9.8168e-03],
          [ 2.5449e-02,  3.1345e-02, -7.2176e-02],
          [-3.5438e-02,  1.0525e-01, -5.5673e-02]],

         [[ 2.3608e-03, -9.5793e-03,  6.8177e-02],
          [-1.1851e-02,  6.3794e-02,  1.9248e-02],
          [-9.1156e-04,  6.5888e-03,  7.9514e-03]],

         [[-4.4835e-02,  1.2097e-01,  3.3541e-02],
          [ 1.1298e-01,  1.8815e-02,  2.8020e-02],
          [-9.1277e-02,  1.0499e-01,  3.9977e-03]],

         [[-2.0921e-03, -2.9671e-02,  8.5602e-03],
          [ 6.1289e-02, -5.5110e-02, -4.1811e-03],
          [-5.4757e-02,  3.5556e-02,  1.9024e-04]],

         [[ 1.9399e-02, -2.2278e-02,  1.1773e-02],
          [ 1.2187e-05,  4.7719e-02, -8.5241e-03],
          [-2.4985e-03, -4.4478e-03, -2.7288e-02]],

         [[-7.8784e-02, -3.1829e-02,  8.4421e-02],
          [-9.6289e-02,  1.1688e-01, -2.0849e-02],
          [-4.6706e-02,  5.9342e-02,  5.0769e-02]],

         [[ 9.7066e-03,  4.4835e-02,  6.2859e-02],
          [ 2.3340e-03, -5.2028e-02,  2.2082e-02],
          [-1.0887e-01,  6.8908e-02, -3.3248e-02]],

         [[ 2.0841e-02,  4.1121e-02, -1.9276e-02],
          [ 6.7009e-02,  9.9211e-02, -1.9009e-02],
          [-1.0234e-01, -4.5562e-03,  2.6761e-02]],

         [[ 1.8311e-02, -6.0896e-02, -1.7458e-02],
          [-1.1856e-02,  7.3446e-02, -3.5173e-02],
          [-6.7956e-02,  5.6479e-02, -5.3639e-02]],

         [[-3.6743e-02, -1.7444e-02,  2.1950e-02],
          [ 1.0039e-01,  8.1627e-03, -5.7132e-02],
          [-9.0422e-02,  7.5892e-03, -8.5635e-02]],

         [[-5.4502e-02,  1.0351e-02,  1.5590e-02],
          [-3.2668e-02,  1.5006e-02, -6.4394e-02],
          [ 2.3863e-02,  1.7530e-02, -1.0676e-03]],

         [[ 1.5398e-02, -2.3076e-02,  5.5899e-03],
          [ 1.0555e-01,  5.8754e-02,  1.5046e-02],
          [-3.9269e-02, -1.3376e-02, -6.5124e-02]],

         [[-1.1496e-01, -1.9146e-02, -1.8398e-02],
          [ 3.3552e-02,  1.6961e-02,  6.7284e-02],
          [ 1.3319e-01, -3.1644e-02, -1.8327e-02]]],


        [[[ 1.9024e-02,  1.6809e-02,  4.3703e-02],
          [ 7.4988e-02,  8.9174e-02,  4.1529e-02],
          [-1.6143e-02, -9.8308e-02,  1.2063e-02]],

         [[-4.1456e-02,  2.4903e-02, -2.2102e-02],
          [ 7.5750e-02,  3.7242e-02,  9.7616e-02],
          [-1.7570e-02,  7.7313e-02, -5.3739e-02]],

         [[-2.4487e-02, -5.4400e-02, -3.4331e-02],
          [ 4.2891e-02,  2.4894e-02,  8.5224e-03],
          [-5.7589e-02, -3.6049e-02, -8.4619e-02]],

         [[-4.5029e-02,  6.9065e-02, -3.9896e-02],
          [ 4.4835e-02,  6.0263e-02,  2.2505e-02],
          [ 3.9904e-02, -2.0152e-02,  3.8819e-03]],

         [[-7.8574e-02, -5.8514e-02,  4.4101e-02],
          [ 7.1531e-02, -4.2894e-02,  4.6218e-03],
          [ 1.0379e-01, -7.2791e-02, -2.3426e-02]],

         [[ 1.1901e-01, -3.7441e-02, -2.0225e-02],
          [-6.5125e-02, -1.5062e-02,  5.4092e-02],
          [ 6.8806e-02,  8.7095e-02, -8.2195e-03]],

         [[ 4.4950e-02,  4.1870e-02, -9.4920e-03],
          [-9.4995e-03,  2.7636e-02, -1.2102e-03],
          [-9.8725e-03, -1.3235e-01,  1.2099e-02]],

         [[-5.6509e-02, -2.0666e-02, -4.7954e-02],
          [ 9.8903e-02,  1.1862e-01, -5.9500e-02],
          [-1.5354e-02,  9.8605e-03, -5.2756e-03]],

         [[-2.1535e-02, -1.5893e-02, -9.3965e-02],
          [ 1.4455e-02, -5.5198e-02, -1.2059e-01],
          [-9.1714e-02, -3.8839e-02, -2.5104e-02]],

         [[ 9.8268e-02,  3.8471e-02,  5.3055e-02],
          [-1.2765e-02, -2.6781e-02, -1.8660e-02],
          [-4.5326e-02,  2.3039e-02, -1.2903e-02]],

         [[ 4.7596e-02,  1.2933e-01, -1.0179e-01],
          [-4.0136e-02, -1.4228e-02, -4.8916e-02],
          [ 6.3877e-02,  6.9723e-02, -2.1513e-03]],

         [[ 1.2779e-01, -8.8923e-02, -1.1307e-03],
          [-1.0151e-01,  1.9615e-02, -3.9897e-02],
          [ 2.0345e-02,  8.0821e-02, -1.9651e-02]],

         [[-1.0277e-01,  4.2036e-02, -2.1167e-02],
          [ 1.1779e-01,  1.3382e-02,  2.3699e-02],
          [-2.6278e-02, -1.6310e-02, -4.4742e-03]],

         [[ 9.2729e-03,  2.6233e-02,  2.3094e-02],
          [-3.4789e-03, -2.6628e-02, -1.4757e-03],
          [ 5.7628e-02, -1.5316e-02,  3.5544e-02]],

         [[ 5.0953e-02, -3.4292e-02,  2.3236e-02],
          [-1.6927e-02,  8.0178e-02,  7.9541e-02],
          [ 5.3670e-02,  3.5296e-02, -2.1810e-02]],

         [[ 5.4003e-02,  1.4387e-02,  1.7871e-02],
          [-4.8728e-02,  7.1595e-02,  4.0873e-02],
          [-2.4821e-02,  5.2888e-02, -4.1516e-02]],

         [[ 9.5073e-02,  7.7150e-03, -5.1465e-03],
          [-5.2492e-02, -6.4161e-02, -7.7623e-04],
          [ 8.6387e-02,  5.8934e-02, -4.3636e-02]],

         [[ 2.8547e-02, -2.5638e-02,  2.5980e-02],
          [-2.3537e-02, -6.4369e-03, -1.8622e-02],
          [-5.5136e-02, -5.3718e-02, -6.2308e-02]],

         [[-4.9009e-02,  7.3857e-02, -3.8393e-02],
          [ 1.0147e-01,  1.8924e-02,  9.9829e-02],
          [ 5.5224e-02, -2.2751e-03,  5.4348e-02]],

         [[ 1.1894e-03,  7.0279e-02, -6.2511e-02],
          [ 5.0098e-02, -5.3292e-04, -5.9783e-02],
          [-8.8867e-03, -9.5858e-02,  3.1424e-02]],

         [[ 2.9854e-02,  9.9589e-02,  3.5261e-02],
          [-2.1133e-02, -5.3405e-02, -8.0584e-02],
          [-7.8910e-02,  5.2530e-02,  6.3253e-02]],

         [[ 4.8042e-02, -2.8757e-02,  5.8940e-03],
          [-8.4778e-02,  3.4340e-02,  6.1020e-02],
          [ 8.8131e-02, -1.8515e-02,  9.6013e-02]],

         [[-4.5010e-03,  5.5686e-02,  6.3779e-03],
          [ 3.9784e-02, -9.1516e-03, -4.6612e-02],
          [ 1.3694e-02, -3.3847e-02, -5.9767e-02]],

         [[ 5.0654e-02,  3.0435e-02,  1.8062e-02],
          [ 3.1624e-02, -6.1239e-02, -3.1621e-03],
          [-7.5451e-02, -4.5861e-02,  8.0792e-02]],

         [[ 5.2295e-02,  8.9084e-02, -2.9484e-02],
          [ 4.6968e-02,  2.8085e-02, -1.1422e-01],
          [-6.2509e-02, -7.5780e-03, -5.1819e-02]],

         [[ 2.9186e-02, -1.7676e-02, -1.3069e-03],
          [-2.3576e-03,  4.6317e-02,  1.4271e-02],
          [ 4.2536e-02,  4.7882e-02, -3.8076e-02]],

         [[ 8.7520e-02,  5.2993e-02, -3.5764e-02],
          [ 4.1147e-02, -8.0349e-02, -8.1138e-03],
          [ 1.3695e-01,  6.8869e-02,  1.2358e-01]],

         [[-1.2102e-02,  2.6512e-02,  5.0533e-02],
          [-1.4409e-02,  5.2890e-02,  6.7745e-02],
          [ 2.2747e-03, -6.1208e-02,  1.5833e-02]],

         [[ 1.1528e-01,  8.3256e-02,  6.5467e-03],
          [-1.6222e-02, -9.3574e-02,  4.0826e-02],
          [ 4.0482e-02,  3.6465e-02,  8.3762e-02]],

         [[ 5.6368e-02,  2.3465e-02,  1.4691e-02],
          [ 1.0483e-03,  8.9891e-02,  6.1491e-03],
          [ 5.4019e-02, -6.3333e-02,  7.1373e-02]],

         [[ 4.1974e-02, -2.5150e-02,  7.7128e-02],
          [-4.4775e-03, -4.6724e-02,  4.3871e-03],
          [ 7.0164e-02,  5.2312e-02,  9.6724e-03]],

         [[ 1.2528e-01, -2.1597e-02, -7.3027e-04],
          [-2.2341e-02, -4.0565e-02, -1.1821e-01],
          [ 4.3988e-02,  4.7343e-02,  2.8785e-02]]],


        [[[-2.9000e-02, -2.1912e-02,  2.0557e-02],
          [-4.8347e-02,  2.0163e-02, -2.1522e-02],
          [-1.0543e-02,  3.0884e-02,  3.8163e-02]],

         [[ 2.4953e-02, -6.0609e-02, -3.0948e-03],
          [-5.2911e-02, -2.4118e-02, -1.2912e-02],
          [-3.0171e-02, -7.4285e-02,  7.3140e-02]],

         [[-4.5869e-02, -4.2180e-03, -4.7471e-02],
          [ 8.9796e-03, -7.3019e-03, -1.7938e-02],
          [-2.7853e-02, -3.5469e-02,  3.4563e-02]],

         [[-1.3668e-02, -3.4048e-02,  3.3747e-02],
          [-5.9112e-02, -1.9491e-02,  4.0488e-02],
          [-4.7196e-02,  9.7890e-03,  2.9951e-02]],

         [[ 2.7642e-02,  2.5375e-02, -4.4562e-02],
          [-7.9942e-02, -2.9809e-02, -6.1268e-02],
          [-7.9288e-03, -2.6943e-02, -3.1007e-02]],

         [[-1.6467e-02,  1.2771e-02, -8.5584e-02],
          [-1.8277e-02, -6.1896e-02, -7.7618e-02],
          [ 2.8101e-02,  2.2956e-02, -3.4300e-02]],

         [[-3.3173e-02,  3.6858e-02, -1.1841e-02],
          [-1.6329e-03, -5.5796e-02,  3.4441e-02],
          [ 4.4033e-02, -9.3581e-04, -6.9571e-02]],

         [[-3.1048e-03,  8.2535e-03, -4.1239e-02],
          [-5.0625e-03,  8.5731e-02, -1.8080e-02],
          [-3.1647e-02, -8.6363e-03, -7.1344e-02]],

         [[ 9.7471e-03, -4.9722e-02,  2.4605e-02],
          [-5.0580e-03,  2.9864e-02, -1.9638e-04],
          [ 5.6555e-02,  3.4021e-02,  2.2797e-02]],

         [[ 1.0872e-02,  3.0536e-04, -6.7238e-02],
          [ 1.3924e-02,  6.6164e-02, -1.9010e-02],
          [ 1.3672e-02, -1.5380e-02, -4.6999e-02]],

         [[ 4.0905e-02, -3.3991e-02, -6.5694e-02],
          [ 1.4792e-02,  5.4615e-03, -4.4667e-02],
          [ 1.0920e-02, -8.7345e-03, -3.8489e-02]],

         [[-5.4024e-02, -7.9995e-02, -4.8117e-02],
          [ 2.0384e-02, -2.6342e-02, -1.8728e-02],
          [-3.7528e-03, -2.1256e-02, -3.0992e-03]],

         [[-4.8864e-03, -9.8621e-03,  3.3553e-02],
          [-3.0841e-02,  2.8988e-02,  5.5152e-02],
          [ 6.5346e-02, -1.1910e-03, -9.8216e-03]],

         [[ 2.2818e-02, -2.9496e-02, -9.8967e-04],
          [-1.7664e-02, -2.3620e-02, -3.9141e-02],
          [ 3.2630e-02,  1.3641e-02,  2.2736e-02]],

         [[-6.1885e-02, -6.0489e-03, -1.5880e-03],
          [-3.9487e-02, -3.5027e-03, -2.6662e-02],
          [-2.0043e-02,  1.8413e-02,  6.7995e-03]],

         [[-1.2944e-03, -4.4183e-02, -3.8322e-02],
          [-1.9143e-02,  2.5919e-02,  7.2477e-03],
          [-2.6205e-02, -7.9575e-02, -6.9321e-02]],

         [[ 4.9483e-02, -1.9196e-02, -2.5035e-02],
          [ 5.4373e-02, -2.9960e-02, -1.0009e-02],
          [-1.2713e-02, -8.4407e-03,  9.4580e-03]],

         [[ 4.2923e-02, -3.5286e-03, -3.3016e-02],
          [ 6.5893e-02, -2.6760e-02, -3.4542e-02],
          [-9.8819e-03,  3.7872e-02, -1.6559e-02]],

         [[-4.2930e-02, -6.9175e-02,  9.1752e-03],
          [-3.4106e-02,  1.3954e-02, -4.3232e-02],
          [-1.1098e-02, -4.4476e-02,  2.2287e-02]],

         [[-2.4046e-02,  1.2823e-02, -5.7934e-02],
          [-2.2408e-02, -4.7338e-02, -7.1187e-04],
          [-2.1031e-02, -9.6171e-03, -4.7817e-02]],

         [[ 2.5303e-03,  4.3552e-02, -4.9009e-02],
          [ 2.2376e-02, -2.8070e-02, -5.3707e-03],
          [-1.8904e-03, -4.2068e-02, -5.4789e-02]],

         [[-1.6544e-02, -7.3640e-02,  1.1561e-02],
          [ 2.5485e-02, -8.0691e-02,  1.9563e-02],
          [-2.7302e-02, -5.9806e-02,  1.8805e-02]],

         [[ 5.1734e-02,  3.7292e-02, -2.7626e-02],
          [ 2.9496e-02,  4.0379e-02, -4.1291e-02],
          [ 5.7916e-02, -4.4918e-02,  5.7970e-02]],

         [[-6.8328e-02, -7.5902e-02, -3.3170e-02],
          [-5.8859e-02, -8.8100e-02, -4.6567e-02],
          [-4.6508e-02, -3.5704e-02,  1.4000e-02]],

         [[-1.8210e-02, -5.0158e-02,  3.7713e-02],
          [-4.2038e-02,  3.0961e-02,  1.4315e-03],
          [-1.8628e-03, -3.2461e-02, -3.0202e-02]],

         [[ 1.5421e-02, -3.2686e-02,  5.6034e-03],
          [-2.2730e-02,  2.7084e-02,  6.0688e-02],
          [-5.3507e-02,  8.7493e-02, -4.1025e-02]],

         [[-8.9136e-03, -6.8115e-02, -1.3223e-02],
          [-9.5476e-03,  8.7293e-03, -6.9727e-02],
          [-6.9104e-02, -3.3608e-02, -3.0439e-02]],

         [[ 2.3320e-02,  4.7810e-02,  9.9444e-04],
          [ 2.3732e-02, -8.7407e-02, -6.2988e-02],
          [-2.6550e-02, -3.2617e-03, -4.6879e-02]],

         [[-6.4147e-02,  2.1738e-02, -8.3297e-02],
          [ 1.1423e-02,  5.4529e-02, -6.1097e-02],
          [ 4.4715e-02, -5.5919e-02,  2.3606e-02]],

         [[-6.3853e-02, -5.3062e-02, -5.5942e-02],
          [-6.1087e-02, -4.3258e-02, -7.5333e-02],
          [-8.4459e-02,  1.1331e-03,  2.4660e-02]],

         [[-1.1765e-02, -1.9436e-02, -5.8445e-02],
          [-3.3403e-03, -6.3063e-02, -3.0538e-02],
          [-4.9621e-02,  1.5763e-02, -7.9592e-02]],

         [[-3.2157e-02,  4.5683e-02, -1.6015e-03],
          [ 5.5207e-02,  1.3559e-02, -2.8653e-02],
          [ 4.6723e-02, -3.5389e-02, -7.9237e-02]]],


        [[[-1.3498e-02,  4.4076e-02, -4.5500e-02],
          [ 6.7334e-03, -5.5226e-02,  5.3186e-02],
          [-5.0701e-02,  7.3921e-02,  3.0930e-02]],

         [[ 5.2571e-02, -3.4046e-02, -4.3234e-03],
          [-3.4370e-02,  3.6946e-02, -1.7840e-02],
          [ 4.7518e-02, -7.4251e-02,  2.4644e-03]],

         [[ 5.2146e-03,  3.4045e-02, -6.0668e-04],
          [ 6.4916e-02, -8.0428e-02, -8.3103e-02],
          [-7.0470e-02,  9.7705e-02, -2.4223e-02]],

         [[ 7.8299e-02, -3.5498e-03, -4.7071e-02],
          [-1.0540e-01, -7.0381e-02,  2.1214e-02],
          [ 4.1940e-02, -8.2131e-02, -2.2533e-02]],

         [[-4.1448e-02,  4.5622e-02,  4.7484e-02],
          [ 2.0708e-02,  1.3798e-02,  6.0132e-02],
          [-2.8580e-02, -3.9010e-02,  2.9487e-02]],

         [[-7.5346e-02,  3.1755e-02,  7.8523e-02],
          [ 2.9393e-02, -3.9553e-02, -3.4239e-04],
          [ 6.8775e-02, -1.0082e-01, -1.0194e-02]],

         [[-1.0236e-02, -4.2709e-02, -8.1052e-02],
          [-1.1113e-01, -4.5824e-02,  4.3365e-02],
          [-5.7361e-02,  2.4845e-02,  5.4515e-02]],

         [[ 8.4079e-02,  4.2538e-02, -6.4032e-02],
          [-1.3558e-02, -2.2265e-02,  5.2724e-03],
          [-7.7219e-02, -7.2297e-02, -6.8500e-02]],

         [[-4.4922e-02, -5.5536e-02, -5.9322e-02],
          [-4.5380e-02, -6.9149e-02,  1.0841e-02],
          [ 6.0962e-02, -5.7115e-02, -7.7892e-02]],

         [[ 7.2236e-02, -1.2336e-01, -1.2705e-01],
          [-1.1736e-01,  8.1442e-02,  1.2444e-01],
          [-6.3952e-02, -7.3095e-02,  4.3985e-02]],

         [[-3.4216e-02, -7.0248e-02, -5.3920e-03],
          [ 4.2467e-02,  1.0323e-01,  5.8997e-02],
          [-4.6742e-02,  3.9532e-02,  5.1649e-02]],

         [[-8.2207e-02, -5.7502e-03, -3.3843e-03],
          [ 9.6997e-02,  6.8970e-02, -1.0496e-01],
          [ 6.6672e-02,  2.5438e-02, -2.1696e-02]],

         [[ 5.1444e-02, -7.4210e-02, -4.4423e-02],
          [-3.9854e-02,  4.5603e-02,  4.3645e-02],
          [-3.8453e-02, -1.0333e-02, -3.5620e-02]],

         [[ 2.1037e-02, -2.7778e-02,  5.4537e-02],
          [ 9.8172e-02, -1.7050e-02,  8.5366e-03],
          [ 3.7258e-03,  5.7209e-02,  3.4083e-02]],

         [[-6.5052e-02,  3.7388e-02,  7.2423e-02],
          [ 9.5114e-02, -3.5834e-02,  3.1020e-02],
          [ 2.3519e-02, -4.3703e-02, -6.8940e-02]],

         [[-1.2009e-01, -1.2960e-03,  1.4421e-02],
          [ 5.0790e-02,  2.5063e-02,  8.0187e-03],
          [ 7.1103e-02,  6.0999e-02,  1.1308e-02]],

         [[-7.6160e-02, -3.2876e-02,  1.1492e-01],
          [ 4.7976e-02, -3.5097e-02, -7.3065e-02],
          [-3.4254e-02, -8.6435e-03, -4.0343e-03]],

         [[-1.1827e-01, -1.0163e-02, -7.7785e-02],
          [-6.0300e-02,  4.1621e-02, -1.7560e-02],
          [-1.0220e-01, -7.4443e-02, -4.4663e-02]],

         [[-1.0638e-03,  1.2274e-02, -7.6140e-03],
          [-3.1681e-02, -4.8140e-02, -2.0633e-02],
          [ 3.2643e-02,  7.0957e-02, -1.0209e-01]],

         [[-2.1032e-02, -5.0272e-02,  1.0763e-01],
          [ 6.7225e-02, -7.3175e-02,  4.5554e-02],
          [-6.1429e-02,  1.1565e-01, -5.7200e-03]],

         [[-2.4896e-02, -1.0869e-01,  5.1912e-02],
          [-2.0591e-02,  8.1267e-02, -7.3724e-02],
          [ 5.2124e-04, -4.5220e-02,  2.6878e-02]],

         [[-9.6597e-02,  4.1075e-02,  9.6141e-02],
          [ 5.6170e-02, -1.2908e-01, -3.4717e-02],
          [-3.0181e-02,  7.0732e-02,  8.6759e-02]],

         [[-5.4490e-02, -3.9944e-02,  8.8074e-02],
          [ 2.0913e-02,  3.5377e-02, -1.5723e-02],
          [-1.0601e-01,  6.1593e-02,  4.0411e-02]],

         [[ 4.8402e-02, -7.5503e-02, -5.6550e-03],
          [-1.0957e-01,  6.2586e-02, -3.1647e-02],
          [-8.4054e-02, -2.3308e-03,  6.9676e-02]],

         [[-1.1840e-02, -9.7593e-02, -6.3060e-02],
          [-4.7671e-02,  5.7259e-02,  4.8159e-02],
          [-4.3629e-02, -2.5802e-02, -4.2361e-03]],

         [[ 1.7852e-02,  3.5789e-02,  6.8133e-02],
          [ 3.4244e-02, -1.3507e-01, -3.2063e-02],
          [-2.2747e-02, -2.1839e-02, -1.2671e-02]],

         [[-2.2714e-02, -2.5629e-02,  6.3346e-02],
          [ 4.0023e-03,  4.8742e-02,  1.1436e-02],
          [-1.2643e-01, -4.5621e-03, -1.0936e-02]],

         [[-9.7355e-03, -6.0193e-02, -4.2014e-03],
          [-6.3571e-02, -4.0479e-02,  3.7610e-02],
          [-7.9533e-02,  5.4858e-02,  8.7372e-03]],

         [[-3.5434e-02, -9.2660e-02,  2.0543e-02],
          [ 9.5053e-02,  8.5193e-02, -2.7377e-03],
          [-1.8397e-02,  1.1066e-02, -5.1613e-02]],

         [[ 2.9576e-02,  2.9499e-03,  2.2393e-02],
          [ 7.5504e-02, -5.5997e-02,  3.8238e-02],
          [ 4.2221e-03,  1.0799e-03,  5.8235e-02]],

         [[-1.9130e-02,  1.7397e-02, -2.7757e-03],
          [ 2.1500e-02,  1.8801e-02,  4.3660e-02],
          [-3.8465e-02, -9.5229e-03, -3.5868e-03]],

         [[-2.6916e-02,  8.3155e-03,  2.8844e-02],
          [-6.3310e-02,  1.2406e-02,  1.6432e-02],
          [ 7.4752e-02,  2.4825e-02,  6.3421e-03]]],


        [[[-5.7835e-02, -6.4219e-02, -8.9135e-02],
          [ 8.1238e-03, -1.0265e-01,  1.1301e-02],
          [ 8.9432e-02,  8.2686e-02, -9.5422e-03]],

         [[-3.9773e-02,  1.0950e-02, -3.4521e-02],
          [-4.9218e-02, -5.4740e-02, -1.7527e-02],
          [ 1.6619e-02, -8.1840e-02, -5.2550e-02]],

         [[ 5.8837e-02, -1.7066e-03, -8.0935e-02],
          [-1.0113e-01, -3.5474e-02,  4.2719e-02],
          [-1.2383e-02,  2.7005e-02, -3.8905e-02]],

         [[ 3.7037e-02,  1.0097e-02,  5.4327e-02],
          [-1.0053e-01, -8.3708e-02, -3.4884e-02],
          [-4.3989e-02, -7.6968e-02,  1.9988e-02]],

         [[ 6.2018e-02,  2.5988e-02,  5.5224e-02],
          [-4.3446e-02, -1.6253e-02,  7.9981e-03],
          [-4.7015e-02,  9.6406e-03,  5.4929e-02]],

         [[-3.7366e-03,  4.8729e-02, -7.0795e-02],
          [ 1.0256e-01, -1.9127e-03, -5.6515e-02],
          [-1.6528e-02, -8.8861e-02, -7.0955e-02]],

         [[-6.4355e-03, -1.7528e-02,  5.8530e-02],
          [-2.2362e-02,  8.2339e-04, -5.8073e-02],
          [ 3.9710e-02,  6.0420e-02,  3.1086e-02]],

         [[ 8.0680e-03,  8.7305e-02,  6.2539e-02],
          [-6.6911e-02,  2.9605e-03, -9.6899e-02],
          [-4.2952e-02,  5.1666e-02,  5.5054e-02]],

         [[-7.9425e-03,  9.3804e-02, -1.2603e-02],
          [-8.5386e-02,  6.7149e-02, -4.8058e-02],
          [-1.2943e-02,  6.1194e-03,  5.9359e-02]],

         [[-8.9554e-02,  3.4762e-03,  2.6710e-02],
          [-6.6722e-02,  4.6388e-02, -1.0435e-02],
          [ 6.5290e-02,  6.3183e-02, -8.4021e-04]],

         [[-3.3965e-02, -1.2674e-02, -8.9691e-03],
          [ 6.8093e-02,  1.4471e-01,  3.5701e-02],
          [-3.1581e-02,  1.1385e-02, -6.4499e-02]],

         [[-5.3260e-02,  4.9209e-02, -3.3048e-02],
          [ 1.2318e-02,  4.3772e-02, -1.4084e-02],
          [-3.7159e-02, -1.3741e-02, -8.6918e-02]],

         [[ 4.5871e-02, -6.9025e-02,  2.0669e-02],
          [-5.5723e-02,  6.0371e-02,  2.6073e-02],
          [-1.0508e-02,  3.7439e-02, -4.3581e-02]],

         [[-2.0193e-02, -7.1199e-02, -8.2523e-02],
          [ 1.2585e-01,  7.6807e-02, -2.5102e-02],
          [-6.9021e-02, -3.3344e-02, -6.2024e-02]],

         [[-4.9195e-02, -1.0873e-02, -2.5282e-02],
          [ 2.5498e-03, -5.1202e-02, -5.1968e-03],
          [-4.1126e-02, -7.4489e-02, -5.1890e-02]],

         [[-5.0796e-03, -1.2348e-02,  4.5868e-02],
          [ 3.6583e-02, -8.5733e-03,  4.7146e-02],
          [ 4.0814e-02, -3.6681e-02, -2.0045e-02]],

         [[-1.1691e-01,  3.3416e-02,  2.4001e-02],
          [ 7.7773e-02,  2.0421e-02, -3.2964e-02],
          [ 4.8323e-02, -1.4412e-02, -1.2633e-02]],

         [[-8.4645e-03,  5.2228e-03,  9.7909e-02],
          [ 6.7507e-02,  8.7861e-02, -5.4829e-02],
          [-3.1270e-02,  2.3319e-02,  2.2776e-02]],

         [[ 1.2761e-02,  2.6920e-03,  2.4821e-04],
          [-1.9083e-02,  4.0123e-02, -5.8593e-03],
          [ 6.0806e-02, -4.1524e-02,  5.8248e-02]],

         [[-1.0786e-01, -1.6604e-03, -8.3327e-02],
          [ 1.0445e-02,  7.8083e-02,  5.1290e-03],
          [-9.6339e-03,  7.4707e-02, -4.8995e-02]],

         [[-9.6099e-02, -6.9461e-02, -7.4014e-02],
          [ 1.5176e-02,  8.0453e-02,  6.7096e-02],
          [-5.7185e-02, -5.9031e-02,  2.6860e-03]],

         [[-9.2855e-02,  9.1245e-04, -5.2402e-02],
          [ 9.2558e-02, -3.7608e-02, -6.9167e-03],
          [-2.4875e-02,  3.1253e-02, -1.9989e-02]],

         [[-2.5069e-02,  3.5483e-02,  4.1559e-02],
          [ 4.9042e-02,  1.3314e-02, -1.3985e-02],
          [ 5.4324e-02,  2.4288e-02,  7.6990e-02]],

         [[ 3.1879e-02, -2.5791e-02, -4.2544e-02],
          [-2.0797e-02, -1.4425e-02,  1.8645e-03],
          [-1.2257e-03, -7.3244e-02, -4.7372e-02]],

         [[-2.3837e-02, -1.4284e-02, -7.0858e-02],
          [-1.6697e-02,  6.5709e-02,  1.0262e-02],
          [ 2.3190e-03, -2.6200e-02, -5.0828e-02]],

         [[ 7.5566e-03,  3.3255e-02,  4.0630e-02],
          [-2.4009e-02, -8.5241e-03,  2.6678e-02],
          [ 1.1631e-02,  5.9826e-02, -3.3457e-04]],

         [[-7.0863e-02, -7.8607e-02, -1.4556e-02],
          [ 1.2390e-02,  7.7679e-02,  2.5275e-02],
          [-5.7344e-02, -5.8735e-02,  3.7995e-02]],

         [[-7.6868e-02, -8.1808e-02,  7.0650e-03],
          [-7.2233e-03,  6.7258e-02,  1.7470e-02],
          [ 5.1843e-03,  7.8407e-02,  8.9055e-02]],

         [[-4.6775e-02, -4.2219e-02,  1.1097e-02],
          [ 2.4500e-04,  8.0813e-03, -3.6369e-02],
          [-8.8916e-02, -2.0381e-02, -1.7742e-04]],

         [[-3.5278e-02, -1.3111e-02,  4.5265e-02],
          [-4.8789e-02, -6.4525e-02,  5.1993e-02],
          [ 4.8136e-02,  7.0970e-02,  5.6012e-02]],

         [[ 2.7187e-02, -3.9418e-02,  3.3975e-03],
          [ 2.2500e-02,  3.4068e-02,  8.9945e-03],
          [ 2.2903e-02, -8.6085e-02, -2.5314e-02]],

         [[-3.4319e-02,  7.1363e-02, -1.8464e-02],
          [ 5.3841e-02,  3.9596e-02, -8.3443e-02],
          [ 2.7447e-02, -2.2367e-02,  4.1377e-02]]],


        [[[ 5.4258e-03, -2.0261e-02, -8.0633e-02],
          [ 5.8092e-02,  7.9394e-02,  1.1345e-02],
          [-8.8621e-02, -6.6782e-02, -1.8759e-02]],

         [[ 1.8539e-02,  3.9482e-02, -3.7800e-02],
          [ 4.5763e-02,  6.7567e-02,  4.3823e-02],
          [-1.5123e-02,  5.8392e-03,  2.1377e-02]],

         [[ 1.2895e-02,  3.4102e-02, -3.5298e-02],
          [-1.4053e-02, -8.5413e-03, -9.5217e-02],
          [-5.1839e-02,  3.1541e-02,  7.4454e-03]],

         [[-5.2841e-03,  2.1423e-02, -5.0133e-02],
          [ 9.8010e-02,  1.9993e-02, -5.6099e-03],
          [ 4.0376e-02, -5.2851e-02,  3.1476e-02]],

         [[-3.0311e-02,  6.8569e-02, -1.5701e-02],
          [ 6.6062e-02,  6.0683e-02,  4.5405e-02],
          [-1.9449e-02,  4.7034e-02, -2.4603e-02]],

         [[ 5.6796e-02,  7.9494e-02, -5.8783e-02],
          [ 2.8874e-02, -1.1707e-02, -7.9796e-02],
          [ 7.2034e-02,  4.7854e-02,  1.2548e-02]],

         [[ 4.7934e-02, -8.5681e-02, -8.9657e-02],
          [ 1.6977e-02,  3.9727e-02,  6.1039e-02],
          [-3.6186e-02, -5.3153e-02, -2.2164e-03]],

         [[-4.0768e-02, -1.0812e-01, -1.2206e-01],
          [ 1.4073e-01,  8.9915e-02,  3.6540e-02],
          [-6.3495e-02, -7.2651e-02,  1.1770e-02]],

         [[ 4.2960e-02, -6.9063e-02, -4.1702e-02],
          [ 7.1030e-02, -4.6381e-02,  8.0695e-02],
          [ 7.7631e-03, -5.1323e-02,  6.3263e-02]],

         [[ 4.7474e-02, -5.8441e-02,  1.8113e-02],
          [-1.1146e-02,  1.8552e-03,  6.9289e-02],
          [-4.2419e-02,  4.8597e-04,  1.3418e-02]],

         [[ 1.1446e-01,  1.0836e-01,  5.1197e-02],
          [-3.5385e-02, -5.8285e-02, -1.0144e-01],
          [ 3.8987e-02,  4.7533e-02,  1.6901e-02]],

         [[ 5.3694e-02,  4.0760e-03,  2.5772e-02],
          [ 2.9360e-02, -3.8925e-02, -3.8303e-02],
          [ 2.8519e-02,  1.2638e-01,  9.6040e-02]],

         [[ 2.6145e-03,  3.8393e-02, -5.2313e-02],
          [ 6.0678e-02, -8.5012e-02, -3.2800e-02],
          [ 1.1709e-02,  6.6176e-02,  5.3000e-02]],

         [[ 8.6165e-02,  9.0635e-02, -4.6758e-02],
          [-9.4259e-02, -1.8997e-02, -1.2320e-01],
          [ 1.2652e-01,  1.2258e-01, -2.5434e-02]],

         [[-3.9429e-04, -2.6774e-02,  7.7502e-02],
          [ 4.7293e-02,  1.2777e-02,  3.8048e-03],
          [ 7.6818e-02,  8.1730e-03, -6.2587e-02]],

         [[ 1.2322e-02, -2.9598e-02,  7.6971e-02],
          [-3.5692e-03,  1.7338e-02,  1.9305e-02],
          [ 4.3465e-02,  6.8426e-02,  6.6278e-02]],

         [[ 9.7447e-02,  2.7065e-02,  6.1975e-03],
          [-7.6276e-02, -6.7648e-02, -1.1164e-02],
          [ 1.1674e-01,  4.5852e-02,  2.2262e-02]],

         [[ 4.1263e-02, -6.1086e-02,  1.2446e-02],
          [ 6.1887e-03, -2.7543e-02,  9.0253e-02],
          [ 9.1957e-03, -1.1979e-01,  4.6248e-02]],

         [[ 6.2049e-02,  1.9757e-02, -3.2718e-02],
          [ 5.2264e-02,  5.5758e-02, -1.6768e-02],
          [-4.1173e-02,  9.3471e-02, -5.6621e-02]],

         [[ 3.4171e-02,  1.0760e-02,  9.3636e-02],
          [-5.5187e-02,  1.2431e-03, -5.6132e-03],
          [ 8.8503e-02,  7.4496e-02,  6.3564e-02]],

         [[ 2.7750e-03,  7.4066e-02, -2.2200e-02],
          [-1.0826e-01, -6.0513e-02, -3.1577e-02],
          [ 1.6648e-02,  1.0756e-01,  5.5972e-02]],

         [[ 3.1755e-02,  9.8549e-02,  3.9237e-02],
          [ 2.4870e-02,  5.9005e-02, -2.1467e-02],
          [ 1.1330e-01,  7.1837e-02,  4.0566e-02]],

         [[-1.6751e-02,  4.6536e-02,  3.0913e-02],
          [-2.0599e-02, -2.4389e-02,  2.2796e-02],
          [-4.8096e-02, -8.3923e-02, -1.4734e-02]],

         [[ 3.0980e-02,  6.2571e-02,  7.8497e-02],
          [-5.2926e-02, -3.6013e-02,  1.0487e-02],
          [ 1.0085e-02,  4.6378e-02,  2.7388e-02]],

         [[ 3.1289e-03,  5.1250e-02, -1.7719e-04],
          [-1.1778e-01, -4.6833e-02,  3.8565e-02],
          [ 6.1716e-02,  6.6457e-02,  3.2891e-02]],

         [[ 5.4375e-02,  4.9786e-02, -4.1800e-02],
          [ 7.0020e-02,  1.3143e-01,  4.2404e-02],
          [ 1.9789e-02, -4.0711e-02, -5.1208e-02]],

         [[ 3.7479e-02,  5.4157e-02,  5.5413e-02],
          [-1.2553e-01, -5.1099e-02, -7.2442e-02],
          [ 7.4509e-02,  8.7819e-02, -9.9656e-03]],

         [[ 4.5460e-02,  2.8073e-02, -5.9388e-02],
          [-3.7874e-02,  9.7718e-02, -1.1009e-01],
          [-4.6557e-02, -1.6183e-02, -3.2729e-02]],

         [[ 4.6726e-02,  9.7435e-02,  4.3400e-02],
          [-9.7803e-02, -6.4492e-02, -1.1145e-01],
          [ 2.9140e-02,  5.2690e-02,  4.6737e-02]],

         [[ 7.2962e-02,  3.2037e-03, -1.9283e-02],
          [ 3.5469e-02, -7.3126e-03,  2.0855e-02],
          [ 2.2040e-02,  4.3437e-02, -5.1067e-02]],

         [[ 1.0764e-01,  2.6805e-02,  1.5666e-03],
          [ 2.2863e-02, -2.5835e-02,  1.8721e-02],
          [ 6.2374e-02,  1.0889e-01,  5.3720e-02]],

         [[ 8.4794e-02, -4.1152e-02,  2.3582e-03],
          [ 9.1135e-02, -2.9184e-02,  4.0277e-02],
          [-7.7833e-02,  5.1831e-02, -2.1865e-03]]],


        [[[-3.3282e-02, -7.9023e-02, -4.3986e-02],
          [-1.3418e-02,  9.4162e-02, -2.0290e-02],
          [-1.6771e-02, -9.6220e-02, -5.8274e-02]],

         [[-5.3331e-02, -6.4092e-02, -2.5403e-02],
          [ 6.5286e-02, -9.8614e-02,  1.3070e-01],
          [-8.4673e-02,  2.7644e-02,  3.9775e-02]],

         [[ 2.3863e-02, -1.8692e-02,  2.1380e-02],
          [-4.8901e-02, -6.4647e-02,  6.1139e-02],
          [ 6.8454e-02, -1.9937e-02,  6.9870e-02]],

         [[-2.8009e-02, -6.9545e-02, -6.4157e-02],
          [-1.0176e-02,  3.1725e-02,  7.1676e-02],
          [-5.2963e-03,  5.6869e-02,  7.3127e-02]],

         [[-1.0101e-01,  3.3221e-02, -8.9562e-02],
          [-2.9229e-02,  4.1752e-02,  1.2361e-03],
          [ 5.8982e-02,  2.1345e-03, -8.7889e-02]],

         [[ 6.2585e-02,  9.5237e-02,  4.1888e-02],
          [-1.0450e-01,  2.2667e-02, -6.3263e-02],
          [ 2.1984e-02,  9.9468e-02,  2.7170e-02]],

         [[ 3.4230e-03, -1.0446e-01,  1.3153e-01],
          [ 9.4058e-02,  5.9758e-02, -9.5385e-03],
          [-9.0744e-03,  2.9869e-02,  1.3006e-01]],

         [[-8.5905e-02, -1.1877e-01,  8.8836e-02],
          [ 4.5071e-02,  1.8072e-02,  1.2146e-01],
          [ 1.9961e-02, -5.2220e-02,  5.4174e-02]],

         [[-5.8710e-02, -5.9384e-02,  8.8035e-02],
          [ 3.4931e-02, -3.5370e-02,  4.6597e-02],
          [-7.7563e-02, -4.6813e-02,  6.1416e-02]],

         [[-8.6825e-02, -7.6700e-02,  6.2034e-02],
          [ 1.2063e-01, -2.4718e-02,  6.0627e-02],
          [-3.8814e-02, -5.0254e-02,  1.3051e-01]],

         [[ 1.1269e-02,  1.2510e-01,  4.9979e-02],
          [ 1.3053e-02, -7.5029e-02, -1.2707e-01],
          [ 7.0219e-03,  3.7324e-02,  5.4211e-02]],

         [[ 3.1774e-03, -8.2139e-03,  3.0975e-02],
          [ 9.1387e-03, -2.7858e-02, -1.8178e-02],
          [ 9.4169e-02,  1.1472e-01,  7.6273e-02]],

         [[-1.6217e-02,  5.4139e-02, -2.4532e-02],
          [ 3.0452e-02, -4.3775e-02,  9.4162e-02],
          [ 7.5984e-03, -5.1032e-02, -3.2647e-02]],

         [[-2.4450e-02,  1.2911e-01, -1.0818e-01],
          [ 5.4928e-03, -9.9316e-02, -4.8572e-02],
          [ 5.7907e-02,  8.5087e-02, -2.2912e-02]],

         [[ 7.2329e-04,  1.2225e-02, -1.4228e-02],
          [-4.5307e-02, -1.4518e-02,  5.3177e-02],
          [ 1.6177e-03,  9.9426e-02, -9.6112e-02]],

         [[ 4.1600e-02,  4.0000e-02,  1.5134e-02],
          [ 1.5615e-02,  3.3434e-03,  1.8395e-02],
          [-3.7775e-02,  7.2676e-02,  7.2789e-02]],

         [[ 9.4239e-02,  1.3718e-01, -5.9003e-03],
          [-1.0159e-01, -4.5598e-02,  2.9742e-02],
          [ 1.0019e-01,  5.9567e-02, -3.1044e-02]],

         [[-3.1364e-02, -1.1496e-02,  9.6609e-02],
          [-1.3917e-02, -3.8515e-02, -3.6901e-02],
          [ 7.6817e-02, -9.9558e-02,  1.0370e-01]],

         [[-2.5025e-02,  8.2897e-02, -1.0646e-01],
          [ 7.3875e-02, -4.0322e-02, -5.0415e-02],
          [ 6.3746e-02, -3.9887e-03, -1.0427e-02]],

         [[ 1.5596e-02,  1.2482e-01,  6.4696e-03],
          [-8.4564e-02, -8.6977e-02, -2.3160e-02],
          [ 1.0741e-01,  1.9650e-03,  2.4003e-02]],

         [[-5.9459e-02,  9.6110e-02,  5.0893e-02],
          [ 1.7927e-02, -7.3191e-02, -9.2220e-03],
          [-9.9874e-02,  3.6425e-02,  1.0038e-01]],

         [[ 1.2387e-01,  5.8057e-02, -6.5886e-02],
          [-3.1434e-02,  7.1396e-02, -4.8554e-02],
          [ 1.3189e-01, -1.8521e-02, -5.4223e-02]],

         [[ 5.7589e-02, -3.8138e-03,  1.2719e-02],
          [-3.6362e-02, -4.3441e-02, -3.1647e-02],
          [ 4.4527e-02, -1.9673e-03, -3.8024e-02]],

         [[-9.1144e-03,  8.5070e-02, -3.4247e-02],
          [ 5.3417e-02,  3.5967e-02, -3.1445e-02],
          [ 2.3302e-02,  6.3515e-02, -2.8075e-02]],

         [[-3.9301e-02,  3.5844e-02,  7.6469e-02],
          [ 8.4574e-02, -1.3887e-01,  3.6244e-02],
          [-1.7148e-02, -1.7976e-02,  3.4678e-03]],

         [[ 1.5884e-02,  9.6930e-02, -1.3011e-01],
          [-2.7169e-02,  2.7982e-02,  1.2578e-02],
          [ 7.4484e-02,  2.6282e-02, -8.1135e-02]],

         [[-7.7897e-02,  4.0005e-02,  2.9468e-02],
          [-2.3082e-02, -2.4113e-03, -9.0629e-02],
          [ 7.8886e-02,  6.8174e-02, -1.2119e-02]],

         [[-6.1636e-02,  6.3532e-02,  5.5353e-02],
          [-2.0925e-02,  8.2857e-02, -7.9036e-02],
          [ 2.4303e-02, -8.7958e-03, -5.3295e-03]],

         [[ 3.4201e-03,  1.2547e-01, -2.3910e-02],
          [-6.0422e-02, -5.5193e-02, -1.2365e-01],
          [ 1.9403e-03,  8.1672e-02, -9.2559e-03]],

         [[ 2.0350e-02,  7.1315e-02, -1.0892e-02],
          [-4.3088e-02,  2.6993e-02,  3.3222e-02],
          [ 7.5250e-02,  6.3537e-02, -3.2949e-02]],

         [[-2.4287e-02,  9.0019e-02,  3.0099e-02],
          [-2.5179e-02, -8.2377e-02, -8.1787e-02],
          [-4.0933e-02,  9.7625e-02,  2.0411e-02]],

         [[ 6.2360e-02, -8.0176e-02,  6.4868e-02],
          [-2.3461e-02,  1.0596e-02,  5.2269e-02],
          [-1.0736e-01,  3.0957e-02,  1.0795e-01]]],


        [[[ 7.5298e-03,  4.9317e-02, -5.4100e-02],
          [-4.9309e-02, -1.0656e-01, -2.3644e-02],
          [ 1.5323e-02,  2.4311e-02,  1.6806e-02]],

         [[ 5.7922e-02, -2.7913e-02,  3.0335e-02],
          [-4.6011e-02, -5.5533e-03,  6.2334e-02],
          [-1.9384e-02, -3.9723e-02,  3.5685e-02]],

         [[-4.1791e-02,  6.9336e-04, -5.1828e-02],
          [-9.8278e-02, -7.1951e-02, -2.4941e-02],
          [ 4.4909e-03, -7.1752e-03, -6.0983e-02]],

         [[-6.2732e-03,  1.9484e-02, -4.0340e-03],
          [-1.2184e-01, -2.4692e-02, -5.4207e-03],
          [ 1.5850e-02, -3.1912e-02, -2.4581e-02]],

         [[ 5.5023e-02, -1.1732e-02,  4.2590e-02],
          [ 6.6264e-02,  2.1335e-02, -1.5046e-03],
          [-5.3811e-02,  1.4627e-02,  4.9602e-02]],

         [[ 1.9536e-03, -7.1692e-02,  1.1302e-02],
          [ 1.0620e-01, -7.6064e-02,  1.6040e-02],
          [-2.1610e-02, -6.7689e-02, -6.6593e-02]],

         [[-7.0460e-02,  1.3165e-02,  1.0935e-02],
          [-8.7494e-02, -4.2292e-02, -1.1525e-01],
          [ 4.7832e-02,  8.3762e-02,  3.3533e-02]],

         [[ 9.5281e-02,  8.7297e-02,  8.3311e-02],
          [-8.6181e-02, -1.3051e-01, -9.9166e-02],
          [-5.3944e-02, -1.5536e-02, -2.8751e-02]],

         [[ 2.8813e-02,  8.7517e-02, -2.2088e-03],
          [-9.1792e-02, -5.6518e-02, -8.9139e-04],
          [ 5.1254e-02,  6.3829e-02,  3.1110e-02]],

         [[-5.5174e-02, -7.7032e-02,  4.7148e-02],
          [-8.5000e-03,  2.3944e-02, -1.2337e-03],
          [-3.3044e-02,  4.3901e-02,  4.5526e-02]],

         [[-3.6186e-02, -8.6961e-02, -5.8954e-02],
          [ 5.5157e-02,  5.1039e-02,  9.0485e-02],
          [-7.6668e-02, -4.2682e-02, -8.3614e-02]],

         [[ 2.9008e-02, -1.7198e-02,  2.1831e-02],
          [ 2.5820e-02, -3.2811e-02,  4.9659e-02],
          [ 3.4683e-02, -1.5642e-02, -4.6887e-02]],

         [[ 4.4646e-02, -1.0001e-01,  6.1768e-02],
          [-6.0728e-02, -6.3416e-02, -6.2605e-02],
          [-7.3079e-02,  5.5195e-02,  4.6491e-02]],

         [[-9.1042e-02, -9.3366e-02, -1.2195e-01],
          [ 5.2215e-03,  1.6116e-02,  9.9428e-02],
          [-5.9583e-02, -6.9318e-02, -6.3361e-02]],

         [[ 5.9195e-02,  6.1784e-02, -4.2745e-02],
          [ 2.7093e-02,  3.4810e-02,  2.0607e-02],
          [ 2.2935e-03, -6.1924e-02,  5.4104e-04]],

         [[ 4.1123e-02,  2.1395e-02,  6.4190e-02],
          [-3.1012e-02,  3.3908e-02,  1.2664e-03],
          [ 7.0109e-02, -7.7351e-02, -1.5059e-02]],

         [[-9.3766e-02, -5.9033e-02, -2.6175e-02],
          [ 4.1806e-02,  7.7306e-02,  5.6585e-02],
          [-8.3250e-02, -8.8996e-02, -3.0068e-02]],

         [[-6.2351e-02,  9.9470e-02,  3.4134e-02],
          [-7.5397e-02, -2.9672e-02, -6.9652e-02],
          [-5.7609e-03, -3.9014e-03, -8.4600e-02]],

         [[-2.4852e-02,  4.0369e-02, -2.6399e-02],
          [ 4.0485e-02,  4.6439e-02, -2.7938e-02],
          [-4.3527e-02,  3.9948e-03,  3.4089e-02]],

         [[-4.3293e-02, -6.3678e-02, -1.0190e-01],
          [ 8.4578e-02, -7.3760e-03,  5.7695e-02],
          [-1.9083e-02,  5.8135e-03, -1.9925e-02]],

         [[-7.6750e-02, -2.6626e-02,  2.0505e-02],
          [ 1.0381e-01,  4.4216e-02,  3.3286e-02],
          [-7.8383e-02, -5.8458e-02, -2.2336e-02]],

         [[-1.3327e-02,  1.5806e-02, -7.6679e-02],
          [ 6.0501e-02, -9.0123e-02, -2.2877e-02],
          [-4.5698e-02,  1.4510e-02, -6.7598e-02]],

         [[-2.9235e-02, -3.7720e-02, -7.0888e-02],
          [-4.5351e-02, -2.4526e-02,  2.5322e-02],
          [-4.4418e-02,  8.8335e-02,  1.2343e-02]],

         [[ 1.9123e-02,  5.3109e-03, -2.0273e-02],
          [ 3.4007e-02,  2.1407e-02,  2.6202e-02],
          [-1.0723e-02, -4.1115e-02, -2.2252e-03]],

         [[-3.1488e-02, -2.9140e-02, -4.0308e-02],
          [ 4.2804e-02,  4.6667e-03, -6.6896e-02],
          [-1.5279e-02,  3.9422e-02, -1.1006e-02]],

         [[ 7.2173e-02, -4.3850e-02,  3.3892e-02],
          [ 9.0161e-03, -4.2828e-02,  3.9031e-02],
          [-3.3355e-02, -2.5261e-03,  6.2383e-02]],

         [[-3.9761e-02, -1.1391e-01, -6.8016e-02],
          [ 4.0863e-02,  8.9735e-04,  3.5218e-02],
          [-3.0774e-02,  9.7984e-03, -8.5301e-02]],

         [[-7.9863e-02, -1.3397e-01, -7.5622e-02],
          [ 1.3038e-02,  2.4249e-02, -2.7757e-03],
          [-4.4330e-02, -3.4531e-02, -6.6113e-02]],

         [[-3.1669e-02, -3.9193e-03, -5.0024e-02],
          [ 3.4745e-02,  3.4789e-02,  7.5235e-02],
          [-8.7000e-02, -9.9306e-02, -2.5859e-02]],

         [[ 3.0564e-02, -3.0244e-02,  1.3527e-02],
          [-2.7868e-02,  4.1107e-02,  6.1981e-02],
          [ 2.4249e-02, -5.3205e-03, -4.6274e-02]],

         [[ 5.0577e-02, -6.7418e-03, -3.4402e-02],
          [ 7.4614e-02,  4.6884e-02,  8.0753e-02],
          [-9.3757e-02, -7.4299e-02, -3.2129e-02]],

         [[ 2.3381e-02,  1.1117e-01, -7.8453e-02],
          [-3.6377e-02, -2.9616e-02, -6.1035e-02],
          [ 5.9142e-02, -8.7852e-03, -9.4243e-03]]],


        [[[-7.0215e-02,  1.4731e-02, -5.9991e-02],
          [-7.4467e-02,  6.7929e-02,  4.2103e-02],
          [-1.0836e-02,  3.7091e-02,  3.2729e-02]],

         [[ 8.2207e-03,  1.5169e-02, -3.7679e-02],
          [-1.1949e-03, -3.5520e-02, -3.1193e-02],
          [-3.6151e-02, -7.2270e-02, -7.1590e-02]],

         [[-1.6033e-02, -7.1093e-02,  1.1893e-02],
          [-5.1258e-02, -6.4429e-02, -1.1892e-01],
          [-8.9223e-02,  5.3292e-02,  6.2581e-02]],

         [[ 1.6174e-02, -1.8148e-02, -1.1799e-01],
          [ 1.2742e-02, -4.7859e-02, -4.5778e-02],
          [ 4.2108e-02, -1.2463e-01, -9.1687e-02]],

         [[ 4.2086e-02,  4.6975e-02, -3.6872e-02],
          [-1.6216e-02,  2.8050e-02, -5.5600e-02],
          [-1.4207e-02,  1.4878e-02,  2.9636e-02]],

         [[-5.5037e-02,  5.0754e-02,  3.1944e-02],
          [ 3.3046e-02, -3.9702e-03, -3.6994e-02],
          [ 4.2224e-02, -3.9888e-02,  2.0462e-02]],

         [[-3.7860e-02, -1.2397e-01, -2.8073e-02],
          [ 6.2976e-03,  5.5834e-02,  9.8357e-03],
          [-7.6402e-02, -1.2146e-01, -3.3020e-02]],

         [[ 4.7160e-02, -1.1358e-01, -6.8848e-02],
          [ 6.9544e-03, -4.0561e-02, -1.5989e-02],
          [-2.7612e-02, -2.4000e-02, -4.9723e-03]],

         [[ 3.6181e-02, -1.0208e-01, -6.0371e-02],
          [-1.0602e-01,  1.7900e-02,  5.4220e-02],
          [ 7.5913e-02, -1.1100e-03,  1.5033e-02]],

         [[ 2.9195e-03, -1.1166e-01, -2.2270e-02],
          [-3.1923e-02, -2.4110e-02,  1.0121e-01],
          [-7.7503e-02, -5.8247e-02, -6.6991e-02]],

         [[-3.7129e-02,  3.3232e-03,  1.0668e-01],
          [-1.7692e-02,  2.1155e-03, -4.0516e-02],
          [-1.0218e-01,  2.2185e-02,  6.0411e-02]],

         [[-7.5941e-02,  7.5772e-03, -2.4113e-02],
          [ 2.0209e-02, -3.6324e-03, -2.2778e-02],
          [-1.7004e-02, -3.9337e-02,  2.1051e-02]],

         [[ 1.3255e-02, -5.1523e-02, -1.0465e-01],
          [-3.9692e-02, -2.7804e-02,  2.8386e-02],
          [-2.9826e-02,  7.8739e-02, -1.7878e-02]],

         [[-2.8801e-02, -6.6191e-02,  5.9342e-02],
          [ 1.0369e-02, -3.2304e-02,  2.3459e-03],
          [ 5.0911e-03,  1.0222e-01,  1.3805e-02]],

         [[ 2.4656e-03,  4.7873e-02, -4.1817e-02],
          [ 4.5174e-02,  3.2551e-02,  4.0070e-03],
          [ 8.5970e-02,  6.3715e-02, -5.6903e-02]],

         [[-6.0242e-02,  2.6242e-02,  3.4043e-02],
          [-1.6797e-02, -1.0970e-02,  1.3050e-02],
          [ 3.3680e-02, -5.3083e-02, -2.3704e-02]],

         [[-6.0669e-02,  8.7143e-02,  1.0241e-01],
          [-3.6170e-02,  2.7830e-02, -4.1605e-02],
          [ 7.2559e-02,  5.2725e-02,  6.3410e-02]],

         [[-4.0320e-02, -1.4674e-03, -8.7253e-02],
          [-4.9895e-03, -8.2455e-02, -9.3597e-04],
          [ 6.5947e-03, -1.1153e-01,  5.2847e-02]],

         [[ 6.8768e-02,  2.7174e-02,  3.2940e-02],
          [-5.8368e-02, -3.6368e-02, -1.0234e-02],
          [ 2.0856e-02,  4.4614e-02, -8.2717e-02]],

         [[ 9.8119e-03,  3.1606e-02,  8.3938e-02],
          [-2.7607e-02, -4.7683e-02,  7.7946e-03],
          [-9.0074e-02,  6.8211e-02,  1.0842e-01]],

         [[ 4.6652e-02, -9.2956e-02,  7.9160e-02],
          [-3.5853e-02,  9.7222e-03, -2.9721e-02],
          [ 4.3200e-03, -4.5733e-02, -2.8366e-02]],

         [[-4.5439e-02,  1.0518e-01,  1.2410e-01],
          [ 9.0039e-02, -2.5850e-02, -5.9968e-02],
          [ 1.1514e-02, -4.3649e-02,  8.9983e-02]],

         [[-8.2563e-02,  2.7177e-02,  2.0189e-02],
          [-2.5765e-05,  1.7103e-02, -3.9163e-03],
          [-5.1735e-02,  8.3774e-02,  3.1986e-02]],

         [[ 3.6395e-02,  1.9724e-02,  5.5282e-02],
          [-5.1772e-02, -3.9764e-02, -3.9124e-03],
          [ 1.0173e-02,  8.7773e-02,  5.0125e-02]],

         [[ 7.4366e-02, -2.3026e-02, -3.7272e-02],
          [-3.5630e-02,  2.6713e-02,  1.0236e-01],
          [-6.1238e-02,  1.8026e-02,  4.3291e-02]],

         [[-6.7042e-02,  7.7157e-02, -5.2955e-02],
          [ 9.5176e-02, -8.8767e-02, -2.3706e-02],
          [ 7.7054e-03, -8.2085e-02, -5.6844e-02]],

         [[ 3.1210e-02, -1.1657e-01,  1.1584e-01],
          [-5.0448e-02, -7.3817e-03,  4.4296e-02],
          [-9.2096e-02,  4.2287e-02,  2.4939e-02]],

         [[-6.1249e-02, -5.5414e-02, -6.2262e-02],
          [-3.1128e-03,  1.5679e-03, -4.4620e-02],
          [-1.0798e-01,  2.5729e-02, -5.4885e-03]],

         [[-1.2078e-02, -6.5494e-02,  2.6759e-02],
          [ 3.1387e-02, -3.2674e-02,  5.4296e-03],
          [ 2.3013e-02,  5.3501e-02,  1.8830e-02]],

         [[ 4.6387e-02,  1.4137e-02, -4.3684e-02],
          [ 5.4560e-02, -2.9885e-02,  2.0589e-02],
          [-2.8576e-02,  1.0720e-03, -1.9855e-02]],

         [[-5.3040e-02, -3.7731e-02,  3.8957e-02],
          [-3.4439e-02,  4.4372e-02, -6.5669e-02],
          [ 1.5763e-02, -2.0215e-02, -7.7294e-03]],

         [[-3.7524e-02, -3.5054e-02, -1.5068e-02],
          [-4.2647e-02,  7.9414e-02,  5.8107e-02],
          [-6.6799e-03,  2.0301e-02,  3.5399e-02]]],


        [[[ 3.9768e-02, -5.4839e-02, -2.3059e-02],
          [ 2.8657e-02, -6.3676e-02, -2.4720e-02],
          [-3.5924e-02, -1.5144e-02,  2.6321e-02]],

         [[-5.1335e-03, -2.4362e-02, -8.0389e-02],
          [-6.8040e-02, -7.0069e-02, -2.1603e-03],
          [-1.1116e-02, -5.0405e-02, -4.8595e-02]],

         [[-5.7529e-04,  2.9736e-02, -2.7981e-02],
          [-7.2786e-03,  2.1772e-02, -3.2691e-02],
          [-3.4376e-02, -1.4617e-02,  2.0164e-02]],

         [[-8.6065e-02,  5.5385e-02,  1.2881e-02],
          [-8.2884e-02, -8.2119e-02, -3.6315e-02],
          [-3.8394e-03,  1.7324e-02,  3.6906e-03]],

         [[-7.0948e-03,  1.8242e-02, -5.0546e-02],
          [-1.5881e-02, -5.1451e-02, -4.6831e-02],
          [ 2.3963e-02, -7.0218e-02, -9.6684e-03]],

         [[-2.7072e-02,  6.6626e-02, -6.3134e-02],
          [-5.5025e-02, -9.1269e-03, -2.6137e-02],
          [-6.3307e-04,  1.2466e-02, -4.5093e-02]],

         [[-5.5889e-02,  7.1528e-03,  4.1995e-02],
          [-6.0303e-02,  1.5920e-02, -4.8657e-02],
          [ 7.7193e-03, -2.0428e-02,  7.2935e-02]],

         [[-7.1871e-02, -2.5259e-02,  4.9551e-02],
          [ 1.3130e-02, -6.3894e-02, -5.9661e-02],
          [ 2.2767e-02,  2.4667e-02,  2.1342e-02]],

         [[ 4.2429e-02, -4.0850e-02, -6.2511e-04],
          [-6.8206e-02, -6.5461e-02, -4.1207e-03],
          [ 6.9316e-04, -4.8049e-02,  5.1524e-02]],

         [[ 1.6514e-02, -8.7270e-02,  8.1700e-02],
          [-3.7346e-04, -7.8250e-02, -3.1672e-02],
          [ 7.3727e-03, -9.8766e-03, -2.2746e-02]],

         [[-1.1634e-02,  2.2902e-02,  1.3496e-02],
          [-7.7477e-02, -6.3820e-02,  1.6124e-02],
          [-4.8905e-02,  1.1256e-02,  7.5764e-02]],

         [[-1.4935e-02, -5.2788e-02, -8.8366e-02],
          [ 4.1137e-03, -1.6338e-02, -6.1373e-02],
          [ 2.2356e-02, -5.3611e-02, -4.5323e-02]],

         [[ 1.7013e-03, -6.2473e-02,  4.8635e-03],
          [-7.1492e-03,  1.1065e-02, -2.7911e-02],
          [-6.1864e-02,  6.4350e-03, -1.2855e-02]],

         [[ 1.4537e-03,  7.0410e-02, -5.1516e-02],
          [-7.0600e-02,  7.3878e-02, -9.6630e-04],
          [-5.9270e-02, -3.5542e-02,  2.2005e-02]],

         [[-3.2334e-02, -2.1346e-02, -5.4958e-02],
          [-2.1765e-02, -4.4907e-02, -2.8704e-02],
          [-5.6861e-02,  2.2807e-02, -8.0638e-02]],

         [[-5.2468e-02, -6.7379e-02,  2.4657e-02],
          [-4.6684e-02, -7.2306e-02,  9.4308e-03],
          [ 2.6446e-02, -3.0158e-02, -8.4763e-03]],

         [[-2.9234e-02,  4.8794e-02, -1.7138e-02],
          [-3.3945e-02,  4.3350e-02,  2.7379e-02],
          [-6.0287e-02, -4.8430e-02, -6.0395e-02]],

         [[-6.7064e-02,  3.3010e-02,  2.6265e-02],
          [ 6.4754e-02, -1.5797e-02, -2.8197e-02],
          [ 4.4132e-02,  1.3687e-02,  2.6522e-02]],

         [[-7.0384e-02, -1.2005e-02, -4.2502e-02],
          [ 2.5807e-02,  5.2431e-03, -9.3821e-03],
          [-3.9924e-02, -4.1819e-02, -1.9613e-02]],

         [[ 2.1331e-02,  5.3833e-02, -1.3742e-02],
          [-7.0092e-02,  6.3302e-02, -5.4332e-02],
          [-5.5788e-02, -8.8770e-03, -7.7848e-02]],

         [[ 2.3569e-02, -1.1759e-02, -3.6511e-02],
          [ 3.9572e-03,  2.2591e-02,  2.5226e-02],
          [-2.8306e-02,  2.8930e-02, -4.9753e-02]],

         [[ 5.7937e-04,  4.3526e-03, -3.4842e-02],
          [ 1.8243e-02,  2.2664e-02,  9.0889e-03],
          [-1.6034e-02,  8.4308e-02, -5.3515e-02]],

         [[ 1.1457e-03, -2.7011e-03,  4.6095e-02],
          [ 5.2799e-02, -5.5222e-02,  1.7246e-02],
          [-1.0890e-02,  7.9364e-02, -1.2608e-02]],

         [[ 2.6083e-02,  2.5308e-02, -1.6524e-02],
          [-3.2128e-02,  1.4560e-02, -1.1226e-02],
          [-3.2218e-02, -1.3647e-02, -5.6901e-02]],

         [[-3.1763e-02, -7.2582e-02, -9.4704e-03],
          [-1.1504e-02, -1.1506e-02,  1.4101e-02],
          [-7.1116e-02, -2.5198e-02,  3.6494e-03]],

         [[ 1.1767e-02, -8.4804e-03, -1.9801e-02],
          [ 5.2903e-03,  1.9694e-02, -1.6091e-02],
          [ 2.2790e-02, -2.2404e-02, -2.2820e-02]],

         [[ 2.2276e-02,  2.8140e-02, -6.4891e-02],
          [-8.5524e-02,  4.7633e-02, -1.8841e-03],
          [-4.5928e-02,  6.8658e-02,  1.0578e-02]],

         [[ 1.0869e-02,  5.6196e-02, -3.7726e-02],
          [-2.1014e-02, -2.1865e-02,  2.8411e-02],
          [-5.9639e-02,  1.9338e-02, -2.6678e-02]],

         [[-3.6103e-02,  8.2459e-03, -2.5316e-02],
          [-8.4418e-02,  2.0100e-02, -7.5174e-02],
          [ 1.0460e-02,  7.7730e-02, -7.1711e-02]],

         [[-5.9501e-02, -4.5153e-02, -3.7054e-02],
          [-4.5656e-02,  2.8201e-02, -3.2060e-02],
          [-7.4908e-02, -4.6227e-02, -1.0622e-02]],

         [[-2.1227e-02, -4.9377e-02, -2.6669e-02],
          [-4.6723e-02,  3.8609e-03, -1.5181e-03],
          [-3.8878e-02, -5.7514e-02,  2.1133e-02]],

         [[ 2.9321e-02, -7.6109e-02,  8.0783e-02],
          [-1.5048e-02,  2.8539e-02,  8.0720e-02],
          [ 1.7248e-02, -4.5534e-02,  9.4210e-03]]],


        [[[ 3.7659e-03, -2.7087e-02,  6.5737e-03],
          [ 3.3041e-03,  1.4049e-02,  3.6812e-02],
          [-1.0139e-01, -8.3461e-02,  1.3809e-02]],

         [[-3.3165e-03, -6.3271e-03,  5.1503e-02],
          [ 3.0901e-02,  5.3166e-02, -2.1185e-03],
          [-6.0858e-03,  8.7556e-02,  3.3556e-02]],

         [[-2.5756e-02, -3.7204e-02,  1.5997e-03],
          [-2.2575e-02,  1.1500e-01, -1.1405e-01],
          [-1.3604e-02, -8.4054e-02,  7.9079e-02]],

         [[-1.2590e-03, -3.7038e-02, -1.4405e-02],
          [ 7.8117e-02,  8.5449e-02,  9.2032e-02],
          [ 1.3546e-01,  1.0814e-01, -1.4509e-03]],

         [[ 2.7085e-02,  3.2244e-03,  5.7634e-03],
          [ 6.9419e-02, -1.5272e-03,  4.7319e-02],
          [ 2.4748e-02, -8.7998e-03,  3.4194e-03]],

         [[ 4.5808e-02, -1.0391e-01,  7.5343e-02],
          [-9.2746e-02, -1.7874e-02,  7.0633e-02],
          [ 3.2872e-02,  1.1166e-01,  1.0402e-01]],

         [[ 6.4436e-02,  2.8461e-02, -1.2968e-02],
          [ 1.2385e-01,  4.5179e-02,  4.9795e-02],
          [ 6.0705e-02, -3.7413e-02,  5.9279e-03]],

         [[-1.7345e-02,  2.9983e-02, -5.3375e-02],
          [ 1.2595e-01,  1.3247e-01, -2.5112e-02],
          [ 9.8248e-02,  7.6659e-02,  2.1591e-02]],

         [[ 2.3468e-02, -3.1186e-02, -5.1632e-02],
          [ 6.0313e-02,  2.8224e-02, -1.1883e-01],
          [-3.1446e-03,  7.8111e-02, -3.6302e-02]],

         [[ 7.5483e-02,  1.2716e-01, -6.5340e-02],
          [ 5.1501e-02, -4.6798e-02, -1.0608e-02],
          [-2.8640e-02, -4.1121e-02, -1.0581e-01]],

         [[ 3.1656e-02,  4.8596e-02,  1.4357e-03],
          [-1.7069e-02, -9.6629e-02, -5.5395e-02],
          [ 4.7051e-03, -1.5387e-02, -6.6867e-02]],

         [[ 1.8932e-02,  1.3348e-02, -7.4295e-03],
          [-6.8566e-02, -4.3236e-02, -9.3841e-04],
          [-3.1026e-02,  6.1699e-02, -8.4704e-03]],

         [[-2.9133e-02,  8.7122e-02,  3.1014e-02],
          [ 1.3448e-01,  1.9170e-02, -6.6843e-02],
          [ 2.0755e-03, -2.2144e-03,  7.0308e-02]],

         [[ 1.2435e-01,  1.1824e-02,  7.3219e-02],
          [-3.6409e-02, -1.0276e-01, -2.2937e-02],
          [-5.3878e-02, -1.7528e-02,  3.9000e-02]],

         [[-3.1311e-02,  8.3368e-03, -2.9752e-02],
          [-1.5188e-02,  6.9436e-02, -9.9442e-04],
          [-6.5225e-03,  3.3659e-02,  9.3453e-02]],

         [[-2.5456e-02, -4.9589e-02, -9.0740e-04],
          [ 1.4261e-02, -8.7038e-04, -2.9116e-02],
          [-5.8927e-02,  7.9366e-02,  3.9448e-02]],

         [[ 1.2884e-01, -2.2824e-02,  7.7747e-02],
          [-1.4128e-02, -4.9982e-02,  4.4552e-02],
          [ 2.1108e-02, -2.1588e-02,  4.9516e-02]],

         [[ 8.7946e-03,  8.9026e-02, -9.4313e-02],
          [ 8.9322e-02,  6.5992e-02, -6.2526e-02],
          [ 2.7019e-02, -3.3135e-02, -4.7351e-02]],

         [[-1.5217e-02, -4.7019e-02,  2.0829e-02],
          [ 9.8975e-02,  1.5891e-02, -2.2714e-02],
          [ 7.6041e-02,  6.6297e-03,  3.6782e-02]],

         [[ 4.8459e-02,  8.5233e-02,  8.3286e-02],
          [-1.8617e-02, -5.1267e-02, -3.9079e-02],
          [-5.4261e-02, -1.3710e-01,  3.6915e-03]],

         [[-8.6164e-03,  9.6929e-02, -3.6008e-02],
          [-1.1237e-01,  1.2957e-03, -7.9167e-02],
          [-8.0269e-02, -4.9107e-02, -5.5928e-03]],

         [[ 6.7824e-02, -2.8460e-04,  5.3951e-02],
          [-4.2368e-02,  5.3838e-02,  1.1609e-01],
          [ 3.0272e-02, -1.0612e-02,  5.0526e-02]],

         [[-1.3829e-02,  6.1245e-02,  1.5581e-02],
          [ 2.3756e-02, -2.2482e-02,  7.3827e-02],
          [ 6.0291e-02, -8.3850e-03, -6.8522e-02]],

         [[ 6.1394e-02,  4.0777e-02, -4.4355e-02],
          [ 2.5654e-02, -9.9908e-02,  3.4705e-02],
          [ 2.0131e-02, -6.1466e-02, -3.0898e-02]],

         [[-1.6173e-04,  1.0641e-01, -3.2814e-02],
          [-2.6713e-02, -9.8115e-02, -9.1817e-02],
          [ 8.1359e-02, -1.6838e-02,  4.9087e-02]],

         [[-1.0340e-02, -3.8441e-02,  2.1337e-02],
          [ 4.2526e-02,  7.9528e-02,  8.5316e-02],
          [ 3.4000e-02,  3.0970e-02, -2.8833e-04]],

         [[ 5.4398e-02,  6.6448e-02, -3.6252e-02],
          [-8.0045e-02, -6.1073e-03, -2.2881e-02],
          [ 1.0415e-01, -4.6532e-02,  3.6461e-02]],

         [[ 1.3130e-01,  6.7986e-02, -2.7687e-02],
          [-2.2692e-02, -4.1410e-02, -4.4369e-02],
          [ 3.0607e-02,  1.4996e-02,  1.8247e-02]],

         [[ 1.0341e-01,  1.6548e-02, -2.0909e-02],
          [-1.2692e-01, -7.7195e-02, -8.6789e-02],
          [ 2.7615e-02,  6.4509e-02, -2.4422e-03]],

         [[ 6.6648e-02, -5.8131e-03, -3.0116e-02],
          [-1.2369e-02,  7.7239e-02,  3.5184e-02],
          [-7.2719e-02,  3.2931e-03, -4.3273e-02]],

         [[ 2.6855e-02,  4.0429e-02, -2.6510e-02],
          [-4.7200e-02, -7.9520e-02,  4.0759e-02],
          [-2.5228e-02,  4.7991e-02,  7.1521e-02]],

         [[ 1.5550e-02, -3.7294e-02, -6.1062e-02],
          [ 1.6450e-02, -7.1545e-04,  4.7867e-02],
          [-1.1278e-02, -1.5987e-02, -2.7122e-02]]],


        [[[-2.9205e-02, -9.5312e-03,  4.6658e-02],
          [ 5.3103e-02,  3.2484e-02, -2.8581e-02],
          [-2.3366e-02, -1.8976e-02, -3.3201e-02]],

         [[ 1.1724e-02,  5.6833e-03, -2.5968e-02],
          [-6.7398e-02,  3.2234e-02, -3.2697e-02],
          [ 1.9872e-02,  3.6813e-02,  4.6594e-03]],

         [[-2.4764e-03, -3.1457e-03,  1.8694e-02],
          [ 5.5380e-02, -2.2701e-02, -8.3869e-02],
          [-2.2289e-02, -4.5867e-02, -3.8715e-02]],

         [[ 4.2970e-02,  4.4933e-02,  3.2612e-02],
          [ 4.5863e-02,  9.9792e-02, -5.9087e-02],
          [ 1.3987e-02,  7.1445e-03, -6.4184e-02]],

         [[-5.0802e-02, -1.0983e-02, -5.2397e-02],
          [ 4.0768e-02, -7.9514e-02, -2.9187e-03],
          [-6.5828e-02, -1.0291e-02, -7.5481e-02]],

         [[-5.5753e-02, -5.7525e-03, -8.0224e-03],
          [-1.3994e-01,  1.1257e-01,  9.0392e-02],
          [ 2.3054e-02,  6.1926e-02, -5.3350e-02]],

         [[-5.5585e-02,  8.7141e-02, -7.6907e-02],
          [-1.5805e-02, -2.4846e-02, -9.0751e-02],
          [-3.7080e-02,  4.8333e-03, -1.3763e-02]],

         [[-9.0157e-02,  5.5238e-02, -7.6915e-02],
          [ 4.6460e-02, -2.0360e-02, -6.6362e-02],
          [-3.2877e-02,  2.7617e-03, -3.0398e-03]],

         [[-4.6606e-02,  6.1556e-02, -1.0930e-01],
          [-1.9857e-04,  1.1466e-02, -1.0044e-01],
          [ 2.5508e-02,  6.3384e-02, -1.0117e-02]],

         [[-5.5300e-02,  2.7563e-02, -6.1740e-02],
          [-4.2907e-02, -6.9194e-02, -7.4019e-02],
          [-1.3770e-02, -4.4238e-02,  1.3722e-02]],

         [[-4.7648e-02, -5.6359e-02, -3.8904e-03],
          [-8.8775e-02, -2.9724e-02,  1.0823e-01],
          [-1.1353e-01, -1.2433e-01, -5.3940e-02]],

         [[-1.8933e-03, -6.4892e-02, -1.2668e-02],
          [-2.5085e-02,  6.2413e-02,  4.4523e-02],
          [-2.5064e-02, -5.6812e-02,  6.1231e-03]],

         [[-1.2202e-02, -3.1564e-02,  9.8984e-02],
          [ 3.7632e-02,  7.9713e-02, -2.7429e-02],
          [-6.8695e-02, -3.6213e-02, -4.3257e-02]],

         [[ 4.7174e-02, -6.0821e-02,  3.2357e-02],
          [-1.6321e-02,  8.3788e-02,  2.5165e-02],
          [-3.2511e-02,  1.1891e-02, -2.5823e-02]],

         [[-7.6576e-02, -1.0267e-02,  3.9377e-02],
          [-2.1786e-02,  7.4281e-02,  8.3004e-03],
          [-4.7680e-03,  1.6057e-02, -4.6537e-02]],

         [[ 7.0832e-03, -3.5499e-02, -7.0715e-02],
          [-9.2954e-02, -6.9539e-03, -4.4985e-03],
          [-1.4110e-03, -3.0282e-02, -4.0342e-02]],

         [[-1.8681e-02,  6.2764e-03, -2.4776e-02],
          [-7.9131e-02,  1.2593e-01,  9.6531e-02],
          [-3.5531e-02, -8.1123e-02, -6.7659e-02]],

         [[-2.3901e-02,  5.0952e-03, -9.0078e-02],
          [ 1.6429e-02,  1.1058e-02,  5.0349e-02],
          [ 3.2180e-02,  2.0671e-02, -1.0986e-01]],

         [[ 1.1720e-02, -7.8413e-02,  3.4949e-02],
          [-1.8797e-02,  7.0830e-02, -2.5487e-02],
          [-9.1843e-02,  1.0447e-02,  3.6646e-02]],

         [[ 1.4078e-02, -7.1357e-02, -3.0281e-02],
          [-4.9502e-02,  3.1067e-02,  6.5075e-02],
          [-8.4876e-02, -2.3383e-02,  1.7829e-02]],

         [[ 1.2825e-03, -4.6913e-02,  3.2709e-03],
          [-1.0391e-01, -4.4620e-02,  1.0099e-01],
          [ 9.2456e-03, -7.4687e-02,  4.3886e-02]],

         [[-5.6739e-02, -3.0400e-02,  2.2613e-02],
          [-6.4676e-03,  1.0274e-01,  1.2887e-01],
          [-8.7351e-02,  2.2339e-03, -5.5379e-02]],

         [[-5.4723e-02,  5.4250e-02, -2.2295e-02],
          [-7.9028e-02,  4.5366e-02, -1.4699e-02],
          [-6.5822e-02, -3.5518e-02, -1.6361e-02]],

         [[-5.9672e-02, -6.5879e-02,  2.5189e-02],
          [ 2.2213e-03, -1.0565e-01,  2.4501e-02],
          [-6.6660e-02, -7.4295e-02, -2.9284e-02]],

         [[ 6.0181e-03,  2.7246e-02,  4.1100e-02],
          [-9.1554e-02,  6.3611e-03, -1.9236e-02],
          [-2.8715e-02, -1.1392e-01,  6.0379e-02]],

         [[-1.3796e-02, -6.1085e-02,  1.4703e-02],
          [-5.8786e-02,  2.9475e-03, -3.0288e-02],
          [-5.4216e-02,  6.0301e-02, -4.6334e-02]],

         [[ 2.5665e-02, -2.4813e-02,  2.7628e-02],
          [-7.0707e-02,  4.8779e-02, -4.9467e-03],
          [-2.2374e-02, -6.5082e-02, -1.0812e-01]],

         [[ 5.8320e-02, -8.3748e-02, -4.9709e-02],
          [-3.3300e-02, -3.1080e-02,  9.1606e-02],
          [-3.0493e-02, -4.9275e-02, -9.4891e-02]],

         [[ 1.4770e-02, -2.4259e-02, -8.5777e-02],
          [-8.0408e-02,  4.8908e-02,  8.6537e-03],
          [-1.0896e-01,  9.5721e-03, -1.1340e-01]],

         [[-5.9374e-02, -7.3824e-02, -1.7704e-02],
          [ 1.3324e-02,  4.9114e-02, -1.3279e-02],
          [-6.6645e-04, -7.4305e-02,  4.3659e-02]],

         [[-6.5408e-03, -4.2831e-02,  1.9023e-02],
          [-1.8224e-03, -5.9920e-02,  5.5169e-02],
          [-1.0937e-01, -8.1979e-02, -3.1072e-02]],

         [[-3.8284e-02,  6.7560e-02, -4.9323e-02],
          [-3.1110e-02, -7.0159e-02,  9.1695e-02],
          [-2.0687e-02, -9.5353e-02,  1.0261e-01]]],


        [[[ 3.2111e-02, -6.9804e-02, -1.8944e-03],
          [-7.2096e-02,  2.7609e-02,  4.8558e-03],
          [-5.9787e-02,  3.6114e-02,  2.1180e-02]],

         [[-1.6441e-02, -1.2134e-03, -3.9128e-03],
          [-3.7765e-02, -1.1599e-02, -7.2121e-02],
          [ 2.6424e-02,  7.4662e-03, -1.7465e-02]],

         [[-5.6239e-02, -6.5391e-02, -1.5087e-04],
          [ 2.9952e-02,  1.8010e-02, -1.6452e-03],
          [ 7.9762e-03,  3.1695e-02,  4.0454e-02]],

         [[-1.3520e-02, -2.1356e-02,  2.0913e-02],
          [-3.7654e-02, -4.8543e-02,  7.7646e-04],
          [-8.0465e-02,  1.5704e-02, -3.2459e-02]],

         [[-2.0460e-02,  6.6882e-04, -6.7929e-02],
          [-2.0681e-02,  1.9490e-02,  1.5225e-02],
          [-4.6095e-02, -3.4837e-02,  2.3620e-02]],

         [[-5.6073e-02, -6.8073e-03, -4.1961e-02],
          [-6.5999e-02,  1.3580e-02, -9.8800e-03],
          [-6.4529e-02,  3.0693e-02, -4.5522e-02]],

         [[-1.0524e-02, -4.3072e-02, -2.5589e-02],
          [-4.3458e-02, -6.8238e-02, -1.3877e-02],
          [-5.1495e-02, -3.6406e-02, -6.6447e-02]],

         [[-2.7672e-02, -2.2476e-02, -3.7993e-02],
          [-4.4439e-02, -5.5993e-02,  1.2849e-02],
          [-5.0609e-02,  3.6525e-02,  1.8544e-02]],

         [[ 4.9898e-02, -2.9933e-03,  2.2599e-02],
          [-5.3792e-02, -1.0209e-02, -6.7291e-02],
          [-4.8501e-02, -5.5902e-02, -5.7866e-02]],

         [[-1.3720e-02, -3.0950e-03,  8.2189e-03],
          [-2.9788e-02,  3.1745e-02,  1.6747e-02],
          [ 3.5479e-02,  2.4898e-02, -2.4716e-02]],

         [[ 2.1649e-03,  6.6171e-04, -5.6394e-02],
          [ 3.1225e-02, -1.7377e-02,  1.2605e-02],
          [ 2.6968e-02,  3.3830e-02, -3.7251e-02]],

         [[ 3.1735e-02, -5.6137e-02,  9.7920e-03],
          [-1.7413e-02,  2.0501e-02, -4.5115e-02],
          [-2.3243e-02,  3.3137e-02, -1.2776e-02]],

         [[-2.7184e-02, -7.8219e-02,  4.8061e-02],
          [-5.8236e-02, -3.5695e-02,  1.7576e-02],
          [-8.1354e-03, -4.9689e-03, -6.0640e-02]],

         [[ 1.6725e-02, -7.7116e-02,  2.9308e-02],
          [ 2.9137e-02,  4.8989e-02, -3.4260e-02],
          [ 2.4291e-02,  4.0170e-03, -4.5362e-02]],

         [[ 2.3878e-03, -2.7206e-03,  4.1852e-02],
          [-5.4849e-02, -3.7376e-02,  8.7787e-03],
          [-5.6502e-02, -6.8489e-02, -3.2069e-02]],

         [[-5.8411e-02,  2.6152e-02, -8.2001e-02],
          [-2.4092e-02,  1.4291e-02,  2.4310e-02],
          [-6.9623e-02, -1.4542e-02, -5.7937e-02]],

         [[ 1.9328e-02,  2.0732e-03, -1.6980e-02],
          [-3.1403e-02,  1.2413e-02, -2.5950e-02],
          [-3.3941e-03, -4.1715e-02, -2.3984e-02]],

         [[-3.2491e-02,  2.7237e-02,  2.5462e-02],
          [ 3.6155e-02, -2.5211e-02, -4.0755e-02],
          [-4.0825e-02, -1.3778e-02, -5.1117e-02]],

         [[ 1.3429e-02,  1.1798e-02,  1.8899e-03],
          [-4.9629e-02, -1.8328e-02, -1.1550e-02],
          [-3.8932e-02,  1.2807e-02,  8.4363e-03]],

         [[-3.4004e-03,  2.5427e-02, -4.3478e-02],
          [-3.1130e-02,  2.8928e-02,  2.2052e-03],
          [-4.3642e-02, -4.7256e-02,  1.8639e-02]],

         [[-6.5221e-02,  3.2654e-03, -5.9399e-02],
          [-6.2258e-02, -2.0161e-02,  6.6817e-03],
          [-5.9038e-02,  3.7911e-03,  8.8644e-03]],

         [[-1.3060e-02, -9.4103e-04,  3.3920e-02],
          [-6.4822e-02, -3.6151e-02, -4.8065e-04],
          [ 4.2328e-02,  3.1142e-02,  3.5528e-02]],

         [[-1.8024e-02,  3.0935e-03,  4.2166e-02],
          [ 1.6101e-02, -2.2218e-04, -2.6476e-02],
          [-3.9731e-02, -4.8492e-02, -5.8714e-02]],

         [[ 1.9635e-02,  2.7921e-02, -1.2591e-02],
          [-3.5807e-02, -5.5855e-02,  6.9096e-03],
          [-4.2401e-02, -3.6358e-04, -5.7464e-02]],

         [[ 2.7747e-02, -1.8613e-03, -2.8413e-02],
          [ 2.2572e-03, -4.6466e-02, -5.1427e-02],
          [-5.5746e-02, -6.0085e-02, -6.9883e-03]],

         [[-2.4158e-02, -6.9314e-02, -8.8167e-02],
          [-1.8772e-02, -5.5850e-03, -6.1831e-03],
          [-2.2804e-02,  1.6918e-02, -5.4152e-02]],

         [[ 3.3655e-02, -1.8777e-02, -4.3241e-02],
          [-3.2924e-02,  9.9881e-03, -2.5749e-02],
          [-4.5690e-02, -1.2506e-02, -5.9114e-02]],

         [[-3.0818e-02, -1.0151e-02,  1.6768e-02],
          [-1.9883e-02, -4.8705e-02,  3.0393e-02],
          [-5.1524e-02, -6.9493e-03, -4.1476e-02]],

         [[-3.8828e-02, -5.4269e-02, -5.5128e-02],
          [-4.1266e-02,  6.9997e-03,  4.1912e-02],
          [ 2.3371e-02, -5.3991e-02,  1.3729e-02]],

         [[-6.0271e-02, -6.9683e-02, -1.5944e-02],
          [-6.5267e-02, -1.2773e-02, -2.1890e-02],
          [-2.6568e-02, -2.3725e-02, -6.7477e-02]],

         [[-5.3551e-02, -5.1606e-02, -5.4459e-02],
          [-4.0433e-02,  4.6017e-03, -5.1083e-02],
          [ 1.4219e-02,  4.2143e-02, -1.7795e-02]],

         [[-3.6274e-02,  1.1668e-03,  1.9949e-02],
          [ 1.0739e-02, -4.5620e-02, -3.0767e-02],
          [-6.8144e-02, -4.4046e-02, -3.6525e-03]]],


        [[[-7.1379e-02, -2.1594e-02,  3.0490e-02],
          [-4.0512e-02, -4.8733e-02,  1.7222e-02],
          [-1.3534e-01,  1.2539e-03, -8.6967e-03]],

         [[-4.0158e-02, -3.8015e-02, -2.6648e-02],
          [-9.8723e-03, -4.1474e-03,  1.2032e-02],
          [ 5.7677e-02,  6.6146e-02, -5.3600e-02]],

         [[ 5.4928e-04, -2.6052e-02,  1.5168e-02],
          [ 5.3676e-02,  9.5257e-03, -5.4000e-02],
          [-8.4746e-02,  4.7106e-02, -1.7668e-04]],

         [[-1.0427e-01,  1.2639e-02,  3.4147e-02],
          [ 5.6254e-02, -4.9286e-02,  6.8023e-03],
          [ 9.9504e-02,  9.2321e-03, -3.0842e-02]],

         [[ 1.2624e-02, -6.6361e-02,  2.8162e-02],
          [ 5.4686e-02, -7.8854e-02,  1.3118e-02],
          [ 2.0497e-02, -8.1267e-02,  1.1648e-01]],

         [[-3.4953e-02, -1.0932e-01,  7.8491e-02],
          [-6.5682e-02, -6.3655e-02,  4.8842e-02],
          [-2.1645e-02, -6.0655e-02,  7.3500e-02]],

         [[-4.7204e-03,  9.7250e-02, -4.9205e-02],
          [-1.0624e-01, -2.7927e-02, -1.0117e-01],
          [ 3.1207e-02,  9.7907e-02, -1.3850e-01]],

         [[-2.4205e-02,  9.9100e-02,  6.3700e-02],
          [ 3.6158e-02,  6.2050e-02, -1.0940e-01],
          [ 4.5125e-02,  8.1404e-02, -1.1997e-01]],

         [[-1.0679e-02,  2.9774e-02, -6.1502e-02],
          [-4.7548e-02,  1.7688e-02, -4.1472e-02],
          [-1.0633e-02,  9.0375e-02, -3.5933e-02]],

         [[-2.3417e-02,  5.0439e-02, -8.2028e-02],
          [-1.0963e-01, -6.9751e-03, -9.1271e-02],
          [ 2.8403e-03,  6.2922e-02, -1.3725e-01]],

         [[ 3.7905e-02, -3.4499e-02,  4.7862e-04],
          [ 1.2326e-02, -5.8721e-02,  2.7118e-02],
          [-8.5920e-02, -1.7214e-03, -3.1583e-02]],

         [[ 3.1380e-02,  1.1929e-02,  1.8465e-02],
          [-3.7662e-02,  3.8757e-02, -4.4519e-02],
          [-2.8521e-02, -9.0430e-03, -3.8834e-02]],

         [[ 6.1030e-02,  9.2240e-03,  1.1559e-01],
          [ 2.5172e-02, -2.6654e-03,  3.8495e-02],
          [-5.9067e-02, -1.3316e-03, -9.0960e-02]],

         [[-2.7897e-02, -9.9728e-02,  1.0243e-01],
          [ 1.1221e-02, -2.0729e-02,  5.9567e-02],
          [-9.5273e-02, -5.1704e-02,  4.2379e-02]],

         [[-3.6063e-02,  1.5258e-02, -6.3500e-03],
          [-1.7114e-02,  2.6083e-02,  3.1665e-02],
          [ 5.3245e-02, -1.4542e-02,  5.9531e-02]],

         [[ 3.5606e-02,  3.4807e-02, -1.1109e-02],
          [-5.6958e-03,  9.2041e-02, -3.9654e-02],
          [ 1.6044e-02,  5.8464e-03, -8.7545e-02]],

         [[-1.1354e-02, -4.3570e-02, -1.6833e-02],
          [-3.5484e-02, -4.5059e-03,  2.4878e-02],
          [-1.0621e-01,  6.4075e-03, -1.6432e-02]],

         [[-8.5289e-02,  4.5282e-02, -3.0312e-03],
          [-6.5540e-02,  5.1215e-02, -5.8463e-02],
          [-3.5005e-02,  2.2611e-02, -8.3593e-02]],

         [[ 4.0919e-02, -2.8797e-02, -6.9099e-03],
          [ 7.1598e-02, -2.8827e-02,  1.5766e-02],
          [-5.3075e-02, -4.0519e-02,  9.9242e-02]],

         [[ 1.5426e-02,  2.2514e-03, -6.4385e-02],
          [ 5.5331e-02,  1.6356e-02, -3.9283e-02],
          [-3.7774e-02, -3.7833e-02, -1.2151e-02]],

         [[ 6.0942e-02, -1.3101e-01, -1.8035e-02],
          [-2.8788e-02,  2.1206e-02,  9.3520e-02],
          [-1.6425e-02, -3.8742e-03, -3.4631e-02]],

         [[-7.0617e-02, -5.6151e-02,  2.2076e-02],
          [ 1.1675e-01, -1.7395e-02,  8.1624e-02],
          [ 6.6185e-03, -1.7154e-02,  4.4862e-03]],

         [[-2.2543e-02, -5.0735e-02,  4.9130e-02],
          [-8.3357e-04,  1.5184e-02,  6.3657e-02],
          [ 4.5084e-02, -2.1326e-02,  5.7796e-02]],

         [[ 2.1000e-03, -7.9681e-03, -2.8501e-02],
          [ 2.3433e-02, -4.7776e-02,  4.1224e-02],
          [-1.3848e-02, -9.4446e-02,  4.8540e-02]],

         [[ 4.7813e-02, -8.4085e-02,  2.2749e-02],
          [ 4.1768e-02, -3.0048e-02, -2.2307e-02],
          [-1.1136e-01,  8.3433e-02, -7.8933e-02]],

         [[-4.5237e-02, -6.6291e-02,  6.1629e-02],
          [ 3.9457e-02,  7.5556e-03,  4.0059e-02],
          [ 5.1280e-02, -5.0341e-02,  5.3160e-02]],

         [[ 8.1411e-02, -1.2134e-01,  4.1251e-02],
          [-2.4271e-02, -3.8605e-02,  5.8641e-02],
          [-1.1225e-01, -7.4094e-02,  1.2315e-02]],

         [[ 4.2036e-02, -5.0543e-03,  2.0365e-03],
          [ 8.4657e-02, -1.1886e-01,  7.9315e-02],
          [-8.2498e-02, -4.8640e-02,  1.1231e-01]],

         [[-2.6338e-02, -4.6338e-02,  3.7287e-02],
          [-8.0681e-02, -1.1719e-02, -4.2099e-05],
          [-3.5214e-02, -6.8453e-02,  7.4481e-02]],

         [[-6.4134e-02,  2.4307e-02, -6.2349e-02],
          [-5.5774e-03,  9.0200e-03,  4.6235e-02],
          [ 3.2164e-02, -8.3466e-02,  8.0663e-02]],

         [[-5.8838e-02,  1.0814e-02,  1.7853e-02],
          [ 4.0860e-02, -6.7929e-03,  4.9736e-02],
          [ 2.5443e-02, -1.6225e-03,  5.4837e-02]],

         [[ 3.0843e-02, -1.6502e-02, -1.0477e-01],
          [-7.6371e-02,  4.5997e-02, -3.8771e-02],
          [-4.8896e-02, -5.8382e-02, -9.7355e-02]]],


        [[[ 1.2953e-02, -5.7465e-02,  3.2691e-02],
          [ 7.1639e-02,  5.4227e-02,  2.9457e-02],
          [ 6.4966e-03, -1.1234e-02, -5.0060e-02]],

         [[-4.3750e-02, -8.0452e-03, -8.5676e-02],
          [-6.6913e-02,  7.7816e-02, -8.8412e-03],
          [-3.4502e-02,  2.3681e-02,  2.9041e-03]],

         [[-6.3916e-03, -1.3379e-02, -5.2582e-02],
          [ 1.3182e-03,  8.0203e-02, -8.6400e-02],
          [-7.3622e-02, -2.1983e-02, -2.4451e-02]],

         [[-1.6971e-02, -1.3625e-02,  2.2260e-02],
          [ 6.4947e-02, -4.5098e-02,  2.7238e-02],
          [ 4.2651e-02, -2.5994e-02, -3.3923e-02]],

         [[-1.6093e-02, -4.8110e-02, -6.8297e-02],
          [-5.8949e-02,  1.4175e-02,  6.8761e-03],
          [-3.1112e-02, -8.4010e-02, -4.7717e-02]],

         [[-1.9325e-02,  1.2252e-02,  3.3882e-02],
          [-1.4688e-02, -4.1643e-02,  4.4448e-02],
          [ 7.0509e-03,  4.0080e-02, -1.5233e-04]],

         [[ 5.1486e-02,  3.6252e-02,  1.0589e-02],
          [-8.5543e-03,  5.7150e-02, -8.1334e-02],
          [-5.9902e-02, -2.0654e-02, -1.3557e-02]],

         [[-8.4787e-02,  3.4750e-03,  1.5708e-02],
          [ 2.1551e-02, -2.0162e-02,  2.7613e-02],
          [-4.9691e-02, -2.0098e-02,  5.4470e-03]],

         [[ 2.7375e-02,  2.9951e-02, -7.0243e-02],
          [-3.8552e-02,  7.1086e-02,  2.8966e-02],
          [-7.9378e-02,  5.3977e-02,  1.2190e-02]],

         [[ 5.5477e-02,  7.8748e-02, -6.3158e-02],
          [ 6.4798e-02, -2.7121e-02, -6.6397e-02],
          [ 1.9517e-02, -2.1653e-02,  2.8388e-02]],

         [[ 6.5072e-02, -1.8681e-02, -6.8054e-02],
          [-2.0225e-02,  1.6677e-02, -7.2658e-02],
          [-2.8339e-02, -1.3726e-02, -6.4873e-02]],

         [[-2.8060e-02, -4.1970e-02,  2.3850e-02],
          [ 8.0420e-03,  5.0792e-02, -3.1777e-02],
          [ 9.1102e-03,  5.4105e-02, -8.1697e-02]],

         [[ 2.3909e-02,  3.9553e-02,  8.8496e-02],
          [ 3.3568e-02, -3.3170e-02, -6.0663e-03],
          [-5.8634e-02,  5.5103e-03, -5.5700e-03]],

         [[ 1.2198e-02,  5.6053e-02, -3.1141e-03],
          [-7.1027e-02, -1.9140e-02, -4.8446e-02],
          [ 2.4147e-02, -5.3954e-03,  8.2696e-02]],

         [[ 2.8404e-02, -7.6257e-02, -4.7568e-02],
          [-3.5908e-02,  1.0074e-02, -1.3007e-02],
          [-7.0884e-04, -1.0844e-02,  7.5452e-03]],

         [[-2.4880e-02,  1.6351e-02, -7.9603e-02],
          [-8.7037e-02, -7.1419e-02, -6.6955e-02],
          [-3.7108e-03,  4.7793e-02,  1.2425e-02]],

         [[ 3.1494e-02,  7.6358e-02, -1.7517e-02],
          [-4.3989e-02, -2.6979e-02, -4.3876e-02],
          [-4.7913e-02, -7.2193e-02,  8.2714e-02]],

         [[ 3.2942e-02, -2.3404e-02,  1.1173e-02],
          [-2.0764e-02, -1.4187e-02, -4.0153e-02],
          [-8.7941e-03,  4.0512e-02,  4.6457e-03]],

         [[-8.7557e-02,  5.6117e-03, -6.7290e-02],
          [ 7.7278e-02, -5.1033e-02,  1.7807e-02],
          [-2.4605e-02, -7.7906e-03, -2.7815e-02]],

         [[ 4.7500e-02, -1.4155e-02, -6.8705e-02],
          [-5.0510e-02, -2.4565e-02, -4.0396e-02],
          [-3.7595e-02, -3.7412e-02, -4.2402e-03]],

         [[-1.4457e-02,  2.6136e-02, -2.5829e-02],
          [ 2.9112e-02,  2.5622e-02,  2.9087e-02],
          [-7.6336e-02,  2.8239e-02,  8.7229e-02]],

         [[ 1.5032e-02,  5.0684e-03, -5.4095e-02],
          [-7.2756e-02,  1.5683e-02,  5.4939e-03],
          [-3.4667e-03, -5.8901e-02,  1.6476e-03]],

         [[ 5.8215e-02,  5.4785e-02,  3.7634e-02],
          [-1.5979e-02,  2.9282e-02,  3.3270e-02],
          [ 8.2706e-02, -1.4777e-02,  2.7107e-02]],

         [[-3.7074e-02, -3.4851e-02, -7.4787e-02],
          [-7.3039e-02, -5.8511e-02, -1.4468e-02],
          [ 2.4595e-02, -4.3399e-03, -6.7477e-04]],

         [[-4.0424e-02,  4.4485e-03,  5.7304e-02],
          [-3.8568e-02, -5.3432e-02, -5.4249e-02],
          [-3.9563e-02, -3.1652e-02, -2.2797e-02]],

         [[ 2.7075e-02, -7.9396e-02, -7.6465e-02],
          [ 2.6563e-03,  8.7451e-05,  9.8849e-03],
          [-6.8855e-02,  1.1626e-02, -8.0728e-03]],

         [[ 5.9661e-02,  2.8280e-02, -3.3602e-02],
          [-3.3256e-02, -3.8300e-02, -2.9927e-02],
          [-7.5271e-03, -2.0837e-02, -1.6750e-02]],

         [[-2.1132e-02, -7.1065e-02, -2.6912e-02],
          [-5.4570e-03, -8.4003e-03,  5.7544e-02],
          [-3.4003e-02, -5.4456e-03, -3.3929e-02]],

         [[ 4.7435e-02,  1.4890e-02, -7.1216e-02],
          [-8.2283e-02,  2.4189e-03, -6.7384e-02],
          [-3.1807e-02, -6.1526e-02,  3.9891e-02]],

         [[-7.5832e-02, -5.1319e-02,  2.2298e-02],
          [-4.3339e-02, -3.2454e-02, -6.7572e-02],
          [-5.6210e-02, -7.5146e-02,  8.6754e-02]],

         [[-8.4280e-02,  1.2410e-02,  4.4405e-03],
          [ 7.1902e-03, -1.7250e-03, -9.2412e-03],
          [-8.7924e-02, -6.6238e-03,  4.1562e-02]],

         [[-8.4320e-02,  7.4491e-02, -2.5212e-02],
          [-3.7431e-02, -1.4096e-02, -5.1544e-02],
          [-6.5282e-02, -8.2672e-02, -2.9104e-02]]],


        [[[ 1.4095e-03, -6.7235e-02,  4.9469e-02],
          [ 5.6497e-02,  5.4456e-02, -4.3666e-02],
          [-4.6561e-04, -5.4088e-02,  1.9567e-03]],

         [[-4.5699e-02, -1.4952e-02,  4.4376e-02],
          [-7.2813e-02,  3.5314e-02,  2.1330e-02],
          [ 4.3419e-02,  1.2117e-02,  5.0878e-02]],

         [[-3.9513e-03, -1.8860e-02,  2.0116e-02],
          [-3.7921e-02,  1.1984e-01,  1.0382e-01],
          [ 1.4063e-01,  5.3249e-02, -6.1140e-02]],

         [[ 6.2213e-02, -1.8841e-02,  1.3390e-01],
          [-5.3418e-04,  3.9863e-02,  3.9273e-02],
          [-5.1509e-02,  4.8252e-02,  6.7415e-02]],

         [[-1.8756e-02, -1.5067e-02,  4.5109e-03],
          [ 1.7322e-02,  9.3010e-02, -4.4665e-02],
          [ 7.4305e-02,  3.3290e-02, -3.1978e-02]],

         [[ 4.3295e-02, -3.4852e-02, -2.9968e-02],
          [-3.1746e-02,  1.6907e-02, -8.0157e-03],
          [-3.5904e-02, -6.4225e-02, -3.9760e-02]],

         [[ 4.5433e-02,  1.0643e-01,  6.3114e-02],
          [ 5.3501e-02,  2.7805e-02, -8.0569e-02],
          [ 9.4322e-02,  1.2838e-02,  2.1023e-02]],

         [[-7.3736e-02,  1.3864e-01,  1.2149e-01],
          [-5.9698e-02, -2.6285e-02,  3.9052e-02],
          [ 5.1549e-02,  1.4399e-01,  6.7339e-02]],

         [[-2.4134e-02,  7.1286e-02,  7.8250e-02],
          [ 8.2873e-02, -3.2132e-03, -6.9918e-02],
          [-6.5616e-03,  1.7638e-02,  5.5049e-02]],

         [[-6.6143e-02,  8.3214e-02,  3.2411e-02],
          [ 1.2732e-01, -1.7777e-02, -3.6796e-02],
          [ 4.4258e-02,  1.3892e-02, -6.7230e-02]],

         [[-1.0050e-02, -2.1238e-02, -1.4016e-02],
          [ 1.2496e-01, -5.3263e-02,  3.5036e-03],
          [ 1.5233e-03, -7.9716e-02, -9.9994e-02]],

         [[-8.5426e-03,  3.8590e-02, -1.1439e-02],
          [-9.4823e-03,  5.3219e-02,  1.1150e-01],
          [-1.2413e-01, -4.5799e-03, -4.2430e-02]],

         [[-8.2158e-02,  2.4950e-02,  9.5934e-02],
          [ 9.4057e-02,  1.0219e-01, -1.1055e-01],
          [ 9.4893e-02, -1.6238e-03, -1.3703e-02]],

         [[-2.7641e-02, -2.6727e-02, -1.6088e-02],
          [-6.3949e-03,  8.1547e-02, -1.1605e-02],
          [-9.0918e-02, -8.1857e-02,  1.9253e-02]],

         [[ 7.1144e-02,  3.7162e-03, -7.6817e-02],
          [-6.0491e-02, -8.9416e-02,  8.1861e-02],
          [-3.1937e-03,  3.0951e-02, -5.3410e-03]],

         [[ 7.5201e-02,  1.6133e-02,  1.3851e-02],
          [-6.2908e-02, -4.4981e-02,  1.0880e-01],
          [-1.1566e-01, -2.4100e-02, -2.9850e-02]],

         [[-2.2657e-02, -5.1951e-03, -1.9661e-02],
          [ 1.6004e-02, -6.5957e-03,  5.5833e-02],
          [-2.8788e-02, -5.7349e-02, -3.6935e-03]],

         [[-1.4785e-02,  2.8114e-02,  1.0137e-01],
          [ 5.0373e-02,  6.6024e-03, -3.7974e-02],
          [ 9.5989e-02,  3.4197e-02, -7.7005e-02]],

         [[-7.3904e-02,  4.3301e-02,  1.6276e-02],
          [-2.9724e-02,  9.6746e-02,  2.7575e-02],
          [ 7.3012e-03,  2.1262e-02,  2.6833e-02]],

         [[ 8.3961e-03, -1.0591e-01, -5.1593e-02],
          [ 1.3208e-01,  7.0419e-02,  2.1960e-02],
          [ 2.3125e-02, -1.3624e-01, -8.5561e-02]],

         [[-1.0519e-01,  4.6525e-03,  2.6517e-02],
          [ 3.1519e-03, -4.5876e-03,  3.2749e-02],
          [-5.1071e-02,  3.5596e-02, -2.7164e-02]],

         [[-2.3000e-03,  9.7391e-03, -3.5327e-02],
          [-2.5420e-02,  2.9575e-02, -5.0594e-02],
          [-7.2850e-02,  8.4133e-02, -3.6155e-02]],

         [[ 5.9880e-02,  5.4717e-03, -6.7050e-02],
          [ 6.0886e-02, -6.1584e-02,  5.0088e-02],
          [ 1.4344e-01, -3.8231e-02, -9.5411e-02]],

         [[-7.5843e-02,  6.0853e-02, -2.0058e-02],
          [ 1.3484e-01, -9.7160e-03, -2.7848e-02],
          [ 3.9368e-02, -6.2177e-02, -8.3737e-02]],

         [[ 1.0104e-02, -4.3369e-02,  4.3190e-02],
          [ 1.2587e-01,  8.4269e-02,  4.3581e-03],
          [ 7.8273e-02, -1.1113e-01, -1.7670e-02]],

         [[-1.9936e-02, -3.1024e-02, -1.4539e-03],
          [-6.2998e-02,  6.2605e-02,  2.3083e-02],
          [ 3.2301e-02,  4.4373e-02,  9.0835e-03]],

         [[-6.4764e-02, -1.6284e-02, -7.7390e-02],
          [ 4.6884e-02,  2.8102e-02, -2.8371e-02],
          [ 8.2272e-02, -3.6325e-02, -3.4925e-02]],

         [[ 1.4872e-02, -8.9037e-02,  2.2252e-02],
          [ 9.7967e-02,  1.2254e-01,  3.5846e-02],
          [ 1.3920e-01, -5.6080e-02, -1.3890e-01]],

         [[ 8.7691e-03, -1.4827e-02, -8.1309e-02],
          [ 8.8386e-02,  8.2485e-02,  4.3237e-02],
          [-9.0015e-02, -1.1075e-01, -2.8196e-02]],

         [[ 5.4823e-02, -2.4774e-02, -3.9738e-02],
          [-9.1200e-02, -9.2219e-03,  2.1612e-02],
          [-5.7777e-02,  1.0260e-02, -4.7816e-02]],

         [[-4.9356e-02,  2.9409e-02, -1.1725e-02],
          [ 1.0106e-01,  4.2520e-02,  3.3720e-02],
          [-1.0805e-01, -4.1409e-02, -4.6170e-02]],

         [[-2.1706e-02,  4.4714e-02,  1.4270e-02],
          [ 3.7578e-02, -2.6019e-02,  4.6181e-02],
          [ 4.6402e-03,  4.9677e-02,  7.6292e-02]]],


        [[[ 2.7623e-02,  6.3556e-03,  2.5328e-02],
          [-3.6038e-02, -1.6782e-02,  4.6805e-02],
          [-8.9686e-05, -2.8705e-03,  5.7088e-03]],

         [[-2.2256e-02,  2.2662e-02, -3.7405e-02],
          [ 4.1440e-02,  8.3954e-03, -1.2086e-02],
          [ 1.8292e-02, -3.4812e-02, -3.3638e-03]],

         [[-6.1732e-03, -8.0586e-02, -1.7615e-02],
          [-4.7373e-02, -5.7688e-02,  8.4699e-02],
          [ 2.0748e-02, -4.5290e-02,  6.4311e-02]],

         [[-2.0585e-03,  2.9503e-02, -1.0982e-01],
          [-2.5523e-02,  7.0138e-02, -1.4672e-02],
          [ 4.8741e-02,  8.4613e-02,  3.6746e-02]],

         [[ 3.0400e-02, -3.8730e-03,  3.3724e-03],
          [-9.2657e-03,  5.7765e-03, -1.0597e-02],
          [ 3.9853e-02,  5.3842e-02, -5.6194e-02]],

         [[ 3.1089e-02,  4.2012e-02,  1.9185e-02],
          [-2.7089e-02,  4.3561e-02,  6.3434e-02],
          [-3.5307e-02,  4.2480e-02, -5.1966e-02]],

         [[-8.2715e-02, -6.9702e-03,  5.2030e-02],
          [ 9.5589e-02, -5.4002e-03,  2.5668e-02],
          [ 8.0261e-02, -1.8698e-02,  1.4740e-01]],

         [[ 3.2682e-02, -5.5001e-02,  4.2551e-02],
          [-6.4655e-02, -1.1033e-01,  7.8743e-02],
          [ 1.2561e-01, -1.4264e-02,  1.2124e-01]],

         [[-5.2297e-02, -6.8913e-02,  3.1624e-02],
          [ 8.6924e-02, -2.4429e-04, -1.2981e-02],
          [ 3.2652e-02, -8.1226e-02,  5.6723e-02]],

         [[-1.3066e-01, -9.4971e-02,  1.2146e-01],
          [ 1.3294e-01,  1.3087e-02, -1.1080e-02],
          [ 4.4070e-02, -4.0397e-04,  9.4538e-02]],

         [[-6.3207e-02, -6.7706e-02,  1.6569e-02],
          [ 2.7754e-02,  3.2519e-03, -7.1410e-03],
          [-1.3084e-02, -1.2836e-02,  3.7244e-02]],

         [[ 2.0134e-02,  6.3490e-02, -4.1620e-03],
          [-3.8539e-02, -2.4731e-02,  1.6773e-02],
          [-5.3885e-02, -1.9392e-02,  6.8025e-03]],

         [[-1.1201e-01, -1.1319e-01, -1.8622e-02],
          [ 6.7569e-02, -4.6576e-03,  5.9242e-02],
          [ 7.0434e-02, -9.5523e-04, -7.0877e-03]],

         [[-9.7280e-02, -6.4063e-02,  1.0941e-02],
          [ 5.9310e-02,  7.5082e-02,  4.7236e-02],
          [ 9.6931e-03, -6.8997e-02, -1.0011e-01]],

         [[ 1.9318e-02, -1.9924e-02,  4.3706e-02],
          [-6.4921e-02,  3.3134e-02,  1.8036e-02],
          [ 8.4890e-03, -1.4068e-02,  5.8491e-03]],

         [[ 4.7379e-02, -1.8101e-02, -2.2540e-02],
          [ 2.6243e-03, -6.1397e-02,  7.7911e-02],
          [ 5.0514e-02,  5.0616e-02,  2.6625e-02]],

         [[-3.5418e-02,  4.0460e-02, -1.2751e-02],
          [-1.6266e-02,  1.3517e-02,  2.7579e-02],
          [ 3.4080e-02,  4.2902e-03, -1.0521e-01]],

         [[-8.9239e-02, -7.3348e-02,  7.5943e-02],
          [-1.0804e-02,  2.7296e-02, -2.8889e-02],
          [ 2.1997e-02,  2.2374e-02,  9.3906e-02]],

         [[ 8.7916e-03,  4.2955e-02, -6.9272e-02],
          [ 1.0996e-02,  8.5374e-02, -1.7823e-02],
          [ 3.0145e-02,  6.4747e-02, -9.1879e-02]],

         [[ 1.1800e-02, -5.2527e-02, -7.3981e-03],
          [-6.4449e-03,  3.7004e-02,  9.3080e-02],
          [ 1.0379e-01, -2.0763e-02, -9.4692e-04]],

         [[-1.0059e-01, -1.1622e-01,  7.0105e-02],
          [ 1.5044e-02,  1.4807e-02,  5.6194e-03],
          [ 1.6332e-02, -1.5831e-02,  3.8118e-02]],

         [[-9.2775e-02, -2.9061e-02,  2.3354e-02],
          [-3.9643e-02,  6.7963e-02,  1.4528e-02],
          [ 4.7439e-03,  8.5771e-02, -6.7045e-02]],

         [[-2.9527e-02,  3.5584e-02, -4.5032e-02],
          [-3.2626e-02, -4.3777e-02, -1.1555e-02],
          [ 4.5437e-02, -4.0804e-03, -3.7717e-02]],

         [[-2.3826e-02, -3.6310e-02,  3.6494e-02],
          [ 4.5869e-02, -2.2359e-02,  2.6451e-02],
          [-4.5816e-04,  1.2240e-02, -2.5922e-02]],

         [[-5.7918e-02, -8.6600e-02,  1.1550e-02],
          [ 8.3750e-02,  3.3794e-02,  6.2741e-02],
          [ 7.0119e-02, -9.8400e-02,  3.6304e-02]],

         [[ 7.5811e-02, -5.7987e-02,  4.2387e-02],
          [-7.8525e-02,  6.3759e-02,  3.6749e-02],
          [ 7.1636e-02,  7.8545e-02, -7.4557e-02]],

         [[-8.0250e-02, -2.2430e-02,  1.0889e-02],
          [ 8.1887e-02,  4.6617e-02, -4.7744e-02],
          [ 3.7904e-02, -5.4750e-03, -7.1421e-02]],

         [[-1.2086e-01,  6.4594e-03,  5.8589e-03],
          [ 3.5177e-02,  1.3467e-01, -1.4530e-02],
          [ 7.8517e-02,  1.0743e-01, -9.9674e-02]],

         [[ 8.9208e-03, -1.7379e-02, -2.4380e-02],
          [ 3.6373e-02,  6.2603e-02,  2.6652e-02],
          [-5.6667e-02, -4.8035e-02, -1.0822e-01]],

         [[ 2.4478e-03,  4.0231e-02, -3.0358e-02],
          [ 1.8503e-02,  9.2314e-03, -4.8103e-05],
          [-1.9942e-03,  5.5604e-02, -8.0873e-02]],

         [[-3.4526e-02, -1.8938e-02, -2.9264e-02],
          [-5.0411e-02,  4.9516e-02, -3.9402e-02],
          [-7.1620e-02,  2.0718e-02, -1.0765e-01]],

         [[ 4.4496e-02, -3.2788e-02,  1.2784e-01],
          [ 1.0596e-01,  2.5743e-02,  7.1000e-02],
          [ 2.6841e-02, -1.9536e-02,  1.2295e-02]]],


        [[[ 3.0043e-02,  8.4442e-02,  3.3236e-02],
          [ 8.9516e-02,  8.4645e-02,  9.1223e-02],
          [ 2.5057e-03, -5.9352e-02, -3.5709e-03]],

         [[ 8.1925e-03,  5.6206e-02, -6.7028e-03],
          [ 5.7818e-02,  1.2628e-02,  3.4205e-02],
          [ 6.3758e-02,  1.5133e-03, -2.7711e-02]],

         [[ 4.1920e-02,  1.2992e-02, -2.6048e-02],
          [-1.6749e-02, -7.4490e-02, -2.8103e-02],
          [-1.0330e-02, -6.4269e-02, -4.0529e-02]],

         [[-1.6936e-02,  1.4929e-02, -3.6506e-02],
          [ 8.0082e-02,  4.7323e-03, -4.4195e-03],
          [ 1.1020e-01,  6.0532e-02,  4.7285e-02]],

         [[-8.0367e-02, -1.7384e-02,  2.6420e-03],
          [-3.0251e-03, -1.9641e-02, -6.1458e-02],
          [-1.4895e-02,  3.1635e-02, -3.0673e-02]],

         [[ 5.1381e-02,  5.9814e-02,  3.3320e-02],
          [-7.9786e-02, -9.3326e-04, -4.7445e-02],
          [-4.1304e-02,  9.7169e-02,  4.4226e-02]],

         [[-3.5636e-02, -5.6093e-04, -2.7221e-02],
          [ 1.0077e-01,  4.6054e-02,  8.5334e-03],
          [ 5.3665e-02, -1.7809e-02, -4.5905e-02]],

         [[ 1.6505e-02, -1.3901e-02, -6.6082e-03],
          [ 7.2881e-03,  7.8438e-03, -4.8724e-02],
          [ 1.0385e-01,  1.8684e-02, -1.0551e-03]],

         [[-4.9483e-02, -3.2997e-02,  7.0100e-03],
          [ 4.3647e-02,  2.1907e-02,  4.0601e-02],
          [-3.0815e-02, -4.1281e-02, -5.7439e-02]],

         [[-7.5049e-03, -8.3380e-02, -7.5355e-02],
          [ 1.8970e-02,  2.4129e-02, -1.2845e-02],
          [-4.8157e-02, -8.9178e-02, -2.0478e-02]],

         [[-1.3870e-02,  2.4915e-03,  2.7506e-02],
          [-3.1416e-02, -6.2013e-02, -6.8761e-02],
          [ 1.2888e-02, -5.9277e-02,  2.1238e-02]],

         [[ 5.0286e-02,  6.2157e-02,  2.5817e-02],
          [ 4.4949e-03,  1.3658e-02, -1.4516e-02],
          [-3.3151e-02,  1.6426e-03, -1.0616e-02]],

         [[-3.0450e-02,  5.7960e-02, -8.0353e-02],
          [ 8.4989e-02, -3.1366e-02, -5.7758e-02],
          [-2.9057e-03, -5.5432e-02,  2.5837e-02]],

         [[ 1.3097e-02,  1.1875e-01, -1.3928e-02],
          [-1.9122e-02,  7.2895e-03, -6.3611e-02],
          [-2.2902e-02, -7.1631e-02, -3.5807e-02]],

         [[-5.6812e-02,  5.3584e-03, -5.8869e-04],
          [ 2.1073e-02, -3.3243e-02, -1.5869e-02],
          [ 3.6433e-03,  4.5093e-02,  1.1017e-02]],

         [[ 2.6538e-02,  1.6890e-02,  5.0976e-02],
          [-5.5675e-02, -7.0136e-02,  4.9339e-03],
          [-4.8449e-02,  5.7714e-02,  2.7974e-02]],

         [[ 8.4576e-02,  1.0763e-01,  6.0258e-02],
          [-5.0088e-02, -9.7870e-02, -8.6465e-03],
          [-3.1386e-02,  4.3930e-02, -1.3630e-02]],

         [[ 8.0258e-02, -6.3021e-02, -2.3462e-02],
          [ 3.7531e-03, -1.0140e-01, -8.8931e-02],
          [ 2.3632e-02, -4.3053e-02, -1.5964e-02]],

         [[-5.9255e-02, -3.6019e-02, -8.0495e-02],
          [-9.9085e-03, -1.4737e-02, -6.1393e-03],
          [ 6.7786e-02, -4.1423e-02,  2.0581e-03]],

         [[ 1.2845e-02,  8.9125e-02,  6.0762e-02],
          [-8.7397e-03, -2.8020e-02, -4.9900e-02],
          [-6.1486e-02, -8.8969e-02,  2.5310e-02]],

         [[-2.6063e-02,  3.1831e-02,  8.7576e-02],
          [-4.5019e-04, -9.1931e-02, -5.0863e-02],
          [-8.0577e-02, -7.8847e-02,  7.5785e-02]],

         [[-4.0109e-02,  1.4387e-02,  5.7996e-02],
          [ 6.5813e-03,  3.1979e-02,  1.7784e-02],
          [ 1.2573e-02,  7.5932e-03,  5.5064e-02]],

         [[ 4.9815e-02, -3.1554e-02, -4.6119e-02],
          [-2.9199e-02,  5.2196e-02, -2.1829e-02],
          [-5.5986e-02, -6.8519e-02,  2.6351e-02]],

         [[ 1.0693e-02, -5.4913e-02, -5.1585e-02],
          [-3.2874e-02,  3.7489e-02,  3.0243e-02],
          [-9.0895e-02, -5.7056e-02,  3.2748e-02]],

         [[-4.1123e-02, -4.0410e-02,  4.3772e-03],
          [-2.3572e-02, -5.0242e-02,  5.0600e-02],
          [-7.0266e-03, -1.2537e-01,  5.6976e-02]],

         [[-1.4365e-02, -2.1367e-02,  2.8865e-02],
          [-7.9938e-02,  2.0108e-02, -4.1249e-02],
          [ 4.0940e-02,  6.2795e-02, -2.8090e-02]],

         [[ 1.9869e-02,  7.7724e-02,  1.2002e-02],
          [ 4.9612e-02, -1.4096e-02,  7.4774e-03],
          [ 5.5051e-03, -2.4040e-02,  3.7342e-02]],

         [[-5.7496e-02,  6.7562e-02,  3.4791e-02],
          [ 4.8087e-02,  3.4494e-02, -7.6975e-03],
          [-4.6295e-02, -3.6152e-02,  2.4511e-02]],

         [[-6.6585e-03, -1.6413e-02,  7.5745e-02],
          [-6.2002e-02, -3.5994e-02, -7.3269e-02],
          [-4.9173e-02,  4.7072e-02,  1.9389e-02]],

         [[ 5.0297e-02, -2.9755e-02, -3.5809e-02],
          [ 3.0714e-02,  9.9829e-04,  5.9037e-02],
          [-3.2224e-02,  3.5978e-02,  7.8044e-03]],

         [[-3.1285e-02,  4.6827e-02, -1.3325e-02],
          [ 1.9729e-02, -4.7694e-02, -2.1016e-03],
          [-5.8161e-02, -2.5992e-02, -7.3620e-02]],

         [[ 2.5908e-02, -2.9452e-02,  6.5300e-02],
          [-5.4328e-03, -2.6576e-02,  9.7539e-02],
          [-3.4071e-02,  6.1081e-02,  2.3802e-02]]],


        [[[ 2.4600e-02,  8.3551e-03, -2.8752e-02],
          [ 2.8538e-02, -6.6401e-03, -4.0330e-02],
          [-8.3746e-02, -7.4965e-02, -8.2963e-02]],

         [[ 2.6772e-03, -1.7590e-02, -2.5518e-03],
          [ 5.0235e-02,  9.2285e-02, -5.7736e-02],
          [ 6.8252e-02,  3.6194e-02, -5.5439e-02]],

         [[ 8.1122e-03,  4.7376e-03, -3.8486e-02],
          [ 5.8278e-02,  1.0068e-02, -4.8846e-02],
          [-1.1917e-02, -6.0009e-02, -2.8167e-03]],

         [[-2.5508e-02, -3.5762e-02,  8.9068e-02],
          [ 2.1095e-02,  5.1093e-02,  1.9293e-02],
          [ 4.5668e-02, -7.8030e-03, -6.0989e-02]],

         [[ 5.8753e-02, -3.1325e-02,  6.5854e-02],
          [ 3.8451e-02, -1.8534e-03, -8.4585e-03],
          [-6.3946e-02, -8.6337e-02,  3.3145e-02]],

         [[-4.9683e-02, -1.0700e-01,  1.5506e-03],
          [-4.9404e-02, -8.3372e-02,  2.7440e-03],
          [ 1.4821e-02,  7.9763e-02,  6.9416e-02]],

         [[ 6.5188e-02,  2.3202e-02, -9.5406e-02],
          [ 9.6578e-03,  4.6209e-02,  1.6928e-02],
          [-9.6683e-02, -6.2849e-02, -5.5741e-02]],

         [[-9.5667e-03, -2.3872e-02, -2.4662e-02],
          [ 1.2453e-01,  1.2901e-01, -1.6230e-02],
          [ 7.8342e-03,  3.9193e-02, -9.3035e-02]],

         [[-4.6082e-02,  1.7919e-02, -3.9632e-02],
          [ 7.8137e-03, -1.3840e-02, -8.3878e-02],
          [-1.1653e-01,  8.5969e-03, -1.1709e-01]],

         [[ 1.0230e-01,  6.2370e-02, -7.3892e-02],
          [-6.6113e-02, -5.3407e-02, -2.4495e-02],
          [-8.0856e-02, -1.8162e-02, -9.6404e-02]],

         [[ 6.2750e-02,  6.9730e-02, -1.0395e-01],
          [-6.5994e-03, -3.8643e-02,  8.9083e-04],
          [ 5.9596e-02,  1.9764e-02, -7.6863e-04]],

         [[-3.6579e-02, -1.2288e-02,  4.5301e-02],
          [-1.0867e-01, -1.1838e-02, -8.2329e-02],
          [ 5.5637e-02,  4.1697e-02, -1.6433e-02]],

         [[ 3.7995e-02,  1.0381e-01,  6.3373e-03],
          [ 4.2305e-02,  1.6613e-02,  4.0262e-02],
          [-6.6648e-02,  2.7068e-02,  2.7357e-02]],

         [[ 3.8927e-02, -2.2320e-02,  7.3613e-02],
          [-3.5343e-02, -3.4111e-02, -2.4282e-02],
          [ 3.6527e-02,  7.1855e-02,  3.8898e-02]],

         [[ 2.8685e-02, -7.2339e-02,  3.8397e-02],
          [-2.7564e-02,  6.0542e-02, -4.8079e-02],
          [ 9.5623e-02,  3.0036e-02,  1.7741e-02]],

         [[-7.3162e-02, -5.0810e-02, -2.7071e-02],
          [-8.0617e-02, -6.2611e-03,  4.0311e-03],
          [ 1.9539e-02,  4.3211e-02, -4.1603e-02]],

         [[ 1.6981e-02, -4.6250e-02,  1.2205e-01],
          [-1.1432e-01,  1.6422e-02, -1.2079e-04],
          [-3.5227e-02,  2.8022e-02,  9.7559e-02]],

         [[ 7.5282e-02,  3.1054e-02, -1.2165e-03],
          [ 6.1728e-02,  3.9815e-02, -3.1901e-02],
          [ 7.1745e-02,  4.5537e-02, -4.8671e-03]],

         [[-2.9487e-03, -7.0186e-02,  9.8825e-03],
          [-2.6314e-03, -6.2960e-02,  4.9577e-02],
          [-5.0066e-02, -4.7072e-02,  5.8221e-02]],

         [[ 1.2545e-01, -2.0841e-04,  8.1352e-02],
          [ 1.9342e-02, -7.4472e-02,  9.4284e-03],
          [-1.2439e-02, -2.7241e-02,  9.2677e-02]],

         [[ 1.2845e-02,  6.1067e-02, -6.7227e-02],
          [-4.4822e-02, -7.0385e-02, -5.1409e-02],
          [-3.9240e-03,  1.0119e-01,  3.6745e-02]],

         [[-2.2263e-02, -1.0384e-01,  4.7406e-02],
          [ 8.2413e-02,  3.5265e-02,  2.4469e-02],
          [ 5.0467e-02, -4.9661e-02,  3.8988e-02]],

         [[ 2.4357e-02, -4.4675e-03,  5.5424e-02],
          [-2.7668e-02, -3.1724e-02,  1.9236e-02],
          [-1.3756e-02, -1.7353e-03,  1.7303e-03]],

         [[-2.7504e-03, -5.1724e-02,  5.6146e-02],
          [-4.0094e-02, -9.2212e-02,  5.1286e-03],
          [ 8.3522e-03, -3.0196e-02, -1.4096e-02]],

         [[ 8.5343e-02,  5.2244e-02, -5.2122e-02],
          [ 2.3507e-02, -3.6234e-02, -1.6273e-02],
          [ 6.8473e-03,  5.8833e-02,  2.0620e-02]],

         [[-7.7246e-02, -3.9426e-02, -2.7648e-04],
          [ 4.5403e-02,  1.1675e-01,  2.9951e-02],
          [ 2.6454e-02,  9.6972e-03,  8.4315e-02]],

         [[ 1.0033e-01,  2.4949e-02,  4.1323e-02],
          [-9.5832e-03, -6.9964e-02, -9.3259e-03],
          [ 8.4973e-04,  6.0276e-02,  9.2849e-02]],

         [[ 2.0716e-02,  6.5818e-02,  3.1355e-02],
          [-1.6991e-02, -3.7169e-02,  1.0017e-01],
          [-3.3369e-02, -8.5460e-02,  7.6756e-02]],

         [[ 1.1592e-01,  3.0100e-02,  7.2305e-02],
          [ 1.0770e-03, -3.2404e-02, -7.7808e-02],
          [ 2.9919e-02,  5.9993e-02,  9.5532e-02]],

         [[ 1.7058e-03,  5.7606e-03, -2.2152e-02],
          [-1.8972e-02,  1.7392e-02,  4.7545e-02],
          [-4.8744e-02, -8.7839e-02,  5.1583e-02]],

         [[ 5.6844e-02, -6.6681e-04,  6.5305e-02],
          [-8.0175e-02, -3.7878e-02,  3.9804e-02],
          [ 4.0259e-02,  5.8452e-02,  3.0264e-02]],

         [[ 5.1105e-03, -4.0573e-02, -9.6532e-03],
          [-2.2202e-02,  6.4503e-02, -5.0590e-02],
          [-4.1496e-02, -3.0491e-02, -1.8809e-02]]],


        [[[ 8.5083e-02, -8.2917e-03, -5.6576e-02],
          [ 1.1769e-01,  9.6967e-02,  3.5307e-02],
          [ 2.4550e-02, -5.3976e-02, -7.8512e-02]],

         [[-6.8854e-02, -2.5504e-03,  3.0331e-02],
          [ 5.5407e-02,  3.9502e-02,  2.7816e-02],
          [-2.4876e-02,  2.3172e-02,  6.6620e-02]],

         [[-2.8924e-02,  3.9915e-02, -9.6240e-03],
          [ 1.3819e-03, -1.0320e-02,  7.6008e-03],
          [ 8.9532e-02,  5.1009e-03, -2.5558e-02]],

         [[-7.0231e-02, -5.3716e-02, -9.8492e-03],
          [ 5.0365e-02,  1.0836e-01,  7.6825e-02],
          [ 1.3423e-02,  1.0676e-01, -3.0552e-02]],

         [[-7.1154e-02, -4.0305e-02, -2.3573e-02],
          [ 4.2797e-02, -1.2659e-02, -7.2540e-02],
          [ 8.0169e-02,  3.8693e-02,  9.0682e-03]],

         [[ 4.7898e-02,  7.4056e-02,  1.4519e-02],
          [-1.0738e-01,  8.5278e-03, -3.7381e-02],
          [ 2.6247e-02,  1.3623e-01,  6.2501e-02]],

         [[ 2.9128e-02, -6.3867e-02,  2.5875e-03],
          [ 1.2053e-01,  9.5244e-02,  3.3507e-02],
          [-2.6221e-02, -8.5524e-03,  1.4833e-02]],

         [[-7.4028e-02, -8.2084e-02, -3.6343e-02],
          [ 4.7014e-02,  1.1378e-01,  8.7006e-02],
          [-5.4448e-02, -1.7828e-02,  3.8333e-02]],

         [[-2.6521e-02, -7.5963e-02,  8.5121e-04],
          [ 2.6110e-02,  1.2661e-02,  1.1838e-01],
          [-5.5288e-02, -9.4361e-03, -9.7486e-03]],

         [[ 4.4240e-02,  6.9023e-03,  5.6440e-03],
          [ 5.5585e-02,  1.8044e-02,  5.9239e-02],
          [-9.8612e-03, -2.8693e-02, -2.4023e-02]],

         [[ 6.8943e-02,  6.6809e-02,  2.8894e-02],
          [ 1.9067e-02, -2.9224e-02,  1.7450e-02],
          [ 2.5155e-02, -6.6017e-02,  4.6469e-02]],

         [[ 6.1417e-02,  5.5341e-02,  2.8465e-02],
          [ 2.3268e-02,  2.2950e-02, -3.5211e-02],
          [-4.0593e-02,  7.4110e-02,  7.2320e-02]],

         [[-8.2024e-02,  3.1500e-02, -4.8567e-03],
          [ 8.5846e-02, -2.0156e-02, -1.4748e-03],
          [-3.2569e-02, -7.0790e-02,  3.0033e-02]],

         [[ 3.6365e-02,  9.8719e-02, -1.1833e-04],
          [ 1.2860e-02, -1.6805e-02,  8.3077e-03],
          [ 9.3831e-02,  9.1840e-02, -2.0087e-03]],

         [[ 4.6222e-02, -9.4119e-03, -3.8974e-02],
          [ 6.7110e-03,  3.3910e-02,  1.4550e-04],
          [ 2.1847e-02,  6.4867e-02, -4.2769e-02]],

         [[ 2.4961e-02, -4.7880e-02,  4.0091e-02],
          [ 1.8079e-02, -5.5160e-02, -2.9266e-02],
          [ 3.2099e-02,  1.4599e-02,  1.3947e-02]],

         [[ 1.0245e-01,  9.5373e-02, -2.3508e-02],
          [-5.9736e-02, -8.2226e-03, -3.7102e-02],
          [ 3.1201e-02,  8.7344e-02,  1.0616e-01]],

         [[ 5.1628e-02, -2.6744e-02,  5.6204e-02],
          [ 2.9962e-02, -2.1007e-02, -5.3457e-02],
          [-2.0101e-02, -3.7429e-02,  7.2186e-02]],

         [[-7.0844e-02,  9.2367e-02, -6.8837e-02],
          [ 6.8977e-02,  1.3175e-02, -6.7720e-02],
          [ 1.6221e-02,  1.5619e-02,  9.5263e-03]],

         [[ 5.7807e-02,  7.7922e-02,  4.2167e-03],
          [ 3.7503e-02,  1.4002e-02, -3.2250e-02],
          [ 4.7308e-02,  4.3694e-02,  5.3145e-02]],

         [[ 1.6100e-02,  3.7526e-02,  4.1828e-02],
          [-2.7479e-02, -3.9551e-02, -8.3159e-03],
          [ 8.1748e-03, -6.5366e-04,  5.3066e-02]],

         [[ 1.2948e-01,  7.7364e-02,  1.2008e-02],
          [-8.6465e-02,  1.3684e-01, -8.4013e-02],
          [ 1.0740e-01, -1.4568e-02, -1.4567e-02]],

         [[ 4.6463e-02,  1.8111e-02, -4.9645e-02],
          [ 3.2485e-02, -3.2597e-02,  7.8687e-03],
          [ 7.2910e-02, -3.8650e-02, -9.1791e-02]],

         [[-1.1590e-02,  6.8231e-02, -4.0428e-02],
          [ 7.7631e-02, -5.6079e-02, -6.3095e-02],
          [-2.9675e-02, -1.4568e-02,  1.0380e-02]],

         [[-4.0580e-02,  1.5455e-02,  5.4767e-03],
          [ 3.2826e-02, -6.0187e-02,  2.4657e-02],
          [-6.5881e-03, -3.3482e-02,  7.4509e-02]],

         [[-2.0403e-02, -6.6486e-02, -2.8214e-02],
          [-4.6472e-02,  1.2612e-03, -7.0271e-02],
          [ 5.3393e-02,  6.0628e-02, -7.5604e-03]],

         [[ 8.8709e-02,  1.1379e-01, -8.1829e-02],
          [-5.5620e-02, -4.7270e-02, -9.7803e-02],
          [ 9.1531e-02,  5.3481e-02,  7.7524e-03]],

         [[ 3.0644e-02,  5.6001e-02, -2.4360e-02],
          [ 2.8310e-02,  4.9164e-02, -5.0470e-02],
          [ 6.0779e-02,  4.1492e-02, -9.0271e-02]],

         [[ 5.4825e-02,  1.0893e-01, -1.3899e-02],
          [-8.0483e-02, -2.2642e-02, -8.5398e-02],
          [ 2.8763e-04,  1.0650e-01,  7.1348e-02]],

         [[ 3.9803e-02,  4.2693e-02, -2.2428e-02],
          [-4.2448e-02,  3.4487e-02,  1.7405e-02],
          [-6.1034e-02,  2.1337e-02,  1.6977e-03]],

         [[ 6.0881e-02, -9.0248e-03,  2.2707e-02],
          [ 6.2531e-03, -3.7330e-02, -4.0123e-02],
          [ 1.6930e-02, -1.0921e-02, -2.1328e-02]],

         [[ 5.1435e-02, -2.3015e-02,  3.1018e-02],
          [ 9.6929e-02,  1.8793e-02,  9.3456e-02],
          [-8.4940e-02,  3.3749e-02, -2.2109e-02]]],


        [[[-2.6175e-02, -5.8841e-02, -3.8184e-02],
          [ 7.7568e-02,  2.8144e-02, -1.0850e-03],
          [-4.2836e-03, -9.0102e-02, -3.3070e-02]],

         [[-6.8065e-02, -6.6654e-02,  5.6904e-03],
          [ 8.4454e-02,  5.9243e-03,  3.0402e-02],
          [ 5.1901e-03,  6.1010e-02,  3.4305e-02]],

         [[ 3.5571e-02, -4.9212e-02,  3.9721e-02],
          [-3.8087e-02,  9.3026e-02,  1.6967e-03],
          [-1.0758e-01, -6.4470e-02,  1.2826e-01]],

         [[-5.1048e-03, -6.9169e-02,  2.0391e-02],
          [ 4.4668e-03,  2.6243e-06,  3.1671e-02],
          [-1.2044e-02, -4.1012e-02, -3.3609e-02]],

         [[ 1.3520e-02, -7.2916e-03,  9.8651e-04],
          [ 1.6413e-02, -2.2534e-02,  3.4048e-02],
          [ 4.6482e-02, -6.4837e-02,  6.5618e-02]],

         [[ 3.8543e-02, -1.0781e-02, -5.5395e-02],
          [-3.6443e-02, -2.1971e-02, -6.2218e-02],
          [-2.2758e-02, -5.6209e-02,  1.8710e-02]],

         [[ 4.3258e-02, -1.2846e-05, -7.5669e-02],
          [ 4.4919e-02,  5.0018e-02,  4.2332e-02],
          [-3.8329e-02,  7.5852e-03, -9.6170e-02]],

         [[-8.9980e-02,  5.2849e-02, -2.7541e-02],
          [ 7.9110e-02,  6.4914e-02, -2.4699e-02],
          [-4.8145e-02,  4.9225e-02, -3.7154e-02]],

         [[-1.1180e-02,  3.7575e-02, -5.2537e-02],
          [ 7.7339e-02,  5.4485e-02, -9.8093e-02],
          [-1.0673e-01,  7.6593e-02, -5.1270e-02]],

         [[ 7.2865e-02,  3.7800e-02, -1.1277e-01],
          [ 2.8109e-03,  5.2284e-02,  8.8025e-02],
          [ 4.1595e-02,  1.7506e-02, -9.7341e-02]],

         [[ 1.1021e-01,  6.5875e-02, -1.0521e-01],
          [-1.1872e-02, -2.2076e-02,  2.7894e-03],
          [ 2.2814e-02,  4.1651e-02, -1.0089e-01]],

         [[ 1.3779e-02,  4.9394e-02,  1.9697e-02],
          [-7.8943e-02,  1.0729e-01, -1.8198e-03],
          [ 3.4588e-02, -1.5297e-02,  1.1019e-03]],

         [[-6.1181e-02,  1.1186e-01,  2.6632e-02],
          [ 9.4621e-02, -1.6515e-02, -2.5670e-02],
          [-9.2661e-02, -1.2771e-01,  4.0960e-02]],

         [[ 5.7750e-02,  7.0640e-03, -4.6524e-02],
          [-3.9014e-02, -6.0764e-02, -2.9303e-02],
          [-1.1981e-04, -7.4403e-02,  5.5139e-02]],

         [[-5.5038e-02,  4.5057e-04,  1.8581e-02],
          [-6.7531e-02,  9.7934e-02, -6.0474e-02],
          [ 5.5114e-02, -7.4947e-02, -3.5303e-02]],

         [[-9.3992e-03, -6.9098e-02,  1.1134e-02],
          [-1.0598e-01,  8.5614e-02, -8.0019e-02],
          [-2.3193e-02,  3.9956e-02,  6.0421e-03]],

         [[ 1.0730e-01, -1.6222e-02, -8.3094e-02],
          [-5.0665e-02,  2.2929e-02, -1.2485e-02],
          [ 6.4804e-02, -5.3557e-02,  7.7395e-02]],

         [[ 2.5018e-02,  2.0558e-02, -1.3756e-02],
          [ 5.4569e-02, -3.9857e-02,  7.1767e-02],
          [-2.0753e-02,  1.0978e-02, -2.2335e-02]],

         [[-2.9861e-03, -5.1908e-02,  3.1390e-02],
          [ 1.4455e-01,  6.5603e-03, -7.1810e-02],
          [ 3.2868e-03,  3.8591e-03,  1.4437e-01]],

         [[ 3.3338e-02,  2.8366e-02, -2.4194e-02],
          [-6.3768e-02, -9.6312e-03, -1.0821e-02],
          [-4.2472e-02, -1.1268e-01,  5.5040e-02]],

         [[-3.8310e-02,  1.3320e-01, -8.3993e-02],
          [-3.7857e-02, -9.4596e-02, -9.7315e-03],
          [-9.0407e-03,  7.4466e-03,  6.9076e-02]],

         [[-9.3423e-04, -6.0265e-02, -4.8993e-02],
          [-3.7021e-02,  5.2214e-02,  5.4186e-02],
          [ 6.5310e-02, -5.5482e-03,  1.6750e-02]],

         [[ 2.8501e-02,  1.8064e-02, -4.6918e-02],
          [-4.6471e-02,  6.2799e-02,  5.5722e-02],
          [ 1.0437e-01,  7.6051e-02, -3.3583e-02]],

         [[ 3.5664e-02, -2.2203e-02, -1.5236e-02],
          [ 4.2803e-02, -2.9365e-02, -5.4892e-02],
          [-4.1836e-02,  1.7131e-02,  4.8262e-02]],

         [[-2.7629e-02,  4.3942e-02, -3.9770e-02],
          [ 7.4382e-02, -5.3817e-02, -3.0013e-02],
          [-1.1179e-01,  5.9907e-02,  6.4509e-02]],

         [[-3.6729e-02, -3.9904e-02,  1.3856e-02],
          [ 4.5992e-02,  9.8834e-02,  8.0382e-03],
          [ 8.2846e-04, -3.1732e-02,  2.7522e-02]],

         [[ 1.2229e-01,  1.2672e-01,  1.6295e-02],
          [-4.0428e-02, -6.0637e-02,  4.2970e-02],
          [ 1.3320e-01,  3.0828e-03,  1.2732e-02]],

         [[ 7.8300e-02, -2.8861e-02,  4.9254e-02],
          [ 7.8785e-03,  8.1242e-03,  4.0460e-02],
          [ 3.0945e-02, -1.8435e-02,  4.7943e-03]],

         [[ 1.1551e-01, -1.3389e-02, -2.0533e-02],
          [-7.0863e-02, -4.9776e-02,  6.4003e-03],
          [ 8.5663e-02,  3.4718e-02,  9.3894e-02]],

         [[-2.8858e-02, -9.8458e-02,  2.1126e-02],
          [-6.2544e-02,  3.6276e-02,  5.2027e-04],
          [-4.7311e-02, -1.0150e-01,  6.3407e-02]],

         [[ 1.4970e-02, -6.6069e-02, -6.3239e-02],
          [-3.0079e-02, -5.3243e-02,  3.1631e-02],
          [ 5.6205e-02,  9.8767e-02,  7.6857e-02]],

         [[ 7.7623e-02,  8.2599e-02, -2.8398e-02],
          [-3.7611e-03,  2.4041e-02,  2.5002e-02],
          [-1.0217e-02,  9.5661e-02, -2.6496e-02]]],


        [[[-4.5519e-02,  5.4415e-03,  1.9275e-02],
          [-2.8315e-02,  1.2442e-02,  1.0444e-02],
          [ 6.9894e-02,  5.8422e-02,  2.8704e-02]],

         [[ 6.2475e-02, -5.2288e-02,  1.0348e-02],
          [-7.1182e-02, -6.6033e-02,  4.6870e-03],
          [ 1.1279e-01, -2.4097e-02, -9.1299e-02]],

         [[ 1.0412e-02, -5.9134e-02, -4.5174e-02],
          [ 3.9591e-03,  1.9182e-02,  4.9365e-03],
          [-9.6919e-02,  1.4709e-01,  3.7123e-02]],

         [[ 8.7379e-02,  7.4033e-02, -4.6923e-02],
          [-4.7044e-02, -1.1698e-01,  1.9479e-02],
          [ 3.1605e-02, -1.2469e-01, -1.1826e-02]],

         [[ 7.8503e-03, -8.6053e-03, -1.8830e-02],
          [-5.9530e-02, -4.7814e-02,  1.0044e-02],
          [-3.8493e-02,  3.9082e-02,  4.8877e-02]],

         [[-6.8230e-02,  1.1198e-02,  1.0644e-01],
          [ 1.3185e-01, -1.1216e-01,  1.0517e-02],
          [ 2.6459e-02, -5.6890e-02,  7.7662e-02]],

         [[-3.2281e-02, -8.2195e-02, -1.0146e-01],
          [-7.5901e-02,  7.4399e-02, -2.7715e-02],
          [-1.1652e-02, -2.9685e-02, -2.3794e-02]],

         [[ 5.3918e-02,  5.2874e-02, -4.8614e-03],
          [ 1.2391e-02, -3.7639e-02,  2.5710e-02],
          [-5.4457e-02,  5.2395e-02, -3.4183e-02]],

         [[ 1.0797e-02, -2.2439e-02, -4.3187e-02],
          [-6.2181e-02,  1.0213e-01, -1.6218e-03],
          [ 1.1498e-01, -4.3248e-02, -3.3281e-02]],

         [[ 1.0958e-01, -2.3960e-02, -2.5306e-02],
          [-1.2658e-01,  1.1689e-01,  5.6455e-02],
          [ 6.6480e-02, -2.1662e-02, -1.0410e-02]],

         [[ 1.1094e-02, -1.5426e-02, -2.2341e-04],
          [ 6.1417e-03,  9.6195e-02, -6.6485e-02],
          [-3.9804e-02, -4.3478e-02,  3.4700e-02]],

         [[ 1.1654e-02, -3.0551e-02,  4.3741e-02],
          [ 3.8778e-02,  3.7687e-02, -9.0957e-02],
          [-4.0199e-03,  2.7220e-02, -6.7854e-04]],

         [[ 7.9314e-02, -2.4322e-02, -1.7660e-02],
          [-4.3884e-02, -5.7830e-02,  2.3706e-02],
          [ 5.6935e-02,  3.3881e-02, -7.9912e-02]],

         [[-6.9036e-03,  5.1096e-02,  3.9053e-02],
          [ 9.2615e-02, -1.8823e-02,  2.2166e-02],
          [-4.5502e-02,  2.4830e-02,  1.4360e-01]],

         [[-8.4926e-02,  9.9280e-02, -1.5258e-02],
          [ 6.9480e-03, -7.3139e-02,  5.6344e-04],
          [ 6.0233e-02,  3.7829e-02, -2.1303e-02]],

         [[-1.5827e-02,  5.2220e-02,  4.0748e-03],
          [ 8.5822e-02,  3.2164e-03, -2.1576e-02],
          [ 8.1051e-02, -2.8242e-02,  4.8664e-02]],

         [[ 5.6165e-02,  7.0198e-02, -3.6196e-03],
          [ 1.1334e-01, -3.1619e-02, -1.5595e-02],
          [-4.9324e-02,  8.1606e-02,  1.1232e-01]],

         [[-1.1496e-02,  4.0316e-04, -9.6145e-02],
          [ 2.8077e-02,  6.1567e-02, -5.1643e-03],
          [-7.6756e-02, -9.7509e-02,  5.3536e-02]],

         [[ 7.7589e-02,  3.9692e-02,  2.3315e-02],
          [-3.4580e-02, -3.0045e-02,  5.9936e-02],
          [ 4.6816e-02,  4.4869e-02, -1.6485e-02]],

         [[ 4.2664e-02, -1.2110e-02,  3.8173e-03],
          [ 1.0452e-01, -1.2674e-02, -5.4480e-02],
          [-1.0445e-01,  1.2605e-01,  5.0276e-02]],

         [[ 1.1385e-01, -8.0683e-02,  8.5729e-03],
          [ 1.1085e-02,  3.6025e-02,  3.8957e-02],
          [-2.7664e-03,  4.3674e-02,  6.3924e-02]],

         [[-9.8104e-02,  1.2479e-01,  9.0498e-02],
          [ 1.1079e-01, -3.8005e-02, -4.2838e-02],
          [-5.4659e-02,  4.2661e-02,  3.8020e-02]],

         [[-2.4238e-02,  2.6315e-02,  5.4161e-02],
          [ 2.9213e-02,  1.0218e-02,  1.7022e-02],
          [ 5.7451e-03,  3.7745e-02, -1.1909e-02]],

         [[ 4.6181e-02, -8.3929e-02,  4.9809e-02],
          [-2.6928e-02,  1.2094e-02,  7.7353e-02],
          [ 3.0986e-02, -1.0534e-02, -1.1327e-02]],

         [[ 4.9789e-02, -1.1180e-02, -8.2899e-02],
          [ 2.0316e-02,  4.0599e-02, -2.2287e-02],
          [ 5.0847e-02, -8.3178e-03, -7.1615e-02]],

         [[ 3.8265e-03,  8.7877e-02, -3.8754e-02],
          [ 2.6671e-02, -3.9028e-02,  2.4208e-02],
          [ 1.1449e-02,  6.2868e-02,  3.6313e-02]],

         [[ 3.5610e-02, -3.7163e-02,  4.6060e-02],
          [ 9.7176e-03,  4.0420e-02,  5.7095e-03],
          [-1.0166e-01,  2.6617e-02,  3.1448e-02]],

         [[-3.8546e-02, -2.9275e-02, -3.6343e-02],
          [ 2.9826e-02,  6.0978e-02,  4.1320e-02],
          [-7.3305e-02,  5.2130e-02,  5.3171e-02]],

         [[ 5.8804e-02, -7.4155e-02,  7.8913e-02],
          [ 7.5565e-02,  4.9906e-02, -4.0583e-02],
          [-2.8628e-02, -2.0749e-02,  5.2863e-02]],

         [[-5.6819e-03,  5.0410e-02, -7.3634e-03],
          [ 2.5217e-02, -6.1997e-02, -6.4580e-03],
          [-3.2750e-02,  1.2301e-02,  6.5766e-02]],

         [[-1.0003e-02, -6.8887e-02,  7.8610e-03],
          [-3.0840e-04,  3.5925e-02, -2.9642e-02],
          [ 1.2696e-02, -6.0514e-02,  5.5266e-02]],

         [[-1.4801e-02,  5.7944e-02, -6.7704e-02],
          [-4.2277e-02,  3.2675e-02,  6.4896e-03],
          [ 7.8282e-02,  2.3461e-02, -5.7884e-02]]],


        [[[-6.0504e-02, -3.5292e-02, -3.1089e-02],
          [ 1.0808e-02,  5.4339e-02, -2.5794e-02],
          [ 4.7179e-02,  1.0685e-01,  2.7393e-02]],

         [[-5.5142e-02,  3.8248e-03,  4.7597e-02],
          [ 4.5806e-03, -6.5700e-02,  5.4036e-02],
          [-8.8737e-02, -6.6490e-02,  3.2369e-02]],

         [[ 1.4429e-02,  1.2713e-02, -2.4385e-02],
          [-4.5163e-02, -9.2194e-02,  7.5336e-02],
          [ 3.2440e-02, -5.2232e-02,  3.1479e-02]],

         [[-8.3972e-02, -2.9957e-04, -2.9836e-02],
          [-8.9465e-02, -5.4485e-02, -2.1890e-02],
          [-6.1352e-02, -9.2542e-02, -3.3512e-02]],

         [[-4.2352e-02,  4.7481e-02,  9.0443e-03],
          [ 4.7825e-02, -3.1749e-02, -3.0759e-02],
          [-7.4110e-02,  7.4850e-02, -3.4613e-02]],

         [[-1.4339e-02,  1.0023e-01, -7.5184e-04],
          [ 5.6894e-03,  3.6436e-02, -1.0445e-01],
          [ 1.7307e-02, -2.4316e-02, -1.0577e-01]],

         [[-1.9014e-02, -8.7525e-02,  5.6381e-02],
          [-2.0787e-02, -3.3135e-02,  3.9314e-02],
          [-4.4153e-03, -4.4419e-02,  7.8704e-02]],

         [[ 4.7002e-02, -1.2225e-01,  2.8657e-02],
          [-1.2447e-02, -2.1306e-02,  1.0706e-01],
          [-1.1391e-01, -5.1184e-02,  1.3688e-01]],

         [[-2.3265e-02, -2.8412e-02,  1.2770e-01],
          [-6.0039e-03, -9.1918e-03,  9.7179e-02],
          [ 1.6493e-02, -3.9475e-02,  2.9622e-02]],

         [[-7.0621e-02,  1.2895e-02,  9.6689e-02],
          [-2.4964e-02, -2.1929e-02, -2.7997e-03],
          [-5.3555e-02,  5.6150e-02,  2.8346e-02]],

         [[ 3.3935e-02,  9.8236e-03,  9.3419e-02],
          [ 2.8034e-02,  6.7073e-02, -1.3452e-02],
          [ 6.6891e-02, -9.8822e-03,  7.8101e-02]],

         [[-2.2715e-02, -4.0884e-02, -3.9730e-05],
          [-1.9610e-02, -2.3752e-02, -3.7356e-02],
          [ 7.0654e-03,  7.5054e-04, -8.1986e-02]],

         [[-6.5062e-02,  2.5982e-02, -7.4097e-02],
          [-1.8371e-02, -1.1377e-01,  4.8208e-02],
          [ 9.0530e-03, -2.7732e-03, -2.2800e-02]],

         [[-5.4585e-02,  2.2033e-02, -7.9956e-02],
          [-5.3400e-02, -6.4440e-02,  3.2446e-02],
          [ 8.8402e-02,  7.8983e-02, -4.4852e-02]],

         [[-3.3073e-02,  6.1706e-02,  3.2191e-02],
          [ 7.3458e-03,  5.3021e-02,  2.1531e-02],
          [-2.2176e-02,  1.5435e-02, -4.7808e-02]],

         [[ 5.3800e-02, -7.7839e-02, -1.8340e-02],
          [ 3.1460e-02,  1.4617e-03,  2.7149e-02],
          [ 5.8245e-02, -8.5214e-02, -2.2535e-02]],

         [[ 3.7306e-03,  2.2511e-02, -7.1382e-03],
          [-2.3142e-02,  3.5767e-02, -4.3391e-02],
          [ 6.6422e-02,  8.5469e-02,  2.8138e-02]],

         [[ 1.6602e-02, -9.3900e-03,  6.6627e-02],
          [-6.4615e-02,  3.6550e-02,  7.4820e-03],
          [-2.9128e-02, -6.3161e-03,  1.0915e-01]],

         [[-4.0981e-02, -2.0872e-02,  1.2916e-02],
          [-1.3598e-02,  2.9816e-02, -8.9688e-03],
          [-5.1968e-02,  5.7026e-02, -3.6911e-02]],

         [[ 1.7983e-04,  7.1378e-02, -1.2782e-02],
          [-6.6728e-02,  1.8593e-02,  3.2498e-02],
          [-1.2814e-02,  3.8676e-02, -2.2299e-02]],

         [[-7.3922e-02,  5.4101e-02,  3.6274e-02],
          [ 2.3471e-02, -6.2902e-02, -5.1482e-02],
          [ 2.3110e-03,  4.5299e-03,  2.5984e-02]],

         [[ 2.4417e-02,  5.5069e-02, -2.3399e-02],
          [-1.9161e-02,  1.3357e-01, -3.2724e-02],
          [-3.2195e-02, -2.3912e-03, -7.0045e-03]],

         [[-1.4134e-02, -3.6292e-02,  4.4830e-04],
          [-4.4998e-03,  5.2942e-02, -6.0939e-02],
          [-2.3717e-02,  6.0708e-02,  7.4180e-02]],

         [[-1.1586e-03,  4.9039e-02,  3.8912e-02],
          [-1.5097e-02,  6.4546e-02, -5.7695e-02],
          [ 5.1570e-02,  5.2844e-02, -2.9340e-02]],

         [[-7.3972e-02,  2.1831e-02,  8.3414e-02],
          [ 2.7611e-02, -1.8229e-02,  1.0681e-01],
          [ 1.5248e-02,  4.7386e-02,  4.3676e-02]],

         [[-1.8056e-02,  2.8750e-02, -4.3719e-02],
          [-4.1019e-02, -3.0595e-03, -7.3676e-02],
          [ 4.2377e-02,  1.1960e-02, -7.5834e-02]],

         [[-8.2933e-02,  4.2803e-04, -3.8915e-02],
          [ 1.6013e-02, -7.2239e-02,  3.0430e-03],
          [ 8.3024e-02, -2.8544e-03, -7.2290e-02]],

         [[ 4.0975e-02,  1.9855e-02, -7.3711e-02],
          [-1.7957e-02,  5.5643e-02, -7.1333e-02],
          [ 6.0015e-02,  1.3597e-01, -1.0350e-01]],

         [[-3.0315e-02,  5.9553e-02, -3.1216e-02],
          [-5.1210e-03, -3.9903e-02,  3.6246e-02],
          [ 2.7777e-02,  7.5258e-02, -8.5923e-03]],

         [[-1.4088e-03,  4.2638e-02, -4.3820e-02],
          [-1.2376e-02,  1.9384e-03, -3.9866e-02],
          [ 4.4050e-02,  3.1240e-02, -2.3861e-02]],

         [[-4.2364e-02, -2.1478e-02,  3.4314e-02],
          [-1.4555e-02, -2.9505e-02, -9.4119e-03],
          [-4.1674e-03,  9.0211e-02, -1.8044e-02]],

         [[ 5.4565e-02, -6.1885e-02,  9.0603e-02],
          [ 7.8804e-02,  5.9432e-02,  6.3905e-02],
          [ 2.1072e-02, -2.5641e-03,  3.7319e-02]]],


        [[[-1.7812e-02,  2.6211e-02,  1.3080e-02],
          [-9.6855e-03,  1.8606e-03,  3.8489e-03],
          [ 6.6577e-02,  1.3018e-01, -6.1083e-02]],

         [[-6.6128e-03,  3.4020e-02,  2.8101e-02],
          [-5.2638e-02, -1.0569e-01, -3.1744e-02],
          [-7.0476e-02, -6.7086e-02,  3.6910e-02]],

         [[-4.0947e-02, -6.3239e-02, -1.7089e-02],
          [-4.9084e-02, -7.6645e-02,  7.8556e-02],
          [-1.4525e-03,  8.9473e-03, -1.5449e-02]],

         [[ 3.5186e-02,  5.0871e-02, -9.0122e-02],
          [-1.0824e-01, -1.0716e-01, -1.2691e-01],
          [-2.1703e-02,  4.7221e-02,  2.5480e-02]],

         [[-3.1157e-02,  4.2320e-02, -1.0123e-01],
          [-2.9156e-02, -4.3016e-02, -5.2841e-02],
          [ 3.1438e-02, -4.8396e-03, -1.0325e-01]],

         [[ 1.1644e-02,  1.4036e-02, -3.4804e-03],
          [ 3.3711e-02,  1.2663e-02, -5.2658e-03],
          [-3.5706e-02, -8.5669e-02,  1.4500e-02]],

         [[-5.3666e-02, -7.0432e-03,  9.4524e-02],
          [-3.4344e-02,  1.3096e-02,  2.7046e-02],
          [ 8.7668e-02,  2.3196e-02,  8.8650e-02]],

         [[ 3.6457e-02, -5.1408e-02,  9.0609e-02],
          [-4.4090e-02, -3.6624e-02, -1.0891e-02],
          [ 3.4752e-02,  8.4137e-03,  5.8060e-02]],

         [[ 1.9456e-02, -1.4832e-02,  7.1019e-02],
          [ 2.4506e-02, -3.6437e-02,  6.2708e-02],
          [ 7.7962e-02, -1.8256e-02,  1.0625e-01]],

         [[-2.0153e-02, -5.4086e-02,  9.8284e-02],
          [-6.4401e-02,  7.9169e-02, -2.2697e-03],
          [ 3.9456e-02, -1.1853e-02,  4.9747e-02]],

         [[-1.6180e-02, -9.0743e-02,  3.0722e-02],
          [-5.3890e-03,  3.9299e-02,  6.5687e-03],
          [-6.2314e-04,  7.5440e-02,  1.9578e-02]],

         [[ 4.0942e-02,  2.0239e-03,  3.6506e-03],
          [ 2.2355e-02, -4.3051e-03,  3.2696e-02],
          [-7.6354e-02, -1.1235e-01, -6.1736e-02]],

         [[ 3.0317e-03, -7.8063e-02,  1.7135e-03],
          [-3.2470e-02,  3.1690e-02,  4.1476e-02],
          [ 3.6188e-02,  4.0820e-02, -3.4974e-02]],

         [[ 1.3869e-02, -1.9452e-02,  1.5444e-02],
          [-1.2477e-02,  6.0621e-03,  5.1192e-02],
          [ 1.3904e-03,  3.0102e-02, -1.2671e-02]],

         [[ 2.9239e-02,  6.2592e-02, -2.1609e-02],
          [-3.5478e-02,  3.4032e-03, -9.4303e-03],
          [-2.0084e-02,  5.1625e-02, -7.5210e-02]],

         [[ 4.6419e-02,  5.2629e-02,  6.8106e-02],
          [ 4.8319e-02, -5.2033e-02,  5.5532e-02],
          [ 5.5138e-02, -2.6516e-02, -3.3247e-02]],

         [[ 2.8282e-03, -1.6547e-02, -6.5347e-02],
          [ 1.0731e-01,  8.2478e-02, -2.0813e-02],
          [ 2.7257e-02, -4.9075e-02, -2.3730e-02]],

         [[-5.9516e-02,  1.3773e-02,  4.5856e-02],
          [ 3.2974e-02, -5.5369e-03,  3.4341e-02],
          [-3.5325e-02,  1.9502e-02,  1.2127e-01]],

         [[-1.8206e-03,  4.3551e-02, -8.5613e-02],
          [-3.1801e-02,  3.0736e-02, -3.3018e-02],
          [ 5.2171e-02,  4.7499e-02,  1.2852e-03]],

         [[-5.1971e-02,  2.0423e-02, -2.4545e-02],
          [ 4.7075e-02,  7.4537e-02,  8.7604e-02],
          [ 1.1690e-01,  1.1307e-01, -4.5014e-03]],

         [[-3.0048e-02, -8.0013e-02,  3.6808e-02],
          [ 3.5351e-02,  9.6080e-03,  1.7105e-02],
          [ 1.9494e-02, -7.0881e-02,  8.4138e-03]],

         [[ 5.9718e-03,  6.4581e-02, -7.2864e-02],
          [ 6.7259e-02, -6.8244e-02,  1.7556e-03],
          [ 6.8877e-02,  1.6002e-02,  2.3170e-02]],

         [[ 2.4250e-02,  3.1015e-02, -2.0498e-02],
          [-5.3762e-02, -4.3462e-02,  1.4881e-02],
          [ 4.4069e-02,  2.7169e-02, -1.1608e-02]],

         [[-8.3019e-02, -1.4242e-02,  8.2617e-03],
          [ 2.6228e-02,  1.0420e-02, -3.4477e-02],
          [-2.3683e-02,  1.0989e-02, -7.9206e-02]],

         [[-7.9099e-02, -5.4426e-02,  2.8436e-02],
          [ 6.7045e-02,  9.8652e-02,  9.2922e-02],
          [-2.7415e-02,  2.8578e-02,  3.8341e-02]],

         [[ 6.5964e-02,  8.9825e-03, -3.8723e-02],
          [-9.1530e-02, -7.6909e-02, -7.1892e-02],
          [-7.3949e-02,  6.3661e-03, -2.3465e-02]],

         [[-6.0740e-02, -9.5717e-03, -5.7068e-02],
          [-4.4611e-03,  4.7687e-02, -2.2644e-02],
          [-2.6128e-02, -3.2260e-02, -9.7782e-02]],

         [[-6.4545e-02,  1.7858e-02,  1.3916e-02],
          [ 2.4689e-02,  9.6002e-02, -7.2236e-02],
          [-4.6423e-03,  7.8419e-02, -7.9981e-02]],

         [[-6.3336e-02,  1.9209e-02, -7.2263e-02],
          [ 4.1181e-02,  6.2226e-02, -5.2414e-02],
          [ 2.2862e-02,  1.8604e-03, -9.4722e-02]],

         [[ 5.3463e-02,  5.4778e-02, -3.3603e-02],
          [ 5.7994e-02,  4.9633e-02, -4.5253e-02],
          [ 5.3428e-02,  4.4564e-02, -1.0558e-01]],

         [[-5.1582e-02,  1.5571e-02, -4.9437e-02],
          [ 3.4257e-02,  3.8027e-02,  3.3996e-02],
          [ 2.7753e-02, -5.0092e-02, -4.9307e-02]],

         [[ 3.7827e-02,  9.0956e-04,  2.1749e-02],
          [ 2.6549e-02, -6.7512e-02, -5.0127e-03],
          [ 1.2704e-02,  7.3030e-02,  1.0794e-01]]],


        [[[-4.9473e-02, -1.8203e-02, -1.9099e-03],
          [-4.0265e-02, -1.0561e-01, -6.2607e-03],
          [-3.2546e-03,  1.9723e-02, -8.3392e-03]],

         [[ 2.4825e-02, -3.6435e-02,  2.2163e-02],
          [ 1.4741e-04, -3.7078e-02, -1.6790e-02],
          [ 6.1702e-03, -8.1293e-02, -7.8856e-02]],

         [[ 6.0565e-03,  3.8373e-02,  3.3296e-02],
          [-4.8362e-02, -3.9682e-02, -3.6264e-03],
          [ 4.0262e-02, -6.4232e-03,  2.9576e-02]],

         [[ 6.0293e-02,  2.8995e-03,  8.2086e-02],
          [-1.0332e-01, -9.4122e-03,  3.0135e-02],
          [-3.9780e-02, -3.0534e-02, -3.5856e-02]],

         [[ 6.6281e-02, -4.4676e-02,  2.1700e-03],
          [ 2.4673e-02, -3.2471e-02,  1.3608e-03],
          [-4.7909e-02, -2.8756e-02,  9.5278e-02]],

         [[-5.9961e-02, -1.9851e-02, -6.9045e-02],
          [ 6.9114e-02, -2.9078e-02,  4.4147e-02],
          [-2.0371e-02, -1.1866e-01,  1.1805e-02]],

         [[-3.2094e-02, -3.8773e-03, -3.1227e-02],
          [-1.0696e-01, -2.4263e-02, -2.7056e-02],
          [-2.3290e-04,  1.2726e-02, -2.4927e-02]],

         [[ 9.4899e-03,  3.3314e-02, -1.7217e-02],
          [-4.1818e-02, -7.0679e-02, -3.7233e-02],
          [-6.2236e-02,  5.7111e-02, -9.4034e-02]],

         [[ 1.5391e-02,  1.9697e-02, -1.4590e-02],
          [-9.4065e-02, -4.3154e-02, -5.1916e-02],
          [-4.3831e-02,  1.4322e-02, -3.0712e-02]],

         [[-2.2420e-02,  7.4995e-02,  2.3151e-02],
          [-3.4003e-02,  2.0909e-02, -7.3272e-02],
          [-2.7212e-02,  7.3081e-02, -7.0228e-02]],

         [[-8.1507e-02, -1.7414e-02, -8.8070e-02],
          [ 9.2923e-03,  4.7037e-02,  1.4057e-02],
          [-4.2264e-02,  7.7239e-02,  3.6880e-02]],

         [[-5.4116e-02,  4.1077e-02, -2.1168e-02],
          [-1.1278e-02, -1.1268e-02,  2.3855e-02],
          [ 3.3485e-03, -4.5532e-02, -3.1644e-02]],

         [[-1.4669e-02, -4.7627e-02,  3.8086e-03],
          [-8.3838e-02, -3.7824e-03,  3.0720e-02],
          [-9.1295e-02,  5.3780e-02, -6.6621e-03]],

         [[-9.8376e-02, -1.1002e-01, -7.3376e-02],
          [ 9.6984e-02, -7.1750e-03,  2.1167e-02],
          [-1.9392e-02, -2.0624e-02, -4.1435e-02]],

         [[-8.2853e-03, -3.4083e-02, -1.4399e-02],
          [-1.7789e-02,  4.0773e-02, -1.8492e-02],
          [ 2.0015e-02, -9.1696e-02,  3.0482e-02]],

         [[-3.6141e-03, -3.9010e-02,  5.0182e-02],
          [-9.5849e-03, -2.4733e-02, -2.6782e-02],
          [ 6.4537e-03, -7.5004e-02, -5.2689e-02]],

         [[-8.6647e-02, -6.6628e-02, -6.2692e-02],
          [ 3.8313e-02,  9.0186e-02,  1.0670e-01],
          [ 4.0616e-02, -6.6934e-02, -3.3629e-02]],

         [[-8.7900e-02,  1.0417e-01, -2.6940e-02],
          [-6.8236e-02,  1.0597e-02, -8.3923e-02],
          [-6.3251e-02,  1.1580e-01, -1.4375e-02]],

         [[ 9.9399e-03, -2.4343e-02,  3.9160e-02],
          [ 5.2582e-02, -2.0445e-02, -3.2248e-02],
          [-2.4544e-03, -4.2755e-02,  6.3550e-02]],

         [[-2.4374e-02, -6.3592e-03, -7.4321e-02],
          [ 8.2264e-02,  5.2211e-02,  2.9257e-02],
          [-5.6421e-02, -1.4051e-02,  1.5270e-02]],

         [[-5.6386e-02, -7.0763e-02, -1.9377e-02],
          [ 4.5167e-02,  7.6350e-03,  1.0293e-01],
          [-5.9575e-03, -4.1457e-02, -4.7817e-02]],

         [[-4.2177e-02, -2.6441e-02, -8.5197e-02],
          [ 4.8137e-02, -5.0271e-02,  2.6205e-02],
          [-6.6403e-02, -6.4889e-02, -1.6912e-02]],

         [[ 3.1710e-02,  3.4849e-02, -7.4321e-03],
          [-6.3015e-03, -5.6387e-02,  5.1971e-02],
          [-6.2915e-02,  9.9256e-03,  8.0627e-02]],

         [[ 2.9309e-02, -6.6501e-02,  4.4432e-02],
          [ 3.2391e-03,  1.2368e-02,  4.8716e-02],
          [-6.9449e-02,  2.7053e-02, -3.5743e-02]],

         [[ 1.3137e-02, -4.0962e-02, -3.8826e-02],
          [-8.1736e-04, -2.7884e-02, -5.3640e-02],
          [-2.4289e-02,  2.1426e-02, -1.4379e-02]],

         [[ 1.5367e-02, -2.3709e-02,  5.9149e-02],
          [ 2.5393e-02,  2.6336e-02,  3.3143e-02],
          [ 2.2922e-02, -4.0168e-03,  4.1902e-02]],

         [[-7.2590e-02, -7.9790e-02,  3.2718e-03],
          [ 4.2159e-02, -1.1028e-02,  9.6518e-02],
          [-4.1786e-02, -4.8291e-02, -2.4120e-02]],

         [[-7.7842e-02, -6.1067e-02,  2.0417e-02],
          [-1.0525e-02, -4.9350e-02,  1.0407e-01],
          [-3.0579e-02,  3.7996e-02,  1.3680e-02]],

         [[-7.3140e-02, -1.2082e-02, -3.4158e-02],
          [ 8.5767e-02,  8.4663e-02,  6.2056e-02],
          [-2.4413e-02, -2.0588e-02, -1.9196e-02]],

         [[ 4.1100e-02,  3.7905e-02, -6.2923e-04],
          [ 1.1777e-02, -3.8562e-03, -3.6554e-03],
          [-2.7324e-02, -2.0928e-02,  6.1520e-02]],

         [[ 1.1829e-02,  3.5564e-03, -1.5598e-02],
          [ 7.4415e-02,  2.9143e-03,  3.7751e-02],
          [-5.2063e-02, -5.6582e-02, -4.2066e-02]],

         [[-3.3662e-02,  6.5308e-02, -8.7993e-04],
          [ 3.6940e-02, -2.0776e-02, -1.0440e-01],
          [ 3.9541e-02,  8.1113e-03, -7.8600e-02]]],


        [[[ 3.3213e-02,  4.2263e-03, -5.6365e-03],
          [ 5.5169e-03, -3.5884e-02, -9.3314e-04],
          [ 3.0950e-03,  3.7696e-02, -5.3114e-02]],

         [[-4.7536e-02, -2.1678e-02, -6.6546e-02],
          [-7.0662e-02,  1.5913e-02, -7.8961e-02],
          [-3.9740e-02, -3.9885e-02,  2.6336e-02]],

         [[ 3.0490e-02,  2.4493e-02, -7.9524e-04],
          [-7.2163e-03, -3.3088e-02, -2.5622e-03],
          [-4.7175e-03,  1.9778e-03, -4.6995e-02]],

         [[ 3.8493e-02,  2.0092e-02,  4.2880e-02],
          [ 2.5376e-02, -1.0453e-02, -5.8875e-03],
          [-5.5219e-02, -2.6589e-02, -5.2285e-02]],

         [[-5.8803e-02, -4.5290e-02, -7.9387e-03],
          [ 3.1443e-03, -5.6879e-03,  6.4649e-03],
          [-7.8941e-02, -7.1549e-02, -1.3713e-02]],

         [[ 4.8483e-02, -4.1866e-02, -3.8247e-02],
          [ 1.2884e-02, -5.3562e-02, -2.5573e-02],
          [-7.2908e-03,  4.6734e-02,  2.6466e-02]],

         [[-1.2032e-02, -3.5534e-02, -2.0636e-02],
          [-3.2085e-02, -3.2772e-02,  4.6286e-02],
          [-5.3008e-02,  6.3359e-03,  1.5313e-02]],

         [[-3.7015e-02, -3.9103e-03, -4.9914e-02],
          [ 1.2457e-02, -3.1775e-02, -4.9090e-02],
          [-2.3583e-02, -2.6990e-02, -2.6860e-02]],

         [[-4.4343e-02,  3.2523e-02,  5.6243e-02],
          [ 7.8645e-03, -4.1815e-02, -3.8801e-03],
          [-7.7061e-02, -4.6472e-03, -4.8569e-03]],

         [[ 1.2540e-02, -3.9700e-02, -9.7380e-03],
          [-9.7917e-03, -6.5928e-04, -3.2428e-02],
          [ 1.5689e-02, -2.9446e-02,  4.5970e-02]],

         [[-2.9231e-02, -4.1365e-03,  1.8451e-02],
          [ 7.5136e-03, -4.3429e-03, -4.3561e-02],
          [-7.1869e-02, -2.4375e-02, -4.0124e-03]],

         [[-6.2557e-02, -4.8520e-02, -6.9182e-02],
          [-7.2573e-02, -1.1891e-02, -4.4910e-02],
          [-4.5147e-02,  7.9185e-03, -4.0912e-02]],

         [[ 3.1273e-02, -5.5451e-02, -9.9176e-03],
          [-2.6982e-02, -5.6313e-02, -5.5999e-02],
          [-3.6210e-02, -3.1554e-02, -4.1032e-02]],

         [[ 4.7990e-02, -8.4200e-03, -4.9924e-02],
          [-4.4866e-02, -5.1423e-03, -1.4666e-02],
          [-3.0125e-02, -4.4639e-02,  5.7758e-02]],

         [[-1.8138e-02, -1.9619e-02,  2.0356e-02],
          [-3.6746e-02, -6.1710e-02, -9.5609e-03],
          [-1.4211e-02, -6.9182e-02, -3.5806e-02]],

         [[-5.3927e-02,  2.2580e-02,  2.0347e-02],
          [-6.4208e-03, -1.1543e-02,  6.1689e-03],
          [-6.2223e-02, -9.6768e-03,  2.3739e-02]],

         [[-2.2359e-02,  2.0287e-02,  3.3961e-02],
          [ 5.2418e-02, -1.3175e-02, -9.3709e-04],
          [ 1.0637e-02,  3.4231e-02, -1.3344e-02]],

         [[ 6.4741e-03,  4.3353e-02,  4.6747e-02],
          [-1.7656e-02, -1.0964e-02, -3.0107e-02],
          [-8.2223e-02,  4.9542e-02,  7.5608e-03]],

         [[-4.4204e-02,  7.9490e-04, -6.7709e-02],
          [ 5.1500e-02, -2.1248e-02, -7.0846e-02],
          [-6.7467e-03,  6.4990e-03, -4.5320e-02]],

         [[-2.0862e-03,  9.5940e-03, -4.6184e-02],
          [ 3.6006e-02, -1.6227e-02, -5.0997e-03],
          [-6.3136e-04,  2.9176e-02, -4.6652e-02]],

         [[ 1.1914e-02, -1.1794e-02,  2.4353e-02],
          [-2.1544e-02, -2.5110e-02,  4.3712e-02],
          [ 4.7845e-02, -4.6147e-02, -5.6491e-02]],

         [[ 1.2554e-02, -5.1043e-02, -2.5953e-02],
          [-4.3661e-02,  2.6035e-02,  4.9663e-02],
          [-3.4782e-02, -5.3934e-02, -5.7592e-02]],

         [[-1.8038e-02,  1.6716e-02, -5.8662e-02],
          [-5.7700e-02, -2.5820e-02,  2.9975e-04],
          [ 1.7933e-02, -4.7088e-03,  4.8982e-03]],

         [[ 1.1513e-02, -6.5153e-03, -4.8549e-02],
          [-2.9204e-03, -8.5423e-02,  1.1545e-02],
          [-8.0046e-02, -4.6377e-03, -4.9478e-02]],

         [[ 5.2419e-02, -1.8563e-02, -2.1926e-02],
          [-1.3339e-02,  5.2690e-02, -6.3592e-03],
          [ 7.4199e-03, -1.0943e-02, -5.0097e-02]],

         [[-3.9364e-03, -2.9252e-02,  1.3144e-02],
          [-6.0808e-02, -5.8250e-02, -8.5987e-02],
          [ 2.1204e-02, -8.6321e-02, -2.4426e-02]],

         [[ 3.2423e-03, -3.6748e-02, -5.7222e-02],
          [-3.5207e-02,  5.9013e-03, -3.5398e-03],
          [-6.4772e-02, -6.9837e-02,  1.1194e-02]],

         [[-5.6259e-02, -5.3835e-02, -2.7900e-02],
          [-5.2799e-02,  1.3864e-02,  3.8882e-02],
          [-2.7468e-02,  6.0206e-03,  1.1412e-02]],

         [[-4.7926e-02, -5.2448e-02, -3.1060e-02],
          [-1.7784e-02,  2.8493e-02,  4.9646e-02],
          [-1.7776e-02, -8.2997e-02, -4.6599e-02]],

         [[-1.7013e-02, -1.8033e-02, -2.2032e-02],
          [ 2.1472e-02, -7.1413e-03, -7.1462e-02],
          [-2.7490e-02,  3.6845e-03, -5.7055e-02]],

         [[-3.6670e-02, -7.4316e-02, -5.8257e-02],
          [-3.6727e-02, -2.1834e-03,  7.0199e-03],
          [-1.1899e-02,  1.8756e-02, -3.4938e-02]],

         [[-5.5472e-03, -5.7060e-02, -5.0057e-02],
          [ 6.6169e-03,  3.8195e-02, -4.2237e-02],
          [-8.6874e-03, -5.9684e-03, -1.1836e-02]]],


        [[[-4.4452e-02,  2.9498e-02, -8.3428e-02],
          [-2.8800e-02,  2.8926e-02,  1.8068e-02],
          [ 4.0674e-02,  5.8801e-02,  7.7874e-03]],

         [[-1.2138e-02,  6.1157e-02, -6.6134e-02],
          [ 1.2991e-02, -3.8173e-02,  2.0180e-02],
          [-6.0666e-02, -5.5760e-02, -2.9584e-03]],

         [[-2.6331e-02, -1.3154e-04, -9.1021e-03],
          [-4.0450e-02,  5.7759e-02,  7.3460e-03],
          [ 3.0195e-02, -2.7695e-02,  7.8056e-02]],

         [[-7.4381e-02, -6.3472e-02, -5.8125e-02],
          [-4.4477e-02,  1.9497e-02, -6.1435e-03],
          [-1.9212e-02, -3.2691e-03, -9.8915e-02]],

         [[-1.5067e-02,  5.3268e-02, -4.0887e-02],
          [ 1.5854e-02, -3.0647e-02, -4.4624e-02],
          [ 2.8270e-02,  6.5717e-02, -6.8922e-02]],

         [[-5.9750e-03,  1.0636e-02, -4.8646e-02],
          [-1.1052e-01, -9.5224e-03, -5.7942e-02],
          [-3.7933e-02,  8.2693e-02, -8.8793e-02]],

         [[-1.9295e-02,  5.1190e-02,  3.3349e-02],
          [-9.5787e-03,  1.1548e-03,  2.9608e-02],
          [-4.3324e-03, -2.4934e-02,  1.8548e-02]],

         [[-1.9361e-03, -4.8742e-02,  1.0871e-01],
          [ 1.0028e-02, -3.5153e-02,  7.7361e-02],
          [-7.1092e-02, -9.0952e-02,  1.4122e-01]],

         [[ 5.3021e-02,  3.0897e-02,  9.8986e-02],
          [ 2.3717e-02, -2.6041e-02,  2.3879e-02],
          [-1.0409e-01,  5.7665e-03, -5.6742e-03]],

         [[-4.9513e-02,  3.5890e-02, -9.1156e-03],
          [-1.5257e-02, -3.3815e-02,  7.2267e-02],
          [-4.1672e-02,  1.9515e-02,  1.0569e-02]],

         [[ 5.2250e-02,  4.4877e-02,  5.1241e-02],
          [ 7.4896e-03, -1.0508e-02, -6.3278e-02],
          [ 8.3204e-02,  3.2226e-02,  7.1726e-03]],

         [[ 6.9648e-02,  6.6671e-02, -9.7416e-03],
          [-3.9516e-02, -6.7112e-02, -4.4275e-02],
          [-2.5947e-02, -1.0564e-02, -9.6033e-04]],

         [[-4.5030e-02,  7.3756e-03, -5.7624e-02],
          [-5.3076e-02, -4.8886e-02,  2.6040e-02],
          [ 5.7953e-03, -6.0402e-02,  4.6794e-02]],

         [[ 1.3437e-02,  2.6977e-02, -2.1133e-02],
          [-2.1037e-02,  5.4833e-02, -1.0884e-01],
          [ 4.8512e-02,  5.6110e-02,  9.2305e-03]],

         [[-7.7386e-03,  5.2846e-02, -3.5325e-02],
          [-2.9002e-02,  6.1444e-02,  4.4105e-03],
          [-4.7250e-02, -2.5328e-02, -7.1043e-02]],

         [[-2.7619e-02,  1.5688e-02,  3.9298e-02],
          [ 4.1942e-02, -8.6384e-03,  1.0618e-02],
          [-4.9173e-02, -7.9762e-02,  6.4230e-02]],

         [[-4.9297e-02,  2.4149e-02,  7.1285e-03],
          [-4.3142e-02,  8.1368e-02, -5.4970e-02],
          [ 1.3764e-02,  1.0383e-01, -3.0084e-02]],

         [[-1.8865e-02, -4.2421e-02,  5.1946e-02],
          [ 5.1969e-02,  2.3531e-02,  4.2327e-02],
          [-3.1464e-02, -5.2849e-02,  1.3974e-01]],

         [[-6.0959e-02,  7.7767e-02, -4.2580e-02],
          [-6.6748e-02,  1.3206e-02, -5.8117e-02],
          [-5.5151e-02,  9.2401e-02, -1.2267e-01]],

         [[ 4.9860e-02,  3.5062e-02, -4.0988e-02],
          [-7.4885e-02,  4.4383e-02, -2.7083e-02],
          [ 7.1815e-02,  6.1540e-02, -6.4123e-03]],

         [[-3.3772e-02,  3.7800e-02,  1.5245e-03],
          [-8.2514e-03, -1.0459e-01, -7.1681e-02],
          [ 1.2619e-02,  2.7267e-02,  3.1389e-02]],

         [[ 1.5611e-02, -8.5160e-03, -7.5375e-02],
          [-9.5170e-02,  7.1631e-02, -9.3268e-02],
          [ 3.7716e-02,  1.9390e-02,  8.3662e-03]],

         [[-8.0107e-02,  9.7212e-03, -6.1918e-02],
          [-1.9577e-03,  4.9286e-02,  2.4489e-02],
          [ 4.0326e-02,  4.4917e-02, -1.2785e-01]],

         [[-3.0593e-02,  6.7444e-02,  3.3574e-02],
          [-1.9345e-02, -1.6447e-02, -4.5443e-02],
          [ 7.0392e-02,  7.5101e-02,  4.2249e-04]],

         [[ 2.4501e-04,  3.7261e-02,  3.8110e-02],
          [ 5.0732e-02,  1.8864e-02,  3.9574e-02],
          [ 2.9555e-02,  5.6821e-02,  7.0547e-02]],

         [[-3.7447e-02,  5.4845e-02,  2.4422e-04],
          [-3.2796e-02,  5.9420e-02, -8.9656e-02],
          [-5.9319e-02,  4.4040e-02, -7.9334e-02]],

         [[-6.4457e-02,  2.1334e-02, -9.0339e-02],
          [-6.3359e-02, -1.1327e-02, -1.4777e-02],
          [ 5.0407e-02,  3.8439e-02,  4.4904e-02]],

         [[ 1.2267e-02,  5.6242e-02, -6.7819e-03],
          [-9.6393e-02,  5.0007e-02, -7.1618e-02],
          [ 4.2065e-02,  6.0135e-02, -9.6715e-02]],

         [[-3.5114e-02,  9.6176e-02, -9.7245e-02],
          [-1.5053e-03,  1.9941e-02, -2.7810e-02],
          [-9.5529e-03,  7.5505e-02, -8.4574e-02]],

         [[-2.1061e-03,  6.2808e-02, -7.0680e-04],
          [-1.8599e-02,  7.3448e-02,  1.5136e-02],
          [ 2.3339e-03, -3.0332e-02, -4.9959e-02]],

         [[ 2.9618e-02, -2.4211e-02,  5.9094e-02],
          [ 2.1626e-02,  4.1803e-02, -5.8713e-02],
          [ 1.8050e-03,  9.8804e-02, -8.1682e-02]],

         [[-2.3578e-02, -2.1440e-02,  1.1110e-01],
          [ 2.1952e-02, -2.4709e-02,  8.8446e-02],
          [ 1.0765e-02, -4.7670e-02,  3.5121e-02]]],


        [[[-4.3228e-02, -1.5880e-02,  7.7272e-03],
          [-3.0416e-02, -3.2540e-02, -2.0228e-02],
          [ 8.3330e-02,  3.3010e-02,  5.3921e-02]],

         [[-3.6560e-02,  9.5499e-04, -1.8080e-02],
          [ 1.6478e-02,  2.6444e-02,  1.3084e-02],
          [-4.7916e-02,  1.8970e-02, -3.0132e-02]],

         [[ 2.1909e-02,  1.9700e-02,  1.3806e-02],
          [-4.8128e-02, -7.0403e-02,  5.7675e-02],
          [ 7.9750e-02, -2.2442e-02,  2.5703e-02]],

         [[ 2.2417e-02,  3.0740e-02, -3.6387e-02],
          [-3.3930e-02, -9.7942e-03, -6.5927e-02],
          [-3.1482e-02,  5.4763e-03, -2.6883e-02]],

         [[-1.1287e-02, -3.9226e-02, -1.0935e-02],
          [ 2.8818e-03, -3.6713e-02, -3.0130e-02],
          [-6.5043e-02,  1.1648e-02, -5.6587e-02]],

         [[ 2.8468e-02, -2.2739e-02, -4.0781e-02],
          [ 4.6273e-02, -5.0521e-02, -5.8417e-02],
          [-3.6831e-03, -2.2657e-02, -3.4108e-02]],

         [[-2.0243e-02,  1.9475e-02,  1.7438e-02],
          [-5.5791e-02, -2.3202e-02, -7.6924e-02],
          [-7.4371e-02,  3.1102e-03, -7.6746e-02]],

         [[-1.1064e-02,  1.6513e-02, -1.8166e-02],
          [-8.7677e-02, -7.6899e-02, -6.6525e-02],
          [-8.5937e-02,  3.7163e-04, -1.9602e-02]],

         [[ 2.2613e-02, -2.2600e-02,  3.6446e-02],
          [ 1.4133e-02, -5.0463e-02, -6.7563e-02],
          [ 3.4801e-02, -6.8650e-02, -4.2656e-02]],

         [[ 2.3042e-02, -4.7647e-03,  3.4780e-03],
          [-3.5643e-02, -1.5821e-02, -7.7451e-02],
          [-3.3724e-02, -8.2937e-02,  1.4276e-02]],

         [[ 2.8233e-02,  2.5423e-02,  3.3450e-02],
          [ 2.4793e-02,  4.5544e-02,  2.5011e-02],
          [-8.6975e-02, -2.0499e-03, -5.0429e-02]],

         [[-5.9487e-02,  1.1191e-02, -2.7367e-02],
          [ 4.2354e-03,  3.1369e-02, -3.2345e-02],
          [-5.4779e-02, -1.1219e-02, -2.3229e-02]],

         [[ 4.5114e-02,  9.5715e-03, -6.2437e-02],
          [-5.8635e-02,  6.1233e-02, -4.9370e-03],
          [-1.9212e-02, -4.9515e-02,  1.3336e-02]],

         [[ 1.7364e-02,  1.8411e-03,  1.7481e-03],
          [ 7.3429e-02,  4.3725e-02,  2.5130e-02],
          [-3.8212e-02, -1.7046e-03, -5.3676e-02]],

         [[-7.9976e-02, -1.6153e-02, -1.6638e-03],
          [-8.5452e-02, -8.1530e-02,  3.3711e-03],
          [-6.5251e-02,  3.0087e-03,  1.9086e-03]],

         [[-3.1548e-02, -6.8447e-02, -5.5355e-02],
          [-2.4943e-02, -6.4892e-02,  2.3886e-02],
          [-6.7579e-02, -7.6512e-03, -1.4301e-02]],

         [[-1.9787e-02, -1.9721e-02, -7.0036e-02],
          [ 2.4498e-02, -2.7615e-02, -9.0501e-02],
          [-5.8796e-02, -2.8142e-02, -4.2999e-02]],

         [[-1.8610e-03,  5.4923e-02,  4.5529e-02],
          [ 8.5010e-02, -6.5877e-02, -2.9128e-02],
          [-8.1574e-02, -5.5681e-02, -8.0665e-02]],

         [[ 1.6262e-02, -2.5694e-02, -1.1551e-02],
          [ 2.7197e-02, -4.7878e-02, -3.6942e-02],
          [-2.5974e-02, -2.4617e-02,  2.5128e-02]],

         [[-1.1737e-02,  1.9297e-02,  2.2986e-02],
          [ 2.2510e-02, -1.6255e-02, -6.5916e-02],
          [ 1.2002e-02,  2.5834e-02,  2.7661e-02]],

         [[-7.1067e-03, -1.7547e-02, -5.5451e-02],
          [ 5.1386e-02, -2.8818e-02,  1.9946e-02],
          [-2.1412e-02, -1.9488e-02,  1.6749e-02]],

         [[-4.1502e-02,  4.1574e-02, -1.6599e-02],
          [-2.2210e-02, -3.4967e-02, -3.8205e-02],
          [ 2.5681e-02,  3.4364e-02, -2.7097e-02]],

         [[-2.3303e-02,  4.0546e-02, -2.1643e-02],
          [-5.0812e-02, -5.8615e-02, -1.2338e-02],
          [-4.0367e-02,  5.7563e-02,  7.6431e-02]],

         [[-8.2973e-02, -3.4824e-02, -2.7089e-02],
          [-7.7482e-02,  3.1281e-02, -2.5606e-02],
          [-5.8444e-02,  9.1809e-03, -4.0925e-02]],

         [[-2.2956e-02,  4.9336e-03,  5.6226e-02],
          [-3.3730e-02, -1.6898e-02,  2.5874e-02],
          [-6.6721e-02, -8.4074e-02, -5.1171e-02]],

         [[-9.3566e-03, -1.0967e-02, -6.4578e-02],
          [ 9.4342e-03,  1.6016e-02, -6.2833e-03],
          [ 2.5735e-02, -3.3126e-02, -2.9761e-03]],

         [[-2.3318e-03, -1.6164e-03, -1.6659e-02],
          [ 7.1708e-02,  8.6415e-02, -6.8672e-02],
          [-1.7836e-02,  8.6857e-03, -8.6735e-02]],

         [[-3.8353e-02,  2.4976e-02, -5.1212e-02],
          [ 4.1603e-02, -2.7234e-02,  2.5548e-02],
          [ 3.3329e-03,  2.1289e-03, -1.7872e-02]],

         [[-2.4009e-02,  2.2537e-02, -2.2518e-02],
          [-1.7815e-02, -6.6953e-04, -8.7825e-02],
          [-3.2068e-02, -2.8553e-02, -3.9082e-02]],

         [[-3.9566e-03, -7.2202e-02,  4.7999e-03],
          [ 1.6482e-02, -8.8198e-02, -1.0450e-02],
          [ 1.3780e-02,  1.6931e-02, -4.9865e-02]],

         [[-3.5002e-03, -2.4435e-03,  7.5761e-03],
          [ 3.9028e-02, -5.3514e-02, -2.1997e-02],
          [-8.3110e-03, -7.4346e-02,  2.0861e-02]],

         [[-3.5790e-02, -2.7977e-02,  7.2892e-02],
          [-7.5370e-02,  5.0418e-02, -8.0020e-02],
          [-3.0088e-02, -5.8655e-02, -8.4875e-02]]],


        [[[-5.7930e-02,  4.8176e-02, -2.7908e-02],
          [-4.3236e-02, -5.1644e-02, -7.3402e-04],
          [-1.4582e-02,  2.3053e-02, -2.4194e-02]],

         [[-1.3589e-02, -7.6691e-02,  1.5285e-02],
          [-4.3761e-02, -7.7327e-02, -7.8749e-02],
          [ 1.8828e-02, -5.0608e-02,  1.0546e-02]],

         [[ 5.0203e-02, -5.0283e-02, -1.8171e-02],
          [ 5.8632e-02, -2.4840e-02, -4.2858e-02],
          [ 5.0090e-02, -3.4357e-02, -5.7784e-03]],

         [[ 5.1033e-03,  5.6602e-02,  1.1723e-02],
          [ 5.1136e-02, -4.3679e-03, -2.1033e-02],
          [-2.6084e-02,  4.4243e-02, -3.5953e-02]],

         [[ 2.5623e-02, -3.2479e-02, -7.7193e-02],
          [-7.4108e-02, -6.6627e-02,  8.6601e-03],
          [-1.2093e-02, -8.3662e-02,  1.0554e-02]],

         [[ 4.9617e-02, -2.6490e-03, -5.0022e-02],
          [-2.0609e-02,  3.1411e-03, -2.8953e-02],
          [-7.9000e-03, -1.0929e-02,  2.0810e-03]],

         [[-4.9047e-02,  4.8907e-02,  2.4026e-02],
          [-5.7909e-02, -4.2436e-02, -6.1393e-03],
          [ 4.1599e-02, -1.5670e-02,  1.0345e-02]],

         [[-1.7603e-02, -4.8993e-02, -5.8201e-02],
          [-5.2001e-02, -5.4760e-03,  1.1986e-02],
          [-4.0671e-02,  2.0658e-02,  3.0452e-03]],

         [[ 5.6161e-02,  5.5740e-02,  2.1583e-02],
          [ 2.7354e-02,  4.3489e-03,  2.7479e-02],
          [ 4.4859e-02, -1.3401e-02, -1.7376e-02]],

         [[-1.7881e-02,  2.7625e-02,  3.9700e-02],
          [ 2.6077e-02, -3.2551e-02, -4.0127e-02],
          [-3.0065e-02, -6.9869e-03, -3.9547e-02]],

         [[-2.7363e-02,  2.8537e-02, -2.7518e-02],
          [-1.0721e-02, -8.6810e-02, -8.3377e-02],
          [-4.3870e-02, -3.0631e-02, -7.0612e-02]],

         [[-6.6424e-02,  8.6848e-03, -5.0065e-04],
          [-5.2983e-02,  6.6433e-04,  2.7864e-02],
          [-4.6845e-02, -4.8799e-02, -1.7319e-02]],

         [[ 2.7416e-02, -9.7768e-03, -2.0915e-04],
          [-7.1602e-02,  4.4489e-02, -7.2746e-02],
          [-4.9759e-02, -2.7015e-02, -1.9578e-02]],

         [[-1.4202e-02, -1.5902e-02,  4.8456e-02],
          [-2.2358e-03,  2.4502e-02, -3.9859e-03],
          [ 3.6261e-02,  3.6180e-02,  2.9858e-02]],

         [[-2.7569e-02, -7.8283e-02, -4.0913e-02],
          [-6.4401e-02,  2.4534e-02,  4.4718e-02],
          [-1.2586e-02,  1.6697e-02, -2.7536e-02]],

         [[ 2.0533e-02, -5.6759e-02, -6.9324e-02],
          [-1.4256e-02,  6.9645e-03, -2.1703e-02],
          [-4.3696e-02, -3.7734e-02,  4.8526e-02]],

         [[-1.4302e-02, -3.1707e-02,  2.8577e-02],
          [ 1.9910e-03, -7.5784e-02,  3.4850e-02],
          [-4.0918e-02,  5.7539e-02, -3.0106e-02]],

         [[ 5.5729e-02,  2.0639e-02,  5.6643e-02],
          [-2.1838e-02, -5.3624e-02,  1.9588e-02],
          [-4.3504e-02, -1.2699e-02, -5.2801e-02]],

         [[-3.3330e-02,  1.3365e-02,  1.1287e-02],
          [-5.1104e-02,  4.2634e-02, -2.4702e-02],
          [-6.5302e-02, -8.4233e-02,  2.4219e-02]],

         [[-1.0550e-02, -4.1716e-02, -4.1361e-02],
          [-7.1488e-02, -7.4008e-02, -2.9623e-02],
          [ 1.8606e-02, -2.3748e-02, -8.6802e-02]],

         [[-5.3071e-02,  3.5344e-02, -6.9043e-03],
          [-1.4302e-03, -2.2819e-02, -2.5251e-02],
          [ 2.6360e-02, -6.1330e-02,  4.4699e-02]],

         [[-4.3371e-02,  5.6617e-05, -3.0398e-02],
          [ 1.9397e-02, -5.1126e-02,  1.6997e-02],
          [ 8.4440e-04, -4.0368e-02, -6.6145e-02]],

         [[-4.4105e-04,  5.4357e-03, -1.8464e-02],
          [-5.0768e-02,  9.7393e-04,  4.6625e-02],
          [-5.9811e-03, -3.5107e-02,  3.0030e-03]],

         [[-1.0190e-02,  1.1618e-02, -8.3303e-02],
          [-3.0992e-02, -6.4110e-02,  1.1784e-02],
          [-3.8390e-03, -7.1566e-03, -4.1517e-02]],

         [[-3.5942e-02, -5.6620e-02, -4.5534e-02],
          [-4.4503e-02,  4.1547e-02,  1.1705e-02],
          [-2.9721e-02, -2.5318e-02, -5.1873e-02]],

         [[-7.4230e-02, -2.6542e-02, -2.6219e-02],
          [-2.0050e-02, -2.1366e-02, -2.1239e-02],
          [-8.5489e-02,  2.6312e-02, -5.3515e-02]],

         [[-2.6656e-02, -5.9275e-02, -4.9054e-02],
          [ 2.0567e-03, -5.7130e-02,  7.4736e-03],
          [ 1.1978e-02, -5.1586e-02,  5.7737e-04]],

         [[ 2.4208e-02,  1.6028e-02, -4.0808e-02],
          [-1.2201e-02,  4.9273e-02, -7.5694e-02],
          [-2.4273e-02,  1.9037e-02,  1.8364e-02]],

         [[-5.9016e-02,  1.3667e-02, -7.8075e-03],
          [-6.2943e-02, -8.4045e-03, -7.1325e-02],
          [ 9.4953e-03,  5.5594e-02,  6.6863e-03]],

         [[-3.8979e-02, -5.0602e-02,  6.6352e-03],
          [-1.6463e-02,  7.0402e-03, -7.8447e-02],
          [ 6.3972e-03, -2.6532e-02,  8.9598e-03]],

         [[-4.4888e-02, -6.5605e-02, -2.5938e-02],
          [-7.6345e-02, -8.3784e-02, -8.6487e-02],
          [ 1.1323e-03, -6.5861e-02, -4.0633e-02]],

         [[-7.8306e-03, -3.8651e-02, -4.1137e-02],
          [-1.2203e-02, -3.0380e-02,  7.6598e-03],
          [ 5.1168e-02,  9.2510e-03,  5.2448e-03]]],


        [[[-3.1904e-02, -1.4304e-02,  8.1483e-03],
          [-1.9141e-02, -5.2254e-02, -1.9390e-02],
          [ 9.9949e-03,  6.2801e-02, -1.9330e-03]],

         [[ 4.1381e-03,  2.1275e-02,  8.0999e-03],
          [-8.7308e-02, -1.7956e-02,  5.5276e-03],
          [ 3.9863e-02, -5.8895e-02, -7.0305e-02]],

         [[ 5.4910e-02, -5.9023e-02,  1.4313e-02],
          [ 4.3268e-02,  2.5638e-02, -4.4705e-02],
          [-1.3193e-01,  1.1630e-02,  2.2153e-02]],

         [[ 7.0150e-02, -1.3019e-02,  1.6930e-02],
          [-1.0952e-01, -9.3545e-02,  3.7886e-02],
          [ 2.6309e-02, -4.7484e-03,  2.6900e-02]],

         [[ 6.3982e-02,  7.3919e-03,  5.0667e-02],
          [-2.9665e-03, -8.8375e-03,  7.5960e-02],
          [-2.2365e-02, -3.4827e-02,  8.9642e-02]],

         [[-8.5304e-02, -1.0060e-01,  1.9413e-02],
          [ 2.0103e-02, -6.2393e-03, -1.2554e-02],
          [-3.8622e-03, -4.0823e-02, -2.9706e-02]],

         [[ 5.1303e-02,  2.0725e-02, -1.1961e-01],
          [-8.6088e-02, -7.7253e-03, -6.0198e-02],
          [-3.2039e-02,  1.4909e-02, -6.5220e-02]],

         [[ 3.7414e-02,  8.6599e-02, -3.3024e-03],
          [-1.2673e-01, -3.9138e-02, -1.1319e-01],
          [-6.2183e-02,  2.1215e-02, -5.5429e-02]],

         [[ 5.6410e-02,  3.1908e-02,  2.2568e-02],
          [-2.1387e-02, -2.9953e-02, -1.0624e-01],
          [ 5.6789e-02,  1.5787e-02, -8.6868e-02]],

         [[ 2.4295e-02,  5.8683e-02, -8.0885e-02],
          [-1.2204e-01,  5.3561e-02, -5.7175e-02],
          [-5.8412e-02,  5.6942e-02, -3.7487e-02]],

         [[-2.2012e-02, -6.9750e-02, -8.5943e-02],
          [ 1.6817e-03,  1.0294e-01,  2.5656e-02],
          [-6.4466e-02,  3.6799e-02, -7.3250e-02]],

         [[ 1.8033e-02,  1.0572e-02, -1.7998e-03],
          [-3.1697e-03, -7.1867e-03,  1.5120e-03],
          [-2.0166e-02, -2.7050e-02, -3.6018e-03]],

         [[ 9.6329e-02, -6.5864e-02,  4.1629e-02],
          [-1.3593e-01, -1.1451e-02, -5.0471e-03],
          [-3.3970e-02,  6.4182e-02, -1.5456e-03]],

         [[ 8.2716e-02, -1.3158e-01,  6.8224e-02],
          [ 4.8196e-02,  2.2680e-02, -1.5119e-02],
          [-5.8019e-02, -2.8613e-02,  9.2498e-02]],

         [[ 1.2721e-02,  7.0440e-02,  2.0395e-02],
          [-7.0446e-03, -3.5576e-02,  8.1654e-03],
          [ 7.2589e-04, -7.5195e-02,  1.9003e-02]],

         [[-4.4477e-02,  5.7814e-02,  1.6120e-02],
          [ 5.4240e-02, -9.4455e-03, -1.6499e-02],
          [ 5.5246e-02, -6.7949e-02,  1.8656e-02]],

         [[-9.1043e-02, -1.1239e-01,  2.0580e-02],
          [ 2.5373e-02,  7.8244e-02,  5.5569e-02],
          [ 3.1689e-03, -5.7281e-02,  2.1782e-02]],

         [[-6.5383e-02,  7.9398e-02, -2.8678e-02],
          [-8.4238e-02, -2.8137e-02, -4.3438e-02],
          [-6.3391e-02,  5.9357e-02, -8.0618e-02]],

         [[ 3.2598e-02, -6.4572e-02,  8.2225e-02],
          [-3.3713e-03, -4.3210e-02,  1.5177e-02],
          [-1.0339e-01, -2.2166e-02,  2.0929e-02]],

         [[ 4.6573e-02, -1.0631e-01,  5.6426e-02],
          [ 5.7665e-02, -5.2865e-03,  1.1291e-02],
          [-7.5148e-02,  9.0727e-03, -1.1093e-02]],

         [[ 1.0029e-02, -2.6830e-02, -9.8313e-03],
          [-1.0666e-01,  3.5830e-02, -7.4262e-02],
          [ 4.8039e-02, -1.4744e-02,  3.3528e-02]],

         [[-8.4948e-02,  2.5985e-02,  4.1522e-03],
          [ 1.2597e-01, -1.9702e-02,  5.2571e-02],
          [-3.4546e-02,  3.2457e-02, -5.0864e-02]],

         [[-4.7521e-02, -5.5397e-02, -4.2142e-02],
          [-4.4611e-02,  1.6820e-02, -1.8276e-02],
          [-4.9605e-02,  5.9933e-02,  3.1492e-02]],

         [[-8.7419e-03, -9.4965e-03,  6.3587e-02],
          [-1.3879e-02,  3.6921e-03,  7.6963e-02],
          [-5.2063e-02,  6.5700e-03,  2.2355e-02]],

         [[ 7.4807e-02, -6.4672e-03,  3.5036e-02],
          [-9.0027e-02,  1.5127e-02, -3.1492e-02],
          [-2.6737e-02,  4.2038e-02, -3.2851e-02]],

         [[-7.2604e-02, -3.0706e-02,  2.5567e-02],
          [ 6.2419e-02, -5.9333e-02, -3.9696e-02],
          [ 2.9836e-05,  1.6748e-03, -3.0327e-03]],

         [[-1.1080e-02, -8.4285e-02,  2.1691e-02],
          [-1.8278e-02,  5.5900e-02,  2.5043e-02],
          [-2.9294e-02, -1.0542e-03, -5.7653e-02]],

         [[ 6.6931e-02, -8.7132e-02, -6.3297e-02],
          [ 2.0006e-02, -1.0721e-01,  6.1018e-02],
          [-1.2229e-01,  3.7260e-02, -1.4922e-02]],

         [[-8.2138e-04, -5.9853e-02, -2.5686e-02],
          [ 2.4289e-02,  5.4019e-02, -2.9754e-02],
          [ 3.0846e-02,  1.0070e-02,  8.5186e-02]],

         [[ 1.9801e-03,  1.7098e-03, -4.2646e-02],
          [ 6.9040e-02, -5.0921e-02, -1.2297e-02],
          [ 4.4924e-02,  6.7470e-02,  3.8873e-02]],

         [[-1.5963e-03, -5.2345e-02, -3.3085e-02],
          [-2.4029e-02, -2.5287e-04,  7.7937e-02],
          [ 5.8959e-02, -7.6338e-02,  4.6206e-03]],

         [[-6.5046e-02,  9.2043e-02, -3.4641e-02],
          [-4.3881e-02,  3.9298e-02, -3.3934e-02],
          [ 9.3945e-02,  8.8859e-03,  6.4680e-02]]],


        [[[-1.8808e-02, -4.8734e-02,  1.6733e-02],
          [ 1.2209e-02, -6.2384e-02, -5.6387e-02],
          [ 6.2562e-02,  1.9435e-03, -7.7625e-02]],

         [[ 1.4974e-02, -2.0122e-02, -1.2632e-02],
          [-6.5250e-02,  8.1128e-03,  7.7262e-02],
          [-5.2713e-02,  2.6419e-03,  6.9192e-02]],

         [[-1.0073e-02, -5.2160e-02, -2.2612e-02],
          [-2.8877e-02, -2.5374e-02,  6.0694e-02],
          [-1.1530e-02,  2.9302e-02,  1.9268e-02]],

         [[-1.0834e-02,  1.2374e-02, -4.1216e-02],
          [ 2.8965e-02, -2.0784e-02, -7.6436e-02],
          [-6.7558e-02, -2.1952e-02, -8.6729e-02]],

         [[ 4.2804e-02, -6.7168e-02, -4.6259e-02],
          [-1.4480e-02,  2.5832e-02, -2.6061e-02],
          [ 1.0635e-02,  9.1807e-02, -2.1915e-02]],

         [[ 6.0323e-02, -1.0527e-02, -4.4702e-02],
          [-1.1383e-02,  7.3589e-02, -9.8264e-04],
          [-3.2323e-02,  6.6906e-02, -9.8667e-02]],

         [[ 2.9566e-02, -3.3259e-02,  1.6759e-02],
          [-4.6668e-03, -2.8512e-02,  7.2978e-03],
          [ 1.2791e-01, -4.4500e-02,  1.0095e-01]],

         [[ 1.0294e-02, -1.4673e-02,  1.7179e-02],
          [-8.3241e-02, -4.0173e-02, -1.9671e-02],
          [ 7.0007e-02,  7.1409e-02,  8.8337e-02]],

         [[-5.5790e-02, -6.4462e-02,  4.4680e-02],
          [-2.2091e-02, -1.0592e-01,  6.9362e-02],
          [ 1.0002e-02, -9.5089e-02, -5.6482e-04]],

         [[ 1.0783e-02, -4.2871e-02, -3.1905e-03],
          [ 2.0460e-03,  1.8941e-02,  8.1417e-02],
          [ 7.9498e-02, -6.9584e-02,  8.5232e-03]],

         [[-1.5226e-02, -1.1717e-01, -1.1920e-03],
          [ 4.4471e-02,  6.0563e-02, -6.5729e-02],
          [ 7.2405e-02, -6.1987e-02, -9.8620e-02]],

         [[-2.4605e-02,  2.8373e-02, -5.8915e-02],
          [ 1.7632e-02, -4.5909e-02, -6.6154e-03],
          [-7.1390e-02,  1.5375e-02, -4.4572e-03]],

         [[ 8.8271e-03, -2.1671e-02, -6.1064e-02],
          [-3.2813e-02,  2.6656e-02, -7.1721e-02],
          [ 1.3787e-01, -2.2669e-02, -2.0459e-02]],

         [[-1.1427e-02, -2.6867e-02,  1.1153e-02],
          [ 1.0600e-01,  3.4818e-02, -9.1634e-02],
          [-6.2272e-02, -4.2155e-02, -7.2528e-02]],

         [[-5.2796e-02, -2.0337e-02, -1.7973e-03],
          [ 4.3414e-02, -4.4805e-02, -1.4433e-02],
          [-2.9658e-02, -3.4281e-02, -3.2424e-02]],

         [[-4.5876e-02, -4.0066e-02,  4.2781e-02],
          [ 4.2807e-02, -3.0325e-02,  8.3384e-02],
          [-1.8794e-02, -6.4600e-02,  4.4947e-02]],

         [[-4.5203e-03,  5.6601e-03, -2.2447e-02],
          [ 1.2363e-02,  1.6981e-02,  1.4329e-02],
          [-3.5627e-02, -9.3840e-02, -4.7331e-02]],

         [[-1.2669e-02, -3.8679e-02,  6.6797e-02],
          [ 9.9628e-02,  5.5267e-02,  1.3283e-02],
          [ 5.0209e-02,  2.6085e-02, -2.5908e-02]],

         [[ 3.7303e-03,  4.2043e-02, -6.7640e-02],
          [-4.1165e-02,  1.3925e-02, -6.8411e-02],
          [ 9.0876e-02, -3.3463e-02, -6.7529e-02]],

         [[ 4.7201e-02, -6.0218e-02,  9.5648e-03],
          [ 1.5895e-02,  9.2743e-03, -6.8265e-02],
          [ 9.0896e-02, -6.0605e-02, -8.1432e-02]],

         [[-4.5950e-02, -3.4331e-02,  7.3245e-03],
          [-9.3310e-03,  3.6019e-02,  5.1392e-03],
          [ 3.7045e-02, -1.2634e-02, -1.3854e-02]],

         [[ 1.4539e-02,  3.0091e-02, -1.0159e-01],
          [-6.0688e-02,  4.1597e-03, -9.5989e-02],
          [-4.5356e-02,  5.5655e-02, -1.1642e-01]],

         [[-3.5384e-02, -3.4731e-02,  2.8411e-02],
          [-5.2476e-03, -2.8505e-02, -5.4158e-02],
          [ 3.4544e-03,  2.9000e-02, -4.3451e-02]],

         [[ 1.9551e-03, -4.9986e-02, -6.5673e-02],
          [ 6.1827e-02,  4.6413e-02,  2.2151e-03],
          [ 9.1321e-02, -2.4059e-02,  4.0058e-03]],

         [[-7.7803e-02, -5.1819e-02,  2.7649e-02],
          [ 6.0907e-02,  1.0241e-01,  9.7044e-02],
          [ 3.0948e-02,  3.8997e-02,  8.5453e-02]],

         [[ 1.5078e-02,  8.1033e-03, -1.9947e-02],
          [-3.9845e-02, -2.2419e-02, -2.8758e-02],
          [-1.9824e-02,  1.1281e-01, -2.5467e-02]],

         [[ 2.6832e-02,  4.5588e-02, -6.2089e-03],
          [ 6.7351e-02,  9.6863e-02, -4.5054e-02],
          [-1.7776e-02,  4.0529e-02, -3.9504e-02]],

         [[ 2.5980e-02, -9.3019e-02, -5.2008e-02],
          [ 2.2269e-02,  6.3368e-02, -3.2912e-02],
          [ 1.0088e-01,  5.8660e-02, -1.2977e-01]],

         [[-4.5667e-02,  3.1203e-02, -1.6999e-02],
          [ 1.0003e-01,  6.5876e-02, -4.2521e-02],
          [-3.6639e-03, -8.5530e-02, -8.1432e-02]],

         [[ 2.4495e-02, -3.3746e-02, -4.2674e-02],
          [-5.2845e-02,  1.5404e-02,  3.4436e-03],
          [-4.4753e-02,  7.1349e-02, -8.3920e-02]],

         [[-1.9000e-02, -5.8659e-02, -1.7737e-02],
          [ 1.6085e-02, -2.7655e-02,  2.3486e-02],
          [-2.0745e-02, -2.5483e-03, -5.8541e-02]],

         [[-2.5160e-02,  3.7186e-02,  6.4180e-02],
          [ 1.1632e-01, -7.2946e-02,  4.2431e-02],
          [-2.6553e-02, -7.7651e-02,  5.0349e-02]]]], device='cuda:0')), ('conv2.bias', tensor([-0.0548,  0.0196, -0.0322, -0.0775,  0.0576, -0.0373, -0.0099,  0.0369,
        -0.0385, -0.0120, -0.0620, -0.0385,  0.0296,  0.0331,  0.0338, -0.0417,
         0.0523,  0.0141,  0.0310,  0.0429, -0.0435, -0.0716, -0.0536, -0.0146,
        -0.0405,  0.0554, -0.0808,  0.0264, -0.0454, -0.0748, -0.0119,  0.0496],
       device='cuda:0')), ('fc.weight', tensor([[ 5.4026e-02,  3.1403e-03, -5.3809e-02,  2.4126e-02,  3.0539e-02,
          4.2633e-02,  8.9041e-02, -4.8867e-02,  4.7139e-02, -3.7122e-02,
          6.9239e-02,  4.3202e-02, -9.3867e-03,  1.0026e-01, -9.1080e-03,
          3.2875e-03,  2.7816e-02, -2.3257e-03,  5.1458e-02,  4.2246e-02,
          5.2280e-04,  1.7980e-02,  4.9152e-02,  3.8775e-02, -2.9215e-02,
          8.3815e-02,  4.6853e-03, -3.8594e-02, -1.4103e-02,  3.7990e-02,
          4.0448e-02,  8.3567e-02, -3.8838e-02, -5.0483e-02, -1.5534e-02,
          4.7335e-02, -7.4426e-02, -4.3658e-02, -3.0519e-02,  1.0499e-01,
         -3.9783e-02,  7.6528e-02,  7.3613e-02,  8.4944e-02, -1.5824e-02,
          8.1519e-02, -7.1046e-02,  4.9284e-02,  6.0206e-02,  9.2178e-02,
          4.0935e-03,  5.5975e-02, -7.3192e-02, -7.3804e-02, -3.9559e-02,
          6.1554e-02,  1.4250e-02,  5.1798e-02,  7.1279e-02,  5.4314e-02,
          4.9544e-02,  4.8055e-02,  8.3890e-02, -5.4031e-02, -3.8403e-02,
          6.2532e-02,  5.0117e-02, -3.5621e-02,  4.9786e-02,  4.4571e-02,
          6.2793e-02,  1.8673e-02,  1.3502e-02,  2.4700e-02,  2.1075e-02,
         -6.8683e-02, -4.5099e-02, -7.4889e-02,  3.1731e-02, -3.9552e-02,
          1.0200e-01, -3.2861e-02,  2.6134e-02, -1.0005e-02, -6.3622e-02,
          9.9044e-02,  2.0692e-02, -2.6910e-02,  1.0934e-01, -6.0330e-02,
         -8.7675e-02,  3.1660e-02, -5.4084e-02,  4.0483e-02,  1.6979e-02,
         -4.0555e-03,  1.6724e-02,  7.4123e-02,  5.9319e-02,  3.8432e-02,
         -2.8027e-03,  4.1495e-02,  5.0419e-03, -1.2101e-02, -6.4195e-03,
          5.0251e-02,  1.2649e-02, -1.6853e-02,  7.4408e-02, -3.7693e-02,
          4.1295e-02,  4.0783e-02, -2.5536e-02,  9.3737e-02,  3.9529e-03,
         -5.2054e-02, -4.0677e-02,  1.6725e-02,  8.3265e-02,  7.5279e-02,
         -1.9169e-02, -5.4399e-02,  7.2619e-02,  4.2125e-02, -6.1924e-02,
          5.9244e-03, -4.7670e-02, -3.0795e-02,  4.6131e-02,  1.4249e-01,
          1.3568e-02, -3.2722e-02,  2.4329e-03, -2.5487e-02, -4.8417e-02,
         -3.4043e-02,  1.0132e-01, -3.7404e-02, -2.9741e-02,  5.3960e-02,
          8.3763e-02,  7.3205e-02, -3.5262e-02,  8.2527e-02, -2.3682e-02,
         -4.3008e-02,  1.0727e-01,  4.1387e-02, -8.4961e-02, -1.9377e-03,
          1.5576e-02, -2.0926e-02,  2.4424e-02, -3.1104e-02,  2.3205e-02,
         -5.2360e-02, -1.9120e-02, -6.8424e-02, -4.6720e-03,  6.5879e-02,
         -5.9932e-02,  6.2533e-02, -1.1333e-02,  2.8445e-02,  9.0864e-02,
          6.7306e-02, -1.4613e-02,  3.8945e-02,  4.4041e-02,  4.8660e-02,
          5.1358e-02,  5.0609e-02,  5.6388e-02,  1.8237e-02,  4.0963e-02,
         -5.4595e-02, -2.6304e-02, -6.8317e-03, -3.4039e-02, -7.1239e-03,
          1.6425e-03, -1.7021e-02, -1.4903e-02, -8.9420e-02,  1.2761e-02,
         -5.7263e-02,  1.4745e-02,  5.4946e-04,  3.4825e-02,  7.2933e-02,
         -1.5484e-02,  5.5180e-02],
        [ 1.0109e-02,  6.3152e-02,  4.3306e-02, -4.4738e-02, -1.7241e-02,
         -1.7560e-03, -3.0475e-02,  9.9483e-06, -2.5467e-02,  9.6337e-02,
         -2.0817e-02, -1.8724e-02,  8.2268e-02,  2.3212e-02,  4.6813e-02,
         -4.3806e-02,  5.6304e-02, -9.6932e-03,  3.3649e-02, -2.0085e-02,
         -5.8077e-02,  8.6017e-02, -3.9621e-02,  7.0848e-02,  2.5865e-02,
          5.8321e-02,  3.9080e-02,  6.0350e-02, -1.9468e-02,  3.6053e-02,
         -9.3388e-02,  6.5612e-02, -5.8904e-02,  6.8540e-02, -3.1279e-02,
          7.5496e-02,  2.7620e-02,  3.4854e-02, -4.5213e-02, -5.8203e-02,
         -6.6162e-02, -4.5479e-02,  5.9615e-02,  2.7130e-02, -1.3562e-02,
          4.0463e-02,  2.7219e-02,  3.0322e-03,  6.7759e-02, -7.2300e-02,
         -5.7150e-02, -3.0247e-03,  4.9040e-02,  7.2349e-02,  2.3911e-02,
         -5.5267e-02,  6.3795e-02,  9.2271e-02,  1.5379e-02, -4.9704e-05,
         -2.4818e-02,  3.1596e-02,  4.7768e-02,  4.9548e-02, -5.9752e-02,
         -1.4897e-02,  1.0653e-01,  4.0388e-02,  8.7895e-03, -4.0164e-02,
         -1.7431e-02,  9.2707e-02,  3.6624e-02,  7.3610e-02,  6.3759e-02,
          1.1729e-01,  7.3609e-02,  1.2531e-01, -6.9852e-02, -5.8727e-02,
          4.7495e-02, -1.0082e-02, -1.3436e-02,  3.3471e-02,  1.6814e-02,
          6.1808e-02,  6.7116e-02,  6.4172e-02,  5.9138e-04,  1.1601e-02,
         -3.7056e-02,  2.9528e-02, -1.2345e-03, -4.3004e-02,  4.8342e-02,
         -8.9162e-02, -3.2726e-02, -4.0800e-02,  4.9221e-02,  5.1812e-02,
         -4.3056e-03,  8.6187e-02, -1.1721e-02, -2.4822e-02,  3.0909e-02,
          1.0932e-01,  6.4083e-02, -1.8679e-03,  4.2246e-02,  6.4968e-02,
         -1.7247e-02,  5.1570e-02,  6.9297e-02,  6.9497e-02,  6.0576e-02,
         -1.4689e-02,  6.8065e-02, -5.8818e-02,  1.1107e-02,  1.8043e-02,
          3.2126e-02,  9.9791e-02,  7.3905e-02, -1.8704e-02,  1.2661e-02,
         -3.9513e-02,  6.1604e-02, -1.7987e-02,  4.7212e-02,  2.2305e-02,
         -8.6950e-02,  6.6703e-02, -1.1136e-01, -3.4880e-02, -4.7141e-02,
          3.4517e-03,  1.3867e-03,  4.8597e-02, -6.0873e-03,  1.2656e-02,
         -5.0327e-02, -5.2328e-02,  1.1438e-02,  3.9143e-03, -1.0737e-01,
          5.3070e-02, -5.7971e-03, -4.2916e-02,  1.9708e-03, -7.4293e-02,
         -9.6771e-03,  4.7224e-02, -2.8321e-02, -3.3671e-02,  3.9006e-02,
         -1.7206e-02, -6.3243e-02, -2.3560e-02, -3.5384e-03, -5.9278e-02,
         -5.9236e-04,  4.3073e-02,  4.5446e-02, -3.7669e-02,  6.4659e-03,
          1.6658e-02,  6.4389e-02, -1.1867e-02,  2.2648e-02,  3.1613e-03,
          2.6911e-02, -1.6023e-02, -3.5146e-02,  5.1269e-02, -6.1596e-02,
         -6.5195e-02,  9.0998e-03, -4.1660e-02,  7.2097e-02,  5.2385e-02,
         -4.8151e-02,  5.9477e-02,  2.9206e-02,  6.2441e-02, -1.3751e-02,
         -4.6879e-02, -4.3829e-02,  1.7319e-03, -3.3986e-02, -3.7686e-02,
          1.2157e-02, -2.1216e-02],
        [-6.8984e-03, -3.5131e-02, -6.3554e-02, -3.8263e-02,  5.6792e-02,
          4.3286e-02,  4.5946e-02,  1.8812e-02,  4.2727e-02, -7.2229e-02,
          3.8461e-02,  5.5467e-03,  2.1596e-02, -1.3432e-02, -4.6424e-02,
          1.1413e-01, -3.4010e-02, -1.7387e-02,  5.9599e-02, -4.4392e-02,
         -5.7123e-02, -4.0339e-02,  6.3913e-03,  5.6101e-02,  7.6854e-02,
          7.4342e-02,  7.1920e-02, -8.2503e-03,  1.8547e-02,  2.6645e-02,
          7.1036e-02,  4.4965e-02, -2.1984e-02,  1.0995e-02,  1.7373e-02,
          6.1950e-02, -9.7641e-02,  2.4558e-02,  3.1890e-02,  2.2209e-02,
         -3.6446e-03,  8.1071e-02, -2.1634e-02, -1.2849e-02, -7.2610e-02,
          1.3898e-02, -5.1338e-02,  8.6893e-02,  7.1861e-02,  1.7225e-02,
          2.5489e-02,  2.7865e-02, -5.2724e-02, -6.8968e-02,  2.5235e-03,
          4.4704e-02,  3.9448e-03, -1.0852e-02,  4.3233e-02,  3.4301e-02,
         -5.9560e-03, -2.5114e-02,  2.5387e-02,  3.8513e-02, -5.9134e-04,
          9.0451e-02,  1.7417e-02, -7.6848e-03,  1.1362e-02,  1.1571e-01,
          1.0277e-01,  1.1617e-02,  3.1492e-02,  9.3686e-02,  8.2760e-02,
          1.6591e-03,  6.2269e-03, -5.0564e-02,  4.6948e-02,  8.4596e-02,
          5.0089e-02, -6.6585e-03, -1.4649e-02, -4.3780e-02,  2.7358e-02,
         -1.3135e-02, -3.9333e-02, -2.5896e-02,  7.3818e-02, -9.5130e-02,
          2.6678e-02, -3.1939e-02, -4.2029e-02, -3.7548e-02,  3.3346e-02,
          3.7009e-02,  1.3342e-02, -1.7867e-02,  9.1164e-02,  6.2168e-02,
         -1.2053e-01, -6.4877e-02,  1.1025e-02,  9.3683e-02,  1.0882e-01,
          8.2370e-02, -3.4579e-02,  6.4806e-02,  8.9325e-02,  4.8106e-02,
         -3.2581e-02,  5.2989e-02, -3.3789e-02,  8.6484e-02, -4.7231e-02,
         -3.9153e-02, -9.2289e-02, -1.8296e-02,  2.5849e-02,  1.0212e-01,
         -6.5411e-02,  1.2540e-02, -3.9056e-02,  8.0752e-02,  4.3723e-02,
          7.6961e-02, -6.6171e-02, -5.7304e-02, -2.4677e-02,  2.0065e-02,
          1.1989e-01,  9.8195e-03, -3.3060e-02, -3.8728e-02, -5.6455e-02,
         -3.2124e-02,  7.7071e-02,  2.1863e-02,  2.8780e-02, -4.1425e-04,
          8.0760e-02,  4.4689e-03,  2.4248e-02,  4.7402e-03, -1.3339e-02,
          1.6118e-02,  5.3066e-02, -8.0820e-02, -9.4576e-02,  3.2125e-02,
         -2.3988e-02,  5.7252e-02,  7.2327e-02, -7.0325e-02,  1.4812e-02,
         -9.9875e-02,  6.3882e-02,  2.2089e-02,  2.6168e-02,  2.6629e-03,
          3.2343e-02, -8.6226e-03, -4.8292e-02, -6.2392e-02,  1.0730e-02,
         -3.8697e-02,  2.7339e-02,  5.0036e-02,  1.9516e-02,  8.2698e-02,
         -3.3436e-02,  4.9210e-02,  7.0761e-02,  3.7667e-02, -3.7593e-03,
         -4.2942e-02,  3.1738e-02, -9.9393e-02,  2.3531e-02, -1.7241e-02,
          3.1567e-02, -6.0558e-03,  5.7203e-02, -5.9420e-02,  5.1421e-02,
         -1.4243e-02,  1.1541e-01, -1.0860e-02,  6.6238e-02,  6.8321e-03,
         -7.9180e-04, -3.0398e-02],
        [ 6.5635e-02,  5.6266e-03, -5.3314e-02, -9.6478e-03,  5.1263e-02,
          6.5050e-02, -3.7242e-02,  2.0053e-02,  6.9022e-02,  3.4195e-02,
          3.5744e-02,  9.1604e-03, -1.1837e-02,  8.6174e-02,  4.8052e-02,
          1.9819e-02, -7.9534e-02,  1.8263e-02,  5.3680e-02,  3.7789e-02,
         -1.2224e-02, -1.1120e-02, -4.1292e-02, -2.5088e-02, -3.8533e-02,
         -1.7320e-02, -3.9681e-02, -5.0638e-02, -2.2577e-02, -7.8236e-02,
          3.8865e-02,  6.2272e-02, -8.3874e-02, -1.0437e-01,  5.2611e-02,
         -5.3658e-02, -5.0959e-02, -4.3686e-02, -3.6008e-02, -3.8880e-02,
          1.3148e-02, -9.6643e-03, -6.6601e-02, -3.9513e-02, -8.0310e-02,
          6.7505e-02,  5.0222e-02,  1.4939e-02, -8.6012e-02,  1.0116e-02,
          5.3228e-02, -1.0104e-01, -6.4841e-02, -9.6194e-02, -8.1445e-02,
         -4.5584e-02, -3.4879e-02, -6.1511e-02, -1.7484e-02, -5.4505e-03,
         -1.0255e-02,  2.3325e-02, -2.1361e-02,  2.9584e-02, -2.6009e-02,
          5.6391e-02, -1.0821e-02,  2.0572e-03,  4.1913e-02, -3.1669e-02,
          2.6822e-02, -1.8285e-02,  2.6580e-02, -1.0471e-01,  2.9161e-02,
         -6.2446e-03, -7.9951e-03,  8.3345e-02,  7.0145e-02, -2.7631e-02,
          1.1394e-02, -3.0911e-02, -5.4418e-02,  4.7067e-02, -3.1577e-02,
          4.5693e-02, -8.1974e-02,  2.5477e-02, -1.3528e-02,  1.6030e-02,
         -6.6659e-02,  2.6867e-02, -3.5967e-03, -1.8633e-02, -2.9634e-02,
          8.9745e-02,  2.3747e-02,  7.3262e-02,  9.0755e-02, -2.4619e-02,
         -3.6800e-02,  9.4013e-03, -3.2111e-03,  5.6247e-02, -1.0615e-03,
          8.9355e-02, -5.1565e-03, -6.0094e-02,  5.6247e-02,  6.7690e-02,
          2.9918e-02, -7.1979e-03, -3.8357e-02, -7.8717e-03,  1.0965e-02,
         -5.3684e-02, -2.7065e-02,  6.6421e-02,  5.6168e-02, -1.2345e-02,
         -5.8591e-02, -6.3471e-02,  8.9377e-02, -6.1320e-04, -8.2338e-02,
          4.4253e-02,  4.1059e-02,  4.4249e-02,  1.1221e-02,  4.2728e-02,
          8.2454e-02, -1.4560e-02,  8.5563e-02, -3.5792e-02, -5.9019e-02,
         -2.8413e-02,  9.1618e-02,  4.0143e-02,  5.8327e-02,  3.3196e-02,
         -2.6121e-02, -2.9900e-02,  1.9359e-02,  4.6792e-02,  9.0265e-02,
          5.2878e-02,  8.7372e-04, -6.8438e-02, -1.0053e-01, -6.9143e-02,
          8.6749e-03, -3.2370e-02, -4.9530e-02,  4.1659e-02,  5.3108e-02,
         -1.8450e-02, -4.0728e-02, -6.3671e-02, -3.1268e-02, -3.8988e-03,
         -5.0247e-02,  5.1417e-02,  3.2236e-02, -1.0439e-01,  3.9446e-02,
          2.0248e-02, -8.3731e-03,  9.3713e-02,  2.0531e-02,  7.9422e-03,
         -2.7382e-02,  4.9369e-02,  4.8397e-02,  5.7476e-03, -2.6579e-02,
          6.0657e-02, -6.8608e-03,  3.9019e-03, -3.2248e-02,  2.5444e-02,
         -5.8060e-02, -3.3333e-02,  2.9926e-02,  4.2433e-02, -1.9320e-02,
         -4.4910e-02,  3.8771e-02, -1.6637e-02,  4.1741e-02,  1.4387e-02,
         -3.5257e-02, -8.4497e-02],
        [ 5.1765e-02, -1.4824e-03, -2.0409e-02,  1.0220e-02,  6.7971e-02,
         -5.2345e-02,  1.2262e-03, -4.0664e-03, -4.4968e-02,  8.8541e-02,
          6.5848e-02, -7.0107e-02,  7.2042e-02, -8.5757e-02,  1.2031e-01,
         -2.6670e-02,  4.2679e-02, -4.1599e-02,  4.9467e-02,  3.3990e-02,
          5.3971e-02,  1.1530e-02, -2.0133e-02, -3.7712e-02, -1.7113e-02,
          5.4303e-02,  2.4281e-02,  2.9527e-02,  4.7272e-02,  5.7349e-02,
         -1.1780e-02,  2.6147e-02,  9.3829e-03, -1.7065e-02, -5.6165e-02,
          6.7406e-02,  7.3592e-02,  9.9639e-03, -4.7237e-03,  2.2021e-02,
         -4.3261e-02, -7.4352e-02,  3.7769e-02, -4.6071e-02,  3.8025e-02,
         -9.0774e-02,  9.0548e-02, -1.1448e-01,  5.0797e-02,  1.9898e-02,
         -3.8306e-02,  5.7305e-02, -2.9092e-02, -3.8559e-02, -7.1612e-03,
         -3.9377e-02,  1.8163e-02,  6.6454e-02,  1.7483e-02,  2.2682e-02,
          1.2491e-02, -4.5211e-02, -1.2915e-02, -1.1526e-02,  4.4671e-03,
         -8.6829e-02,  2.9060e-02,  3.2888e-02,  8.4240e-02, -7.1922e-02,
          1.2186e-02,  9.0580e-02, -1.3549e-02, -2.9718e-02,  1.5306e-02,
         -5.4939e-03,  2.8966e-03,  8.3737e-02, -8.0965e-03, -6.0838e-02,
         -1.0183e-01,  4.5211e-02, -7.6191e-02, -2.9341e-02,  2.7691e-02,
         -6.2530e-02,  6.5876e-02, -5.7416e-02, -1.1850e-02, -2.7899e-02,
          9.5216e-02, -1.4778e-02,  5.7959e-02,  1.6867e-02,  2.0336e-03,
         -9.3186e-02, -3.2179e-02, -2.9568e-02, -2.8904e-02, -3.6565e-03,
          7.6853e-02,  1.0176e-01,  1.1604e-02, -6.6615e-03, -5.2813e-02,
          7.0142e-03,  5.9092e-02,  2.9269e-02,  3.3988e-02, -7.2121e-02,
         -9.0500e-03,  9.1677e-03, -7.1558e-02, -1.1879e-03,  6.6016e-02,
         -9.4562e-03,  9.9834e-02,  1.9515e-02,  1.1677e-02, -6.5394e-02,
          5.3722e-02,  9.0783e-02,  6.0523e-02, -6.6959e-04,  6.5660e-03,
         -7.7246e-02,  9.1359e-02,  4.1848e-02, -7.5508e-02,  3.1501e-04,
          7.2752e-03,  2.0063e-02,  3.0305e-02,  4.3872e-02, -3.1977e-02,
          1.0646e-01, -6.0205e-02,  1.2558e-02,  2.0447e-02,  9.0461e-02,
         -8.8755e-02,  6.4663e-02,  5.9601e-02,  1.3311e-02, -6.9331e-02,
          7.6306e-02, -1.9522e-02,  2.6472e-02,  9.1223e-02, -6.5481e-02,
         -1.4179e-02, -2.9954e-02,  4.2849e-02, -4.2837e-02, -9.7255e-02,
         -4.3594e-03,  3.8086e-02,  2.4892e-02, -3.3874e-02, -2.8124e-02,
          5.1336e-02, -2.5028e-02, -3.7369e-02,  7.9693e-02, -7.9229e-02,
          4.6501e-02,  1.0451e-02, -1.0194e-01,  6.3144e-02,  8.0509e-02,
          2.0826e-02,  1.6634e-02, -8.7882e-02,  8.7109e-02, -5.4543e-02,
         -2.5544e-02, -7.0696e-02,  4.6710e-03, -6.9096e-02, -2.1274e-03,
         -2.7430e-02,  2.3728e-02,  5.8286e-02,  1.3567e-02, -3.1710e-02,
          5.3883e-02, -1.6038e-02,  6.7481e-02,  5.4865e-02,  5.5525e-02,
         -2.4345e-02,  2.1003e-02],
        [-8.9992e-03, -4.1979e-02, -4.0862e-02, -4.0362e-02,  6.1632e-02,
         -6.5799e-02, -4.5862e-02, -5.1976e-02, -6.3511e-02, -9.1019e-03,
         -2.1819e-02,  6.1445e-02,  5.0934e-02, -2.7459e-02, -5.4653e-02,
          7.7255e-02,  6.0006e-03,  8.2685e-02, -6.6427e-02, -5.0492e-02,
         -5.1535e-02, -3.0607e-02,  1.1687e-02,  3.5112e-02, -4.0206e-02,
         -2.3035e-03,  3.6068e-02, -4.7595e-02,  3.1723e-02, -3.0856e-02,
          8.2138e-02,  4.7759e-02,  3.2383e-02, -6.7602e-02, -2.4918e-02,
          1.4097e-02, -2.0049e-02,  5.3669e-02,  6.9638e-02,  9.9445e-03,
         -4.2955e-02,  1.0390e-01, -5.0604e-03, -6.6635e-03, -3.8112e-02,
         -2.2579e-02, -4.6397e-02,  4.8920e-03, -1.3674e-02, -4.7359e-02,
          7.9470e-02, -7.2789e-03,  2.8813e-02, -9.6950e-03,  2.8775e-02,
          3.3216e-02, -5.3198e-02, -1.8118e-02,  6.4202e-02,  9.5429e-02,
         -2.0797e-02,  5.3243e-03,  4.1398e-02, -3.7375e-02,  6.3713e-03,
         -2.3575e-02, -6.2651e-02, -7.9143e-02, -8.6012e-02,  1.2236e-01,
          2.0152e-02,  9.6377e-02,  4.9521e-02,  9.9182e-02,  6.9620e-02,
          5.7549e-02,  1.9927e-02,  9.7703e-04,  6.6240e-02,  3.0387e-02,
          2.5515e-02,  8.6866e-02,  5.7276e-02,  6.7096e-02, -3.6100e-02,
          3.0769e-02,  5.8550e-02,  5.4478e-02,  6.3656e-02, -2.8264e-02,
         -6.1718e-02,  3.9021e-02, -2.4929e-02, -6.8645e-02, -3.3768e-02,
          4.2687e-02,  5.0738e-02,  1.5885e-02,  5.5299e-02,  4.9578e-02,
         -2.7300e-02, -4.6476e-02,  7.3187e-02,  7.5466e-02,  5.3351e-03,
         -2.5192e-03,  6.9479e-03,  1.2002e-02,  4.9821e-02, -6.8858e-03,
          5.2866e-02,  7.0080e-02,  6.3361e-02,  1.0293e-01,  3.6400e-02,
          7.1813e-02,  1.6740e-02, -2.2109e-02,  5.7871e-02,  1.8624e-02,
         -1.1408e-01, -4.8653e-02, -6.3966e-02,  6.6971e-02,  2.3522e-02,
          3.6725e-02, -9.3589e-02, -4.5342e-02,  4.5667e-02,  3.2248e-02,
          6.8980e-02, -6.8861e-02,  5.7774e-02, -3.1284e-02, -4.5664e-03,
          6.1888e-03,  7.4410e-02, -1.3978e-02, -5.6301e-02,  7.6496e-02,
          8.8863e-03,  5.3719e-02, -1.5797e-02,  1.1069e-01, -1.7270e-03,
          3.2455e-02,  9.2547e-02,  1.2110e-02,  2.6637e-02,  4.5591e-02,
         -5.0564e-02, -1.9507e-02,  9.2028e-02,  9.4367e-03,  6.4930e-02,
         -1.6748e-02,  1.0573e-02, -1.4908e-02,  1.7185e-02,  3.5739e-02,
          2.6830e-02, -2.3673e-02,  4.4834e-02, -4.5252e-02,  5.5248e-02,
          7.7210e-02, -3.3355e-03,  8.8839e-02, -3.1376e-02, -3.3602e-02,
         -3.0714e-02,  2.7091e-05, -7.3469e-03, -1.1136e-02, -2.4220e-02,
         -6.4799e-02,  2.9142e-02,  8.3484e-03,  6.6706e-02,  2.1152e-02,
          1.1336e-02,  3.0117e-03,  5.3674e-02, -1.8094e-02, -7.4229e-03,
          4.5262e-02,  1.0156e-01,  1.1338e-02,  4.8315e-02,  3.9922e-02,
          4.0514e-02, -7.9000e-03],
        [ 6.6198e-02, -3.4495e-02, -6.1251e-02,  3.1146e-03,  5.7562e-02,
         -3.2095e-02, -3.6351e-02,  3.0113e-02, -5.4612e-02,  2.7200e-02,
         -7.0657e-02, -8.4677e-03,  8.5674e-02, -1.0157e-02,  7.8014e-02,
         -4.9701e-02,  1.4159e-02,  2.6492e-02,  4.7628e-02,  5.3564e-02,
          3.2289e-02, -8.4314e-02,  1.3456e-02,  2.9202e-02, -9.3279e-02,
         -9.0226e-02, -6.9133e-02, -8.0844e-02,  1.3220e-02, -1.8800e-02,
         -6.0283e-02, -7.0511e-02,  3.6147e-02, -5.4507e-02, -2.7659e-02,
          9.6916e-03,  1.1381e-01, -2.8959e-02, -4.1116e-02,  7.4910e-03,
         -5.2529e-02, -8.1498e-02, -7.9290e-03,  1.7312e-03, -3.4517e-02,
         -1.4966e-02, -1.9235e-02, -8.5105e-02, -1.0753e-01,  7.0521e-02,
         -7.5181e-02,  4.3126e-02, -2.6353e-02, -6.4616e-02,  6.9000e-02,
          2.4682e-02, -9.5061e-02,  5.1189e-02,  6.8813e-02,  1.9017e-02,
          1.4647e-02,  2.2383e-02,  9.2868e-03, -2.5821e-02,  2.4973e-02,
         -7.1640e-02, -3.5017e-02, -5.1971e-03,  1.1783e-01, -9.1223e-02,
         -1.0800e-01,  2.2130e-02, -4.0547e-02, -7.2566e-02, -1.0185e-01,
         -2.3070e-02, -4.8198e-02,  1.0961e-02,  4.9850e-02,  4.0770e-02,
         -2.3893e-02,  2.4508e-02,  2.6632e-02, -8.8189e-02,  9.3218e-03,
         -1.0779e-01,  3.4161e-02,  2.5500e-03, -1.6710e-02,  7.3481e-02,
         -2.0878e-02,  2.1209e-02,  2.0759e-02, -8.9402e-02,  7.6803e-03,
          3.3823e-02,  5.6476e-03, -2.6087e-02,  1.5614e-02, -4.4271e-02,
          4.1415e-02,  1.3821e-02, -1.0103e-01, -9.8954e-02,  1.3217e-02,
          2.9257e-02,  6.6976e-02, -5.1645e-02, -5.8576e-02,  4.4058e-02,
          6.4003e-02, -8.9116e-02,  3.5608e-02, -8.9386e-02,  1.5408e-02,
         -6.4702e-02,  8.4275e-02, -3.4421e-02,  2.3750e-02, -2.7077e-02,
          6.6201e-02,  1.3047e-01,  5.2934e-02, -7.0075e-02,  8.2323e-02,
          1.5173e-03,  1.3263e-01,  2.4005e-02, -9.8103e-03,  1.5709e-02,
         -1.0565e-02,  2.1437e-04, -7.7929e-02,  3.7164e-02,  1.2634e-02,
          2.2879e-02,  1.3033e-02,  5.8234e-02, -6.0999e-03,  8.3576e-02,
         -1.0888e-01, -2.6662e-02,  7.0978e-03, -5.3044e-02, -3.2542e-02,
          2.6723e-02, -2.9112e-02, -6.2163e-02,  1.7580e-02,  2.6043e-02,
          1.0101e-02, -7.5235e-02, -4.3407e-02,  1.0585e-02, -3.7085e-02,
          7.0366e-02, -2.6555e-02,  6.7221e-02,  1.2299e-02,  2.8610e-02,
         -2.6220e-02,  2.8688e-02, -2.7788e-02,  3.7858e-02, -2.8472e-02,
         -9.6210e-02,  4.3830e-02, -8.2586e-02, -7.6470e-02,  3.7433e-02,
         -1.0156e-01, -5.9585e-02,  1.3437e-02, -3.3199e-02,  6.1372e-02,
          1.5220e-02, -1.5635e-02,  2.2824e-02, -2.0827e-04, -5.9591e-02,
         -1.0105e-01, -6.8851e-02,  1.6065e-02,  7.7250e-02, -5.1429e-02,
          8.5520e-02, -7.5052e-02,  1.6801e-02,  6.7147e-02,  3.3696e-02,
          4.5549e-02, -8.4228e-02],
        [ 6.4653e-02, -1.5547e-02,  4.9865e-02,  1.2855e-02, -2.5371e-02,
         -3.5523e-02,  8.3674e-02, -5.8829e-02,  6.6427e-03,  6.3402e-02,
          7.4916e-02,  1.0922e-01, -2.8613e-02,  2.2006e-02,  4.7548e-02,
          8.5340e-02,  1.4871e-02,  4.2198e-02, -1.2594e-02,  1.5360e-02,
         -5.0548e-02,  5.1476e-02,  3.2712e-02, -4.4127e-02,  8.9322e-02,
          2.3552e-02, -4.5435e-02, -4.5479e-02,  7.7638e-02, -3.1483e-02,
          2.5325e-02,  1.3304e-03, -3.0170e-02, -2.1833e-02,  3.2048e-02,
          7.1194e-02, -3.9704e-02,  6.0086e-02, -2.0174e-03,  8.9053e-02,
          9.6174e-03, -2.3244e-02,  2.1546e-02, -9.7847e-03, -4.7136e-02,
          6.1386e-02, -7.0884e-02,  7.9620e-02,  2.9621e-02,  8.4220e-02,
          3.6526e-02, -5.0930e-02,  6.4184e-02,  3.8562e-02,  9.9053e-03,
          3.5579e-02,  1.4564e-02,  4.4869e-02,  1.4271e-01,  1.9288e-02,
         -2.2655e-02,  4.6798e-02, -9.5464e-02, -5.7277e-02,  5.7738e-02,
         -7.7747e-02,  6.1369e-02,  3.9977e-02, -4.8167e-02,  5.8272e-02,
          1.2082e-01,  4.9015e-02, -6.0930e-02,  4.3724e-02,  3.4853e-02,
          1.0567e-02, -5.7192e-02, -1.5844e-03, -3.1740e-02,  6.5054e-02,
         -8.3283e-03,  4.6708e-02, -2.2336e-02, -2.4230e-02, -1.8426e-03,
          1.4059e-01,  7.7476e-02,  6.6464e-02,  2.7501e-02, -5.4163e-02,
         -3.5031e-02, -1.4757e-02, -2.0474e-02,  9.8877e-02,  2.4954e-02,
          8.8558e-03,  4.3068e-02,  7.3177e-02, -1.9841e-02,  7.7996e-02,
          3.4546e-02, -6.0550e-02,  7.3820e-02,  6.6568e-02,  4.8553e-02,
          9.8359e-02, -9.0408e-02, -2.2830e-02,  1.9583e-02,  1.1039e-01,
          8.5464e-03, -1.4003e-02,  7.3898e-02, -1.5702e-02,  6.5768e-02,
          4.1952e-02,  7.8894e-03, -1.7722e-02,  7.4088e-02, -3.6234e-02,
          4.9103e-02, -2.8396e-02,  6.5029e-02,  1.0110e-01, -5.1958e-02,
          1.4858e-01, -3.7848e-04, -3.5515e-02, -4.4228e-02, -3.5809e-02,
          2.8882e-02,  1.0516e-01,  2.4006e-02,  8.9773e-02,  2.9572e-02,
         -1.8308e-03,  1.2573e-01,  2.7721e-02,  8.8549e-02, -5.5817e-02,
          9.3037e-02, -2.1537e-02, -7.6680e-03,  1.1681e-01,  1.2255e-01,
         -2.9090e-02,  2.0161e-02,  7.5285e-02, -5.0552e-02,  1.5078e-03,
         -4.4254e-02,  5.0927e-02, -2.1075e-02,  1.2968e-01,  1.0991e-01,
          4.3455e-02,  1.7081e-02,  4.9998e-02, -3.0583e-03,  6.4713e-02,
         -4.7365e-02,  4.9718e-02,  6.6212e-03,  4.9609e-02,  1.2077e-02,
          2.6176e-02, -2.4706e-02,  1.0843e-01, -7.0687e-02,  3.6215e-02,
          4.0733e-02,  6.3392e-03, -1.5404e-02, -7.8444e-02,  4.4042e-02,
         -6.5298e-02,  6.7711e-02,  2.1102e-02, -3.8756e-02,  5.5539e-03,
         -3.2970e-02,  1.4610e-01, -5.3893e-02,  1.8893e-02,  9.9133e-02,
          6.9220e-02,  7.6725e-02,  5.5739e-03,  4.6127e-02, -3.8389e-02,
         -2.7073e-02, -3.1743e-02],
        [ 6.9145e-02,  4.9097e-02,  4.2895e-02, -2.7577e-02, -1.4664e-02,
         -4.0198e-02,  1.3563e-02, -3.9012e-02, -3.8051e-02,  4.4476e-02,
          1.0073e-01, -5.7455e-02,  7.4259e-02, -3.9649e-02,  7.7948e-02,
         -6.9183e-02,  1.4803e-02,  6.1086e-02,  3.0811e-02,  1.5638e-02,
          3.1143e-02, -6.4975e-02, -4.5014e-02, -1.2843e-02,  9.2398e-03,
          1.7015e-03,  3.0987e-02,  7.1119e-02, -3.2949e-02, -1.4362e-02,
         -8.2381e-02,  5.1600e-02,  2.6372e-02,  6.0871e-02,  6.1861e-02,
          3.2933e-02,  5.6738e-02,  6.4476e-02,  5.4748e-02,  5.1070e-02,
          3.2093e-02, -6.2537e-02,  8.3763e-02, -5.9540e-03,  3.0868e-02,
         -3.2967e-02,  1.0978e-03, -7.4916e-02,  3.2435e-02,  6.8937e-04,
          3.0891e-02,  4.1466e-02, -3.6827e-02,  5.4449e-02,  3.0493e-02,
          2.9739e-02,  3.1229e-02,  5.6898e-02,  6.0154e-02,  3.1172e-02,
          4.8663e-03, -4.2301e-02,  7.8303e-02, -2.8945e-02, -7.8657e-02,
         -9.8660e-03,  4.4122e-02,  4.7886e-02,  8.8215e-03,  1.6905e-03,
         -2.1273e-02, -2.8495e-02,  1.2287e-01,  1.0128e-01, -8.2786e-03,
          6.2792e-02, -2.7561e-02,  1.2222e-02, -5.0112e-03,  2.1808e-02,
         -6.4256e-02,  4.1160e-02, -1.8245e-02, -2.2499e-02,  3.8796e-02,
          6.8747e-02, -2.8697e-02,  8.3993e-02,  1.5101e-02,  7.2751e-02,
          7.2188e-02,  5.8667e-02, -2.0022e-02, -9.9231e-02, -3.0847e-02,
          4.1177e-03, -6.6058e-02, -1.9301e-02,  6.3415e-02, -2.8608e-02,
          8.4073e-02, -1.1691e-02,  3.2576e-02, -3.4366e-02, -4.9279e-02,
          8.5017e-02,  1.1862e-01, -2.4733e-02,  1.0801e-02, -1.8955e-02,
          2.4567e-02, -3.2565e-02,  3.4354e-02,  1.7276e-02,  4.5942e-02,
         -1.5350e-02, -2.9060e-02, -5.5349e-02, -6.7948e-02,  7.2850e-02,
          4.8747e-02,  5.6527e-02,  4.8515e-02, -3.1080e-02,  4.5178e-02,
         -9.3024e-02,  8.3925e-02,  9.0772e-02,  8.5380e-02, -4.6187e-02,
         -8.9595e-02,  5.2646e-02,  2.0815e-02,  1.3759e-02, -4.9418e-02,
          2.1094e-02, -8.6545e-02, -1.8543e-02, -4.9952e-02,  1.1132e-02,
         -2.0360e-02,  4.5882e-02,  3.6740e-02, -3.4649e-02, -9.0142e-04,
          3.1390e-02,  9.7267e-03,  5.7703e-02,  2.4947e-02, -7.5845e-02,
          1.5996e-02, -2.2332e-02,  5.2735e-02,  3.4870e-02, -1.9799e-02,
          5.2848e-02, -9.8706e-03, -1.3195e-02, -3.7519e-02, -1.0064e-02,
         -3.0736e-02,  1.6016e-02,  1.0976e-02,  2.7015e-02,  2.1259e-03,
         -6.8054e-02,  6.6768e-02, -8.7812e-02,  6.4702e-02,  5.5078e-02,
          8.3740e-03,  4.3494e-02, -2.6889e-03, -1.0321e-02,  2.4359e-03,
          1.7867e-02,  6.0468e-02, -1.3429e-02,  5.6885e-02,  3.8339e-02,
          8.5721e-02,  2.8720e-02,  4.8530e-02,  2.8380e-02, -1.7022e-02,
          6.2604e-02, -1.0164e-01,  3.0316e-02,  4.5669e-02, -4.2311e-02,
          3.0587e-02, -6.9216e-02],
        [ 5.6644e-02,  2.8224e-02, -2.3434e-02, -5.4324e-02, -4.1016e-02,
          1.5514e-02, -8.9447e-02,  6.7195e-02, -6.3294e-02,  5.3368e-02,
          4.0684e-02, -1.3265e-02, -3.3319e-02, -4.5820e-02,  8.9207e-02,
         -1.3399e-01, -3.1417e-02,  5.3994e-02, -1.9360e-02,  1.9592e-02,
          4.3604e-02, -4.7234e-02, -2.8433e-02,  3.8326e-02,  2.8876e-02,
          8.2052e-02,  5.3598e-02,  1.1040e-01,  2.4077e-02, -5.9412e-02,
         -5.4127e-02,  3.5982e-02, -4.5985e-03,  6.3539e-02,  5.7232e-02,
         -3.0077e-02,  7.6929e-02,  4.6229e-02,  3.3886e-02, -8.0415e-02,
         -2.1148e-02,  3.0245e-02,  1.0026e-01, -6.3085e-02,  6.2106e-02,
          1.4721e-03,  2.3602e-02, -2.9517e-02,  5.5832e-02, -3.7799e-02,
          4.5768e-02,  2.2824e-02,  6.7541e-02, -6.2850e-02,  8.7479e-02,
          7.6970e-02,  9.1645e-03,  7.8809e-02, -6.4573e-02, -7.4097e-02,
         -5.3180e-02,  3.8755e-02, -3.9103e-02, -3.1011e-02,  2.0122e-03,
         -1.1995e-02,  3.6135e-02,  5.3754e-02,  1.1844e-01, -2.6693e-02,
         -1.2531e-02, -9.4975e-03, -1.3573e-02,  4.2424e-03, -9.0446e-02,
          6.7475e-02, -3.6718e-02,  8.4422e-02, -8.4154e-02,  4.5830e-02,
         -1.8503e-02, -1.9691e-02, -6.6574e-02, -8.0196e-02,  1.1135e-02,
         -9.1821e-02,  4.6615e-02,  1.8077e-02, -1.1399e-02,  6.9427e-02,
          5.0050e-02, -3.1973e-02,  4.4681e-02,  2.5261e-02,  5.1233e-03,
         -5.0728e-02, -9.8283e-02, -7.6790e-02, -8.0220e-02, -5.3451e-02,
         -8.7025e-04,  1.0134e-01, -4.4666e-02,  8.9050e-03, -6.2074e-02,
         -8.4207e-03,  4.8037e-02,  3.5468e-02, -2.9635e-02, -7.0812e-02,
          6.5087e-02,  5.6175e-02, -4.3641e-02,  1.4305e-02,  3.8099e-02,
         -3.8554e-02,  2.4128e-02, -4.7322e-02, -9.3049e-02,  1.2411e-02,
          9.9089e-02,  8.6489e-02,  4.9593e-02, -6.8331e-02,  4.0361e-02,
          2.1820e-02,  9.2557e-02, -1.5685e-02, -4.3196e-02, -7.9627e-02,
          9.8109e-03,  5.9043e-02, -2.0684e-02, -6.2032e-02, -1.5522e-02,
          5.4100e-02,  2.3943e-02,  3.8254e-02,  1.8563e-02,  5.7112e-02,
         -4.4520e-03, -8.6870e-03,  9.2398e-02, -9.7850e-02, -1.0828e-01,
          8.8885e-02,  1.4362e-02,  8.0427e-03,  1.1782e-01, -8.1023e-02,
         -4.0866e-03, -1.8025e-02, -3.7094e-02,  4.3714e-02, -8.0235e-02,
         -2.8853e-02, -2.3628e-02, -1.2343e-02,  5.7856e-02,  3.3556e-02,
          3.6231e-02, -5.4774e-02,  4.0355e-02, -2.1671e-02, -3.7586e-02,
          7.1867e-03,  3.7097e-02,  4.4036e-05, -4.2338e-02,  3.3348e-02,
          9.6290e-02,  3.5505e-02, -3.8404e-02,  3.9792e-02, -3.3936e-02,
          4.2411e-03, -7.1071e-02,  8.8117e-02,  6.2802e-02, -3.1505e-02,
          4.6092e-02,  5.0399e-02, -5.4181e-02, -3.2181e-02, -8.8433e-02,
         -2.1697e-02, -1.4775e-02,  6.4337e-02, -2.6897e-02, -5.8358e-02,
          9.1849e-02, -4.2708e-02],
        [-4.7281e-02, -8.4584e-04,  6.4313e-02, -9.3248e-03, -2.3016e-02,
         -2.7392e-02,  7.7039e-03,  2.6554e-02,  2.0799e-02,  1.7433e-02,
          7.3164e-02, -7.1637e-04,  6.9147e-02, -6.6245e-02,  2.1504e-02,
         -5.2063e-02, -4.9715e-02, -4.4125e-02, -6.4687e-02,  2.3201e-03,
          3.2450e-02,  3.1066e-02,  1.9345e-03, -3.7824e-02, -6.8521e-02,
         -2.7753e-02,  8.1769e-02,  6.2513e-02,  1.9870e-02, -3.3191e-02,
          1.8625e-02, -2.2285e-02,  3.5422e-02, -4.0136e-03, -5.8023e-02,
          8.8409e-02,  9.7659e-02, -1.1484e-02,  5.3281e-02,  4.9012e-02,
         -5.0144e-02, -8.4489e-02,  1.0858e-02, -4.4141e-02,  5.4209e-02,
         -5.8983e-02,  6.3596e-02, -6.5372e-02, -6.7371e-02,  4.4135e-02,
         -9.3491e-03,  5.2609e-02,  7.7210e-02,  2.6228e-02,  1.3668e-02,
         -5.3366e-02, -4.5694e-02,  2.1669e-02, -6.3275e-02,  4.5734e-02,
         -5.7523e-02,  4.2430e-02,  2.3828e-02,  9.6590e-02, -8.0452e-02,
         -4.2205e-02, -3.9022e-02, -2.9214e-03,  8.6232e-02, -1.0976e-01,
          1.5651e-03,  5.3861e-02,  2.6091e-02,  1.3133e-02, -2.5155e-02,
          7.0778e-03,  1.0962e-01, -3.5864e-02, -1.2812e-02, -7.8410e-02,
         -1.5463e-03,  5.0118e-02,  2.7058e-03,  1.6910e-02,  4.5953e-02,
          1.2969e-02,  3.5914e-02, -3.9079e-02, -6.6343e-02,  1.9280e-02,
          2.2410e-02, -2.6855e-02, -3.4638e-02, -8.2651e-02, -2.9268e-02,
         -5.5470e-02, -6.6014e-02,  1.8405e-02,  1.3757e-02, -6.9123e-02,
          1.1001e-01,  5.1341e-02, -7.5963e-02,  3.3869e-02, -5.4046e-02,
          3.7867e-02,  7.4616e-02, -5.8728e-02,  2.2238e-02, -2.7931e-02,
          8.9211e-03,  1.0041e-02,  2.1077e-02,  4.2602e-02,  7.4511e-02,
          2.2195e-02,  2.1148e-02, -2.4814e-02, -5.2093e-02,  2.6394e-02,
          4.7329e-02, -1.0961e-02,  1.4006e-02, -4.1785e-02, -2.3234e-02,
         -2.2396e-02,  2.7603e-02, -1.9180e-02, -7.4210e-02,  2.9094e-02,
         -5.1395e-02,  2.8865e-02, -9.1253e-02,  1.4203e-02, -2.2611e-03,
          1.0429e-01,  5.1639e-02, -4.6187e-02,  1.5009e-02,  2.5975e-03,
          1.6689e-02, -4.8532e-02,  5.9330e-02,  1.6495e-02, -4.2717e-02,
         -2.2966e-02,  1.9125e-03,  3.5660e-02,  1.5806e-02,  4.5198e-02,
          5.6593e-02,  3.1899e-02,  2.1796e-03, -2.1952e-02,  3.9102e-02,
          1.6547e-02,  7.1777e-02, -2.1427e-02, -2.9896e-02, -6.9536e-02,
         -3.6804e-02,  1.3289e-02,  1.4367e-02,  5.8743e-02, -5.4273e-02,
         -2.7014e-02, -1.0663e-02, -1.0524e-02,  2.6481e-02, -1.0155e-01,
          9.1585e-02,  1.2746e-02,  9.1430e-03,  2.0060e-02,  4.9493e-02,
          2.5503e-02,  6.7868e-02,  9.4210e-04,  3.7954e-02, -2.1493e-02,
         -5.8909e-02,  1.9123e-02, -4.7418e-02,  6.1550e-02,  2.0879e-02,
          3.6539e-02,  2.2203e-02,  3.6279e-02, -5.5031e-02,  1.1426e-02,
          3.8760e-02,  4.8918e-02],
        [-5.0581e-02, -4.4126e-03, -2.1193e-02, -6.4530e-02,  9.8154e-04,
          4.8908e-02,  4.7938e-02, -8.1069e-02,  2.4499e-02, -6.6185e-02,
         -4.8372e-02,  9.6291e-02, -2.9269e-02, -2.4156e-02, -5.3554e-02,
          8.2269e-02, -4.2159e-02, -1.3391e-02,  6.8401e-02,  6.0340e-02,
          3.7533e-02, -1.0048e-02,  3.9501e-02, -4.2024e-02, -5.5350e-02,
          3.7659e-02,  2.9388e-02,  2.5552e-02, -5.8761e-02,  1.5103e-03,
          9.8855e-02,  9.4520e-02,  1.8844e-02,  1.7358e-02,  6.1842e-02,
          6.7126e-02, -2.6154e-02,  1.9591e-02,  3.0959e-02,  9.5424e-02,
          3.3863e-02,  5.7530e-02, -5.8110e-03, -5.7849e-03,  2.8038e-02,
          9.4057e-02, -1.1769e-02,  1.5621e-02,  4.9153e-02,  1.7762e-02,
         -2.4201e-02, -2.0779e-02,  1.9907e-02, -2.9210e-02,  5.4924e-02,
         -3.0567e-03,  1.8684e-02,  7.1904e-03,  5.2764e-02,  3.9695e-02,
         -6.7935e-02,  5.0214e-02,  1.8321e-02, -5.5361e-02,  5.5177e-02,
         -1.8193e-02,  2.8049e-02, -4.3673e-02, -6.5583e-02,  9.1334e-02,
          6.2167e-02, -6.2727e-02, -4.2931e-02, -1.8788e-02,  3.3002e-02,
         -9.8762e-02, -8.1585e-02, -8.6597e-02,  6.1378e-02, -4.2956e-02,
         -2.8761e-02, -9.4009e-03, -2.5775e-02,  3.3387e-02,  5.8443e-02,
          1.4690e-02, -7.8898e-02, -3.3656e-02,  4.2046e-02,  3.6264e-02,
          3.9295e-02, -5.7803e-02, -1.1547e-02, -2.9746e-02, -5.8808e-02,
          8.1344e-02,  4.4491e-02,  5.5488e-02, -2.8923e-02,  3.9116e-03,
         -9.4607e-02, -4.5935e-02, -4.4385e-02, -2.8126e-02,  3.0660e-02,
         -6.1597e-02, -5.8897e-02,  1.7818e-02,  3.8573e-04,  4.4124e-02,
         -7.9633e-03, -1.7043e-02,  3.7533e-02,  3.8311e-02,  4.7640e-02,
          9.0857e-03, -4.8994e-02,  7.1709e-02,  5.5587e-02, -1.5063e-02,
          2.8517e-03, -9.6373e-02, -1.5092e-02,  1.0987e-01,  2.8820e-02,
          1.5667e-02,  2.1203e-02,  1.2062e-02, -4.8146e-02,  5.4135e-02,
          1.1718e-01, -8.6982e-02,  8.3918e-02, -2.1248e-03, -1.8896e-02,
         -9.8971e-02,  1.2380e-02, -3.1456e-02, -3.6557e-02,  2.2465e-02,
          1.1165e-01, -2.8860e-02, -7.5154e-02,  1.2390e-01,  1.3617e-02,
          6.8612e-03,  9.2188e-02, -1.7091e-03,  2.4687e-03, -2.6468e-03,
         -2.5007e-02,  1.6829e-02,  8.0180e-02, -2.8605e-02,  1.0780e-01,
         -7.4514e-02, -4.9733e-02, -6.0505e-02, -9.0293e-03,  6.8950e-02,
         -5.9940e-02, -4.6762e-02,  5.6985e-02, -3.5847e-02,  4.6487e-02,
          5.3846e-02, -1.8953e-02,  9.4508e-02,  3.7553e-02,  1.3570e-02,
         -9.7001e-02,  3.9641e-02,  5.4096e-02, -9.4090e-03,  1.9672e-02,
         -3.6690e-02,  1.0713e-02,  1.3149e-03,  6.2422e-02,  6.7435e-02,
          5.1781e-02,  8.0024e-02,  6.3160e-02, -9.6648e-02,  1.9755e-02,
         -1.7050e-02,  8.9010e-02,  1.4442e-02, -2.3148e-02, -2.5715e-02,
         -7.9407e-02,  8.6967e-02],
        [ 3.0989e-02, -2.4900e-02,  4.3371e-02,  7.0848e-02, -4.1811e-02,
          4.0464e-02,  6.2121e-02,  4.3907e-02,  6.5252e-02, -5.2491e-03,
         -6.2729e-03,  9.9220e-02, -4.0675e-02,  1.1170e-01, -8.3535e-02,
          2.9790e-02, -1.0508e-02,  1.0045e-01,  2.5004e-02,  1.5028e-02,
         -6.7734e-02,  5.1626e-03,  6.5432e-02,  2.0990e-02,  7.6451e-02,
          1.9939e-02,  4.9719e-02, -9.0684e-02,  5.0686e-02,  5.5739e-02,
          6.3916e-02,  1.2013e-01,  6.8884e-02,  2.6279e-02, -6.1174e-03,
         -4.5599e-02, -1.7512e-02, -2.5814e-02,  1.0347e-01,  7.0656e-02,
          4.3043e-02,  6.1517e-02,  6.5080e-02, -1.4001e-02, -8.4879e-03,
          1.1534e-01,  5.7982e-03,  7.4008e-02,  6.1168e-02,  4.1294e-02,
          6.1429e-02, -1.8568e-02,  2.3843e-03, -4.9772e-03, -1.6753e-02,
         -2.8042e-02, -1.8442e-02, -1.4646e-03, -2.9071e-02,  1.0008e-01,
          7.2861e-02,  4.5510e-02,  3.7971e-02,  2.3632e-02,  4.1046e-02,
         -1.2912e-02, -1.2978e-03, -5.1472e-02, -1.5410e-02,  1.1819e-01,
          3.4452e-02,  2.6180e-02,  4.9079e-03,  3.4186e-02,  5.0505e-02,
         -9.5703e-02,  4.1039e-02,  1.3738e-02, -1.5085e-02, -4.1269e-03,
         -8.7661e-03,  1.5574e-03, -2.1945e-02, -3.8825e-02,  2.8859e-02,
          4.4147e-02,  5.5781e-02, -4.1060e-02,  6.2908e-02,  1.5675e-02,
         -2.3338e-02,  4.0023e-02, -7.5926e-03,  3.6740e-02, -4.5571e-02,
          8.4680e-02,  3.3700e-02,  3.7802e-02,  1.1035e-02, -4.9361e-02,
          1.1198e-02, -5.7405e-02,  8.3435e-02,  3.1675e-02,  7.8031e-02,
          1.7558e-02, -8.4815e-02,  7.0688e-03,  1.1271e-01,  1.4810e-02,
          8.1600e-02,  9.5336e-02,  8.7039e-02, -1.0226e-02,  6.2441e-02,
          8.6455e-02, -2.9443e-03,  9.8162e-02,  8.1146e-02,  4.4815e-02,
          4.9743e-02, -3.5872e-03,  5.4121e-02,  2.7907e-02,  2.9067e-02,
          7.8864e-02,  4.5990e-02,  4.9159e-03, -8.9599e-02,  2.9415e-02,
          9.3777e-02, -3.0759e-03, -6.6616e-03, -3.1905e-02, -2.7118e-03,
         -9.1111e-02,  7.4464e-03,  2.4430e-02, -2.7349e-02,  9.9149e-03,
          1.0478e-01,  7.9634e-02,  5.3667e-02, -3.7605e-03,  1.1204e-01,
          2.5380e-02, -4.0299e-03, -5.6004e-02, -5.1147e-02,  8.3699e-02,
          4.1844e-02, -9.9196e-04,  3.0416e-02, -1.4730e-02,  1.4185e-01,
         -5.0755e-02,  1.9635e-02, -4.1826e-02,  4.9465e-02,  1.7474e-02,
         -5.8664e-02, -3.1052e-02,  2.1531e-02,  3.6265e-02,  1.2778e-01,
         -1.0219e-02, -3.3122e-02,  9.5646e-02,  1.4470e-02,  3.4446e-02,
          8.5235e-03,  4.5864e-02,  9.3928e-02, -4.0491e-02,  5.2611e-02,
         -7.1805e-02, -3.7650e-02, -5.8284e-03,  1.8785e-02,  7.1576e-02,
          8.0341e-02,  1.0111e-01, -4.4844e-02,  1.8423e-02, -2.1372e-03,
          3.4896e-02,  6.9211e-02,  8.7448e-02,  1.0479e-01, -5.0510e-02,
          1.4182e-02,  3.2258e-02],
        [ 3.1239e-02,  5.9662e-03, -5.8983e-02, -4.8294e-02, -3.3505e-03,
          3.7231e-02,  9.8247e-02, -7.3105e-02,  5.2259e-02, -1.2801e-01,
          1.8802e-03, -8.3899e-03, -1.9764e-02,  1.7681e-02, -3.2621e-02,
          1.2439e-02, -1.3885e-03,  2.2526e-02,  6.9896e-02,  1.1905e-02,
          6.9356e-02,  8.0667e-02, -9.2859e-03,  3.5341e-02,  3.0798e-02,
          4.1052e-03,  6.5922e-02, -1.5473e-02, -2.1983e-02, -4.0594e-02,
          1.0106e-01, -1.2269e-02, -1.1763e-03, -5.8014e-02,  6.0823e-02,
         -1.7199e-02,  5.1376e-02,  7.9445e-02,  4.6261e-02,  7.3425e-02,
          4.0257e-02,  7.7711e-02, -2.1657e-03,  8.5189e-02,  1.7664e-02,
          6.0680e-02,  4.1687e-02,  4.8157e-02,  5.3892e-02,  9.9481e-02,
          3.2641e-02,  6.5924e-03,  1.8080e-02, -7.2753e-02,  2.6974e-02,
          3.5297e-02, -2.4944e-02,  3.1529e-02,  6.5854e-02,  2.5827e-02,
          1.5005e-02,  4.3129e-02,  9.1924e-02,  3.5373e-02,  7.6713e-02,
         -4.2429e-02, -1.5229e-02,  3.7344e-02, -1.0086e-01,  5.6293e-02,
         -3.4638e-02, -8.3246e-03, -6.2221e-02,  5.6895e-02, -4.0821e-02,
          2.8244e-02, -7.1151e-02, -5.4520e-02,  9.2886e-02,  5.0983e-02,
          8.3596e-02,  1.0704e-03,  7.5344e-02,  1.7621e-02, -1.0797e-02,
         -2.1621e-02,  5.0472e-02,  6.6167e-03,  1.0394e-01, -6.0457e-02,
          3.8016e-02,  2.6075e-02,  4.5596e-02,  9.9374e-02, -5.2293e-02,
         -1.1722e-02,  4.3572e-02, -1.4280e-02,  4.3169e-02,  1.3449e-02,
         -5.8918e-02, -1.0695e-01,  8.9179e-02,  4.9036e-02, -1.6538e-02,
          5.9734e-02,  6.6545e-03,  4.6788e-02,  5.4699e-02,  7.9400e-02,
          6.1575e-02,  5.3873e-03,  1.3161e-03,  1.9339e-02, -7.8289e-02,
         -4.2979e-03,  1.1613e-03,  1.1930e-01,  7.7410e-02,  7.2886e-02,
          1.8128e-02, -1.0910e-01,  3.2789e-02,  7.6305e-02, -3.8491e-02,
          9.6212e-02, -5.1727e-02, -1.8777e-02, -3.7476e-02,  1.0314e-01,
         -1.0933e-02,  2.4850e-02,  7.4931e-02, -1.4901e-03, -6.0116e-02,
         -4.0657e-02,  8.1239e-02, -2.8803e-02, -1.9366e-02, -6.4808e-02,
         -9.3744e-03, -1.9949e-02, -6.5297e-02, -1.4569e-02,  5.3159e-02,
         -5.8069e-02, -1.9120e-02, -6.3454e-02, -1.0651e-01,  4.8819e-02,
          1.1790e-02,  4.2434e-02,  2.1013e-02, -5.8285e-02,  7.2797e-02,
         -5.6186e-02, -2.6420e-03,  5.4869e-02,  3.4530e-02,  2.9441e-02,
          6.8549e-02,  6.9368e-02, -3.7981e-02, -6.5886e-03,  4.8729e-02,
          4.6308e-02,  1.2451e-02,  3.5610e-02,  7.3860e-03,  1.6447e-02,
          3.1842e-02,  2.9236e-03, -3.4142e-02,  6.5223e-03, -5.3262e-02,
          1.6867e-02, -5.0455e-02, -4.4010e-02, -5.1780e-03, -3.6732e-02,
          7.1692e-02, -2.8939e-02, -1.6115e-02, -4.3305e-02,  6.3186e-02,
         -7.5106e-02,  1.0119e-01,  4.2867e-02,  1.4796e-02,  7.4386e-03,
         -6.3854e-02,  3.2422e-02],
        [ 5.3958e-02, -6.6555e-02,  1.8678e-02,  6.3882e-02,  5.0884e-02,
         -4.9568e-03,  6.3829e-02, -4.6324e-02, -1.5222e-02, -7.5792e-03,
          1.9984e-02,  9.0913e-02,  2.8675e-02,  8.7300e-02,  2.0915e-02,
          1.1714e-01,  8.8693e-02, -1.2298e-02,  5.5424e-02,  7.9682e-03,
         -7.0498e-03,  5.5835e-02,  5.5863e-02,  6.0897e-02, -2.7373e-02,
          7.6101e-02,  6.0023e-02, -8.8386e-02,  4.6715e-02,  5.4982e-02,
          9.1673e-02,  4.6219e-02,  7.3865e-02, -5.5959e-02,  3.9713e-02,
         -1.2280e-02, -5.3267e-02, -5.8923e-02,  4.9102e-02,  7.6076e-02,
          5.4756e-02,  2.5198e-02,  4.3820e-02, -4.0022e-02,  4.1196e-02,
          7.2311e-03, -9.2426e-02,  6.6516e-02,  2.0451e-02,  4.2154e-02,
          7.6163e-02,  3.4149e-02, -6.1636e-02, -3.7671e-02,  3.9804e-02,
          2.3951e-02, -3.7370e-03, -1.8182e-02,  9.6779e-02,  5.5351e-02,
          4.0439e-04, -3.7100e-02,  5.3994e-02,  8.4826e-02,  7.9000e-02,
          4.2130e-02,  5.0464e-02,  8.6715e-03, -8.7097e-02,  9.9714e-02,
          5.4801e-02,  8.8327e-02, -6.2397e-02,  9.9558e-02,  9.4344e-02,
         -1.5740e-02,  6.7874e-02, -2.6533e-02,  4.1506e-02, -3.5224e-02,
         -1.9887e-02,  5.2912e-02,  6.8297e-02, -5.1450e-02, -3.1202e-02,
          5.7023e-02,  2.3968e-02, -2.9784e-02,  9.4083e-02, -8.3191e-02,
         -1.1131e-02,  1.8001e-02, -6.2901e-02,  8.6177e-02, -4.0736e-02,
          6.7457e-02, -2.8468e-03, -3.7669e-02,  6.2061e-02, -2.8012e-02,
          2.1889e-02, -6.7918e-03, -1.1463e-02,  3.9627e-02,  9.0501e-02,
          1.0945e-02, -1.0894e-01,  4.6520e-03,  6.7768e-02, -3.7176e-03,
         -7.3585e-03,  1.0729e-02,  5.0704e-02,  8.5525e-02, -3.1116e-03,
          7.0397e-02, -2.8770e-02, -2.8093e-02,  6.4603e-02,  2.8241e-02,
          2.4868e-02,  8.7578e-03,  6.6684e-02,  9.9776e-02,  5.3076e-02,
          3.2994e-02, -2.8846e-02,  7.9051e-02, -5.5045e-02,  1.4130e-02,
          2.8710e-02, -3.0807e-03,  7.1467e-02,  5.3458e-02, -3.5774e-02,
         -3.8862e-02,  9.5893e-02,  4.0181e-02,  3.5784e-02, -1.6361e-03,
          8.9542e-03, -4.7624e-02, -4.1825e-02,  1.0659e-01, -1.8893e-02,
         -3.4571e-02,  6.5850e-03,  2.8192e-02, -2.8267e-02,  5.4632e-02,
          7.2935e-02, -1.4671e-03,  5.7195e-02,  4.6394e-02, -3.4978e-03,
         -5.5403e-02, -3.8176e-02,  1.0309e-02,  5.2030e-02, -1.0374e-02,
          1.3116e-02, -2.8628e-02,  2.4396e-02, -1.8292e-02,  8.9504e-03,
          3.1159e-02, -7.5987e-02, -1.1927e-02, -1.7822e-02, -4.2380e-02,
         -3.5025e-02,  5.7157e-02,  5.3670e-02,  2.2556e-02,  5.9294e-02,
         -2.1345e-02,  5.2907e-02, -5.6887e-02,  3.4975e-03, -9.1616e-03,
          9.5513e-02, -1.2764e-02,  5.3681e-02, -9.4004e-03,  8.8062e-02,
         -1.5107e-02,  3.4489e-03, -3.1161e-03,  8.6600e-02,  7.3153e-03,
         -8.4903e-02,  3.3884e-02],
        [ 5.0592e-02,  1.2765e-02, -3.4386e-03,  5.5863e-02,  5.6908e-02,
         -7.2645e-03, -6.8998e-02, -6.0508e-02, -1.1571e-02,  2.1301e-02,
          3.0621e-02, -6.6271e-02, -4.5462e-02,  3.7469e-02, -1.0002e-01,
         -5.4209e-02, -4.6881e-02, -4.6213e-02, -6.7045e-02,  7.2392e-03,
         -5.4179e-02, -1.5936e-02, -7.1926e-02,  2.1305e-02, -9.7515e-02,
          1.8854e-02, -3.3694e-02, -7.7374e-02, -5.2186e-02, -7.6988e-02,
          1.9083e-02, -4.3353e-02, -3.8692e-02, -4.5927e-02, -9.5688e-02,
         -9.1357e-02, -6.8684e-02, -4.2049e-02, -1.7214e-02,  7.5973e-03,
         -8.2691e-02, -6.6447e-02, -1.0022e-01, -7.6233e-02,  3.5728e-02,
         -9.9953e-02, -7.2517e-02, -8.3625e-02, -4.4573e-02, -5.9080e-02,
          3.1312e-02, -8.6173e-02, -1.7269e-02, -1.5513e-02, -3.1770e-02,
          2.5271e-02, -4.3341e-02, -7.7902e-02,  1.0160e-02, -9.7103e-02,
         -4.0708e-02,  6.9074e-02, -3.0275e-02,  2.1298e-02,  7.0530e-02,
         -7.7782e-03, -3.2398e-02,  1.3039e-02,  1.2261e-02,  2.6519e-02,
          1.6384e-03,  2.9927e-02,  2.6177e-02, -3.1408e-02, -7.9812e-03,
         -1.9421e-02,  1.2757e-02, -6.3811e-02, -6.4248e-02,  6.6844e-02,
         -5.3653e-02, -3.7508e-02,  1.6706e-02,  4.2113e-02,  2.4809e-02,
         -7.4257e-02,  1.9978e-02, -4.5553e-02, -1.7478e-02,  3.3111e-02,
         -3.5467e-02,  5.2989e-02,  4.8396e-02, -5.2017e-02,  6.3880e-02,
         -3.6575e-02, -7.8872e-02, -4.6151e-02, -7.8448e-02, -2.7667e-02,
         -8.9113e-03, -6.5771e-02, -9.5575e-02,  4.2333e-02, -1.0009e-01,
         -6.7047e-02, -3.3141e-02, -5.5059e-02,  3.8236e-02,  2.6778e-02,
          3.3197e-02, -9.5663e-02, -1.0110e-01, -8.8468e-02, -5.0318e-02,
          1.1242e-02, -5.3394e-02, -6.4074e-02, -4.9796e-02, -6.6454e-02,
          3.9051e-02,  1.5544e-02, -9.1213e-02, -8.5213e-02,  1.4434e-02,
         -7.1538e-02,  6.9514e-02, -8.0879e-03, -6.5417e-02,  3.3009e-02,
          1.3317e-02, -8.7720e-02, -2.4874e-02, -5.2349e-03, -1.0711e-02,
         -7.9271e-02,  3.5805e-03,  3.0409e-02, -8.0098e-02,  3.6275e-02,
         -9.5878e-02, -7.8374e-02, -8.9187e-02, -9.0842e-02,  4.0061e-02,
         -2.3876e-02, -6.8918e-02, -4.4128e-03,  2.1377e-02,  5.8615e-03,
         -7.9238e-02, -1.1729e-02, -2.0186e-02, -5.6563e-03,  5.2908e-03,
         -6.9677e-02, -5.0916e-02, -6.9522e-02,  5.6180e-02,  3.5931e-02,
          6.6432e-02, -4.4097e-02,  4.4856e-03,  3.4787e-02,  3.9981e-03,
          1.1599e-02, -7.2594e-02,  1.8972e-02,  3.3592e-02, -3.2992e-02,
          4.6174e-02, -4.6601e-02, -3.9737e-02, -4.4913e-02,  2.9248e-02,
          2.3720e-02,  5.4232e-02, -9.1054e-03,  4.0785e-02, -3.9880e-02,
         -6.4116e-02, -2.3705e-02, -5.4452e-02, -1.2954e-02, -3.1368e-02,
         -9.9524e-02,  3.2944e-02, -1.8760e-02, -4.7569e-02,  6.2250e-02,
         -3.6968e-02,  4.9088e-02],
        [-1.1271e-02, -2.2990e-02,  4.8161e-02,  7.1934e-03,  1.6119e-03,
          2.4368e-02,  2.7006e-02, -5.2070e-03,  7.4699e-02, -1.1342e-01,
         -7.0580e-02,  1.6649e-02, -3.7516e-02, -1.3974e-02,  3.4925e-02,
          3.9711e-02, -3.2689e-02,  1.6124e-02, -7.0502e-02,  3.5154e-02,
          4.8396e-02,  5.0255e-02,  2.6511e-02, -1.8945e-02,  2.2159e-02,
          1.4420e-02,  1.6081e-02, -7.3359e-02, -4.5979e-03,  3.4905e-02,
          7.6974e-02,  8.2991e-02,  3.5707e-02, -6.2907e-02, -1.3954e-02,
          1.1284e-05, -5.3347e-04,  1.0010e-01,  6.7408e-02,  4.3465e-02,
         -2.2821e-02,  7.1818e-02, -3.9339e-02,  7.4052e-02,  1.5749e-02,
          6.3614e-02, -4.3074e-02,  6.9430e-03,  9.2734e-02,  4.3230e-02,
          1.4175e-04, -9.1063e-02, -4.0164e-02,  1.1297e-02,  2.5491e-02,
          4.0554e-02,  9.3395e-02,  3.1944e-02, -2.7232e-02,  5.7235e-02,
          4.4066e-02, -1.5371e-02,  7.1386e-02,  5.4399e-02, -2.2142e-02,
          3.3776e-03, -3.9976e-02, -5.5684e-02, -8.6902e-02,  2.8842e-02,
          5.4076e-02,  3.5090e-02, -4.4205e-02,  2.1064e-02,  4.1836e-02,
         -4.9887e-02,  4.8645e-02,  1.6995e-02, -2.5490e-02,  7.1723e-02,
         -3.4632e-02, -2.3265e-03,  8.9172e-02,  1.8643e-02,  3.7295e-02,
          6.5153e-02, -8.0702e-02,  7.0805e-02,  1.0259e-01,  1.1583e-02,
         -2.8602e-02, -9.8348e-02, -5.6250e-02,  7.0899e-03,  1.1662e-02,
         -3.8567e-02,  3.6550e-02,  3.3746e-02,  5.1856e-02,  3.9591e-02,
         -6.7087e-03, -1.1364e-01,  5.8216e-02,  3.2226e-02,  7.3354e-02,
          6.3387e-02, -2.8258e-02,  6.0787e-02, -4.9654e-02,  5.4855e-02,
          6.6485e-02, -6.2822e-02, -4.6745e-02,  6.6785e-02, -3.4061e-02,
         -4.1470e-02, -2.5930e-02,  2.5031e-02,  9.0147e-03, -3.4597e-02,
         -1.1002e-01, -1.0830e-02,  3.2019e-02,  8.0738e-02,  2.0601e-02,
         -4.0356e-03,  1.2116e-02,  6.6686e-02, -1.8310e-02,  8.3657e-02,
          7.1123e-02,  5.3839e-02,  3.5382e-03, -4.6337e-02,  2.4031e-02,
         -3.9838e-02, -1.0825e-02,  5.5925e-02, -5.4725e-02, -1.7000e-02,
          1.2163e-01, -5.6631e-03,  7.9491e-05,  5.7695e-02, -1.0213e-02,
         -8.8197e-02,  8.2602e-02,  1.4926e-02, -2.3947e-02,  4.8545e-02,
         -2.3870e-02, -1.1657e-02,  1.0625e-02, -2.1444e-02, -1.0711e-02,
         -6.2592e-02, -1.2322e-02, -6.1906e-02,  6.3803e-03,  7.0897e-02,
          7.1747e-02, -9.5896e-03,  1.2475e-02,  8.4390e-03,  1.0853e-01,
          6.4806e-02,  2.7602e-02,  1.6125e-03, -1.9533e-02,  4.9976e-02,
          8.5255e-03,  2.9800e-02,  7.7369e-02, -7.6745e-02, -6.6295e-02,
          3.6664e-03, -6.1370e-02, -2.2186e-02, -3.2737e-02, -1.5852e-02,
         -1.9941e-02,  8.8795e-02, -1.2120e-03, -8.2786e-02,  8.0039e-02,
         -2.2829e-02,  4.9797e-02, -6.8617e-02,  5.9746e-02,  4.9862e-02,
         -3.0219e-02, -2.4628e-02],
        [-3.0351e-02,  5.1435e-02,  2.0177e-02, -5.1573e-02, -6.2871e-02,
          4.0620e-02, -4.2937e-02,  8.7539e-03,  7.1015e-02,  6.0946e-02,
          1.2472e-02, -6.4289e-02,  8.5886e-03, -4.0714e-02,  3.9706e-02,
          6.8357e-02, -7.0561e-02,  4.0575e-02,  4.0582e-02, -1.4558e-02,
          6.3762e-03,  4.1862e-03, -1.6024e-02, -8.9701e-03, -9.0923e-02,
         -1.9168e-02, -7.9123e-02, -3.4394e-02, -7.0678e-02, -8.6024e-02,
          3.6355e-02, -4.6330e-02, -1.4312e-02, -3.9243e-02,  4.9684e-02,
          1.2215e-02, -3.8794e-02,  7.9147e-03,  2.0143e-02, -1.7149e-02,
          2.1695e-02, -9.3375e-02, -6.5093e-02, -4.6418e-02,  1.1838e-02,
         -8.5856e-02,  3.2554e-03, -9.5360e-03, -8.2807e-02, -5.7997e-03,
          2.0372e-02, -2.6359e-02, -8.0744e-03, -4.2156e-03, -6.1496e-02,
         -2.7943e-02, -7.5166e-02,  2.8137e-02,  4.3880e-02,  4.5690e-02,
          2.5973e-02, -3.7768e-02, -1.4097e-02, -1.7593e-02,  3.0084e-03,
         -5.5516e-03, -8.3972e-02, -6.6723e-02, -8.3578e-02, -1.2975e-02,
          1.8204e-02, -6.3475e-02, -7.8129e-02, -1.6699e-02,  4.9581e-02,
          3.8015e-02, -2.0676e-02, -9.2758e-02, -5.6130e-02, -4.8752e-02,
          5.1887e-02,  2.0587e-02,  3.3913e-02,  1.9406e-02, -4.2757e-02,
         -4.6449e-02,  1.0922e-02,  1.6265e-02, -7.1652e-02, -3.3968e-02,
          2.6291e-02,  5.5199e-02, -6.6476e-02, -6.2605e-03,  5.9934e-02,
         -1.4018e-02,  2.5965e-02, -5.5545e-02, -8.4429e-03, -7.2708e-02,
          1.3429e-02, -2.2676e-02, -3.3274e-02, -9.8496e-03, -6.6086e-02,
          3.3397e-02,  1.8178e-02,  4.9704e-02, -5.6919e-02,  3.3293e-02,
         -5.2555e-02, -3.5021e-02,  1.9725e-02,  4.3197e-02, -2.6666e-02,
          4.3333e-03,  2.7376e-02,  3.7572e-02,  4.1370e-02, -5.4165e-02,
         -2.6842e-02, -4.6523e-02, -2.1971e-02, -5.0689e-02, -3.5767e-02,
          4.3889e-02,  2.6094e-02, -1.2167e-03,  5.0656e-02, -4.4770e-02,
          2.7736e-02, -7.5289e-02, -4.0613e-02, -3.5470e-02,  1.4703e-02,
          3.6186e-02,  3.4805e-02,  9.2409e-03, -1.0712e-02,  3.8173e-02,
          3.6976e-02, -3.8315e-02,  4.4796e-02, -8.5804e-02, -2.1167e-02,
          2.4082e-02,  2.7485e-02, -4.0861e-02, -9.0850e-02, -2.8623e-02,
         -2.3910e-02, -7.7107e-02, -8.4754e-02,  4.3508e-02,  1.1907e-02,
          1.3022e-02, -6.0347e-02,  2.4326e-02, -1.7407e-02, -1.5794e-02,
          4.2723e-02,  2.9102e-02, -3.0235e-02, -6.9890e-02, -5.9541e-02,
         -6.8059e-02, -3.2064e-02, -1.7612e-02, -7.1242e-02, -3.1801e-02,
          6.6613e-02, -5.5299e-02,  7.4869e-03, -2.4652e-03, -5.7280e-02,
          1.7708e-02,  5.0033e-02, -9.2597e-03,  9.3669e-03,  6.2446e-02,
          4.6026e-02,  2.1246e-02,  3.4086e-02, -8.6047e-02, -9.2399e-02,
         -1.9327e-02,  6.6926e-03, -1.4318e-02,  2.1528e-03, -2.2290e-02,
         -6.0002e-02,  2.2755e-02],
        [-5.3782e-02,  4.5188e-02, -3.5397e-02,  1.7233e-02,  5.5980e-02,
         -7.0783e-02, -2.7874e-02,  8.7389e-02, -3.7160e-03,  1.2980e-01,
          3.4741e-02, -3.5235e-02,  9.8190e-02, -6.8737e-02,  3.3988e-02,
          1.3077e-02, -4.6065e-02, -3.9200e-02,  1.5078e-02,  3.6732e-02,
         -2.1660e-02, -1.6220e-02,  4.7026e-02, -5.5521e-02,  4.7385e-02,
          6.5359e-02,  5.0354e-02,  8.8916e-02,  1.5973e-02,  3.1063e-02,
          4.6336e-02, -4.6249e-02,  6.9440e-03,  8.9290e-02,  5.4450e-02,
         -6.2026e-02,  6.9921e-02, -2.8883e-02,  1.6388e-03,  1.6051e-03,
         -4.1177e-02, -9.1250e-02,  8.7271e-02, -2.2970e-02,  7.0069e-03,
         -6.1015e-02,  8.1031e-02, -8.7760e-02,  6.3523e-02,  6.5357e-02,
         -3.0987e-02, -7.6071e-03,  7.9291e-02,  1.3658e-02,  5.7641e-02,
         -3.3661e-02, -1.5730e-02, -2.5532e-02,  3.1637e-02, -1.2953e-02,
         -2.0281e-02, -9.1437e-02,  3.1072e-02,  2.3074e-02,  1.2715e-02,
         -6.6750e-02,  2.8781e-02, -6.9134e-02,  3.9616e-02, -7.8652e-02,
         -9.6867e-02,  7.6745e-02,  5.0628e-03, -4.9898e-02, -4.4989e-02,
          6.2398e-04, -3.6188e-02,  1.2110e-01,  2.3171e-02, -5.3155e-02,
         -3.2378e-02, -2.2185e-02, -7.4960e-02,  6.0259e-02, -3.6478e-03,
         -6.2292e-02, -1.6489e-02, -8.1803e-03, -7.9235e-02,  4.8757e-02,
         -2.3784e-03,  8.7628e-02,  4.5039e-02,  2.0107e-02, -3.0927e-02,
         -2.8466e-02,  5.7067e-02, -3.8814e-02,  5.8401e-02, -1.6490e-02,
          1.5016e-02,  1.2788e-01, -1.9835e-02,  9.6064e-03, -1.0394e-01,
         -2.2260e-03,  2.6112e-02, -3.3751e-02, -3.1837e-02, -3.6296e-02,
          8.8667e-02,  4.4351e-02, -3.0581e-03,  7.2330e-02,  7.9939e-02,
          4.5588e-02, -3.6082e-02, -2.2982e-02, -8.9092e-02,  2.4187e-02,
         -1.0262e-02,  6.0474e-02,  4.9471e-02, -6.1135e-02,  4.8247e-02,
         -1.1321e-01,  5.5366e-02, -8.1224e-02, -9.3385e-03,  2.5950e-02,
          3.7964e-03,  8.9912e-02, -1.8813e-03, -6.8292e-02,  5.9500e-02,
          6.3565e-02, -7.6073e-02,  3.9032e-02,  8.0636e-02, -2.4530e-02,
         -4.0457e-02,  4.3416e-02,  2.7430e-02, -1.1919e-01, -8.6697e-02,
          1.0029e-02,  1.3353e-02, -2.4922e-02,  8.5424e-02, -3.9644e-02,
          2.3250e-02, -6.4650e-02,  3.8153e-02,  1.0057e-02,  1.8121e-02,
          7.0532e-02,  2.8687e-02, -4.1246e-02, -5.9922e-02, -8.8248e-02,
         -5.6041e-02, -2.0583e-02,  5.0657e-02,  9.0770e-03, -1.5765e-02,
         -2.5305e-02,  7.5637e-02, -3.7120e-02,  7.6817e-02, -2.4683e-02,
         -3.5674e-02, -2.9581e-02, -6.9601e-03, -2.5108e-02, -2.0376e-02,
          3.8565e-02, -1.9839e-02, -1.5858e-02,  3.6302e-03, -7.0494e-02,
         -2.9305e-02,  3.7650e-02, -5.0045e-02,  5.4511e-02, -4.3801e-02,
         -2.5172e-02, -9.0758e-02, -3.7385e-02,  3.2833e-02,  3.4806e-02,
          6.5749e-02, -6.1406e-02],
        [-6.1449e-02,  3.7534e-02,  7.9673e-03,  6.3194e-02,  3.3403e-02,
          7.3386e-03,  5.0559e-02, -4.7117e-03, -2.5263e-03, -5.0635e-02,
         -7.7057e-02,  4.7387e-02, -5.7038e-03,  9.6858e-03, -3.8650e-02,
         -7.6799e-02, -6.2025e-02,  4.8961e-02,  4.9888e-02, -1.2217e-02,
         -2.7349e-02, -5.6257e-05,  8.8994e-02, -6.1206e-03, -6.2591e-02,
         -8.0736e-02, -3.6500e-02, -6.3802e-02, -8.7414e-02, -9.5808e-03,
          1.2228e-02, -6.0179e-02, -3.6357e-02, -8.8817e-02,  7.6394e-02,
          3.5293e-02, -8.1008e-02,  5.7468e-02, -2.7525e-02, -8.4013e-02,
          8.8720e-02, -2.9293e-02, -2.9549e-02,  1.8469e-02,  3.6685e-02,
          3.0193e-02,  2.2573e-02,  5.3934e-02, -7.0457e-03, -1.1853e-02,
         -5.8488e-02, -1.0538e-01, -2.0249e-02,  3.1609e-02, -3.6755e-03,
          4.0908e-02,  3.6107e-02, -4.4946e-02, -1.0379e-02, -2.7668e-02,
          8.8990e-02, -2.4583e-02, -3.2078e-03,  4.1979e-03,  6.5827e-02,
          2.6214e-02, -5.3728e-02, -5.4054e-02,  1.2226e-02, -7.2457e-02,
         -7.2562e-02, -7.7498e-02, -2.6386e-02, -5.9938e-02,  1.5377e-02,
         -3.6193e-02,  9.7073e-02, -2.6558e-02, -7.1654e-02,  6.2250e-02,
         -2.4298e-02, -8.0579e-03,  2.7976e-02,  3.4058e-02, -7.4401e-02,
         -4.0688e-02, -8.5077e-02,  6.8841e-02, -2.9451e-03,  1.2446e-02,
         -7.1855e-02,  4.8004e-03, -3.5474e-02,  2.3938e-02, -8.2022e-02,
          8.7780e-02, -2.6993e-02, -3.6053e-02,  4.6121e-02,  5.6247e-03,
         -3.1800e-03, -4.9796e-02, -7.2539e-02, -5.4204e-02,  1.9450e-03,
         -1.2145e-02, -2.0889e-02, -2.2216e-02, -8.2798e-02,  1.1893e-02,
          3.7877e-02, -1.1647e-01,  2.4887e-02, -5.5958e-03, -1.4138e-03,
          1.4695e-02,  2.5687e-02, -5.4202e-02, -1.0991e-01, -8.1547e-02,
         -7.2054e-03, -1.1194e-01,  6.9422e-02, -1.9337e-02, -6.3771e-02,
         -2.0011e-03, -4.9138e-03,  3.9113e-02, -1.0956e-02,  2.4653e-02,
         -1.8780e-03, -3.3923e-02, -7.1114e-02, -3.2095e-02, -1.1345e-01,
         -6.5644e-02,  1.4142e-02, -7.6578e-02, -1.0521e-01, -1.8244e-02,
          6.3570e-03, -6.4878e-02, -7.9201e-02,  1.6908e-02,  2.8126e-02,
         -1.1367e-01, -5.5313e-02, -5.6260e-03, -3.3745e-02, -2.6607e-02,
          2.5252e-02, -1.7709e-02, -1.0151e-02, -1.9438e-02,  1.8268e-02,
          3.0959e-03,  6.5391e-02,  3.8289e-02, -6.7416e-02,  4.2788e-03,
         -1.9063e-02, -7.0979e-02, -1.3543e-03,  4.1561e-02, -3.6237e-02,
         -2.3774e-02,  4.0035e-02, -8.6811e-02,  5.5300e-02, -3.5749e-02,
          4.6747e-02,  5.0599e-02,  6.8677e-02, -7.1789e-02,  3.4381e-02,
         -7.0176e-02, -4.4561e-02,  2.4311e-02, -1.6180e-02, -3.7800e-02,
         -2.2973e-02, -8.2408e-02, -4.9094e-02, -4.5349e-02, -1.1797e-03,
         -5.6439e-02,  2.2458e-02, -4.6007e-02, -4.5678e-03, -8.8709e-02,
          1.5142e-02, -9.5553e-02],
        [-5.0654e-02, -7.0651e-03, -4.9301e-02, -4.7520e-02,  6.7490e-02,
          1.6314e-02, -6.6935e-02,  6.3424e-02, -1.8647e-02,  3.7788e-02,
          9.8494e-02,  2.0110e-02,  3.2934e-02,  5.6649e-02,  3.0204e-02,
         -1.0533e-01, -6.5621e-02,  5.5170e-02,  1.1771e-02,  4.9157e-02,
          3.6650e-02, -1.2770e-02, -1.6663e-02,  3.7887e-02,  3.7580e-02,
          2.2098e-03,  2.5105e-02,  7.5265e-02, -7.9177e-03, -8.5656e-02,
          4.6919e-02,  1.2061e-02,  2.5071e-03,  5.3070e-02,  3.8231e-02,
          2.8372e-02,  1.0357e-02, -5.6281e-02, -5.6319e-03, -8.1005e-02,
         -2.8830e-02, -4.3309e-02, -5.6238e-02, -5.2702e-02, -5.6154e-04,
         -6.2473e-02,  8.9848e-02,  1.1581e-02,  7.1296e-02,  5.4564e-02,
         -2.1368e-02,  2.2475e-02,  6.1367e-02,  6.8409e-03,  1.2821e-02,
         -1.9165e-02, -2.6674e-02,  1.3573e-03,  7.4676e-03, -6.8620e-02,
          5.7951e-02, -1.2427e-02,  6.6813e-02,  5.0702e-02, -5.5857e-02,
         -2.2559e-02, -4.1568e-02,  4.2999e-02, -3.0294e-04, -5.2146e-02,
         -6.4966e-02,  2.5621e-02,  4.6350e-02,  3.1013e-02,  6.1342e-02,
          6.1964e-02, -4.1129e-02,  6.6264e-02,  4.2163e-02,  1.4199e-02,
         -2.8404e-02,  1.0180e-02,  3.7862e-02, -5.0888e-02, -1.9416e-02,
         -4.8714e-03,  6.3881e-02, -2.3914e-02, -6.5513e-02,  6.1705e-02,
         -1.9360e-02,  5.5891e-02,  5.7583e-02, -1.5838e-02,  7.1867e-02,
         -2.1357e-02, -3.2054e-02, -4.2142e-02,  1.4570e-02, -4.1456e-02,
          2.2621e-02,  4.9442e-03, -5.5801e-02, -4.9533e-02, -4.8135e-02,
          5.4978e-02, -1.4680e-03, -2.3743e-02,  2.6336e-02,  4.0473e-02,
          7.4142e-02, -3.4889e-02,  5.5827e-02,  3.2883e-02,  4.3648e-02,
          2.0652e-02, -3.7327e-02,  2.3877e-02, -3.6514e-02, -1.5476e-02,
          6.3639e-02,  9.2237e-02, -4.2547e-03, -8.9101e-02, -4.4548e-02,
          2.6607e-02,  7.3848e-02,  7.2761e-02,  4.8441e-02, -9.6260e-02,
          1.2072e-02, -2.1963e-02, -7.9480e-02,  3.5514e-02,  5.4520e-02,
          7.2650e-02,  8.8572e-03, -3.7636e-02,  3.8512e-02,  4.0383e-02,
         -4.9251e-02,  1.3419e-02,  7.5583e-02, -4.9743e-02,  3.6550e-02,
          4.3392e-02,  7.0377e-03, -1.5880e-02, -1.0019e-02, -1.3066e-02,
         -1.3272e-02,  1.7763e-02, -2.1156e-03, -1.1214e-02,  2.9410e-03,
         -1.5218e-02,  1.9633e-02,  3.1650e-02, -1.2009e-02,  6.5531e-02,
          2.8074e-02, -3.5878e-02,  4.1123e-02,  4.8455e-02, -2.4868e-02,
         -5.0124e-02,  6.8930e-02, -9.7041e-02, -6.7545e-02, -1.8518e-02,
          4.0759e-02, -2.3266e-02, -4.2309e-02, -4.4479e-02, -7.0320e-02,
         -5.9115e-02,  1.7578e-02, -5.2381e-02,  6.1436e-02, -4.3548e-02,
          3.6884e-02,  2.0961e-02,  3.2764e-02,  3.4384e-03, -8.9199e-02,
          7.3070e-02, -7.9036e-02,  6.9521e-02, -7.0486e-02,  4.1602e-02,
         -2.1762e-02, -8.0286e-02],
        [ 6.5766e-02, -4.0382e-02, -4.5523e-03, -5.6833e-02, -6.7091e-02,
          8.7790e-03,  5.9596e-02,  8.8660e-02, -5.2915e-04,  5.8709e-02,
         -2.8963e-02, -3.9242e-02,  9.0796e-02, -5.7343e-02,  2.5919e-03,
         -8.8710e-02, -3.2853e-02,  8.0465e-02, -5.1649e-02,  6.1867e-02,
          6.8336e-02, -4.8948e-02,  3.7865e-02, -3.8021e-02, -5.6359e-02,
          3.0637e-02,  8.1484e-02,  7.8950e-02,  6.1183e-02,  7.6388e-02,
         -4.6548e-03,  6.7077e-02,  2.9759e-02,  5.3369e-02,  5.2814e-02,
         -2.9071e-02, -1.2014e-03, -2.4203e-02,  6.5029e-02, -2.2302e-03,
          2.2749e-02, -5.5836e-02, -1.4249e-02, -5.0970e-03, -1.6468e-02,
          4.3857e-02,  2.6622e-02, -3.4322e-02,  6.4873e-02,  3.0886e-02,
         -3.3102e-02, -1.3448e-02,  9.8531e-03,  3.8006e-02,  6.9124e-02,
          2.6785e-02,  2.9089e-02,  4.0428e-02, -8.9845e-04,  7.5330e-04,
          2.6561e-02,  6.0232e-02, -3.4847e-04,  3.6723e-02,  9.5235e-02,
         -1.1964e-02,  9.5258e-03,  5.9830e-02,  5.6365e-02, -7.7946e-02,
         -7.6198e-02,  7.6793e-02,  9.8100e-02,  1.0945e-01,  2.5281e-02,
          8.0099e-02,  7.5053e-02,  1.3137e-01,  4.6363e-02,  7.1350e-03,
         -1.8202e-02, -5.1353e-02, -2.4619e-02, -7.5940e-02,  4.3576e-02,
         -4.6636e-02,  2.9626e-02, -3.7844e-02, -6.8448e-02,  5.5047e-02,
         -3.6952e-02,  3.2976e-02, -1.7051e-03, -4.0311e-03, -6.0484e-03,
          5.3639e-02, -3.5092e-02, -5.8593e-02,  4.7955e-02,  2.9174e-02,
          8.5196e-02,  1.0860e-01,  5.8225e-04, -5.8422e-02, -4.7963e-02,
          4.5864e-02,  1.3288e-01,  3.1171e-02,  7.0895e-02, -1.5687e-02,
          6.7613e-02,  2.2951e-03,  3.7457e-02, -4.4735e-02,  3.4918e-02,
         -4.1499e-03,  1.1045e-02,  1.2946e-02,  2.3199e-02, -2.7936e-02,
          5.2614e-02,  4.2034e-02,  6.4775e-02, -3.7522e-02, -5.7249e-03,
         -2.0469e-02,  6.6910e-02,  5.2517e-03, -2.8243e-02, -2.8246e-02,
          1.6357e-02,  4.7552e-02,  2.4930e-02,  7.1347e-02, -9.6156e-03,
         -1.5955e-02, -7.6453e-03, -3.6445e-03, -6.1267e-03, -3.5389e-03,
          2.6262e-02, -1.9206e-02, -1.6993e-02, -4.1536e-02,  1.5136e-02,
         -4.7061e-02,  8.1515e-03,  5.0381e-02,  8.1821e-02, -1.2306e-02,
          4.1676e-02, -6.6246e-02,  4.9444e-02,  1.8653e-02,  2.8725e-02,
          4.6512e-03,  1.2268e-02, -1.2119e-02, -5.7087e-02,  3.9876e-02,
          1.7890e-02, -3.8040e-02,  8.4583e-02,  5.1554e-02,  3.7370e-02,
         -2.6188e-02, -2.8320e-03, -6.7354e-02, -4.0826e-02,  9.7560e-02,
         -1.2282e-02,  8.8927e-02,  7.9896e-03,  1.8423e-02,  3.2058e-02,
          5.8681e-02, -9.5840e-04,  1.2024e-02, -1.4085e-02, -6.4116e-02,
         -5.2742e-02, -2.6360e-02,  4.9509e-02, -1.2827e-02,  3.9876e-02,
          7.2802e-02,  1.1630e-02, -3.7836e-02, -5.8099e-02,  7.0883e-02,
          9.0209e-02,  3.5255e-02],
        [ 6.0261e-04,  6.7021e-02, -4.9411e-02,  6.4897e-02,  1.5680e-02,
         -1.4566e-02, -1.0746e-02,  7.0474e-02, -8.4806e-02,  2.0628e-02,
          5.3086e-02, -7.9240e-02, -7.5629e-03, -8.8496e-02,  6.3582e-02,
         -8.6260e-02, -3.6844e-04,  1.0797e-02,  5.6837e-02, -1.7800e-02,
          6.0021e-02, -3.2177e-03, -6.4878e-02, -4.7049e-02, -7.4103e-03,
          7.5192e-02, -3.0489e-02,  8.5291e-02, -6.5546e-02,  2.2383e-02,
         -2.1945e-02,  3.3840e-02,  6.1894e-02, -3.0245e-02, -7.4914e-02,
         -3.7400e-02,  5.1555e-03, -4.7588e-02,  3.7239e-02, -6.1512e-02,
         -8.9637e-02, -8.0326e-02,  3.0397e-02, -3.9681e-02,  6.2329e-02,
         -7.5740e-02,  9.4707e-02, -7.0544e-02, -7.0892e-02,  4.7807e-02,
         -4.5080e-03,  8.7193e-02,  1.1424e-03,  3.5761e-02,  7.1596e-02,
          6.3532e-02, -9.8947e-03,  2.5478e-02,  8.0371e-03, -9.3223e-02,
          7.7525e-02, -8.7248e-02,  1.0765e-02,  1.4220e-02, -9.1463e-02,
         -7.3851e-02,  2.3497e-03, -5.2270e-02,  9.2119e-02,  1.1059e-02,
         -8.9965e-02, -4.5497e-02,  4.9308e-02,  3.0306e-02, -2.7061e-02,
          5.1660e-02,  5.3907e-02,  5.5303e-02, -7.1487e-02, -3.2061e-02,
         -6.9600e-02,  1.6299e-02,  4.0097e-02,  5.0081e-02, -1.0471e-02,
          2.6344e-02, -2.2145e-02, -5.8146e-02, -3.1197e-02,  7.8857e-02,
         -2.0332e-02,  7.9160e-02,  5.1280e-02,  8.2089e-02, -2.1773e-02,
         -3.2990e-02,  6.5376e-03,  2.2009e-02, -5.4543e-02, -3.8507e-02,
          4.5501e-02,  1.0128e-01,  4.9416e-02, -7.0698e-02,  2.6699e-02,
          2.1360e-02,  4.8956e-02, -1.3880e-02, -3.4608e-02,  2.9525e-03,
          2.2388e-02, -5.6933e-02, -3.9865e-02, -2.9140e-02, -9.6218e-03,
         -5.3988e-03,  9.0446e-02,  2.8050e-02,  1.6662e-02, -3.2356e-02,
         -8.8435e-03,  2.8420e-02,  4.8496e-02,  2.9903e-02,  1.0406e-01,
         -1.0221e-01,  2.1386e-02,  2.2487e-02, -3.4497e-03,  2.6004e-02,
         -8.8543e-02,  2.0701e-02, -6.7447e-02,  2.7460e-02,  3.9880e-02,
          8.0584e-02, -3.7283e-03, -3.0435e-02,  7.2390e-02,  4.1522e-02,
         -1.1078e-01,  3.2515e-02,  8.0349e-03, -1.1152e-01, -4.0770e-02,
          4.8668e-02, -4.5013e-02,  2.9328e-02, -5.6407e-04, -3.9301e-02,
         -6.6851e-02,  2.9187e-02,  5.6790e-02,  4.4434e-02, -4.9538e-02,
          6.0628e-02, -6.5638e-03,  3.3443e-02,  4.1150e-02, -1.1709e-02,
          6.9775e-02, -3.7782e-02,  8.4132e-02,  6.6325e-02, -3.8829e-02,
          5.9416e-02,  6.4985e-02, -5.3125e-03,  8.9236e-02,  9.2230e-02,
          5.7294e-02,  2.6489e-02, -1.7878e-02,  7.6520e-02, -2.7548e-02,
         -2.0107e-02, -6.2666e-02,  9.4004e-02,  2.0073e-02,  4.3418e-02,
         -1.6130e-02, -2.5449e-02, -1.3654e-02,  1.0536e-01,  1.1457e-02,
         -2.6441e-02, -5.0586e-02,  1.9121e-05, -7.2497e-02, -9.3512e-03,
          3.3379e-02, -2.9108e-02],
        [-3.5108e-02,  2.5371e-02,  5.4348e-02, -3.5412e-02, -5.5849e-02,
          1.9462e-02, -4.3615e-02, -6.9024e-02, -4.4542e-02,  4.5249e-02,
         -7.0419e-02, -5.6010e-03,  5.0779e-03,  1.3602e-02,  2.6445e-02,
         -6.8640e-02,  4.7810e-02, -3.9531e-02,  7.1528e-02, -3.0461e-02,
          1.9821e-02,  3.8890e-02,  3.5788e-02,  1.8707e-02, -3.8717e-02,
         -6.4460e-02, -6.0554e-02, -6.9189e-02, -1.2090e-02,  4.8447e-02,
          4.9116e-02, -7.0249e-03,  2.4571e-03,  1.0956e-02,  6.8771e-04,
         -2.3179e-02, -4.8546e-02,  1.2347e-02, -3.2439e-02, -4.8294e-02,
         -3.2819e-02, -6.4830e-02, -1.0780e-05, -4.8636e-02,  3.8009e-02,
         -3.2520e-02, -6.2865e-02,  2.5472e-03, -3.8749e-02,  6.9960e-02,
          9.6432e-04,  3.0252e-03,  4.5685e-02,  7.1790e-02, -3.8087e-02,
          2.7651e-02, -4.0680e-02, -1.6929e-03,  2.0705e-02, -1.3637e-02,
          3.0292e-02, -7.2969e-03,  7.0095e-02,  2.1107e-02, -6.2910e-02,
         -2.2028e-02, -7.1001e-02, -2.4827e-02, -5.7985e-02, -2.5533e-02,
         -6.3189e-02,  2.7591e-02,  3.6683e-02,  1.8020e-02, -6.8842e-02,
         -7.1801e-02, -6.9860e-02,  2.8680e-02, -4.1276e-02,  5.3778e-02,
         -6.0514e-02, -8.6520e-03, -5.9947e-02, -6.4616e-02, -4.6726e-02,
         -4.1335e-02,  4.2892e-02, -2.5249e-02,  9.4635e-03, -6.1696e-02,
          4.5056e-02,  6.2735e-02, -2.8321e-02,  4.4675e-03,  4.1368e-02,
         -3.8552e-02, -2.3795e-02, -5.4782e-02,  2.8225e-02, -5.8756e-02,
         -1.0447e-03,  3.2441e-02, -6.6654e-02, -3.1177e-02, -1.0393e-02,
          2.4553e-02, -3.5109e-02, -5.5083e-02,  1.5981e-02, -3.9873e-02,
         -3.5197e-02, -2.2398e-02, -4.9997e-02, -6.7622e-02,  1.6924e-02,
          4.9415e-02,  8.7215e-03, -1.8861e-02, -3.6522e-02, -4.7129e-02,
         -2.6247e-02, -6.2706e-02,  3.3687e-02, -4.9353e-02,  6.2902e-02,
          8.2489e-03,  6.3043e-02, -1.8814e-02, -4.9720e-02,  1.0497e-02,
         -2.1448e-02,  5.4599e-03,  6.8973e-02, -6.2580e-02,  3.0478e-02,
         -5.1361e-02,  5.8873e-02,  4.7660e-02, -3.2312e-02, -4.0742e-02,
         -3.3261e-02,  3.7081e-02, -2.5619e-02,  2.0033e-02, -5.8405e-02,
         -6.0118e-02, -1.5183e-02,  1.3761e-02,  6.6606e-02, -7.6816e-03,
         -2.3767e-02, -6.9853e-02, -8.8967e-04,  1.1015e-02, -2.9154e-02,
          1.9183e-02,  6.0926e-03,  4.0446e-02,  9.5587e-03,  5.5586e-02,
         -2.7929e-02, -2.6009e-03,  3.7515e-02,  3.4839e-02, -4.3230e-02,
          1.8689e-03, -5.6892e-02, -5.1344e-02,  3.9940e-02,  2.9026e-02,
         -2.9168e-02, -2.5059e-02,  4.4782e-03, -2.2523e-02,  4.6105e-02,
          3.9751e-02, -3.0099e-02, -1.7214e-02,  4.4967e-03,  2.6760e-03,
         -5.8968e-02,  3.5444e-02, -5.5057e-02, -7.2155e-02, -4.8916e-02,
          4.7862e-02,  6.9061e-03, -5.2329e-02, -2.4650e-02, -3.6874e-02,
          1.9255e-02,  1.9562e-02],
        [-2.6127e-02, -9.4225e-03,  6.0978e-02, -1.5702e-02, -3.0941e-02,
         -1.8100e-02,  8.9102e-03, -2.4378e-02, -4.0888e-02, -5.0747e-03,
          3.9165e-02,  3.1227e-03, -8.9864e-02,  9.3218e-02, -1.0885e-01,
          2.6231e-02, -3.9682e-02,  6.7942e-03,  2.8372e-02,  1.5630e-02,
          5.3968e-02,  3.4961e-02,  1.2049e-02, -2.5725e-02,  8.5143e-02,
          1.6679e-02, -6.1106e-02, -1.9365e-02,  8.2562e-02, -1.8852e-02,
          1.8744e-02, -1.8506e-02,  2.3712e-02,  3.7015e-02, -1.9441e-02,
          8.7166e-02, -6.9575e-02,  3.0459e-02,  1.2759e-03,  6.3120e-03,
          1.7554e-02,  1.1775e-01, -1.3233e-02, -9.5181e-03, -4.8319e-02,
         -1.0843e-02, -7.8466e-02,  2.2956e-02, -2.2869e-02,  7.9115e-02,
          8.1120e-02, -4.4579e-02,  2.5424e-02,  5.7982e-02,  5.9360e-02,
         -3.2952e-02,  5.1353e-02, -7.9484e-02,  3.9863e-02,  9.7235e-03,
          1.0784e-02,  9.5479e-02,  2.7568e-04,  3.3101e-02,  7.4441e-02,
          9.5937e-02, -3.6057e-02,  5.6980e-02, -1.2136e-02,  1.3163e-03,
          9.2854e-02, -5.9115e-02,  9.7428e-03, -1.2493e-02, -1.3336e-02,
          3.6763e-02, -4.0604e-02, -5.5940e-02,  8.9871e-02, -3.3351e-02,
         -5.3420e-02,  5.5604e-03, -5.4827e-03, -2.6364e-02,  2.3697e-02,
          8.0994e-02, -1.9117e-02, -4.1563e-02,  8.0545e-02, -4.9609e-02,
          3.5795e-02, -5.5011e-02,  1.9337e-02,  7.1821e-02, -9.8124e-02,
          2.1161e-02,  7.9001e-02, -2.9613e-02,  9.1678e-02,  6.7730e-02,
         -9.4341e-02,  1.4521e-02,  9.3134e-02,  7.0497e-02,  6.3339e-02,
          2.9330e-02, -9.8350e-02,  5.4740e-02,  4.2492e-02,  2.4038e-02,
         -4.8858e-02,  6.1609e-02, -4.1546e-03,  4.8467e-02, -4.9332e-02,
          8.8604e-02, -5.6819e-02,  2.2664e-02,  9.3789e-02,  4.9768e-03,
         -1.7591e-02,  2.8719e-02,  1.0313e-02,  6.6237e-02, -7.1694e-02,
          6.4266e-02, -1.0632e-02, -2.9628e-02, -4.0158e-02,  7.7569e-02,
         -2.5231e-02, -5.9086e-02, -3.4021e-02,  3.5973e-02,  3.1936e-03,
         -9.2612e-02,  3.4857e-04,  4.8226e-02, -1.0231e-02, -3.1337e-02,
          7.8995e-02, -3.3471e-02, -8.2341e-02,  1.1323e-01, -5.7405e-03,
          3.8085e-02,  1.0570e-01, -2.6197e-02, -1.1283e-01,  4.8417e-03,
          2.1663e-02,  8.2177e-02,  4.3676e-02,  6.2926e-02, -6.5282e-03,
          2.2473e-02,  1.8791e-02, -4.7056e-02, -3.8655e-02,  4.2619e-02,
         -7.2756e-03,  3.0662e-02,  4.3602e-02, -2.5116e-02,  1.0231e-01,
          1.0086e-01,  2.5805e-02,  9.9930e-02, -5.0389e-02, -8.7546e-02,
         -5.1964e-02, -3.9472e-02,  3.0818e-02,  3.7270e-04,  2.9455e-02,
          6.8862e-02,  6.5087e-02,  5.0586e-02, -3.6245e-02,  2.4632e-02,
         -4.1496e-02,  7.5471e-02,  2.7853e-02, -5.6519e-02,  4.9389e-02,
         -5.0576e-02,  1.0761e-01, -6.1252e-02,  3.6098e-03, -3.7386e-02,
         -6.2619e-03,  5.5604e-02],
        [-7.0477e-02,  5.4925e-02,  4.6317e-02, -5.6362e-02, -4.9808e-02,
          5.9625e-02,  2.7664e-02, -1.4705e-02,  5.1979e-02, -6.6017e-02,
          2.9765e-02, -1.7024e-02, -7.5327e-02,  4.2374e-02,  1.5484e-02,
          1.2365e-02,  1.5965e-02, -7.9201e-03, -5.7389e-02, -3.7278e-02,
          1.1420e-02,  6.6767e-02,  8.6444e-02,  6.5244e-04, -2.6104e-02,
          7.4197e-02, -4.1871e-02,  3.3353e-02,  2.1384e-02,  7.4806e-02,
          1.0544e-01,  2.2625e-02,  3.4351e-02,  4.6736e-02,  8.7042e-02,
         -7.2059e-02,  1.8097e-02,  1.1387e-01,  4.2861e-03,  4.5226e-02,
         -1.9117e-02,  1.1082e-02, -5.1189e-02,  3.6447e-02, -2.7329e-02,
          6.7102e-02, -6.9122e-02,  1.4076e-01,  8.2226e-03,  8.3504e-02,
          4.6797e-03, -4.6620e-02,  3.3189e-02, -3.0284e-02,  3.3106e-02,
          2.5144e-02, -3.1196e-02,  4.6784e-02,  9.7284e-02,  8.5912e-02,
          3.4691e-02, -3.2458e-03,  5.8553e-02,  2.2662e-02, -2.1570e-02,
          2.6314e-02, -7.1654e-02,  3.6897e-02, -1.2196e-02,  1.2583e-01,
          7.2744e-02,  6.0923e-02, -6.4850e-02,  3.7227e-02,  2.9300e-02,
         -5.2136e-02,  1.6466e-02,  2.3690e-02,  5.9885e-02,  5.9905e-03,
         -3.8505e-02, -2.5830e-02, -1.4093e-02,  4.3784e-02, -6.5680e-02,
          4.4840e-03, -1.0764e-02,  1.4120e-02,  1.6843e-02, -8.0367e-02,
         -4.1401e-02,  1.9523e-03,  5.6467e-02, -1.1406e-02, -6.5818e-02,
          7.6712e-02,  8.6861e-02,  1.2729e-02,  1.6969e-02,  9.1570e-03,
         -1.2264e-02, -1.1560e-03,  2.7750e-02,  7.5421e-02,  6.4782e-02,
         -5.7550e-02, -5.6003e-02,  8.3726e-03,  4.7710e-03, -4.2828e-02,
          5.2912e-02, -1.0485e-02,  4.3069e-02,  5.6153e-02,  4.6030e-02,
         -3.4058e-02,  3.0591e-02,  5.8163e-02,  8.8077e-02,  8.2726e-02,
         -4.8708e-02, -1.0675e-01,  2.7698e-02,  7.5647e-02, -8.2353e-02,
          3.1681e-02,  1.0410e-02, -1.8714e-02, -1.8931e-02,  1.0150e-01,
          1.0208e-01,  5.3743e-02, -5.1421e-02,  3.5043e-02, -4.6862e-02,
         -3.3203e-02,  8.0516e-02,  8.7529e-02, -7.6225e-03, -1.4827e-02,
          3.4623e-02,  1.1073e-02, -8.4352e-02, -8.4720e-03, -1.2788e-02,
          3.8789e-02,  7.9268e-03,  1.3648e-02, -1.1748e-01,  1.8817e-02,
          7.6668e-03,  2.4545e-02,  5.5584e-03, -8.1798e-02,  3.7744e-02,
         -6.7843e-02,  5.4033e-02, -1.1254e-02,  2.2703e-02,  8.3108e-02,
          1.8950e-02,  4.0620e-02,  2.2811e-02, -6.0214e-02,  1.6253e-02,
         -5.1308e-04, -2.2100e-02,  6.3331e-02,  1.4165e-03,  2.7194e-03,
          9.1189e-03, -1.5160e-02,  1.0119e-01, -4.9471e-02, -1.7381e-02,
         -1.7585e-02, -2.5459e-02, -7.5117e-02,  6.8339e-02, -5.4479e-02,
          6.2753e-02,  4.3326e-03,  6.2274e-02, -5.8151e-02,  1.1062e-01,
         -8.1353e-02,  2.0531e-03,  1.0460e-02,  3.4374e-03,  7.3859e-02,
         -1.5747e-02,  8.4102e-02],
        [-4.6417e-02, -1.8373e-02, -1.4949e-02,  8.3532e-03,  1.3770e-02,
          3.1801e-02,  9.4249e-02,  8.3507e-03, -1.6067e-02, -9.3731e-02,
         -5.6412e-02,  2.6598e-02,  4.9078e-02, -1.7033e-02, -6.8938e-02,
          1.2496e-02,  2.8305e-02,  8.5392e-02, -6.0414e-02, -4.0329e-02,
         -5.3632e-02,  2.9530e-02,  9.0318e-02, -3.8028e-02,  7.7027e-02,
          1.0459e-01, -2.8483e-02,  1.3651e-02, -4.8027e-03,  2.7044e-02,
          1.6751e-03,  1.0878e-02,  6.8958e-02, -5.7492e-03, -4.0270e-02,
          6.3289e-02,  4.2724e-02,  1.2682e-02,  2.3738e-02, -7.9246e-03,
         -1.6335e-02,  7.2266e-02, -7.0481e-03,  7.7166e-02,  1.2618e-02,
          9.4410e-02, -6.1628e-02,  7.1163e-02,  9.8125e-02, -3.4562e-02,
          6.8188e-02,  1.4893e-02,  4.4296e-02, -8.3668e-02, -6.6533e-02,
          8.0595e-02,  7.0973e-02, -7.5734e-02,  5.2108e-03,  2.5814e-02,
          7.5076e-02, -3.5757e-02, -1.0156e-02, -3.8381e-02,  2.4417e-02,
          1.7242e-02, -7.2045e-02,  7.3756e-02, -1.0170e-01,  1.2476e-01,
         -9.8034e-03,  6.4963e-02, -2.2788e-02, -4.5621e-03,  6.7354e-02,
         -7.7599e-02,  7.7658e-03,  3.8849e-02,  8.5259e-02,  5.7499e-02,
         -1.2962e-02,  8.5113e-02,  2.9655e-02,  1.2898e-02,  6.5530e-03,
          9.0237e-02, -3.6022e-02,  1.8959e-02,  2.1787e-02,  2.0550e-02,
          2.3132e-02, -5.3391e-02,  6.4624e-02,  7.2876e-02, -4.3696e-03,
          9.2581e-02,  1.1089e-01, -6.4540e-03,  7.2472e-02,  4.6860e-02,
         -9.1766e-02,  6.6279e-03,  6.7921e-03,  8.2197e-02,  9.1807e-02,
          4.3470e-02, -3.1701e-03,  7.6593e-04,  7.4708e-02,  2.0301e-02,
         -3.3281e-02, -1.7823e-02, -8.1307e-03,  6.2557e-02, -2.5710e-02,
         -2.3088e-02, -6.7727e-02,  9.5971e-02,  8.5530e-02,  7.5177e-02,
          1.9778e-02,  2.5260e-02,  2.3828e-02,  1.1358e-01, -4.0469e-02,
          8.6597e-02, -8.3138e-02, -4.3969e-02, -4.6053e-03,  5.2150e-02,
          4.7378e-02, -2.0134e-02,  8.3584e-03,  7.4988e-02, -7.8446e-02,
         -4.0891e-03,  3.8123e-02,  5.6147e-03,  2.2884e-02, -7.6401e-02,
          3.1040e-02,  5.3596e-03, -4.7532e-02,  9.6691e-02,  5.5429e-02,
         -7.2159e-02, -1.1024e-02,  4.0281e-03, -9.2602e-02, -3.1351e-02,
          1.9672e-02,  1.0924e-01,  8.2682e-02,  4.4279e-06,  4.7516e-02,
         -6.1644e-02,  3.7666e-02, -4.0442e-02,  2.5258e-02,  1.1504e-02,
          4.6065e-02, -2.0650e-02, -8.4311e-03, -7.3422e-02, -1.6912e-02,
         -3.1532e-03, -7.0080e-02,  9.0091e-02, -2.4585e-02,  9.6080e-03,
         -3.1243e-02,  6.8971e-02,  1.1518e-02, -5.2420e-02,  1.8686e-02,
          2.8024e-02, -7.0500e-02, -6.4575e-03, -4.6732e-02,  3.3222e-02,
         -2.1043e-02,  6.1173e-02,  6.9284e-02, -7.8560e-02,  7.3285e-02,
          1.0083e-02,  1.1196e-01, -1.3740e-02,  4.8243e-02, -8.2265e-03,
         -5.2601e-02,  8.0291e-02],
        [-3.9830e-02, -5.6865e-03, -3.9251e-02, -4.4069e-02,  2.0616e-02,
          2.9266e-03, -3.1500e-02,  2.5736e-02, -2.8038e-02, -2.1129e-02,
          1.5131e-02, -6.3795e-02, -2.3273e-02, -4.4929e-03, -9.3093e-02,
          1.7076e-02, -2.5209e-02, -2.4084e-03, -4.4464e-03,  4.5960e-02,
         -4.9242e-02,  1.1660e-02,  5.3329e-02, -6.3737e-02, -7.0060e-02,
         -1.7317e-02, -8.7072e-02,  2.0291e-02, -1.3341e-02,  5.7172e-02,
          4.2609e-02,  1.6443e-03,  1.1176e-01,  4.1511e-02, -2.6402e-02,
         -1.0109e-02,  4.4545e-02, -2.5928e-02, -1.6729e-02,  6.2863e-03,
         -4.4886e-02,  4.0633e-02, -7.4818e-02, -3.8092e-02,  9.8884e-03,
         -1.5718e-02, -7.7097e-02,  2.3044e-02, -3.1922e-02,  6.9820e-03,
          4.4107e-02, -5.0124e-02, -2.8991e-02, -2.7007e-02, -1.4622e-02,
         -4.3469e-02, -7.5187e-02, -1.9057e-02,  3.1625e-02,  1.1593e-01,
         -4.0535e-02,  7.0086e-02, -5.3969e-02, -2.9571e-02,  2.0442e-02,
         -3.7886e-02,  2.0543e-02,  4.6226e-02, -6.8966e-02,  2.0508e-02,
          5.9274e-02,  4.6692e-02, -2.5683e-03,  5.0128e-02,  4.8528e-02,
          2.2771e-02, -4.6031e-02, -9.5009e-02,  4.8193e-02, -5.5253e-02,
          7.0774e-02, -5.5553e-02, -5.9477e-02, -9.0676e-02, -4.7587e-02,
         -5.5866e-02, -7.8265e-02, -8.2869e-02, -8.4533e-02, -7.0609e-02,
          2.5192e-02, -6.8247e-02, -1.0946e-02, -1.7862e-02,  4.4544e-02,
          6.0909e-02, -3.2976e-02,  4.2264e-02,  1.5626e-02,  8.7853e-03,
         -8.3567e-02, -6.1739e-02,  7.3539e-02, -4.8987e-03,  1.9996e-02,
          2.9619e-02, -9.5478e-02, -8.6714e-03, -5.8400e-03, -6.2830e-02,
          6.5912e-03,  1.4433e-02, -9.4773e-02,  8.2392e-02, -7.0186e-02,
          8.7702e-02, -8.0022e-02, -3.3565e-02, -2.6563e-02, -1.8826e-02,
          7.7496e-03, -6.5397e-02,  2.7700e-02,  1.5612e-02,  1.6233e-02,
          6.4098e-03, -2.0175e-02,  9.0376e-03, -7.0153e-02, -3.0928e-03,
          7.2412e-02, -1.9889e-02, -5.5592e-03, -1.4094e-02, -2.1454e-02,
          1.3972e-02,  2.7968e-03,  1.1574e-01, -2.8210e-02, -4.5162e-02,
          1.1916e-03,  6.5831e-03, -7.5153e-02, -9.3214e-02, -5.7445e-02,
         -4.1403e-03,  1.7124e-02, -8.8281e-02, -2.5416e-02, -7.1719e-02,
         -5.4193e-02,  3.5047e-02,  3.5167e-02,  2.5841e-02,  2.9013e-02,
         -3.1870e-02, -6.4574e-03,  2.9444e-02, -5.1456e-02, -5.2931e-03,
         -6.2273e-02,  4.4518e-02, -6.6181e-02, -5.7718e-02,  7.7467e-02,
          3.1157e-02,  2.9642e-03, -5.2061e-03,  2.5970e-02, -1.9063e-02,
         -2.2073e-02, -2.0015e-02, -3.0226e-02, -6.8289e-02, -5.7065e-02,
         -9.3980e-03,  1.6473e-02, -3.1625e-02,  2.3150e-02, -5.1757e-02,
         -6.9825e-03, -2.2826e-02, -2.6174e-03, -7.3855e-02, -3.6302e-02,
         -9.4107e-02, -4.5906e-02, -6.0073e-02,  8.7275e-02, -2.5931e-02,
         -3.3916e-02, -3.2961e-02],
        [ 1.8463e-02, -8.3674e-03, -5.3592e-02,  3.1306e-02, -7.0265e-02,
          3.7631e-02, -8.2913e-02,  4.4424e-02,  6.3180e-02,  6.6145e-02,
          1.8538e-02, -2.4310e-02,  1.0185e-01, -5.3958e-02,  2.8840e-02,
         -5.3101e-02,  1.5894e-02,  1.3509e-02, -3.0754e-02,  4.0691e-02,
          1.0699e-02, -4.1246e-02, -4.9696e-02, -1.0945e-02,  4.7114e-02,
          4.2503e-02,  2.8899e-02,  7.8711e-02, -3.1686e-02, -5.2520e-02,
         -4.6916e-02, -6.8842e-02, -2.2666e-02, -1.0883e-02,  7.8098e-02,
          8.8751e-02,  7.5552e-02, -1.2901e-02, -5.2355e-02, -8.2568e-02,
         -7.0581e-03, -5.6863e-02, -3.0194e-02,  6.2555e-02,  2.4703e-03,
         -9.3499e-04,  3.4382e-02, -3.2273e-02, -4.8133e-02, -3.2371e-02,
         -2.1460e-02,  3.3931e-02, -2.7037e-02, -3.1716e-02,  6.2547e-02,
          6.2963e-02, -4.7540e-02,  9.4148e-02, -3.2857e-02, -4.3035e-03,
          6.0344e-02,  5.2165e-02,  5.5762e-02, -3.1781e-02, -6.5742e-03,
          5.5107e-03, -2.6987e-02,  9.3292e-02, -1.4837e-02, -6.3795e-02,
         -6.7650e-02, -3.9434e-02,  1.4916e-02, -7.3557e-03,  2.6957e-02,
         -1.3132e-02,  4.8044e-02,  5.1961e-02, -4.2556e-03, -7.3448e-02,
          4.4792e-02,  2.8491e-02, -7.5570e-02, -4.5346e-02, -9.8533e-03,
         -1.9892e-02,  7.8231e-02,  3.0664e-02, -5.1165e-02,  4.1035e-02,
         -4.9339e-02, -7.7708e-03,  4.9876e-02,  6.3771e-02,  7.2866e-02,
          3.3135e-02, -9.9747e-03,  6.5549e-02,  6.8526e-02, -3.3870e-02,
          3.9562e-02,  7.3255e-02, -2.4939e-02,  6.1158e-02, -5.5209e-02,
          1.3030e-02,  5.2606e-02, -5.8173e-02,  8.4692e-02, -1.4416e-02,
          6.1336e-02,  6.2394e-02,  8.2337e-03, -3.7439e-02,  6.5704e-02,
         -4.6057e-02,  6.0693e-02, -6.3947e-02,  4.6162e-03,  3.9228e-02,
          7.0493e-03,  7.0281e-02, -2.2692e-03, -1.0119e-01, -2.1995e-02,
         -7.7137e-02,  5.9071e-02,  1.0324e-01,  2.2725e-02, -1.3456e-02,
         -4.0519e-02,  9.0302e-02, -5.1037e-02,  6.1401e-03, -3.5622e-02,
          5.2602e-02, -7.0062e-03, -3.8132e-02,  8.6384e-03,  8.4842e-02,
          6.1340e-02,  2.9575e-02,  5.9258e-02, -4.1133e-02, -6.1583e-02,
          4.5067e-02, -6.6575e-02,  7.2744e-02,  9.9581e-02, -1.8678e-02,
          2.4796e-02, -3.3758e-02,  1.8310e-02,  1.9572e-02,  6.0431e-02,
          3.0647e-03,  3.5113e-02,  7.0217e-02, -6.0848e-02,  5.6459e-02,
          3.0101e-02,  6.4558e-02, -5.2777e-02,  2.8825e-02, -5.6157e-02,
         -2.9442e-03,  3.1204e-02, -7.5820e-02,  4.5497e-02, -3.0144e-02,
          2.2352e-02,  7.2345e-02,  7.1976e-02,  9.5632e-03,  4.4350e-02,
          9.1658e-03,  1.2373e-02,  1.9087e-02, -7.0707e-02,  2.8906e-02,
         -2.9870e-02,  3.8927e-02, -4.2761e-03,  4.1356e-03, -1.1370e-02,
          7.6272e-03, -2.7950e-03,  5.0513e-02, -8.4592e-03, -6.0892e-02,
          8.5330e-02,  6.8307e-02],
        [-8.3085e-03,  5.9404e-02,  3.6980e-02, -5.2875e-02, -2.0203e-02,
         -5.9259e-02,  5.8701e-02,  6.7111e-03, -6.7949e-02,  2.7904e-02,
          6.8756e-02, -2.3120e-02, -4.4376e-02, -2.9315e-02,  4.8992e-02,
          4.0158e-02, -5.2972e-02,  5.3480e-02,  1.3378e-03, -1.2888e-02,
         -5.0223e-02,  3.7175e-02,  4.2549e-02,  7.6878e-03, -4.8736e-02,
          4.7677e-02,  7.0592e-02,  1.3876e-02, -3.4977e-02, -4.6276e-02,
          1.2506e-02, -6.6107e-02,  3.7435e-02, -5.1661e-03,  3.8698e-02,
          5.5697e-03, -5.4981e-03, -6.4653e-02, -5.4719e-02,  3.3608e-02,
          2.3867e-02, -4.8512e-03, -5.7825e-02,  5.7675e-03, -6.3465e-02,
         -5.0638e-02, -3.7859e-03,  4.3518e-02, -7.1849e-03,  4.0541e-02,
          4.8622e-02,  5.8955e-02, -4.9463e-02, -1.0519e-03,  1.0538e-02,
          6.3994e-02,  1.0412e-02, -5.2694e-02, -3.6212e-02, -4.1242e-02,
          2.1447e-02,  4.4592e-02,  6.7746e-02,  3.9698e-02,  2.3118e-02,
         -6.2433e-02, -3.1788e-02, -2.5333e-02, -6.0491e-02,  1.5939e-02,
          1.2253e-02, -4.8155e-02,  2.1231e-02, -3.8918e-02, -1.1601e-02,
          1.6052e-02,  5.7483e-02, -6.1291e-03,  4.6281e-02, -4.6280e-02,
          1.9999e-02,  4.7798e-02,  5.5631e-02, -6.1942e-03, -6.2413e-02,
          5.1133e-02,  3.8080e-02,  4.7527e-03,  6.2753e-02, -6.4097e-02,
          1.7908e-02, -4.4486e-02,  2.8127e-02,  6.3482e-03, -2.0007e-02,
         -1.8675e-02,  2.4765e-02, -1.4200e-02,  3.2411e-02, -5.1952e-02,
         -7.0498e-02, -6.1189e-02,  4.8810e-02,  9.6500e-03, -8.1166e-03,
         -5.1469e-02,  2.5173e-02,  6.3372e-02, -6.0764e-02, -5.7808e-02,
         -4.7210e-02, -5.6131e-02,  5.1692e-02, -4.6871e-02, -3.8380e-02,
          5.3379e-02, -4.7842e-02,  5.1127e-02,  8.0201e-03, -6.7429e-02,
          6.3258e-02,  6.7420e-02, -2.9524e-02, -4.3368e-02, -4.2069e-02,
         -6.2554e-02,  9.2129e-03,  1.6407e-02,  7.7813e-03, -5.7777e-02,
          3.7200e-02, -6.0049e-02, -6.6668e-02,  3.2145e-02, -2.7130e-02,
          4.8089e-02, -6.1878e-02,  5.3097e-02,  6.2634e-02, -2.2124e-02,
         -6.0444e-02,  3.6664e-02, -1.2052e-02, -3.0624e-03, -1.2296e-02,
          1.4940e-02, -2.0070e-02,  4.5633e-02, -7.0864e-02, -5.2285e-04,
         -1.0225e-02, -6.6887e-02,  2.2188e-02, -1.6366e-02,  6.0059e-02,
         -4.3564e-02, -5.6882e-02,  1.5092e-02, -9.6861e-03,  6.2063e-02,
          3.8534e-02, -4.9540e-02,  6.6481e-02,  5.8158e-02,  3.2157e-03,
         -2.1466e-02, -1.3857e-02, -1.9762e-02, -3.6238e-02,  8.7374e-03,
          4.8083e-02, -5.4491e-02,  6.1015e-02,  2.3702e-02,  1.9629e-02,
          2.4708e-02, -5.9264e-02,  2.8769e-02, -5.8863e-02, -1.7202e-02,
         -4.7623e-02, -6.3398e-02, -1.9089e-02, -4.9109e-02,  1.0540e-02,
          6.8879e-02, -5.7673e-02,  1.5767e-02, -1.6867e-02, -4.2491e-02,
         -9.5733e-03,  3.3647e-03],
        [ 5.9955e-02,  4.5129e-02, -6.8249e-02,  1.5270e-02, -3.7193e-02,
         -1.9815e-02, -6.2167e-02, -9.0747e-03,  7.6737e-02, -5.5565e-02,
          4.1591e-02,  8.5772e-03,  5.1985e-02,  1.2821e-03, -2.2155e-02,
          4.0411e-02,  7.1320e-03,  1.0504e-02,  6.6022e-02,  2.2608e-02,
         -4.2413e-02,  9.1691e-02, -2.7896e-02,  1.8339e-02,  8.3718e-03,
          4.6453e-02, -2.4180e-02, -3.5285e-02,  5.5785e-02,  3.6932e-02,
         -1.1438e-02,  7.4754e-02, -4.2080e-02, -7.1675e-02, -3.5284e-02,
         -5.3715e-02, -7.3212e-04,  4.9165e-02,  8.9958e-02, -4.7923e-02,
          2.1206e-02,  1.4265e-03,  2.7891e-02,  1.7516e-02,  5.3040e-02,
          5.0961e-02, -3.6128e-02,  2.1458e-02,  7.8547e-02,  8.8396e-02,
          7.7768e-02, -8.8510e-03,  3.7293e-02, -3.4148e-02, -3.2488e-02,
         -6.0225e-03, -1.1167e-02,  1.9652e-02,  2.3489e-02,  3.5882e-02,
          3.1670e-02,  5.8878e-02,  7.8812e-02,  9.7318e-02, -2.1050e-02,
          8.2895e-02,  7.2735e-02,  4.3519e-02,  1.4921e-02,  7.2404e-02,
         -2.1313e-03,  5.7088e-02, -7.2029e-02,  3.5847e-02,  3.6992e-02,
          7.0483e-02,  1.3111e-02, -8.2986e-02, -3.6056e-02,  8.9596e-02,
          2.0821e-02,  2.0831e-02,  6.7341e-02,  1.5135e-02,  4.5494e-02,
          8.4462e-02, -2.2569e-02,  6.7198e-02,  7.7475e-02, -8.9324e-02,
         -2.2480e-02, -6.5001e-02, -3.6738e-03,  5.2154e-02,  5.3258e-02,
          8.4174e-02,  6.7563e-02, -6.6847e-03, -3.1454e-02, -2.2609e-02,
         -5.4069e-02,  4.0314e-02,  5.4197e-02, -3.8113e-02,  3.3780e-02,
          4.4965e-02,  1.7952e-02, -3.7386e-02,  7.0397e-02, -5.3986e-03,
          5.6412e-02, -1.3938e-02, -1.0427e-02, -1.4860e-02,  3.7380e-02,
         -2.1666e-02,  8.2437e-03, -4.2214e-02, -2.8875e-02,  4.7288e-02,
         -5.9985e-02, -4.2752e-02, -6.7184e-02, -2.4473e-02, -4.6671e-02,
         -2.7561e-02, -2.9753e-02,  2.8793e-02,  1.7062e-02, -2.9067e-02,
          9.5723e-02, -2.9588e-02, -3.1507e-02,  3.6785e-03, -4.4446e-02,
          1.0277e-02, -4.0949e-02, -7.6180e-03,  2.0034e-02, -7.4055e-02,
          1.0107e-01, -5.6034e-02, -6.4626e-02,  5.2018e-02, -1.4718e-02,
         -1.0769e-01,  1.0022e-01, -2.1570e-02, -8.6264e-02, -3.2776e-02,
          5.8687e-02,  5.4980e-02, -2.2890e-03,  1.9528e-02,  6.2685e-02,
         -1.4820e-02,  4.7786e-02, -5.3833e-03,  1.4561e-02, -3.0717e-02,
          5.4148e-02,  6.6667e-02, -1.7605e-02, -7.0964e-02, -1.2702e-02,
          4.4150e-02, -8.7741e-02,  1.4636e-02,  8.0810e-02,  8.5074e-02,
          2.8252e-02, -1.3540e-02,  5.0797e-02, -1.2988e-02,  4.4218e-02,
         -7.1268e-02, -2.7276e-02, -4.2145e-02, -1.5636e-02,  6.0577e-02,
          4.4348e-04,  2.5925e-02,  7.2454e-03,  2.9245e-03,  7.9101e-02,
          5.4428e-02,  3.2009e-03,  9.2952e-03,  5.4132e-03, -1.7914e-02,
         -7.3498e-02, -4.3073e-03],
        [-1.6768e-02, -1.1370e-02,  5.7872e-02, -1.3638e-03, -5.2671e-02,
         -1.1294e-02,  3.1298e-02,  3.0090e-02,  2.7065e-02,  1.7520e-02,
         -6.2734e-02, -3.7198e-02, -2.8041e-02, -1.4230e-02, -6.0817e-02,
          4.7866e-02,  5.1028e-02,  3.9132e-02, -4.4934e-02, -5.5778e-02,
          7.7299e-03, -3.8729e-02, -1.6920e-02,  6.2400e-02, -8.8947e-03,
          6.0332e-02, -4.8346e-02,  4.3299e-02, -6.5910e-02, -4.6161e-02,
         -7.0599e-02, -2.1742e-02, -5.1570e-04, -6.0171e-02, -4.2217e-02,
         -4.5163e-02,  4.6492e-02, -2.7930e-02,  2.1210e-02, -1.0397e-03,
          6.5514e-03, -2.6354e-02,  2.9038e-02,  7.3577e-03,  6.4245e-02,
          7.4544e-02,  2.0817e-02,  7.3821e-02, -4.6736e-02,  3.8956e-02,
          3.8532e-03, -5.5192e-02,  4.4803e-04, -2.9685e-02,  4.9404e-02,
         -6.1974e-02,  4.3215e-02, -4.6555e-02,  3.2661e-03,  4.0472e-02,
          4.3839e-02,  1.7471e-02, -2.6066e-02, -6.3149e-02,  6.8636e-02,
         -5.7280e-02, -1.9027e-02, -4.8938e-02,  6.4490e-02, -4.0460e-02,
         -7.1206e-02, -1.6219e-02,  9.8675e-03, -5.4482e-02,  4.1886e-04,
          3.5266e-02,  1.4382e-02,  7.0867e-02, -2.2279e-02, -3.5727e-02,
         -4.5663e-02,  6.6090e-02,  4.6265e-02, -6.1441e-02, -6.3142e-02,
         -1.7255e-02, -4.5649e-02, -4.4466e-02, -2.3531e-02, -6.9580e-02,
         -2.6742e-02, -2.4001e-02,  2.8959e-02, -6.7424e-02,  5.2789e-02,
          5.8397e-02,  7.1774e-03, -4.3897e-02,  5.8522e-02, -1.9896e-02,
          4.3323e-02,  7.7739e-03, -3.8658e-02, -2.3221e-02,  4.0607e-03,
         -5.5567e-02,  9.9098e-03, -1.9693e-02,  5.9756e-02,  1.0944e-02,
         -4.3665e-02,  3.5978e-02, -6.0049e-02, -3.4158e-02,  1.3247e-03,
          3.9164e-02, -5.8241e-02, -5.1331e-02, -5.8355e-02, -2.6161e-02,
          5.5606e-02, -5.9674e-02, -5.0219e-02, -1.0675e-02, -1.3752e-02,
         -5.4319e-02, -1.2622e-02,  5.2522e-02, -6.2554e-02, -5.3052e-02,
         -3.5392e-02, -3.6903e-02, -3.0910e-02, -5.3871e-02,  1.2447e-02,
         -7.0131e-03, -6.6605e-02,  4.7029e-02, -3.9978e-02, -2.5066e-02,
          3.8204e-02,  5.1138e-03,  2.9849e-02,  6.0681e-02,  4.0768e-02,
         -6.3447e-02, -4.3168e-02,  4.2076e-02,  5.9028e-02,  4.6247e-02,
          2.9094e-03,  6.6638e-02, -2.3539e-02,  3.1596e-02,  4.3170e-02,
          1.4477e-02, -6.8754e-02,  5.5660e-02,  1.9292e-03, -1.1964e-02,
          2.0356e-02,  4.9151e-02,  5.9498e-02, -6.0181e-02, -2.6958e-02,
         -4.3357e-02,  5.3675e-02, -4.7111e-02,  4.5830e-03, -5.8987e-02,
          4.0654e-02,  1.3315e-02,  2.0830e-02, -2.7156e-02, -7.0615e-02,
         -6.0522e-02, -4.6292e-02, -6.3934e-02,  5.9929e-02,  4.1592e-02,
          5.0127e-02,  2.2127e-02,  6.1767e-02,  5.3831e-02,  3.9028e-02,
         -6.5676e-02,  2.1278e-02, -3.3421e-02,  1.5688e-02, -5.1733e-02,
          4.5788e-02,  7.2595e-02],
        [ 1.4039e-02,  4.6053e-02,  2.1145e-02, -3.0457e-02, -3.0056e-02,
         -9.0009e-03,  1.0083e-01, -4.5948e-02,  4.7524e-02, -9.7505e-02,
          2.3897e-02, -1.4163e-02, -3.1420e-02,  7.9782e-02, -7.2999e-02,
          1.1233e-01,  6.5296e-02,  6.2809e-02, -5.3927e-02, -1.3157e-02,
          2.9742e-02, -2.5261e-02,  4.2309e-02,  6.8143e-02,  9.6703e-03,
         -1.5565e-02,  7.4045e-03, -7.9797e-02,  7.0385e-02,  9.8403e-02,
          7.9562e-02,  7.9957e-02,  3.2770e-02,  1.1901e-02, -4.2474e-02,
         -5.4423e-02,  3.5344e-02, -4.4494e-02,  3.8806e-02, -7.2505e-03,
          5.7555e-02, -2.7962e-02, -2.5108e-02,  4.9379e-02,  5.9040e-02,
          8.7558e-02, -6.2132e-02,  4.7053e-02,  3.2883e-02,  9.3397e-02,
         -1.7937e-02,  6.8071e-02, -1.6412e-02,  3.6635e-02,  3.7008e-02,
          2.8679e-02,  4.6996e-02, -2.3019e-03, -1.7168e-02,  9.2750e-02,
          3.1337e-02,  4.1846e-02,  7.2619e-02, -1.1561e-02,  9.0393e-02,
          3.3582e-02, -8.1383e-02, -2.9394e-02, -3.1371e-02, -5.3227e-03,
         -2.5324e-02,  3.2364e-02,  9.0730e-03,  1.2351e-02,  4.2388e-02,
          2.5999e-02, -6.0480e-02,  1.8545e-02, -2.2257e-02,  7.5723e-02,
          8.6651e-02, -2.5377e-02,  6.1085e-02, -8.0203e-03,  8.8414e-02,
          8.8663e-02, -4.1190e-04,  2.0957e-02,  3.8678e-02, -1.2277e-02,
         -2.5354e-02, -3.2520e-02, -6.9801e-02, -1.4741e-02,  4.8892e-02,
          1.2676e-02,  7.8046e-02,  7.7381e-02, -2.9962e-02,  7.6977e-03,
         -9.9864e-02,  1.1513e-02,  9.0055e-02,  3.3442e-02,  1.0425e-01,
         -2.7777e-02, -1.1201e-01,  1.0247e-02,  7.5820e-02, -6.0312e-02,
         -3.1129e-02,  8.3382e-02,  1.7491e-02,  2.9128e-02, -9.6914e-02,
          3.3128e-02, -1.0146e-01,  9.7659e-02,  7.8044e-02, -3.7228e-02,
          1.4312e-02, -5.3862e-02,  8.1310e-03,  5.8810e-02,  7.1676e-03,
          4.6251e-02, -3.4975e-02,  5.0390e-02, -4.3995e-02,  1.3554e-01,
         -5.6122e-03, -3.4864e-03,  7.6520e-02,  5.9852e-02, -1.2943e-02,
         -1.0538e-01,  1.0399e-01,  3.5879e-02, -4.8755e-02, -9.2822e-04,
          7.2986e-03,  1.4349e-02, -5.5922e-02,  2.2136e-04,  5.4247e-02,
         -2.5083e-02,  6.5167e-02,  4.0464e-02,  1.3063e-02,  8.7545e-02,
          2.3880e-02,  5.8874e-03,  5.0957e-02, -5.6146e-02, -2.2304e-02,
         -4.3984e-02,  5.0553e-02, -2.6265e-02, -1.1790e-02,  5.8811e-02,
         -1.4869e-02,  1.3143e-02,  3.3793e-02,  1.9407e-02,  6.2082e-02,
         -1.7097e-02, -5.8088e-02, -3.5966e-02,  3.9126e-02,  4.6454e-02,
          2.3171e-02,  6.2098e-02, -1.1455e-02,  5.1470e-02,  6.9957e-02,
         -1.4527e-02, -5.0612e-02, -6.1768e-02, -4.8054e-03,  1.9501e-02,
          9.8583e-02, -4.1369e-02, -2.9117e-02, -8.7678e-02,  9.4167e-02,
         -3.5499e-02,  1.3476e-01,  6.8199e-02,  4.6343e-03,  1.2941e-02,
         -3.4805e-02,  7.8989e-02],
        [-1.8606e-02, -5.3531e-02,  6.6554e-02, -6.0979e-02, -7.9440e-03,
          3.9769e-02,  2.8138e-02,  2.6973e-02, -4.2349e-02, -9.9166e-03,
          2.4295e-02, -2.7943e-02, -8.5294e-02, -2.1985e-02,  5.9353e-02,
          1.4828e-01,  1.8663e-02, -2.1353e-02,  1.8735e-02, -1.8776e-02,
         -4.9934e-02, -4.3975e-03,  5.5098e-02,  1.6893e-02, -5.8483e-04,
          3.0165e-03,  4.6795e-02, -8.3479e-02,  8.2764e-02,  6.1454e-02,
          7.8484e-03,  8.1478e-02, -3.2486e-02,  1.4297e-02,  1.0116e-01,
         -5.4460e-02, -3.1692e-02,  4.1231e-02,  1.6618e-02,  7.0381e-02,
          4.9803e-02,  2.3537e-02, -4.4389e-02, -1.4411e-02, -1.0471e-02,
          3.8825e-02,  3.1569e-03,  1.2334e-01,  9.7104e-02,  8.5633e-02,
          1.0170e-02, -4.8266e-02, -5.6083e-02,  5.5528e-02,  2.6436e-02,
         -3.7265e-03, -1.9139e-03, -6.3995e-02,  6.6008e-02,  1.1312e-01,
          3.3379e-03,  5.3892e-03,  3.8661e-03, -5.9891e-03, -3.4774e-02,
          2.6551e-02, -3.1445e-03, -4.8947e-02, -3.7048e-02,  1.0755e-01,
          9.0009e-02,  2.8612e-02,  4.4198e-02,  9.6664e-02,  4.0726e-02,
         -1.1286e-02,  5.2616e-02, -1.4688e-03,  5.4231e-02, -4.8473e-02,
          9.8137e-02,  4.6425e-02,  8.9012e-02, -1.9246e-02, -2.0116e-02,
          6.8721e-03, -7.9872e-02,  5.9087e-03,  4.0154e-02, -8.1198e-04,
         -6.3268e-02, -8.4022e-02,  3.0285e-02,  2.1829e-02,  1.8392e-02,
         -2.3808e-02,  1.0586e-01,  4.3490e-02,  1.0253e-01,  6.7016e-02,
         -7.0129e-02, -3.7367e-02,  6.0543e-02, -2.2174e-02, -1.3490e-02,
         -1.9009e-02, -6.9352e-02,  1.2876e-02,  3.5558e-02,  4.6892e-02,
         -2.5759e-02,  4.7499e-02, -4.8418e-02, -4.3825e-02, -4.7695e-02,
          7.9704e-02,  4.8867e-02,  3.6803e-04, -1.4091e-02,  7.2958e-02,
         -4.1945e-02, -1.6780e-03, -7.0956e-02,  7.1313e-02, -2.7838e-02,
          8.5741e-02, -7.2101e-02, -1.6577e-02,  2.3531e-02,  1.1284e-01,
          5.2745e-02, -3.8575e-02,  3.4588e-02, -2.7396e-02,  4.3229e-02,
         -3.1540e-02,  1.7471e-02,  7.9128e-02, -2.1485e-02,  7.4643e-03,
         -1.7405e-02, -3.2943e-02, -5.5516e-02,  4.5912e-02,  1.0014e-01,
          6.4627e-02,  1.0200e-01,  3.5456e-03, -1.0639e-01,  3.4271e-02,
          3.1284e-02,  1.1198e-01,  4.3556e-02, -2.2966e-02, -4.0769e-04,
         -5.9020e-02,  5.8816e-02, -2.0490e-02,  4.1041e-03,  2.6346e-02,
          4.0417e-02,  1.7192e-02, -3.9274e-02, -5.7270e-03,  5.5951e-02,
          5.3184e-02, -5.7524e-02,  7.4712e-02,  5.3535e-02,  5.7082e-02,
         -2.3030e-02, -6.8296e-03,  1.8031e-02, -2.1952e-02,  7.7814e-03,
          2.5680e-02, -9.2808e-03, -8.2314e-02,  5.4580e-03, -1.5289e-02,
         -4.4311e-02,  8.4209e-03,  8.4101e-02,  1.5656e-02, -1.2787e-02,
         -7.6393e-03,  4.7489e-02, -4.9383e-02,  2.4931e-03,  6.1778e-02,
          4.5849e-02, -6.1278e-03],
        [ 3.6869e-02,  7.3323e-03,  2.7867e-02, -5.9325e-02,  3.3596e-02,
         -7.3871e-03,  3.3708e-02,  2.5120e-02, -1.4306e-03, -4.8097e-02,
         -1.1388e-02, -4.4335e-02,  8.5983e-02, -4.4829e-02,  1.7865e-02,
          8.2472e-03,  5.5146e-03,  1.9141e-02, -1.5838e-03,  6.5061e-02,
         -3.8077e-02,  4.3097e-02, -4.6364e-03, -2.2794e-03, -1.3293e-02,
         -4.5982e-02, -4.7197e-02,  1.0736e-02,  2.2755e-02,  4.7545e-02,
         -2.8110e-03, -6.4030e-02,  5.4371e-02,  7.1958e-02, -7.9746e-02,
          2.4604e-02, -5.3265e-02, -3.6180e-02, -8.8062e-02, -7.6490e-02,
          2.9434e-03, -9.5424e-02,  4.2506e-02, -3.4384e-02,  1.4622e-02,
         -3.6312e-02, -2.6163e-03, -3.6552e-02, -3.1875e-03, -6.0798e-02,
         -1.7203e-02, -4.6563e-02, -3.3619e-02,  4.1639e-02,  3.3212e-02,
         -7.5396e-03, -5.5502e-02,  5.6728e-02,  4.3997e-02,  3.5727e-02,
          2.1696e-02, -1.2908e-02,  3.3337e-02, -5.1468e-02, -3.4975e-02,
         -2.7893e-03,  4.4883e-04, -5.3073e-02,  9.9547e-02,  2.2363e-02,
         -2.0165e-02, -4.0418e-02, -1.4771e-02,  1.9742e-03, -7.0015e-02,
         -2.1941e-02,  4.0314e-02, -3.7275e-02, -1.2616e-02,  1.0616e-02,
          1.8701e-02, -8.8163e-02,  1.0423e-02, -5.3541e-02,  3.1771e-02,
          3.3207e-02,  6.2289e-02, -4.6071e-03,  3.5777e-02,  1.1531e-02,
          2.9046e-02,  2.8358e-02,  2.4968e-02, -5.4548e-02,  4.9702e-02,
          6.3098e-02, -5.9548e-02, -3.7225e-02, -5.2071e-02, -7.6966e-02,
          5.1643e-03,  9.2309e-02, -5.4449e-02,  4.2234e-02, -6.8007e-03,
          1.9583e-02,  2.9727e-02, -2.9862e-02,  6.0262e-02,  2.2289e-02,
          4.6871e-02,  2.1683e-02,  3.8619e-02, -3.5418e-02, -3.5030e-02,
         -8.0951e-02,  9.2500e-02, -5.1577e-02, -7.1652e-02, -8.8838e-02,
          3.3359e-02,  5.0382e-02,  7.4204e-02,  3.8655e-02,  5.6362e-02,
         -9.6660e-02,  5.7005e-02,  1.8858e-02, -8.8747e-05, -2.7947e-02,
          3.5210e-02,  5.1266e-02, -2.3009e-02,  2.6548e-02,  5.9716e-02,
          4.9334e-02,  1.3696e-02,  3.3166e-02,  3.4465e-02,  6.5664e-02,
         -1.3190e-02,  2.0018e-02,  2.8426e-02, -8.0623e-02, -1.2572e-02,
          7.1010e-03,  3.4221e-03, -2.5366e-03,  8.6438e-02, -9.1782e-02,
          2.9645e-02,  5.0892e-02, -5.0085e-02,  3.5762e-02, -7.8384e-02,
          2.0867e-02, -4.7861e-02, -6.6197e-02,  5.5583e-02,  1.6427e-02,
         -4.5289e-02,  4.1984e-02,  5.6408e-02,  9.5140e-03, -1.2359e-02,
         -4.9240e-02,  2.7990e-02,  5.0681e-02, -3.7889e-02, -1.2220e-02,
         -7.0593e-02, -6.5528e-02, -2.8577e-02, -1.9645e-03, -6.3985e-02,
         -3.0734e-02, -2.0857e-02,  1.4409e-03,  2.0391e-02,  2.3403e-02,
         -6.4330e-02,  9.0834e-03, -4.4517e-03, -5.5405e-02,  1.2803e-02,
         -4.4510e-02, -5.2147e-02, -7.7628e-02, -1.8610e-02, -4.9344e-02,
         -1.6209e-02,  4.3685e-02],
        [-1.0564e-02, -2.3516e-02, -1.6651e-02,  1.5047e-05, -2.0180e-02,
         -2.2259e-02,  9.7508e-02,  6.8494e-02,  5.5690e-02, -6.7405e-02,
          1.5500e-02, -3.8382e-02, -2.1237e-02, -8.3098e-02, -9.8732e-02,
         -2.2544e-02, -1.2775e-02, -4.1502e-02,  1.5684e-03, -1.5779e-02,
          6.8550e-02, -4.3292e-03,  3.0702e-02,  7.1061e-03, -4.7404e-03,
         -2.2083e-02,  2.3278e-02,  6.2121e-02, -2.7407e-02,  3.1001e-02,
         -6.4409e-02, -8.1488e-02,  3.8087e-02,  6.5745e-03, -3.3521e-02,
          9.3389e-03, -5.0942e-02, -5.2181e-02, -4.9260e-02, -7.2022e-02,
         -5.6492e-02, -4.9290e-03, -2.3680e-02, -9.1521e-02, -4.0586e-02,
         -5.4078e-02, -5.0618e-02, -5.0206e-02, -2.6828e-02, -8.0431e-02,
          1.4537e-02, -9.0259e-02,  4.9941e-03, -2.0163e-02, -7.0795e-02,
          1.2368e-02, -1.7078e-03, -2.7914e-02,  4.8866e-03, -4.9817e-02,
         -3.3874e-02,  1.6527e-02, -9.3668e-02,  1.5045e-02,  3.4783e-02,
         -2.7674e-02, -7.9705e-02,  1.3833e-02,  3.4252e-02, -3.2194e-03,
         -8.1580e-02, -7.5404e-02, -4.0647e-02,  9.0031e-03, -5.4513e-02,
         -2.6833e-02, -9.1181e-02, -7.4980e-02, -4.3310e-02, -5.3731e-02,
          4.8884e-02, -8.1469e-03,  4.4210e-02, -2.8508e-02, -1.0088e-01,
         -9.7621e-02,  1.0005e-02, -6.9316e-02, -1.3325e-02, -3.1759e-02,
         -4.5195e-02, -2.4876e-02, -6.0330e-02,  5.0232e-02, -5.1140e-02,
         -4.1145e-02, -1.0974e-01, -4.4352e-02,  2.0700e-02, -9.1469e-02,
          5.5251e-03, -6.6608e-02, -7.4966e-02, -2.2404e-02, -8.8136e-03,
         -4.4292e-02, -4.2840e-02,  7.7111e-03, -7.8784e-02,  1.9987e-03,
          4.3533e-03, -6.0701e-02,  1.8001e-02,  3.0244e-02,  3.4209e-02,
          4.4436e-02, -1.6303e-02, -3.4689e-03, -6.9237e-02,  3.1484e-03,
          1.0137e-01, -4.4993e-02,  2.6241e-02, -3.1278e-02, -6.1867e-02,
         -1.0067e-01,  1.0144e-01,  2.1453e-02,  2.2684e-03,  7.1011e-02,
          3.3096e-02,  5.1867e-03, -6.2075e-02, -3.9179e-02, -5.8306e-02,
         -9.9140e-02, -2.7091e-02, -4.5877e-02, -3.3225e-02,  5.7236e-03,
         -4.0143e-02, -9.7563e-02,  2.7624e-03, -2.1433e-02,  3.4766e-02,
         -5.6600e-02, -3.2940e-02, -8.1852e-02, -2.0628e-02,  5.0042e-02,
         -4.5674e-02, -7.2505e-02,  1.1916e-02, -9.1456e-03, -8.3001e-02,
         -1.4986e-02, -2.4811e-02, -5.0786e-02, -5.0773e-02,  3.1284e-02,
         -2.6467e-02,  5.0622e-02, -1.2103e-02,  2.7381e-02, -2.0815e-02,
         -2.5115e-02, -9.3242e-02,  1.1011e-02, -1.8181e-02, -6.1450e-03,
         -9.3967e-02, -9.7862e-02,  1.1435e-02,  6.1024e-03,  4.3249e-02,
         -1.9840e-02, -1.0285e-02, -3.1825e-02, -3.6333e-02, -3.8828e-02,
          1.6463e-03,  6.0544e-03, -6.0010e-02, -6.8589e-02,  5.5775e-03,
          5.0565e-02,  4.3088e-03,  2.1619e-02,  1.3683e-03, -2.1654e-02,
          3.6849e-02,  1.4688e-02],
        [ 3.6664e-02,  4.4904e-02, -6.1708e-02,  2.3527e-02, -1.7179e-02,
          6.0998e-02, -1.2266e-02, -5.9903e-02, -1.1925e-01,  3.5272e-02,
         -1.1315e-01,  1.1510e-02, -1.4791e-02, -5.8171e-02, -5.0802e-02,
         -3.3327e-02,  8.1239e-03,  6.0931e-03, -4.2429e-02,  2.7052e-02,
          4.5645e-02, -3.8880e-02, -4.7555e-02,  9.5508e-03, -8.7303e-02,
         -1.2181e-02,  1.1834e-03, -4.4573e-02, -1.0524e-01, -9.3808e-02,
         -1.8371e-02, -1.1721e-01, -2.4221e-02, -2.2659e-02, -9.0016e-02,
         -5.1595e-02,  6.7052e-02, -1.3806e-03, -1.7626e-02,  2.0886e-03,
          1.6804e-02,  8.3388e-03, -2.5233e-02, -8.2835e-02, -7.1926e-02,
          6.7865e-03, -1.0976e-01, -1.8826e-03, -2.4821e-02, -1.1718e-01,
         -8.4402e-02,  4.2778e-03, -5.3289e-02, -1.2784e-02, -9.1680e-02,
          4.9131e-02, -8.4246e-02,  2.4641e-02, -3.6886e-02, -5.6338e-02,
          2.3565e-02, -1.0026e-01, -6.0799e-02,  3.6807e-02, -6.9337e-02,
         -5.8818e-02,  2.7365e-02, -8.7584e-02, -1.3225e-01, -2.5989e-02,
         -1.0079e-01, -6.0016e-02,  3.5111e-02, -1.1173e-01,  3.8808e-03,
         -1.0870e-01, -2.0799e-02, -7.9483e-02,  1.6957e-02, -4.3681e-02,
         -5.9936e-02, -2.5970e-02, -4.1142e-02, -3.3693e-02, -1.0404e-01,
         -1.9288e-03,  6.9846e-02, -9.9773e-02, -2.0973e-03,  9.3887e-03,
         -2.2191e-02, -6.8430e-02,  2.8213e-02, -3.2041e-02,  9.3541e-03,
         -4.8845e-02, -6.4674e-02, -5.1742e-02,  8.1027e-03, -5.6320e-02,
          6.1268e-02, -1.2250e-01, -4.6645e-02, -2.4252e-02, -4.6332e-02,
         -8.4451e-02,  7.6196e-03, -4.0798e-02, -4.3343e-02, -1.1735e-01,
         -6.0748e-02, -4.3363e-02, -3.5948e-02, -1.5911e-02,  2.7682e-02,
         -5.8844e-02, -2.1577e-02, -3.4128e-02, -6.5868e-02, -9.1438e-02,
          1.7007e-04,  2.6215e-03, -1.5276e-02, -1.2572e-02, -6.3330e-02,
         -8.3405e-02,  1.5928e-02, -1.1078e-01, -2.5518e-03,  9.2611e-03,
         -3.6690e-02, -7.4613e-02, -5.9083e-02, -9.8468e-02, -2.4383e-02,
         -9.2372e-02, -8.1347e-02, -1.3332e-02, -2.5278e-02, -3.0844e-02,
         -2.5769e-02, -1.0818e-01, -2.3422e-02,  3.3142e-02, -1.1434e-02,
         -1.0324e-01,  1.9250e-02, -2.6549e-02,  5.7084e-02, -4.3360e-02,
         -7.3050e-02, -9.7403e-02, -2.8031e-02,  8.8554e-03, -7.9844e-02,
          6.5751e-02, -6.2108e-02, -8.9278e-03, -2.4648e-02, -8.4052e-03,
          1.6946e-02,  1.6505e-02, -3.3821e-02, -6.5863e-02, -2.7760e-02,
         -3.8842e-02,  1.5869e-02,  2.5282e-02, -1.6399e-02, -6.0682e-02,
         -6.9872e-03,  1.7582e-02, -6.3818e-02,  5.1779e-03,  7.0825e-02,
         -5.0999e-02,  3.8033e-02, -4.6911e-02,  3.4140e-02, -5.1564e-02,
         -2.9591e-02, -7.9842e-04, -1.1672e-01,  3.9357e-02,  3.7629e-02,
         -5.1827e-02, -5.2137e-02, -4.9910e-02,  3.5448e-02, -2.0578e-02,
         -6.5948e-03, -3.6421e-03],
        [ 4.6494e-02,  1.2850e-02,  7.0589e-02,  3.2838e-03, -2.4913e-02,
         -6.1067e-02, -1.8156e-02, -6.5497e-02, -6.4796e-02, -1.9456e-03,
          1.4556e-02, -1.4329e-02, -2.8892e-02,  8.4027e-02, -3.7116e-02,
          3.5763e-02,  8.4047e-02,  5.0509e-02,  6.4866e-04,  4.9074e-02,
          6.8337e-03,  3.5954e-02, -6.6560e-02,  5.8771e-02, -2.9613e-02,
          4.6776e-03,  8.4762e-02, -4.6515e-02,  4.8646e-02,  2.0564e-02,
          5.3406e-03,  5.8170e-02, -2.9589e-02, -5.6861e-02,  1.4612e-02,
          8.7578e-02, -3.4513e-02,  6.9545e-02,  8.5560e-03,  7.0349e-02,
          6.8213e-02, -1.0991e-02,  6.9912e-03,  6.2624e-03,  8.2069e-02,
          8.3069e-02, -7.8913e-02,  2.9427e-02,  9.2449e-02,  8.0967e-02,
         -1.2825e-02, -4.3842e-02, -4.1979e-02,  7.4106e-02,  4.4623e-02,
          6.8497e-02,  4.8710e-03, -4.1762e-02, -9.9574e-03,  5.0962e-02,
          7.1616e-02, -4.9189e-02, -2.1943e-02,  2.9800e-03, -3.4986e-02,
          6.3366e-02,  5.4166e-03, -1.4147e-02,  5.1979e-02, -3.8265e-02,
          6.1855e-02, -4.2605e-02, -5.9376e-02,  4.5775e-02, -1.8326e-02,
          3.7660e-02,  1.8488e-02,  3.5694e-02, -4.3352e-02,  7.8714e-02,
         -1.1439e-02,  4.5336e-02,  6.4786e-02,  3.3736e-02, -4.2810e-02,
          6.4677e-02,  3.4171e-03,  1.4494e-02,  4.9445e-02, -3.1941e-02,
          2.2634e-02, -6.8109e-02, -3.2094e-02, -6.0707e-02,  2.3708e-02,
          2.4840e-02, -4.5118e-02,  9.3223e-03,  1.4821e-02, -3.1091e-02,
         -2.3666e-02,  1.5689e-02, -1.5100e-02, -3.9101e-02, -5.1297e-02,
         -4.8813e-02, -6.1713e-02, -1.9092e-02, -4.8953e-02,  5.0247e-02,
         -5.0101e-02,  3.4945e-02,  4.0809e-02,  1.7538e-02, -5.4317e-03,
         -5.1511e-02,  7.5072e-03,  8.2336e-03,  8.6843e-02, -2.2840e-02,
         -6.7478e-04, -1.7520e-02, -5.1085e-02, -4.2276e-02,  4.9953e-02,
         -1.7559e-02, -3.1455e-03,  1.5888e-02, -3.0777e-02,  5.9548e-02,
          1.0068e-02, -2.4078e-04, -1.1269e-02,  1.3004e-03,  4.3152e-02,
         -2.8961e-02,  3.3224e-02,  3.6138e-02,  2.9454e-02,  3.3840e-02,
         -4.3476e-02,  3.5219e-02,  7.4791e-02, -3.7678e-02,  7.9267e-02,
         -3.0046e-02,  4.0571e-02, -2.8448e-02, -4.8794e-02, -3.3369e-02,
          4.4300e-02,  9.4549e-02, -1.6870e-02,  4.2347e-02,  6.2218e-02,
          5.3829e-03, -7.0237e-02,  1.4954e-02,  6.1829e-02,  3.5400e-02,
          1.3172e-02,  9.0135e-03, -2.5532e-02, -4.0021e-02,  8.4401e-02,
          8.6872e-02,  5.7289e-02,  6.0725e-02,  3.8783e-02,  5.9394e-03,
         -2.0095e-02, -4.4093e-02, -1.6984e-02,  6.7729e-02, -3.0656e-02,
         -6.0623e-02, -4.5706e-02, -6.2473e-02, -7.0075e-02, -7.1696e-02,
          2.5374e-02,  5.1148e-02, -4.8919e-02, -6.2540e-02,  1.4868e-02,
          5.9155e-02,  2.6907e-02,  5.6976e-02, -1.6804e-02, -1.3543e-02,
          3.8844e-02,  5.9580e-02],
        [-6.7838e-03, -2.2523e-02,  3.7382e-02,  9.1341e-03, -3.2109e-02,
         -1.6475e-02,  4.1336e-02,  3.4247e-02, -4.9572e-02, -2.0557e-03,
         -3.2591e-02,  3.2129e-02, -4.7055e-02,  3.3314e-02, -4.5694e-02,
          1.1571e-01,  2.0990e-02, -3.5484e-03, -5.4446e-02,  1.0825e-02,
          3.6818e-02, -2.7672e-02, -2.5193e-02,  6.5396e-02,  6.3469e-02,
         -1.2936e-02,  1.6609e-02, -4.0078e-02,  4.1353e-02,  5.5741e-02,
         -2.6048e-02,  7.2666e-02, -3.2329e-02, -1.4328e-02,  8.9422e-02,
         -1.4334e-02, -5.9015e-02, -3.2326e-02,  3.0331e-02, -2.6159e-02,
          3.9475e-03,  7.7832e-02,  4.7992e-02,  3.4819e-02, -2.0347e-02,
          8.8850e-02, -3.2973e-02,  1.0520e-01, -3.4200e-02,  1.7823e-02,
          6.8705e-02, -1.0313e-02, -2.3346e-02,  5.3968e-02,  6.9552e-03,
         -6.4183e-02,  8.3717e-02, -7.4752e-02,  9.9455e-02,  6.8935e-02,
          2.1682e-02,  4.9122e-02,  3.7497e-02,  4.5419e-02, -1.3733e-02,
          7.7477e-02,  1.8946e-02,  4.6694e-02, -1.1377e-01,  9.3123e-02,
         -3.6633e-04, -5.4808e-02, -7.8941e-02,  4.4732e-03,  8.4444e-02,
         -9.0695e-02,  6.9092e-02,  3.6624e-02,  8.8493e-02,  1.0701e-02,
         -4.1328e-02, -2.4438e-02,  8.6288e-02,  2.2109e-02, -8.8064e-04,
          3.6802e-02, -7.2203e-02, -3.1800e-02,  6.0062e-02, -7.2506e-02,
          2.4806e-02, -8.0151e-02, -1.6382e-02,  3.5667e-02, -6.6588e-02,
         -3.0743e-02,  5.2022e-02,  7.4552e-02, -4.6311e-02,  3.1945e-02,
         -2.3581e-02, -3.1038e-02,  2.7803e-02, -1.2696e-02, -2.4646e-02,
         -1.1285e-02, -7.4582e-02,  3.9839e-03,  1.7314e-02,  6.4308e-03,
         -1.7197e-02,  5.3690e-02,  3.9233e-02,  3.2730e-02,  2.3706e-02,
          6.8628e-02, -1.8528e-02,  5.3131e-02,  4.5103e-03, -3.4875e-02,
         -9.5618e-02, -6.1849e-02,  6.0299e-02,  6.0129e-02, -5.8349e-02,
          5.6331e-02,  1.5373e-02,  4.5565e-02, -6.4726e-02,  1.0095e-01,
          6.2319e-02, -3.3728e-03, -9.7492e-04, -1.9808e-02, -1.1521e-02,
         -8.8488e-02,  3.2384e-02,  8.6005e-02, -7.8602e-03, -1.9740e-02,
         -4.6831e-03,  5.4835e-02, -8.6716e-02,  9.2006e-02,  1.0600e-01,
         -2.9220e-02,  1.0679e-01, -4.7040e-03,  9.3258e-03,  5.2709e-03,
         -3.0808e-02,  1.1098e-01, -7.2851e-03, -7.5409e-02, -2.7058e-02,
         -3.0481e-04, -5.2644e-02,  3.7919e-02, -4.9309e-02,  1.2602e-02,
         -6.8984e-02, -3.3925e-03, -5.9134e-02, -9.3885e-02,  5.8168e-02,
          4.0696e-02, -8.8378e-02,  4.7003e-02,  6.5855e-02,  4.4934e-03,
          7.8889e-02,  8.5848e-02,  1.7875e-02,  1.4374e-02,  3.9460e-03,
          2.8697e-02, -3.2043e-02, -1.8068e-02, -6.8480e-02, -5.7544e-02,
          8.0263e-02,  2.7372e-02,  5.6075e-02, -1.0702e-01, -1.8144e-03,
          3.8525e-02,  3.7542e-04, -5.7116e-02,  2.5279e-02,  6.5974e-02,
          3.0911e-02, -1.6092e-02],
        [-5.8220e-02,  9.2245e-03,  6.9208e-02,  4.1956e-02,  3.7528e-02,
         -5.3156e-02, -5.6485e-03,  2.6307e-02,  3.2106e-02, -9.5898e-02,
          1.4242e-02,  4.1694e-03, -1.5023e-02,  1.2812e-01, -9.3324e-02,
          3.7330e-02,  9.6971e-02,  1.1916e-01,  7.0594e-02, -6.6645e-02,
          3.7704e-02,  2.9394e-03,  8.1691e-02,  4.8560e-02,  3.8756e-03,
          7.9481e-02, -4.2026e-02, -1.5500e-03,  3.2886e-02,  2.1277e-02,
         -2.8046e-02,  8.1699e-02, -2.7861e-02, -5.6849e-02,  5.6448e-02,
         -3.6169e-02, -9.0459e-04,  1.5837e-02,  1.1115e-01, -8.2859e-02,
          1.0565e-01,  1.5284e-02,  6.3060e-03,  1.7524e-02,  4.6052e-02,
         -2.7812e-02, -1.5723e-02,  5.4538e-02, -1.1888e-02,  3.7707e-02,
         -4.3169e-02,  1.6079e-02,  1.9400e-02,  4.2985e-02, -5.9137e-02,
          4.6003e-02,  5.8097e-02,  4.4300e-02,  6.0039e-02,  5.7468e-02,
          7.5516e-02, -1.3434e-02,  5.0916e-03, -1.8102e-02, -1.7034e-02,
         -1.3133e-02,  8.5925e-02,  3.9908e-02,  1.6830e-03, -3.6171e-03,
         -3.5427e-02,  5.0889e-02, -2.8114e-02,  7.5217e-03, -3.5109e-02,
          3.4337e-02,  2.0931e-02,  5.2335e-02, -5.1371e-02,  4.7233e-02,
         -7.0026e-02, -4.8370e-02, -7.4097e-02, -5.1594e-03,  4.0920e-02,
          1.1535e-01, -4.3789e-02,  3.7694e-02,  1.3434e-01, -4.6951e-02,
         -9.5911e-03,  2.0751e-02,  3.0153e-02,  9.9545e-02, -6.0635e-02,
         -2.5172e-02, -8.2786e-02, -9.1764e-02,  3.1522e-02,  1.4428e-02,
          3.9602e-02,  3.2589e-03,  2.3851e-02, -6.3756e-02,  7.3243e-02,
          5.4096e-02,  2.1285e-02, -8.6006e-02,  2.2409e-02,  1.1332e-01,
         -7.1774e-03, -2.4431e-02,  1.3997e-01,  3.5804e-02, -1.6120e-03,
          1.0910e-01, -2.5989e-02,  6.5683e-02, -3.4031e-02,  3.9371e-02,
         -9.3114e-02,  5.4732e-03,  2.5647e-02, -1.0850e-01, -9.4485e-02,
          5.1926e-03,  1.5886e-02,  1.0854e-02, -2.3236e-02, -5.3140e-02,
          4.6617e-02, -3.4036e-02, -8.5967e-02,  4.8424e-02,  5.7113e-02,
         -4.7306e-04,  1.8354e-02,  1.1439e-01,  5.9698e-02, -6.4856e-02,
          1.1733e-02, -1.0813e-01,  6.5576e-03,  1.0231e-02, -4.7171e-02,
         -6.0999e-02,  2.3322e-02, -4.5926e-02, -1.0183e-01, -6.0786e-02,
          2.8886e-02,  4.5874e-02, -2.3950e-02,  7.7494e-03,  6.4611e-02,
         -4.4798e-02, -5.3033e-03, -4.3362e-02, -3.1902e-02, -1.9659e-02,
         -2.6920e-03,  1.8434e-02,  2.4472e-02,  8.1762e-03,  9.7077e-02,
         -3.9314e-02, -3.6893e-02,  2.6070e-02, -3.1966e-02, -2.5219e-02,
         -3.6421e-02, -3.4482e-02, -3.1634e-03,  1.6745e-03,  6.8449e-02,
          6.0796e-02,  9.8288e-04, -4.6675e-02, -1.5645e-02,  1.4517e-02,
         -4.8793e-03,  7.7918e-02, -1.0313e-02, -7.4925e-02,  5.0957e-02,
          4.3928e-02, -2.0502e-02, -3.1391e-02,  4.7080e-02,  1.0003e-01,
         -3.3486e-02,  7.6327e-02],
        [-4.6146e-02, -2.9175e-02, -4.3806e-02, -7.0302e-02, -9.6233e-03,
         -1.0032e-03, -2.0651e-02, -2.2720e-02, -2.8709e-02, -5.2514e-03,
         -5.0037e-02, -5.9497e-02, -4.2383e-02, -6.5398e-02, -7.9493e-02,
         -4.2511e-02, -3.2874e-02,  2.6989e-02, -3.3455e-02,  2.8486e-02,
         -2.2187e-02, -2.3930e-02, -6.5855e-02, -5.4846e-02, -2.0778e-02,
         -7.3389e-02, -6.1835e-03, -1.3696e-02, -8.8385e-02, -2.4217e-02,
          1.9341e-02,  7.1240e-03, -6.4865e-02,  1.0781e-02, -1.6759e-02,
         -8.7661e-02,  5.0974e-02, -4.2330e-02, -3.5927e-02,  9.3542e-04,
          2.1404e-03,  4.8466e-02, -6.4743e-02, -7.0114e-02,  4.9986e-03,
         -3.8104e-02,  4.0996e-02,  5.9028e-03, -1.4766e-02, -3.8777e-02,
         -2.8063e-02, -2.9239e-02,  6.6098e-02,  2.4654e-02, -7.9002e-02,
         -5.3582e-03, -8.0394e-02,  2.8073e-02, -7.2073e-02,  1.8504e-02,
          3.7590e-02, -3.2513e-02, -5.0731e-02, -3.5304e-03, -2.3237e-02,
         -7.7783e-02,  2.6601e-03, -3.2468e-03,  1.9447e-02,  1.8729e-02,
          1.4835e-02, -1.8284e-02, -5.9502e-02, -4.8785e-02, -6.9443e-02,
         -6.3336e-02,  3.6425e-02,  2.9240e-02,  6.7982e-02, -3.5534e-02,
         -3.8828e-02, -5.7563e-02,  2.7372e-02, -3.0427e-02, -9.0347e-02,
         -9.7482e-02, -4.2224e-02, -8.2321e-03,  8.3216e-04, -6.2989e-02,
          7.9676e-03, -7.1820e-02, -2.6989e-02,  6.4411e-03,  1.4502e-02,
          1.8612e-02,  1.6659e-02, -9.8175e-02, -4.1518e-02, -7.1923e-02,
          2.4501e-03,  1.7535e-02, -7.1129e-03,  6.9462e-03, -1.6850e-02,
         -4.6459e-02, -9.0156e-02, -3.6458e-02, -8.9300e-02, -9.5179e-02,
         -4.9782e-02, -6.6371e-02, -1.1183e-03, -8.2605e-02,  2.9824e-02,
         -5.5063e-02,  3.8911e-02, -2.4603e-02,  4.8993e-02, -4.0200e-02,
          9.7148e-03,  1.5663e-02, -4.6792e-02, -4.3768e-03,  2.9202e-02,
          8.8011e-03, -7.2541e-02, -9.1205e-02, -9.3303e-02, -6.8378e-02,
          1.3902e-02, -4.0595e-02, -8.0061e-02, -2.5964e-02,  6.0148e-02,
         -1.0038e-01, -2.2339e-02,  6.1904e-02,  4.0081e-02, -7.0943e-02,
          2.5459e-02, -3.7567e-02,  4.8301e-03, -3.0863e-02,  6.5170e-03,
          6.3485e-02, -5.9148e-02,  3.3066e-02,  2.6917e-02, -1.0845e-02,
          9.8711e-03, -1.8182e-02, -5.3356e-02, -8.1053e-02, -9.9848e-02,
          5.0279e-02, -6.7459e-02, -7.4003e-04,  6.2034e-02, -6.2946e-02,
          1.7619e-02,  2.3019e-02, -5.7500e-02,  4.0547e-02, -9.3204e-02,
         -8.3451e-02,  4.2058e-02,  6.8474e-03,  3.3838e-02, -5.3359e-02,
         -4.0264e-02, -9.0049e-02,  1.9714e-02,  5.1264e-03,  2.8318e-02,
          7.0897e-02,  5.1461e-02,  4.9849e-02, -3.0278e-02, -2.5490e-02,
         -5.6658e-02,  1.9727e-02,  3.4460e-02, -7.8364e-02, -8.5288e-02,
          4.6188e-02, -5.7630e-02, -1.8324e-02, -4.7747e-02, -9.6016e-03,
          4.5332e-02, -7.2731e-02],
        [ 4.9640e-02,  6.1087e-02,  3.5067e-02, -4.7826e-02,  4.9563e-02,
         -2.8249e-02,  7.6606e-02,  2.0441e-02, -1.0681e-02, -3.0612e-02,
         -6.5905e-02, -1.5083e-02,  5.0262e-02,  3.4153e-02, -1.6585e-02,
          6.4252e-02, -2.0286e-02,  5.7115e-02,  6.7153e-02, -2.8065e-02,
         -3.7933e-02,  3.6722e-02,  2.9842e-02, -4.6601e-02,  7.0589e-02,
          4.6086e-02, -4.1073e-02, -3.6518e-02, -4.2372e-02, -5.7073e-02,
         -1.5719e-02, -6.1596e-02,  2.8879e-02, -2.5457e-02,  3.9712e-02,
          9.0147e-03, -4.2443e-02, -2.7054e-02, -6.5705e-02, -4.1157e-02,
         -3.4557e-02,  5.4268e-02,  5.0011e-02, -4.0438e-02, -5.7052e-02,
          4.7396e-02,  5.2150e-02,  1.2809e-02,  6.0355e-02, -5.2743e-02,
          2.5836e-02,  4.5751e-02, -5.6452e-02, -2.5089e-02,  3.9741e-02,
         -6.2598e-02,  2.9717e-03,  3.0288e-03, -4.4838e-02,  6.8214e-02,
          3.7284e-02, -4.4134e-02,  3.0315e-02,  3.5476e-02,  4.3490e-02,
          2.7941e-02, -5.5112e-04, -5.3669e-02,  7.6311e-02, -4.5671e-02,
         -7.1791e-02,  3.8691e-02, -5.2034e-02, -3.1825e-02, -7.7438e-04,
         -6.6437e-02,  2.5574e-03,  8.5073e-03,  5.0598e-02,  8.7153e-03,
          1.6973e-02,  2.0202e-02,  6.7080e-02, -1.1665e-03,  3.8362e-02,
         -6.8864e-02,  7.6359e-02,  3.1273e-02, -2.0388e-02, -6.9713e-02,
         -1.5198e-04,  2.9247e-02, -6.3513e-02,  5.5138e-02, -4.1944e-02,
          6.3591e-02,  3.5881e-02,  6.1967e-02,  4.2389e-02, -4.2056e-02,
          5.8340e-02, -5.8727e-02, -2.1941e-02,  3.5667e-03, -1.5100e-02,
         -2.6779e-02, -4.2544e-02, -2.8185e-02,  4.6060e-02, -4.7023e-03,
          4.6259e-02,  6.5367e-02, -3.1167e-02,  4.9797e-02,  3.2344e-02,
         -5.8439e-02, -1.6680e-02, -1.1157e-02,  1.2911e-02, -9.1196e-03,
          3.0966e-02,  3.1588e-02,  5.3333e-02,  6.0610e-02,  3.1416e-02,
         -6.5430e-02,  1.6657e-02, -7.1048e-02,  2.2432e-03, -4.8230e-02,
          5.3865e-02, -9.2059e-03, -4.4887e-02, -3.9716e-02,  1.1531e-02,
         -6.5040e-02, -3.7284e-03,  6.9901e-03,  2.0739e-02, -3.7961e-02,
         -3.0598e-02, -4.7325e-02,  7.6342e-02, -1.7416e-03, -1.4942e-02,
         -4.5994e-02, -2.2947e-02, -1.8243e-02, -2.3587e-02, -3.1817e-03,
          6.3082e-02,  5.9472e-02,  4.4064e-02, -4.4254e-02, -6.2689e-02,
         -2.2958e-02, -1.5687e-02,  2.0457e-02, -1.0987e-02,  8.2928e-03,
         -2.3901e-02,  1.4893e-02, -1.1784e-02,  7.0116e-02, -3.2208e-02,
          3.3108e-02, -5.4123e-03,  6.8337e-02, -4.1294e-02,  3.4803e-02,
         -5.9905e-02, -3.9901e-02,  3.3673e-02, -8.1873e-03,  8.0630e-03,
          2.2946e-02, -5.2773e-02,  6.2029e-02,  2.5362e-02, -4.2198e-02,
         -5.1667e-02,  1.7922e-02, -6.4061e-02,  3.0076e-02,  4.5788e-02,
          4.2072e-02, -5.1307e-02,  7.0738e-02,  5.7519e-02, -4.6510e-02,
          4.0114e-02,  4.0936e-02],
        [-6.2433e-02, -5.7054e-02, -2.4865e-02, -2.8382e-02,  3.0562e-02,
          3.1228e-02,  7.9955e-02, -5.1180e-02, -9.2089e-03, -1.7872e-02,
          1.3893e-02, -3.7364e-02, -2.8412e-02, -1.9321e-02,  1.1654e-02,
          1.3561e-01, -1.2502e-02,  4.4562e-03, -1.7522e-02, -5.8606e-02,
          2.9410e-02, -1.2204e-02,  8.3886e-02,  6.1502e-02,  9.1117e-02,
          3.0273e-02,  9.5500e-02, -8.0655e-02,  4.5778e-02,  7.3337e-02,
          2.8521e-03,  4.9621e-02,  4.9854e-03, -6.0601e-02,  3.2497e-03,
         -6.3953e-02, -2.0283e-02, -1.7235e-02,  6.1640e-02,  5.3279e-02,
         -1.2451e-02,  1.2131e-01, -1.8102e-02, -6.2209e-03,  8.6920e-02,
          6.8321e-02, -3.6164e-02,  1.0485e-01,  7.4164e-02,  7.5580e-02,
          5.5365e-02,  5.3973e-02,  1.0847e-02, -5.0398e-02,  4.0006e-02,
          6.6212e-02,  8.1994e-02,  2.1639e-02, -2.6642e-02,  5.7489e-02,
          2.8208e-02,  8.9421e-02, -8.1938e-03,  7.6557e-03,  3.1089e-02,
         -5.7435e-03, -7.0979e-02, -1.5958e-02,  1.1337e-02,  9.2959e-02,
          1.7516e-02, -9.8199e-03,  5.8522e-03,  7.2417e-02,  8.9381e-02,
          4.1931e-02,  4.5873e-02, -8.0498e-02,  2.2798e-02, -1.1439e-02,
         -3.4537e-02, -1.8766e-02, -3.5808e-02, -3.2307e-02,  3.5065e-02,
          8.8977e-02, -4.0368e-02,  3.2518e-02,  4.1772e-02, -1.0517e-01,
         -6.0018e-02, -9.6961e-02,  6.8352e-02,  4.3636e-02,  2.4691e-02,
          7.2320e-02,  1.0576e-01, -4.0378e-02,  7.4268e-02,  1.6681e-02,
         -9.5471e-02, -6.6708e-02, -4.2662e-02,  7.5914e-02,  1.4934e-02,
          7.0986e-02, -8.3674e-03,  2.2625e-02,  2.3588e-02, -4.6111e-02,
          1.4585e-02,  7.3163e-03,  3.1882e-02, -3.4542e-02,  2.3766e-02,
          8.5617e-02,  3.5830e-04,  1.5282e-02, -2.2314e-02, -1.9605e-02,
         -1.0815e-01, -7.0038e-04,  5.5283e-02,  8.8634e-02,  3.6820e-02,
          1.1121e-01, -9.8924e-02, -5.4651e-02, -4.1137e-02,  1.5083e-01,
          2.5811e-02,  1.5833e-02,  5.2201e-02, -4.0091e-02, -7.7720e-02,
         -8.6361e-02,  8.3406e-02,  6.6588e-02,  1.0084e-02,  2.1417e-02,
          8.6089e-02, -1.6241e-02, -7.0412e-02,  1.2352e-01, -8.9301e-04,
         -2.6656e-02,  7.1278e-02, -7.9714e-02, -8.6097e-02, -3.5212e-03,
         -8.2742e-03, -2.1818e-02,  5.7238e-02,  5.3505e-02,  6.9775e-02,
         -1.0632e-01,  5.5317e-03, -4.2367e-02, -3.5708e-03,  1.9314e-02,
         -5.1205e-03,  3.0344e-02,  1.7955e-02,  3.8522e-02,  8.4534e-02,
          8.9805e-02,  8.6046e-03,  1.2329e-02,  5.2422e-02, -2.9227e-02,
          1.9644e-03,  2.2786e-02,  1.8034e-02, -2.0740e-02, -2.1471e-02,
          5.7113e-02,  1.3220e-02,  3.7135e-02, -4.9082e-02,  1.0225e-02,
          7.3081e-02,  8.8477e-02,  7.0302e-02, -1.0465e-01,  1.9194e-03,
         -4.1305e-02,  1.1532e-01, -9.2763e-03,  3.1816e-02,  4.7779e-02,
         -7.4200e-02,  6.4486e-02],
        [-4.9723e-02, -5.0129e-02, -6.7190e-02, -4.0709e-02, -2.9946e-03,
         -7.1480e-02, -5.4872e-02,  1.8511e-02,  4.4585e-03, -5.0032e-02,
         -9.1643e-02, -5.2285e-03, -5.5724e-03,  2.5676e-02, -7.7104e-02,
          5.0640e-02, -2.8221e-02,  7.5313e-02, -6.4047e-02, -6.9422e-02,
         -3.4294e-02,  7.1319e-02,  2.5621e-02,  1.9383e-02,  3.0039e-02,
         -2.5640e-02, -2.3774e-02, -3.7200e-02,  2.9384e-02, -3.4149e-02,
          6.8047e-02,  7.0767e-02,  5.8503e-02,  2.4703e-02,  7.4269e-03,
          5.1143e-02, -1.8120e-02, -8.9601e-04,  3.3407e-02,  3.1170e-02,
         -3.6379e-02,  5.8722e-02, -8.0290e-02,  4.2040e-03,  1.6851e-02,
         -2.6372e-02, -3.8799e-02,  9.2280e-02, -3.2195e-02, -5.4354e-02,
          6.7015e-02, -2.3357e-02, -4.2732e-03, -6.7571e-02, -3.9476e-02,
         -4.9769e-02,  2.0076e-03, -2.7891e-02,  5.9876e-02,  3.5648e-02,
         -7.1179e-02, -4.0538e-02,  7.8962e-02,  2.9043e-03,  9.6856e-02,
          1.0154e-01, -1.9762e-03, -4.4244e-03, -1.2391e-02,  7.8153e-02,
          4.0590e-02,  5.6992e-03, -7.7072e-04, -2.7179e-03, -3.9463e-03,
          2.4343e-02,  8.8679e-04, -6.3424e-02, -1.9651e-02,  1.2656e-02,
          9.6296e-02, -3.0685e-02,  2.6259e-02, -4.4851e-02,  3.9857e-02,
          5.7899e-02, -1.0059e-01, -2.1629e-02,  1.7433e-02, -1.0415e-01,
         -3.5908e-04, -4.5305e-02, -1.4590e-02, -5.7820e-02,  2.1214e-02,
          5.4943e-02, -4.0953e-02,  9.8828e-03, -2.9693e-02,  6.3163e-02,
         -7.2933e-02, -4.1826e-02, -6.5429e-02, -7.0488e-02,  3.9676e-02,
         -9.4376e-03, -3.0335e-02,  4.7175e-02, -6.6464e-02, -3.7399e-02,
         -7.9167e-02,  6.6950e-02,  6.6358e-02,  4.2443e-02,  7.4916e-03,
         -8.1592e-03,  2.0196e-02,  1.2151e-01,  2.1755e-03,  7.9596e-02,
         -4.5720e-02,  9.2065e-03,  1.3725e-02, -2.4270e-02,  1.9770e-02,
         -1.1600e-02, -3.0551e-02, -3.3042e-02, -1.5398e-02,  4.5045e-02,
          6.9205e-02,  4.4687e-02,  1.8273e-02,  2.2831e-02,  3.9260e-02,
          1.6285e-02,  6.3924e-02, -8.6723e-04, -9.8112e-03, -2.8489e-02,
          4.6062e-02,  4.4077e-03, -8.5411e-02,  9.2308e-03, -2.0717e-02,
          1.3240e-02, -3.1893e-02, -9.9864e-02, -2.5420e-02, -1.5970e-02,
         -4.2458e-03, -2.6763e-02, -2.4533e-02, -4.1983e-02, -3.4958e-02,
         -7.6386e-02,  1.7138e-03,  6.3178e-02, -3.8251e-02, -6.5595e-02,
         -5.6757e-02, -4.9352e-02,  6.0215e-03, -6.0364e-02,  1.0537e-01,
         -4.5783e-02,  1.4920e-02,  6.8523e-02, -3.6242e-02, -9.2483e-02,
         -3.8192e-02, -6.9413e-02,  2.1019e-02,  7.1050e-04,  3.9313e-02,
          1.0934e-02, -4.2820e-02,  1.1669e-02, -1.5316e-02, -6.9786e-02,
          5.6061e-02,  2.2721e-02,  6.5137e-02, -6.2060e-02, -2.8816e-02,
         -4.6770e-02,  7.0674e-02, -5.9673e-02,  1.0379e-01,  7.3742e-02,
         -6.3512e-02,  5.6219e-02],
        [-1.1052e-02, -3.6056e-02,  1.3893e-02,  6.0202e-02, -1.1080e-02,
         -6.9268e-02,  3.0094e-02, -1.3433e-02, -1.1712e-02,  8.9991e-02,
         -3.1005e-02, -3.4709e-02, -1.4867e-02, -9.9019e-02,  4.8515e-02,
         -6.6105e-02, -3.4421e-02,  8.8365e-03, -5.9993e-02, -7.5036e-03,
          6.6624e-02, -3.8844e-02,  1.5702e-02,  7.8246e-03,  3.3176e-02,
          2.1031e-02, -3.8502e-02,  2.4038e-02, -5.6312e-02,  8.3609e-02,
          7.5090e-02, -8.0817e-02, -5.8952e-02,  9.3894e-02,  5.0159e-02,
          3.6602e-02, -6.0605e-03, -1.3062e-02,  2.1350e-02, -6.1521e-02,
         -2.3233e-02, -9.3318e-02,  1.2693e-03, -4.3963e-02,  4.9429e-02,
         -7.9013e-02,  9.9512e-02, -5.2241e-02, -3.5047e-02,  5.7848e-02,
          5.5312e-02,  6.3307e-02, -2.6828e-02,  8.0267e-02, -3.1797e-02,
         -4.4740e-03, -6.8574e-02,  3.6556e-02, -4.0320e-02,  6.2367e-03,
          7.7453e-02, -7.6727e-02, -6.6717e-02, -2.7389e-02,  5.2725e-03,
         -7.9805e-02,  2.9010e-02,  8.4961e-02,  4.9145e-02, -8.1716e-02,
          7.2564e-03,  2.9746e-02,  1.6546e-02, -3.4261e-02,  4.9610e-03,
          1.5253e-03,  3.1599e-02,  5.3453e-02, -7.7537e-02,  8.3089e-03,
          2.3080e-02,  2.2837e-02,  4.2538e-02, -6.9463e-02,  3.1246e-02,
         -5.6350e-02, -3.1248e-02, -2.4868e-02, -3.5655e-02,  9.0660e-02,
          7.2947e-02, -7.1748e-02,  5.5974e-02,  9.7589e-02,  4.2829e-03,
         -2.9619e-02,  1.9156e-03,  4.3472e-02,  2.5187e-02,  3.2044e-02,
          7.4669e-02,  1.8053e-02,  6.2881e-03,  7.1448e-02, -3.6401e-02,
          1.7750e-03,  6.0365e-02,  3.9950e-02, -2.1664e-02, -5.9196e-03,
          1.1088e-01, -1.8028e-02,  1.6491e-02, -5.0319e-02,  8.6469e-03,
         -3.2423e-02,  7.9184e-02, -2.8422e-02,  4.9308e-03,  1.4198e-02,
          4.1675e-02, -6.7404e-03,  8.4041e-03,  2.4685e-02, -1.9127e-02,
          1.6263e-02,  1.0626e-01,  4.1923e-02, -1.4048e-02, -6.1033e-02,
         -1.1375e-01,  2.1196e-02, -4.3595e-02, -9.2549e-03,  4.9869e-02,
          3.0491e-02,  5.2480e-02, -6.4857e-02, -3.9822e-02,  9.1098e-02,
         -1.4257e-02,  8.2740e-03,  3.4116e-03, -4.6264e-02,  5.2184e-02,
          5.7877e-02, -6.8802e-02,  6.9973e-02, -1.9302e-02, -9.0925e-03,
         -7.6470e-02,  3.1838e-02, -5.3502e-02,  3.4789e-02, -5.6300e-02,
          3.0434e-02,  5.7277e-03,  5.1025e-02,  3.7452e-02,  2.2210e-02,
          3.4596e-02, -1.8951e-02, -4.3612e-03,  5.9061e-02, -5.6661e-02,
          2.0464e-02,  9.1651e-03, -5.3669e-02,  4.1320e-02,  1.3398e-02,
          7.8289e-02,  5.4494e-02,  3.6283e-02,  9.5501e-03,  3.8690e-03,
         -2.3948e-02, -7.0190e-02, -5.1259e-02,  2.5860e-02, -2.1022e-02,
          4.2822e-02,  2.1424e-02,  5.1556e-02,  4.1099e-02, -2.0012e-02,
         -4.0600e-02, -7.9561e-02, -3.7188e-02, -6.4465e-02,  8.5054e-02,
          2.0405e-02, -4.2247e-02],
        [-3.7141e-02, -2.2432e-03, -3.8122e-02,  2.5378e-02, -3.8982e-02,
          1.4317e-02, -7.8708e-02, -6.3751e-03, -6.8996e-02, -3.1225e-03,
         -2.9087e-02, -3.9869e-02,  1.0475e-01,  3.1998e-02, -2.2204e-02,
         -4.5888e-02, -4.1569e-03, -5.2834e-02, -4.9692e-02, -2.1581e-02,
          6.1192e-02,  6.1490e-02,  1.8219e-02, -4.3594e-03, -5.8503e-02,
         -5.4096e-02,  1.5204e-02,  3.2754e-02, -7.3188e-02,  1.2908e-02,
         -6.6409e-03,  5.2739e-03, -3.5870e-02, -1.3968e-02,  5.0238e-02,
         -2.5991e-02,  4.6321e-02,  5.3783e-02, -4.7640e-02, -5.9196e-02,
          6.6380e-03, -4.0076e-02,  8.9436e-02, -3.3976e-02,  7.5958e-02,
         -6.2521e-03, -2.3656e-03, -9.4016e-02,  3.9801e-03, -3.7683e-02,
         -3.6699e-02,  4.1178e-02,  2.2351e-02, -4.0625e-02, -2.7271e-02,
          7.2734e-02,  6.3443e-02,  1.6065e-02,  4.5049e-03, -6.3513e-02,
          6.4588e-02, -5.5643e-02,  4.4460e-02, -2.2010e-03,  2.6321e-02,
          2.4008e-02,  5.1380e-02, -2.6752e-02,  1.0297e-01,  8.4768e-03,
         -9.0587e-02,  7.4280e-02, -2.6097e-02,  4.9998e-02, -1.6504e-02,
         -4.6606e-03, -4.3189e-02,  1.2381e-01,  4.6775e-02,  4.0625e-02,
          2.7029e-02, -1.4814e-02, -4.4201e-02,  3.7398e-02, -2.3021e-02,
         -3.6814e-02, -4.0710e-02,  2.2126e-02, -9.5953e-02,  1.8920e-02,
         -3.4495e-02,  6.0906e-02, -7.1680e-02,  8.9907e-02,  4.3610e-02,
          8.6637e-02, -2.5996e-03,  5.3584e-02,  6.6907e-02,  2.5991e-02,
          7.9775e-02,  1.3042e-01,  3.2027e-02, -3.7394e-02, -5.0929e-02,
          1.4132e-02,  1.2466e-01, -3.7489e-02,  2.0816e-02, -1.6509e-02,
          7.4300e-02,  2.5905e-02,  5.4934e-02, -2.3797e-02,  9.8290e-02,
         -9.0287e-03,  4.3806e-02, -4.1772e-02, -1.7869e-02, -4.6325e-02,
         -5.3852e-03,  6.9217e-02, -9.5495e-04, -7.3470e-03, -4.3258e-02,
         -1.3345e-02, -9.2789e-03,  1.0503e-01, -2.5713e-02, -7.1731e-02,
          5.0064e-03,  4.8580e-02,  3.7245e-02, -3.6330e-02,  8.1340e-02,
          4.9363e-02,  4.2074e-02,  1.9451e-03,  7.0151e-02,  4.9390e-04,
         -8.9427e-02, -7.7030e-03,  3.4508e-02,  9.4403e-03, -7.2653e-02,
         -7.1703e-03,  2.3596e-03, -2.6899e-02,  9.5996e-02, -4.6762e-02,
          3.8201e-03, -8.8280e-02, -4.0335e-02,  1.0387e-01, -5.5139e-02,
          9.6642e-02, -4.1356e-02, -1.6190e-02, -1.0508e-02, -2.1127e-02,
         -1.9590e-02, -5.2493e-02,  7.4394e-02,  6.1610e-02, -8.9859e-02,
         -1.8426e-02,  6.7462e-02, -8.3919e-02, -3.5957e-02, -3.5173e-02,
          5.7710e-02, -2.8418e-02, -4.0124e-02, -7.2390e-04,  2.1742e-02,
          1.4289e-02,  7.1124e-02,  8.9617e-02, -1.1825e-02, -7.1303e-02,
         -4.0783e-02,  6.7565e-02,  5.1955e-02,  3.7173e-02, -9.5515e-02,
         -4.6325e-02, -6.8790e-02, -6.5661e-02,  3.0527e-02,  2.8271e-02,
         -1.4019e-02, -1.7601e-02],
        [-3.3800e-03,  4.8221e-02,  4.2615e-02,  4.8521e-02,  3.2782e-02,
          4.2041e-02, -9.6452e-02,  6.8256e-03, -4.4806e-03,  1.0632e-01,
          1.6934e-03, -4.7709e-02,  3.3813e-02, -5.8916e-02, -1.8733e-02,
         -3.1981e-02,  4.8057e-02, -1.9123e-02, -2.1488e-02, -6.7465e-02,
         -1.6394e-02, -3.0398e-02, -4.0725e-02, -4.0019e-02, -6.5037e-03,
         -3.9804e-02,  7.1450e-02, -2.5189e-02,  1.7401e-02, -6.1379e-02,
         -9.7116e-02, -1.9910e-02,  5.8433e-02,  4.0654e-02, -3.5328e-02,
          3.9423e-03, -2.9529e-02, -5.4265e-02, -5.8503e-02,  1.3182e-02,
         -3.7056e-02, -7.6335e-02,  2.2409e-02, -4.0634e-02,  9.4319e-04,
         -1.0905e-01,  9.0695e-03, -8.4184e-02,  4.1756e-02,  6.1454e-02,
         -8.1247e-03,  7.9022e-03,  5.9851e-02, -2.0158e-02, -1.4494e-02,
          7.6811e-02,  4.6802e-02,  8.4584e-02, -2.3149e-02, -5.2061e-02,
          1.9497e-02, -6.5059e-02, -6.5060e-03,  5.6888e-02, -6.5293e-02,
         -5.4451e-02,  7.0125e-02,  1.1029e-02, -1.7038e-02, -1.1827e-01,
          1.2738e-03,  2.1961e-02,  5.8195e-02,  1.1464e-01,  1.0131e-02,
          8.3649e-02,  4.0677e-02,  1.9586e-02, -7.4831e-02, -7.3808e-02,
          4.9589e-02, -1.7222e-02, -1.2571e-02, -3.1891e-02,  8.4994e-02,
          3.5574e-02, -1.0457e-02,  2.8045e-02, -1.0468e-01,  1.1045e-01,
          1.0682e-02,  9.9394e-02,  6.1630e-02, -2.7595e-02, -2.0971e-03,
         -9.9115e-02,  1.8991e-02,  7.7560e-02,  2.7814e-02, -6.1434e-02,
          1.1138e-01,  5.8302e-03, -6.0885e-02, -5.9944e-03, -9.5031e-02,
          2.0488e-02,  1.0419e-02,  2.1247e-02,  4.4516e-02,  2.5972e-03,
         -1.3098e-02, -2.7473e-02,  7.4267e-02,  6.5652e-03,  1.9610e-02,
         -4.1535e-02,  7.5462e-02, -1.5785e-02, -3.3678e-02, -3.5925e-02,
          3.8614e-02,  9.1658e-02, -3.1092e-02, -1.3082e-02,  7.5946e-02,
          1.7017e-02,  7.5815e-02, -8.9272e-02,  5.8938e-02, -2.0905e-02,
         -1.1269e-01, -3.7947e-03,  1.0605e-02,  2.2015e-02,  6.4702e-02,
          3.7572e-02,  5.2265e-03,  3.6156e-02,  7.0411e-02,  4.1343e-02,
          1.8402e-02, -2.1591e-02,  6.0644e-02, -8.7209e-02, -8.0670e-02,
         -4.9554e-02, -2.3896e-02,  9.6834e-03,  3.9152e-02, -1.7842e-02,
          1.9150e-02, -3.3295e-02,  2.0715e-02, -3.8933e-04,  2.9572e-02,
          1.0118e-01,  5.3721e-03, -4.5257e-03,  5.7009e-02, -8.1101e-02,
          4.2412e-02, -4.4983e-02,  8.5971e-02,  4.7753e-02,  1.4661e-02,
         -7.1045e-03,  8.4726e-02, -1.2055e-02,  7.1429e-02,  5.9719e-02,
          8.9041e-02, -5.5853e-02,  3.4549e-02,  2.7713e-02,  6.2630e-02,
          1.9099e-03, -2.0703e-02, -1.5441e-02,  2.6835e-02,  2.9209e-02,
          7.1479e-02, -5.5761e-02,  6.7145e-03,  6.5619e-02, -6.1382e-02,
          5.2680e-02, -6.8359e-02, -4.9398e-02,  3.6198e-02,  5.8548e-02,
          2.8192e-02, -2.1109e-02],
        [ 2.4437e-02, -3.5222e-02,  7.0987e-02, -2.8786e-02,  8.4631e-03,
          6.1729e-03, -2.5759e-03,  2.8958e-02, -3.5006e-02, -3.6109e-02,
         -6.8983e-02,  3.8362e-03, -6.7516e-02,  3.3846e-02, -3.3043e-02,
          4.4936e-03, -1.7701e-02,  5.3561e-03, -2.3012e-02, -5.3803e-02,
         -4.7687e-02, -6.5206e-03,  1.1183e-02, -3.4996e-02, -7.8517e-02,
          5.3880e-02, -6.8820e-02,  1.4280e-02, -6.8400e-02,  9.4278e-03,
         -5.0285e-02, -5.9911e-02, -4.6444e-02, -2.7079e-02, -2.5844e-02,
          1.3083e-02,  5.3771e-02,  3.9892e-02,  5.0339e-02,  6.9546e-02,
          6.5665e-02, -4.5192e-02, -2.5482e-02, -7.9191e-02, -2.5742e-02,
          6.4913e-02,  4.3456e-02, -4.0448e-02, -6.9411e-04,  6.1751e-02,
         -5.5591e-02, -6.0368e-02, -5.7700e-02, -4.6177e-02, -7.8214e-02,
         -6.5434e-02, -5.3760e-02, -3.1364e-02, -8.5551e-03,  1.0809e-02,
          1.1607e-02, -1.6017e-02,  5.7509e-02,  1.3254e-02,  6.9835e-02,
         -5.1949e-02,  5.4953e-02,  4.3693e-03, -2.9039e-02, -5.9428e-02,
          8.9593e-03,  2.5448e-02, -6.8014e-02, -1.4823e-02, -1.2419e-02,
          2.0283e-02,  5.8233e-02, -7.7002e-03, -2.4075e-02, -2.8935e-02,
         -6.5758e-02, -2.0428e-02, -4.5798e-02, -6.6287e-02,  2.9905e-03,
         -3.3253e-02, -6.6524e-02, -1.8850e-02,  1.0634e-02, -6.5943e-03,
         -6.2587e-02, -3.7466e-02, -7.0604e-02, -5.7572e-02,  6.2904e-02,
          2.5074e-02, -2.5361e-02, -6.5468e-02,  5.2743e-02, -8.3202e-02,
          1.0066e-02, -5.3378e-02, -6.4187e-02, -3.6716e-02,  1.3541e-02,
          5.8119e-03, -7.9879e-02,  4.9957e-02, -3.7921e-02, -5.9970e-02,
          3.6460e-02, -5.4841e-02,  3.8916e-02,  4.4819e-02,  2.3584e-02,
          5.4292e-02, -2.0709e-02,  6.0318e-02,  1.6107e-02, -7.1167e-04,
         -5.2191e-02,  1.3065e-03,  3.3118e-02,  1.6075e-02,  3.1401e-02,
         -4.1920e-02, -2.0395e-02, -1.7226e-02, -1.3088e-02, -4.7375e-02,
          2.9000e-02,  4.6717e-02, -9.5591e-03,  2.3372e-02,  9.4430e-03,
         -4.2056e-02, -2.4984e-02, -6.9380e-03, -7.9384e-02,  1.5516e-02,
         -5.6793e-02,  1.7077e-02, -5.1353e-02,  5.4563e-02, -6.6110e-02,
          9.6958e-03,  7.9711e-02, -8.8052e-04,  4.2218e-02, -4.2859e-02,
         -7.3952e-02, -3.6863e-02, -3.5408e-02, -3.1705e-02,  2.0184e-02,
         -2.5223e-02,  3.3159e-03,  1.3130e-02,  4.6415e-02,  5.2443e-02,
         -6.6141e-02,  2.8294e-02,  2.0778e-02,  2.1356e-02,  3.3916e-02,
         -1.2006e-02, -5.6904e-02,  7.1313e-02,  4.9086e-02,  1.6923e-02,
         -3.0859e-02,  2.6192e-02, -6.6942e-02, -1.6979e-02, -7.1103e-02,
         -1.9063e-02,  3.6967e-02, -6.5711e-02,  5.9079e-02, -2.5862e-02,
         -1.9493e-02, -1.8083e-02,  2.3917e-03,  2.6785e-02, -2.6252e-02,
          2.6245e-02,  4.6728e-02, -6.8564e-02,  4.0338e-02,  5.6454e-02,
         -1.8829e-02,  1.4555e-02],
        [ 1.8330e-02, -6.5992e-02, -1.0001e-02, -4.9398e-03,  3.3309e-02,
          3.1606e-02,  6.9048e-02, -2.1346e-02, -1.7987e-04,  3.2719e-02,
         -6.5151e-02,  9.2047e-03, -9.4652e-04,  9.9149e-02, -4.0965e-02,
          2.8385e-02,  9.0774e-03,  7.4067e-02, -7.9906e-03, -6.7408e-02,
          4.8200e-02,  2.5922e-02, -5.8797e-02,  1.0214e-02, -2.9618e-02,
         -3.3354e-02, -4.1920e-02,  2.3843e-02,  2.2472e-02,  1.0553e-01,
          9.5174e-03,  6.8768e-02,  4.5398e-02, -5.0751e-02, -2.9555e-02,
         -1.1738e-02, -4.7675e-02,  1.5652e-02,  1.2314e-01,  9.2060e-02,
          6.4196e-03,  1.1203e-01, -4.1697e-02,  2.1244e-03,  8.8452e-02,
         -3.0124e-02, -7.3348e-02,  1.1268e-01,  3.8974e-02,  4.7576e-02,
          6.0247e-02,  1.9157e-02,  3.3561e-02, -3.0538e-03, -1.2045e-02,
          8.2609e-02, -8.4385e-03, -3.1075e-02, -1.5573e-02, -6.1724e-03,
          2.6668e-02, -6.6523e-03, -3.5475e-02,  7.7368e-02,  2.8545e-02,
          7.8605e-02, -4.3316e-02, -1.4689e-02,  4.0230e-02,  8.6277e-02,
          2.7294e-02, -1.5484e-02, -4.7997e-02,  7.4844e-02,  5.7996e-02,
         -6.3985e-02, -6.3054e-02,  2.8565e-02,  8.6971e-02,  1.5835e-02,
          1.0753e-03, -1.6978e-02,  7.4816e-02, -2.8477e-02, -9.4371e-03,
         -2.4232e-02, -5.3848e-02,  5.0165e-02,  1.0742e-01, -7.9279e-02,
          8.5310e-03,  4.9498e-02,  5.6387e-02,  2.3644e-02,  2.7537e-03,
          3.2447e-02,  7.5466e-02, -6.0750e-02,  7.1668e-02,  9.5720e-02,
         -1.4882e-02, -2.3744e-03, -4.1932e-03,  2.1037e-02,  1.1809e-02,
         -5.3375e-02, -6.4968e-03, -2.8554e-02, -1.7223e-02,  6.6577e-02,
         -1.4127e-03,  8.4160e-02,  6.2276e-02, -6.1045e-03, -1.8614e-02,
         -1.3099e-02, -1.0729e-02,  8.0930e-02,  3.6880e-03, -2.6322e-02,
         -7.7412e-02, -6.0422e-02,  2.4224e-02,  1.2965e-02, -9.9670e-02,
          5.1665e-02,  2.7402e-02,  1.1366e-02,  2.8815e-02, -4.2826e-03,
          1.2741e-02, -3.2981e-02, -6.4633e-02,  6.4243e-02, -6.4571e-02,
          3.5600e-02,  7.3587e-02,  2.8598e-02, -1.2033e-02, -4.5372e-02,
          1.0983e-01,  9.5865e-03, -6.6369e-02,  5.4473e-02, -2.4349e-02,
          4.9203e-02,  4.6799e-02, -2.4998e-02, -1.1817e-01, -1.8681e-02,
          4.8478e-03,  7.0959e-02,  8.3876e-02, -4.3845e-02, -2.1173e-02,
         -2.8434e-02, -5.2070e-02,  1.8650e-02,  4.0651e-02,  6.7722e-02,
         -3.0173e-02,  4.6698e-02,  2.7913e-02, -4.7242e-03,  5.5075e-02,
         -5.6504e-02, -7.1611e-02,  2.0611e-02,  3.2828e-02, -1.3909e-02,
         -3.2005e-02,  1.9240e-02, -3.2488e-02, -8.6059e-04, -4.5183e-02,
          1.4665e-02, -5.9825e-02, -4.7052e-02, -6.3862e-02, -3.4544e-03,
         -2.2237e-03, -6.1012e-03,  3.4494e-02, -6.8674e-02,  9.1164e-02,
          4.8178e-02, -2.5043e-02, -4.2329e-02,  4.1251e-02, -4.1879e-02,
         -6.4688e-02,  4.6962e-02],
        [-3.4732e-02,  3.7436e-03, -4.3798e-02, -2.4599e-02, -1.7711e-02,
          5.0760e-02, -9.6057e-03,  8.9513e-02,  6.5834e-02, -9.2544e-02,
         -2.0394e-02,  7.5955e-02,  3.0183e-02, -6.0928e-04, -3.0612e-02,
          1.0630e-01, -1.4631e-02,  7.9172e-02, -1.7863e-02, -5.1018e-03,
         -2.0958e-03, -6.9402e-02,  2.5131e-03, -4.1964e-02, -4.0867e-02,
          2.8601e-02,  3.8797e-02, -5.1380e-02,  2.3954e-02, -3.6533e-02,
          3.5345e-02, -1.3086e-02,  6.8982e-02, -5.1195e-02,  4.3289e-02,
          5.1289e-02, -2.3986e-02, -1.5900e-02, -1.4705e-02,  5.5626e-02,
          7.0066e-02,  8.8631e-02,  2.9606e-03, -6.4369e-03,  1.0869e-02,
          1.4278e-02, -1.1796e-02,  1.9571e-02,  2.5985e-02,  4.2921e-02,
          7.0975e-03, -3.8709e-02, -6.9030e-03, -6.6163e-02,  2.0042e-02,
         -4.4698e-02,  8.4539e-02, -5.1356e-02, -1.0798e-02, -3.2782e-02,
          5.4676e-02, -7.7626e-03, -1.1975e-02, -2.8822e-02,  1.4825e-02,
         -4.1177e-02, -3.5292e-02, -9.1216e-03, -7.7090e-03,  1.3466e-02,
          6.0314e-02,  9.0214e-03,  2.7838e-02, -1.4848e-02,  4.9835e-02,
          3.3732e-02, -6.0853e-02, -3.4183e-02,  2.6080e-02, -4.2719e-02,
          2.5068e-02,  1.4158e-02,  8.5435e-03,  8.9396e-02, -1.9367e-03,
          8.0514e-02,  4.3302e-02,  5.8077e-02,  8.9171e-03, -7.6806e-02,
         -1.1568e-02, -5.4884e-02, -7.2318e-03,  6.0952e-02, -1.8848e-02,
          6.6258e-02,  8.5079e-02,  5.5819e-03,  3.8825e-02,  7.3262e-02,
         -6.3745e-02, -7.5120e-03,  2.5981e-02, -4.1038e-02,  3.9992e-02,
         -9.6826e-03, -1.4125e-02,  2.1163e-02, -1.5312e-02,  7.2786e-02,
          9.4096e-03,  4.5873e-02,  1.8961e-03,  3.2597e-02, -8.9342e-02,
         -4.0029e-02,  2.3032e-02,  1.1360e-01,  6.9587e-02,  6.8292e-02,
         -3.6161e-02, -4.5082e-02,  2.0208e-02,  6.3416e-02,  1.7187e-02,
          5.7295e-02,  3.1268e-02, -5.0227e-02, -5.6865e-02,  1.2426e-01,
         -1.2774e-02,  6.3039e-02, -1.7264e-02, -3.5622e-02, -4.9418e-02,
         -8.2937e-02, -2.0877e-02, -3.0854e-03,  4.0537e-03, -3.5468e-02,
          5.1880e-02, -8.3724e-03, -3.1284e-03,  1.1395e-01,  1.0752e-01,
          1.1199e-02,  1.9854e-02, -5.7369e-02, -9.2111e-02, -1.1011e-02,
          2.7845e-02,  3.6291e-02,  3.4143e-02, -5.3812e-02, -2.2747e-03,
         -6.3456e-02,  2.9691e-02,  2.4938e-02, -4.5454e-02,  2.4590e-02,
          6.7700e-02, -6.9392e-02,  6.0805e-02,  5.5627e-02,  3.3291e-02,
          3.0116e-02,  5.6553e-02,  2.0201e-02,  1.0287e-01, -1.2462e-02,
         -3.0933e-02,  6.8860e-02,  8.0386e-02, -3.5269e-02, -3.9836e-02,
         -3.4877e-03, -5.8513e-02, -4.6898e-02,  4.5775e-02, -1.1999e-02,
          6.2558e-02,  2.3986e-02, -1.3678e-02,  1.8033e-02,  4.8575e-02,
          6.0233e-02,  8.4151e-02, -1.9487e-02,  7.2559e-02,  4.5675e-02,
         -5.3798e-03,  6.5552e-02],
        [-5.3508e-02, -1.8943e-03,  6.5171e-02,  7.9439e-03, -6.0950e-02,
         -1.3733e-02,  1.9121e-02, -4.3630e-02, -7.0521e-02,  6.3677e-02,
          4.9307e-02, -6.4874e-02,  7.8292e-02,  1.2548e-02,  7.1827e-02,
         -9.5944e-02, -4.4530e-03,  4.1600e-02,  6.3811e-02,  4.3215e-03,
         -4.1390e-02, -7.2187e-02, -1.4776e-02,  1.1333e-02,  2.4118e-02,
          9.6340e-03, -3.6820e-02,  6.5214e-02, -3.5805e-02, -5.9439e-02,
         -4.6239e-02,  2.8205e-02, -5.4432e-02,  2.1491e-02, -6.4024e-03,
         -4.8351e-02,  3.6882e-02,  6.3919e-02,  6.7714e-02, -8.8089e-02,
         -7.1468e-02, -1.9306e-02,  4.5449e-02,  6.0396e-02,  1.1296e-02,
          2.3051e-02,  5.5581e-02,  1.0407e-02,  6.0925e-02, -2.8387e-02,
         -2.0942e-02,  3.1807e-02,  5.3328e-02,  6.2615e-02, -4.3114e-02,
          1.1843e-02,  3.0155e-02,  9.0152e-02,  1.4820e-02, -5.8588e-02,
          7.2539e-03, -4.5086e-02, -2.9657e-02, -6.8063e-02,  2.8902e-02,
         -7.7380e-02,  7.3488e-02, -6.4701e-03,  8.8012e-02, -3.3088e-02,
          3.2974e-02,  2.7766e-02,  7.1723e-02,  8.6500e-02, -2.8344e-02,
          6.6065e-02, -5.2346e-02,  7.2593e-02,  1.3765e-03,  3.9590e-02,
          4.5968e-02,  3.4483e-02, -7.4071e-02, -7.6006e-02,  1.9068e-02,
         -6.9756e-02,  5.7393e-02, -3.5418e-02, -2.7633e-02,  6.7376e-02,
          6.0066e-02,  2.7540e-02, -5.7146e-02, -1.5561e-02,  2.7998e-02,
         -2.1556e-02, -1.0099e-01,  8.5346e-03, -1.7013e-02,  1.1755e-02,
          1.8448e-02, -1.3214e-02, -5.4925e-03, -3.6086e-02, -1.0410e-01,
         -2.2052e-02,  1.3793e-01, -3.8189e-02, -7.5593e-02, -3.9750e-02,
          1.2426e-02,  6.1567e-02, -2.7686e-02, -3.0285e-02,  8.2153e-02,
         -5.6315e-02,  8.4843e-02,  2.6928e-02, -7.5561e-02, -9.3824e-03,
          1.5860e-02,  2.5988e-02,  2.4822e-02, -5.3665e-02,  6.4582e-02,
         -1.5450e-02,  1.6751e-02, -2.2782e-02, -6.1708e-02, -2.1327e-03,
         -8.6938e-02,  3.2879e-02, -5.4663e-02,  2.8899e-02,  9.3247e-03,
          4.3093e-02, -9.0212e-02, -5.5396e-02, -4.6786e-02, -2.0347e-02,
         -8.8359e-02,  7.1817e-02,  4.3390e-02, -6.1072e-02, -4.9493e-02,
         -2.3991e-02, -8.9359e-02,  5.9056e-03,  2.3850e-02, -2.2919e-02,
          5.7065e-02,  3.7712e-02, -7.4000e-03,  7.0890e-02, -1.9056e-02,
          7.5559e-02, -5.0287e-02,  2.9402e-03,  3.8513e-02, -5.9013e-02,
          5.8807e-02,  6.9301e-02, -4.0109e-02, -4.9695e-02, -9.2226e-02,
          4.7265e-02,  2.6156e-02,  4.9706e-03, -3.1437e-02, -3.8173e-03,
          8.0179e-03,  3.0441e-02, -2.4483e-02, -3.8328e-02, -7.0895e-02,
         -5.1031e-02, -7.2298e-03,  8.3420e-03,  6.2698e-02, -6.6329e-02,
          3.1552e-02, -2.4190e-02,  3.7206e-02,  4.9670e-02, -3.2864e-02,
          5.2409e-02, -7.3582e-02,  3.7245e-02, -1.5543e-02, -2.0434e-02,
          3.6594e-03,  6.3062e-02],
        [ 4.7064e-04,  6.3115e-03,  5.7037e-02, -6.7479e-02,  4.8672e-02,
          1.9731e-02,  1.3829e-02, -2.9792e-02,  3.1057e-02,  3.3846e-02,
         -9.0402e-02,  1.5697e-02, -7.3909e-02,  3.7784e-02,  1.8913e-02,
          6.7140e-02,  5.3814e-03, -3.5343e-02, -6.2173e-02, -4.6803e-02,
          5.6599e-02, -4.5275e-02, -5.0225e-02,  2.9842e-02, -1.0098e-02,
         -6.8508e-02, -8.6200e-02, -8.7066e-02, -6.7710e-02,  3.9016e-02,
         -8.5501e-02, -6.4462e-02, -6.3005e-02, -3.7910e-02, -2.5562e-02,
          3.6784e-03, -4.1658e-02, -6.1523e-02,  1.0458e-02, -1.2585e-02,
          5.6456e-02, -4.6370e-02, -8.1623e-02,  4.8780e-06, -5.9115e-02,
         -2.3502e-02, -9.9716e-02,  1.7747e-03, -7.9416e-02, -1.7738e-02,
          2.7372e-02, -2.8204e-02,  4.4123e-03,  3.6236e-02, -7.7921e-03,
         -3.8847e-02, -8.1893e-02, -2.8726e-02, -6.7762e-02, -5.5900e-02,
          6.0927e-02,  7.0859e-02, -7.1672e-02, -1.8692e-02,  3.0819e-02,
          2.2135e-02, -1.2867e-02, -8.0512e-03, -7.5701e-02, -2.7059e-02,
         -4.6655e-02, -4.7806e-02,  3.4783e-03, -3.6850e-02,  1.1432e-02,
          4.8099e-02, -2.6136e-02, -3.6470e-02, -1.7510e-02, -5.1249e-02,
          3.7842e-02,  5.3443e-02, -9.3405e-03,  4.2516e-02, -1.0013e-01,
         -8.0548e-02, -9.4257e-02, -6.3420e-02, -5.7189e-03,  3.7506e-02,
          5.5933e-02, -2.4574e-03,  6.4301e-03,  6.7826e-02,  3.6288e-02,
         -4.1362e-02, -5.9887e-03,  2.6384e-04,  2.7900e-03,  1.2787e-02,
         -1.7543e-02, -4.6402e-02,  2.0256e-02, -6.1977e-02,  2.1691e-02,
          2.7240e-02, -4.1878e-02, -8.9518e-02, -4.7722e-03, -2.4002e-02,
         -4.8632e-02, -1.6675e-02,  2.2188e-02, -7.5861e-02, -8.5237e-02,
          2.8453e-02, -7.8648e-02,  1.0413e-02, -6.7944e-02, -4.8615e-03,
         -1.7250e-02,  1.2131e-02, -5.1055e-02, -7.0296e-02, -3.3983e-02,
         -5.8225e-02, -7.9132e-02, -1.7302e-02, -9.3799e-02, -7.0982e-03,
          7.1330e-03, -4.9183e-03,  6.9124e-03, -6.8092e-02,  2.8812e-02,
         -2.3573e-02, -3.5157e-02, -9.4251e-03, -8.4816e-02, -3.4644e-02,
         -7.2541e-02, -4.1265e-02, -8.2208e-02, -6.9320e-02, -6.0739e-02,
          3.8161e-02, -2.7169e-02,  1.2180e-02,  1.0017e-02, -8.3377e-02,
          3.0935e-02, -4.3085e-02, -3.8970e-02,  2.8715e-02, -7.8746e-02,
         -7.1212e-02, -5.5983e-02,  2.5636e-02, -6.3982e-02, -5.8248e-02,
         -3.3943e-02, -6.7517e-02, -2.4637e-02, -1.0030e-01, -7.3482e-02,
          2.9042e-02,  3.3774e-02, -6.8173e-02,  3.2797e-02,  6.1144e-02,
         -1.9538e-02,  5.9639e-02,  5.9399e-02, -3.5769e-02,  6.7004e-02,
         -6.7218e-02,  4.4063e-02,  2.8626e-02, -4.5297e-02,  1.5142e-02,
         -9.2750e-03, -6.0530e-02, -7.3987e-02, -3.2945e-02, -1.2728e-02,
         -5.5032e-02, -6.8505e-02,  4.0274e-02, -8.4567e-02, -7.1981e-02,
         -2.1304e-02, -5.1842e-02],
        [ 6.9717e-02,  1.3249e-02, -4.3610e-02,  5.0323e-02,  3.2474e-02,
         -6.4810e-03, -1.0028e-03, -3.8930e-03,  5.5008e-02, -1.1047e-01,
         -6.8424e-02, -3.7071e-02, -7.8239e-02,  2.0527e-02, -6.0431e-02,
          9.1607e-02,  5.6572e-02,  2.2194e-02,  1.8171e-02,  5.2451e-02,
         -6.6277e-02,  9.6893e-02,  9.3950e-02,  8.8334e-02,  9.8061e-02,
          7.4387e-02, -3.1528e-02, -8.8111e-02,  9.8447e-02, -1.4699e-02,
          6.4180e-02, -1.1423e-02,  4.8965e-02,  3.3676e-02,  1.8242e-02,
         -2.5947e-02, -8.8405e-02, -1.3500e-02,  7.0742e-02, -1.8616e-02,
         -1.9944e-02,  1.6049e-02, -3.7191e-02,  2.1232e-02,  4.7230e-02,
          1.1212e-02, -4.1496e-02,  8.1335e-02,  7.6086e-02, -1.1874e-02,
          9.8203e-02, -3.5286e-02,  4.7324e-02,  3.7297e-02, -4.3692e-02,
         -5.7527e-02,  4.8152e-02, -4.6822e-02,  3.7227e-02,  2.3596e-02,
          1.9873e-02,  7.3670e-02,  1.8412e-02, -2.0686e-02,  8.5718e-02,
         -2.9363e-02,  1.3743e-02,  4.4304e-02,  2.0588e-02,  1.2589e-01,
          1.0651e-01, -3.4318e-02, -5.4626e-02,  4.5489e-02,  1.0942e-01,
         -7.3045e-02,  4.1060e-04, -1.4183e-02, -4.0792e-02,  6.9410e-03,
          7.8355e-02,  7.5667e-02, -2.5496e-02,  2.0270e-02, -3.6408e-02,
          1.1699e-01, -3.2515e-02,  6.4687e-02,  1.1758e-01,  3.7319e-02,
          6.7930e-03, -2.6514e-02,  4.1455e-02,  2.5201e-02, -3.1252e-02,
          5.1227e-03,  1.1779e-01,  8.0894e-02,  4.2463e-02,  4.7342e-02,
         -4.7633e-02,  5.3779e-03,  8.3705e-02,  1.0945e-01,  5.5120e-02,
          2.9119e-02, -1.3751e-01, -3.6179e-02,  1.8389e-02, -5.0718e-02,
          1.5004e-03,  4.4336e-02,  9.1095e-03,  9.5864e-02,  1.6434e-02,
         -3.6755e-02,  3.7932e-02,  7.9820e-02,  5.0924e-02,  7.7906e-04,
         -3.0093e-02, -5.3022e-03,  6.7082e-02,  1.2701e-01, -1.3624e-02,
          6.2212e-02, -1.0468e-01,  2.5614e-02,  5.2767e-04, -9.2020e-03,
         -6.3263e-04, -6.5947e-02, -1.4677e-02, -4.3349e-02,  4.1339e-02,
         -4.6275e-02,  7.9586e-02,  5.4149e-02, -2.6063e-02,  1.1314e-02,
          5.9606e-02, -4.6580e-04, -2.9427e-02,  1.2111e-01,  1.1537e-01,
          7.1027e-03,  4.2286e-02, -3.9225e-02,  5.6034e-03,  2.3209e-02,
          2.9698e-03,  4.4021e-02,  6.1050e-02, -3.8029e-02,  2.1800e-02,
          1.5855e-02,  1.2927e-02,  3.2441e-02,  4.2590e-02, -3.6978e-02,
         -2.2050e-02,  3.9139e-02,  6.1129e-02, -6.6857e-02,  2.0862e-02,
          7.9807e-02,  2.4905e-02,  8.2121e-02,  7.9735e-02,  9.6374e-02,
         -6.9793e-03,  7.5331e-02,  9.7437e-02, -8.8984e-02,  3.6884e-02,
          2.3872e-02, -1.9929e-02, -1.8907e-02,  1.0969e-02,  5.2467e-02,
         -2.1293e-02,  6.2453e-02,  8.4485e-02, -7.9774e-02,  8.6431e-02,
          2.5939e-02,  3.8695e-05,  4.3498e-02, -2.7614e-02,  2.4212e-02,
         -1.1915e-02,  4.6913e-02],
        [-7.1078e-02,  1.2277e-02, -2.2789e-02, -1.8178e-02, -5.6601e-02,
         -3.2888e-02, -5.7637e-02, -1.2015e-02,  2.4636e-02, -9.3787e-02,
         -2.0443e-02, -3.3766e-02, -1.4535e-02, -2.4094e-02,  4.1657e-02,
         -4.1142e-02,  5.5801e-02, -7.1599e-02, -8.8500e-03,  3.4564e-02,
          3.9696e-02, -4.5977e-02,  3.0247e-02, -7.0661e-02, -7.3176e-02,
          2.7219e-04,  5.0767e-03,  3.5123e-02, -8.7259e-02,  2.8917e-03,
         -8.0468e-02, -8.6083e-02,  9.8395e-02,  3.6492e-02, -1.0175e-01,
          1.1767e-02,  1.6274e-02,  3.1339e-02, -4.6732e-03,  1.7859e-02,
         -6.9199e-02, -2.6157e-03, -9.4760e-02, -3.9551e-02, -5.2176e-03,
         -9.0270e-02, -4.7900e-02, -4.5114e-02, -9.4536e-04, -9.1734e-02,
         -2.9160e-02, -4.4098e-02,  5.4655e-02, -9.4204e-02, -3.3628e-02,
          5.8105e-02, -7.7078e-02, -1.7685e-02, -5.7624e-02, -5.5991e-02,
         -9.1398e-02,  2.4907e-02, -1.4246e-02, -4.1239e-05, -7.1473e-03,
         -7.7259e-02,  1.1283e-02, -2.5848e-02,  1.6787e-02,  4.3781e-03,
          2.6738e-02,  9.9441e-02, -7.7038e-02, -7.3471e-02, -2.4219e-02,
         -2.2903e-02, -5.1155e-02,  2.7899e-03,  4.1142e-02,  6.5075e-02,
         -7.2274e-03,  3.1444e-02,  1.2424e-02,  2.3889e-02,  2.5357e-02,
          4.2080e-02, -6.5154e-02,  1.5099e-03,  2.9156e-02, -7.5494e-02,
          6.1466e-02, -6.7567e-02,  3.9545e-02,  4.1256e-02,  6.1463e-02,
         -1.5928e-02, -9.1378e-02, -6.7597e-02, -1.1074e-02, -9.9709e-03,
         -9.9463e-02,  5.6236e-03, -1.7871e-02, -4.4974e-02, -3.2759e-02,
          4.2148e-03, -4.1116e-02, -4.5503e-04, -2.6334e-02,  9.4190e-03,
          2.1370e-02, -3.1148e-02,  7.0485e-03,  2.4643e-02, -4.8343e-02,
          1.7419e-02,  1.1476e-02, -5.6718e-02,  6.6242e-02, -7.1619e-02,
          5.4650e-02, -3.0148e-02, -4.0260e-02, -5.5215e-02,  1.0545e-02,
         -5.1273e-02,  4.2528e-02, -6.1981e-02,  4.1709e-02, -8.4718e-02,
          1.0978e-03,  2.9461e-02,  3.6677e-02, -7.6776e-02,  3.2611e-02,
         -7.7316e-03,  2.5046e-02, -5.0341e-02, -3.8445e-02,  3.0469e-02,
         -7.1347e-02,  3.6974e-03, -3.1792e-02, -2.1889e-02,  4.1259e-02,
         -2.7687e-02,  6.2478e-03,  4.0938e-02,  5.2501e-02, -6.0041e-02,
         -7.5264e-02,  1.5862e-02, -1.4571e-03, -9.0090e-02,  3.2325e-02,
         -3.5101e-02, -3.7168e-02, -3.4001e-02,  4.9624e-03,  8.0108e-03,
         -4.1275e-02, -4.5996e-02,  6.1822e-02,  3.9456e-02,  4.5207e-03,
          8.9212e-03, -3.7413e-02, -5.3971e-02, -1.7627e-02, -9.8562e-02,
          1.0898e-02, -1.8221e-02, -4.2355e-02, -6.3856e-02,  3.2271e-02,
         -6.9811e-02, -5.1918e-02,  7.1861e-02, -6.0622e-02,  4.9746e-02,
         -5.3213e-02, -7.8174e-02,  3.2561e-02,  2.3319e-02, -3.4522e-02,
          1.3241e-02,  3.5623e-02,  2.5769e-02, -6.9643e-02, -9.9539e-02,
          1.2149e-02, -4.7071e-02],
        [ 3.7923e-02, -7.5169e-03,  5.9816e-02,  4.9058e-02, -5.1268e-02,
         -8.7503e-05, -1.5174e-02,  4.1883e-02, -2.3784e-02,  6.5598e-03,
          5.2344e-02,  1.0656e-02, -6.3932e-02, -4.9745e-02,  6.8421e-02,
          3.8534e-03,  7.1072e-02,  2.0643e-02, -4.2257e-02,  1.0234e-02,
          1.5092e-02,  8.1967e-03,  4.4257e-02,  4.3137e-02, -2.1609e-02,
          1.6043e-02, -4.4660e-02, -6.9250e-02, -1.5102e-03, -3.7906e-03,
         -2.6286e-04, -2.8062e-02, -3.4588e-02,  3.1948e-02,  8.6619e-03,
          6.2884e-02,  1.0255e-02,  2.1071e-02,  2.5220e-03, -6.6254e-02,
         -3.6942e-02, -2.6042e-02, -6.5607e-02, -6.3044e-02, -1.3673e-02,
         -9.8759e-03, -1.7057e-02,  4.1973e-02, -3.1342e-02,  5.1843e-02,
         -3.8543e-02, -3.6574e-02, -6.7706e-02, -1.7904e-02, -3.1779e-02,
          1.1027e-02, -2.7980e-02,  4.0132e-02, -3.3348e-02,  8.4848e-03,
         -5.1432e-02, -7.6357e-03, -6.7313e-02,  9.5719e-03, -3.2637e-02,
          4.5508e-02,  3.5110e-02, -1.0616e-02, -7.0784e-02,  6.8493e-02,
          1.3838e-02,  6.8796e-02,  7.9679e-03, -6.4007e-02,  2.9812e-02,
         -7.0533e-02, -5.2521e-02, -6.0064e-02, -2.4310e-02, -4.1714e-02,
         -2.2073e-02,  4.2706e-02, -5.6280e-02,  4.7363e-02,  5.5673e-02,
         -2.2095e-02,  3.4374e-02,  4.2531e-02,  1.4275e-02, -6.3343e-02,
         -6.6890e-02,  6.6091e-02,  1.4275e-02, -5.4268e-02, -5.9598e-02,
          4.7192e-02, -6.7232e-02, -9.1733e-03,  3.8139e-02, -6.1869e-02,
          6.6640e-02,  7.8820e-03,  1.2072e-02,  3.9329e-02,  5.3011e-02,
         -3.0782e-03,  5.7609e-02, -3.1925e-02,  7.5849e-03, -4.9008e-04,
         -2.5597e-02,  1.7722e-02, -5.3949e-02, -3.9622e-02, -8.9974e-03,
          1.5917e-02, -1.0512e-02,  5.0888e-02, -2.9723e-02,  5.9618e-02,
          5.6579e-02, -3.5513e-02, -3.1427e-03,  5.0037e-02,  1.5925e-03,
         -6.1233e-02,  5.9133e-02,  3.3948e-02, -6.1055e-03, -6.0499e-02,
         -2.4432e-02,  6.7148e-02,  3.4904e-02, -4.7433e-02,  3.1431e-02,
         -6.0791e-02,  6.3268e-02, -2.6954e-02, -1.9332e-03, -3.9891e-02,
         -4.7240e-02,  3.6740e-02, -6.7181e-02,  3.2062e-02,  4.7474e-02,
          2.8531e-02, -5.1630e-02,  3.3362e-02, -4.8469e-02,  1.0071e-02,
         -4.1002e-02, -2.0561e-02, -2.9218e-02, -6.2097e-02, -3.7211e-02,
          2.7337e-02, -8.1546e-03, -1.0248e-02,  3.3226e-03,  4.6869e-02,
          4.6195e-02,  4.5531e-02,  5.2952e-02, -4.7069e-02,  1.7322e-02,
          5.6867e-02, -5.1888e-03,  5.7636e-02, -2.2138e-02, -2.5538e-02,
          6.6185e-03,  3.2388e-02, -4.8129e-02, -5.7706e-02,  1.0501e-02,
          1.7646e-02,  6.8640e-02,  2.1025e-02, -5.4826e-03, -6.3385e-02,
          1.4968e-02,  6.8134e-02,  4.4938e-02, -2.9246e-02, -5.4175e-02,
          1.9238e-02, -6.1851e-02, -5.6034e-03, -1.8055e-02,  2.7670e-02,
         -1.5871e-02, -1.3652e-02],
        [ 6.4629e-02,  3.8781e-02,  1.2420e-02, -5.0219e-02,  6.7838e-02,
          2.1689e-02,  5.2072e-02,  2.4756e-02,  6.2880e-02, -9.6349e-02,
          3.3015e-02, -2.8484e-02,  7.6153e-02,  4.3232e-02,  9.8538e-03,
          1.2165e-02,  8.8154e-02, -2.6226e-03,  6.2668e-02,  9.5773e-03,
         -3.4882e-03,  7.6532e-03, -1.3948e-02,  3.9700e-02,  2.1428e-02,
          5.8766e-02, -1.9927e-02, -6.2458e-02,  9.2353e-02,  2.2253e-02,
          3.1326e-02,  8.2956e-03,  2.0054e-02,  3.9559e-02, -2.7911e-02,
          5.3027e-02,  5.4251e-02,  2.5765e-02,  2.7047e-02, -3.3942e-02,
          3.6250e-02,  1.0557e-01, -7.9445e-03, -2.8313e-02,  5.0062e-02,
          1.7043e-02, -6.5985e-02,  1.1006e-01, -4.9961e-02,  4.7915e-02,
          8.2769e-02,  1.3136e-02,  4.2311e-02, -6.3304e-02, -1.7894e-02,
         -2.2991e-02, -8.2260e-04,  6.9018e-02,  2.2913e-02, -2.5078e-02,
          4.1083e-02, -1.5701e-03,  2.4681e-02,  8.9355e-02,  8.4003e-02,
          6.8345e-03,  3.9177e-02,  3.6265e-02,  2.3336e-02,  5.2122e-02,
          8.9491e-02, -5.4258e-02, -6.9280e-02, -3.1207e-02, -1.9026e-02,
          1.7909e-02, -1.9459e-02,  3.1671e-02,  3.0325e-02, -1.8760e-03,
         -2.4138e-02,  2.9728e-02,  5.4929e-02,  6.7077e-03,  7.5356e-02,
          6.8201e-02, -7.6122e-03,  9.6429e-02,  1.4909e-02, -6.4869e-02,
         -3.8425e-02,  4.6730e-02,  2.5618e-03,  5.9937e-02, -8.6643e-03,
         -1.7343e-02, -1.5949e-02, -4.1748e-02,  5.7397e-02, -2.9207e-03,
         -3.8144e-02, -1.8385e-02,  3.3894e-02,  4.1603e-02,  7.5196e-02,
          4.6612e-02, -3.5605e-02,  5.6133e-02,  2.3028e-02,  7.1476e-02,
         -1.1223e-03,  3.7468e-02,  8.5418e-03,  7.1629e-02,  3.0958e-03,
         -1.6840e-02, -1.7460e-02,  3.0795e-02,  8.6016e-02, -3.2078e-02,
         -3.8867e-02,  5.7581e-02, -7.0514e-02,  1.9822e-02,  3.7314e-02,
         -1.7529e-02, -6.0176e-02, -6.9749e-02, -2.3716e-02,  1.0586e-01,
          1.6273e-02, -6.8846e-02,  8.9141e-02, -8.4966e-03, -4.3703e-02,
         -9.5577e-02,  5.9437e-02,  2.1620e-02, -1.1583e-02,  4.1056e-02,
          4.0781e-02, -5.4596e-02, -3.7923e-02, -7.6586e-03,  1.0856e-01,
         -2.3821e-02,  7.5116e-02, -1.3050e-02, -5.7939e-02,  6.7257e-02,
          7.0991e-02, -2.4911e-02,  3.2081e-02, -4.0930e-02,  3.5066e-02,
         -4.8836e-03, -7.1155e-02,  2.0010e-02, -4.6878e-02, -5.8755e-02,
         -1.8497e-02,  5.7024e-02,  1.3946e-02,  2.1139e-02,  7.0820e-02,
          8.7701e-02,  1.0342e-02,  1.0256e-01,  7.5518e-02, -3.6863e-02,
          6.7675e-02,  2.7070e-02,  9.9031e-02,  8.2084e-02, -3.2160e-02,
         -3.8124e-02,  4.8224e-02, -5.8006e-02, -1.1405e-02, -1.7219e-02,
         -5.5139e-03,  5.5711e-02,  1.0143e-01, -1.9486e-02,  3.0618e-02,
          4.5444e-02,  5.2062e-02, -2.9657e-02,  4.4859e-02,  2.9805e-02,
         -1.3246e-02,  9.3006e-02],
        [ 2.8019e-02, -5.5670e-02,  2.7646e-02, -4.7955e-02, -1.5321e-02,
         -2.4681e-02, -2.2114e-02, -5.6939e-02,  2.3889e-02, -9.9311e-02,
         -7.8660e-03, -7.3778e-03, -1.0155e-01,  5.5470e-02, -5.4954e-02,
          1.0663e-01,  2.6296e-02,  3.2369e-02,  7.1384e-02,  3.5891e-02,
          8.0107e-03,  1.8931e-02,  3.0744e-02,  8.2942e-02,  8.5499e-02,
          1.4153e-02,  1.1637e-03, -9.2514e-02,  6.9135e-02,  6.2574e-02,
          5.5666e-02, -2.6315e-02,  8.3485e-03, -4.7840e-03,  9.5363e-03,
         -5.5092e-02, -3.5077e-02, -1.6112e-02,  2.1637e-02,  2.5633e-02,
          8.1651e-02,  1.0465e-01,  1.8535e-02,  7.4489e-02, -7.5541e-02,
          2.0380e-02, -1.1500e-02,  1.8002e-02, -4.8671e-02, -4.6063e-03,
          7.3970e-03, -1.7275e-02,  2.0719e-02,  2.8472e-02, -5.2368e-04,
          7.3387e-02,  4.2211e-02, -4.4606e-02,  1.0549e-01,  2.5952e-02,
          8.3808e-02, -6.5723e-03, -4.6696e-02,  2.9185e-02,  1.6151e-02,
          8.7295e-02,  3.9127e-02,  2.4787e-02,  1.1351e-02,  1.0623e-01,
         -1.4068e-02,  1.3431e-02, -9.2805e-02,  9.4131e-02,  5.7063e-02,
          2.1388e-03, -2.7813e-02, -3.4201e-02, -4.7369e-02, -3.2899e-02,
          2.5019e-02, -1.6738e-02,  7.2116e-03, -4.7264e-02, -6.5225e-02,
          7.9777e-02, -8.2092e-03,  4.8183e-03,  3.4216e-03, -3.8015e-02,
          6.8135e-02, -6.7633e-02, -8.2488e-03, -5.8850e-04,  3.9359e-02,
          2.9008e-02,  1.1218e-01,  8.4811e-02,  7.0077e-02, -2.7042e-02,
         -9.1591e-02, -3.0782e-02, -1.6526e-02, -5.7753e-03,  7.8726e-02,
         -2.9136e-02, -8.4970e-02, -3.7818e-02, -6.9091e-05,  3.8057e-02,
          6.8168e-02, -2.7565e-02,  4.4195e-02, -2.4602e-02,  1.5384e-02,
          7.8051e-02, -4.7740e-02,  5.8315e-02,  1.3366e-02, -4.0864e-02,
         -8.3417e-02,  2.7486e-02, -4.5459e-02,  4.5727e-02,  4.7998e-02,
          7.9945e-02, -6.1153e-02,  7.2112e-02, -4.0097e-02,  1.2421e-01,
          3.9859e-02, -7.5710e-02,  8.3802e-02,  5.3783e-02, -1.9467e-02,
         -4.4311e-02,  2.5696e-02, -2.9614e-02, -6.8077e-02,  5.5019e-03,
          6.6054e-02,  8.0844e-02, -8.9659e-02,  5.2143e-02, -1.6711e-02,
          2.3143e-02,  1.0171e-01,  2.7673e-02, -6.8119e-02, -2.1830e-03,
          3.1230e-02,  6.4336e-02,  5.6350e-02, -5.7993e-02,  9.4345e-02,
          1.5659e-02, -3.5325e-02, -1.1757e-02,  2.7755e-02,  9.5296e-02,
          4.5496e-03,  6.0168e-02,  4.2269e-02, -8.1608e-02, -1.7700e-02,
         -3.8948e-02, -8.7451e-02,  1.1229e-01,  2.2889e-02,  5.8839e-02,
          4.7256e-02, -1.2271e-02,  2.2889e-02, -5.6450e-02,  5.1696e-02,
          3.9877e-02, -6.6838e-02, -7.6434e-02, -1.9889e-02,  1.7331e-02,
          1.7429e-02, -3.4600e-02,  1.4831e-02,  1.0089e-02,  9.3238e-02,
         -6.1587e-02,  1.1094e-01,  2.8130e-02,  1.8079e-02, -1.2812e-02,
         -1.2251e-02, -3.2784e-02],
        [-4.8218e-04, -6.3312e-02, -1.1809e-02, -4.2541e-02, -5.5267e-03,
          6.4590e-02,  3.8389e-02, -5.6895e-03, -5.4605e-02, -9.3821e-02,
          1.4718e-04,  4.5857e-02, -8.9290e-02,  6.4903e-02,  3.9105e-02,
          8.9739e-02,  3.5929e-02, -1.9049e-02,  5.2120e-02, -2.1172e-02,
         -2.4797e-02,  4.3011e-02, -4.5681e-03,  3.3540e-02, -8.4820e-03,
         -2.7270e-02,  7.4559e-03,  1.3097e-02,  2.9887e-02, -2.5364e-02,
          8.7684e-02, -2.9713e-02, -9.1055e-03, -1.2487e-02,  6.2699e-02,
          3.8853e-02, -7.7207e-02,  4.3034e-02,  4.5885e-05,  6.0691e-02,
          7.3683e-02,  1.3446e-02,  5.9858e-02, -5.5450e-02, -1.3936e-02,
          1.7372e-02, -3.0587e-02,  7.7713e-03,  2.1655e-02,  5.1735e-02,
          7.0225e-02, -6.6166e-02,  5.7007e-02, -7.8079e-03, -5.2000e-02,
         -3.6590e-02,  9.9689e-03, -4.9791e-02,  8.3812e-02,  7.0790e-02,
          6.3503e-02,  5.3793e-02,  7.0633e-02,  6.8846e-02,  7.3060e-02,
          3.0244e-02, -8.3953e-02,  5.0060e-02, -3.0484e-02,  7.6388e-02,
         -3.1283e-02, -4.8921e-02, -2.7459e-02,  5.9072e-03,  2.5785e-02,
         -2.5760e-02, -6.2946e-02,  1.2217e-02,  4.9402e-02,  3.9733e-02,
         -8.1958e-03,  5.0816e-02, -3.2370e-02,  7.4187e-02,  5.3130e-03,
          1.0124e-01,  3.9884e-02, -3.5187e-02,  9.4280e-02, -3.3357e-02,
         -4.6658e-02,  3.5277e-03,  6.7162e-02,  6.4332e-02, -8.3846e-02,
          1.5243e-02,  5.0393e-02,  3.1973e-02, -3.0028e-02,  2.3988e-02,
         -4.3267e-02,  5.0408e-02,  1.0010e-02,  5.9215e-02,  8.6832e-02,
         -1.6706e-02, -5.8910e-02,  2.6633e-02,  7.7008e-02,  7.2322e-04,
         -9.1618e-03,  4.0196e-02,  3.9145e-02,  7.9644e-02, -6.2735e-02,
          4.6109e-02,  4.6530e-02,  7.4736e-02,  4.5720e-02, -3.9862e-02,
          1.7853e-02, -1.7444e-02,  8.8835e-03,  4.6134e-02, -3.5528e-02,
         -2.7461e-04, -7.2977e-02, -3.5790e-02,  2.8754e-02,  7.4927e-02,
          7.2450e-02, -2.8274e-02,  8.4459e-02, -2.8311e-02, -6.3535e-03,
         -3.7652e-02, -2.9848e-02,  1.6004e-03,  1.8223e-03, -5.0110e-02,
          1.0346e-01, -1.2725e-02, -1.8066e-02,  8.3073e-02,  1.3332e-02,
          1.0607e-02, -7.4770e-03, -2.0510e-02, -9.1133e-02, -1.8724e-02,
         -1.5710e-02,  8.7783e-02, -2.3412e-02, -5.1918e-02, -1.2258e-02,
         -6.1764e-02, -3.8403e-03,  1.3498e-03, -6.4432e-02,  5.8736e-03,
          6.9987e-02,  1.2042e-02,  4.5837e-02, -5.3788e-02,  4.3725e-02,
         -2.8089e-02, -2.7113e-02, -1.3432e-03,  6.4922e-02,  9.2371e-02,
          1.3329e-02, -2.0296e-02, -2.9104e-02, -4.6910e-02,  4.7798e-02,
         -4.6130e-02, -1.1530e-02, -2.3407e-02,  6.3353e-02,  3.5398e-02,
          1.3992e-02,  1.4027e-02,  9.2339e-02, -2.5958e-02, -1.0500e-02,
         -8.1183e-02,  3.2772e-02, -4.3976e-02,  8.1703e-02,  9.0350e-02,
         -5.8386e-02, -2.6782e-03],
        [-2.1367e-02, -6.1248e-02,  1.9266e-02,  8.1718e-03,  1.6843e-02,
          3.2686e-02,  2.9030e-02,  7.4731e-02,  6.0610e-02, -4.2646e-02,
         -2.9518e-02,  3.8427e-02, -1.0718e-01,  5.5149e-02, -9.5725e-02,
         -9.8920e-03,  1.2582e-01,  1.8431e-03, -3.1444e-02,  4.4677e-02,
         -7.0126e-02,  8.9876e-02, -4.1038e-02, -8.0504e-03,  5.6221e-02,
          6.2710e-02,  8.2706e-02,  2.2212e-03,  1.5662e-03,  5.2032e-02,
          2.6717e-02,  9.7483e-02,  4.8340e-02, -8.2856e-02,  4.9735e-02,
          2.6267e-02, -7.7598e-02,  7.4218e-02,  8.9670e-02,  2.8969e-02,
         -1.7625e-02,  5.1556e-02,  9.1872e-02,  7.1991e-02,  1.9953e-02,
          5.9852e-02,  2.6398e-02,  1.1474e-01,  1.0138e-01,  1.2199e-01,
          7.7502e-02,  1.4112e-02, -5.0562e-02,  9.2229e-04, -7.4884e-02,
          7.7316e-02, -4.6963e-03, -5.7912e-02,  8.3468e-02,  8.9153e-02,
          9.3744e-02,  6.5385e-03, -1.9635e-02,  1.6017e-02, -2.5240e-02,
          1.0101e-02,  6.2571e-02,  7.9232e-02, -1.0717e-01,  1.1444e-01,
          1.1982e-02, -4.2122e-02,  6.2469e-02,  1.1091e-01,  1.2394e-01,
          8.4527e-02,  8.3119e-02, -5.4074e-02, -6.5543e-02, -3.0901e-02,
          8.4052e-02, -5.8487e-02,  7.4291e-02, -1.0344e-01, -1.8538e-02,
          2.1146e-02, -2.9883e-02, -1.2289e-02,  4.2073e-02,  2.7749e-02,
          1.8794e-02, -6.6958e-02,  3.5778e-02,  9.6316e-02, -2.4877e-02,
          1.4683e-03,  1.1956e-01,  9.8088e-02,  7.9247e-02, -7.7147e-03,
         -9.8395e-03,  6.7479e-02, -9.6554e-03,  1.3954e-02,  6.1464e-02,
         -2.3957e-03, -4.1505e-02,  2.0588e-02,  5.3279e-03,  4.2253e-02,
          5.6917e-02, -4.6803e-03,  5.1225e-02,  3.6625e-02, -4.2721e-02,
          8.9927e-02, -7.6122e-02,  5.1928e-02,  3.3082e-02,  1.7567e-02,
         -4.6918e-02, -7.2961e-02,  8.8529e-02,  1.6886e-02, -2.9857e-02,
          8.4397e-02, -1.2631e-01, -2.7790e-03,  7.5959e-02,  2.2423e-02,
          1.3688e-01, -3.6202e-02,  9.3442e-03,  1.2108e-01, -2.4375e-02,
         -7.2011e-02,  5.4799e-02,  1.0331e-01,  2.3348e-02, -8.8148e-02,
          5.1730e-02, -2.1261e-02, -1.7205e-02,  1.1375e-01,  2.0483e-02,
         -8.6025e-02,  7.2659e-02, -4.5632e-02, -1.0474e-01,  4.9808e-02,
          5.8356e-02,  1.0782e-01, -1.2491e-04, -2.9732e-02, -6.6461e-04,
         -3.5102e-02,  1.9034e-02, -3.6924e-02,  4.4606e-02,  1.6422e-02,
         -2.0988e-02, -3.8396e-02,  5.0758e-02,  1.0015e-02,  1.3865e-01,
          1.3910e-02, -1.8779e-02, -1.1996e-03,  7.3924e-02,  8.5594e-02,
          8.6930e-02, -2.3078e-02,  3.8651e-02,  7.7233e-02, -2.4962e-02,
         -4.4770e-03,  1.4763e-03, -1.7143e-02, -3.4595e-02, -2.6960e-02,
          5.5793e-02,  7.1155e-02,  2.2900e-02, -7.1959e-02,  1.0258e-01,
          5.2383e-02, -2.0955e-02, -1.6068e-02,  8.1475e-02, -1.5372e-02,
         -3.4689e-02, -2.3301e-02],
        [-2.0667e-02,  3.8781e-02,  4.1038e-02, -2.7997e-02, -5.8871e-02,
         -4.3477e-02,  6.1740e-02, -8.6163e-02,  7.1787e-03, -7.0149e-04,
          6.4331e-03, -6.1726e-04, -1.3623e-02,  1.9921e-02,  3.2512e-02,
          4.7754e-02,  9.5341e-02, -9.1401e-03,  8.9737e-03,  6.8386e-02,
         -6.9596e-02,  4.6339e-02,  6.6234e-02,  7.7533e-03,  1.0344e-01,
          5.7336e-02, -4.1293e-03, -2.4970e-02,  2.6432e-02,  1.9048e-02,
         -2.4207e-02,  8.1964e-02, -2.3702e-02, -8.1808e-02,  1.7056e-02,
          2.7137e-02, -1.1893e-02, -1.2845e-02,  1.6960e-02,  3.9102e-02,
          4.7066e-02,  3.4489e-02,  5.0221e-02,  1.5332e-02,  5.0831e-02,
          5.1379e-02, -8.7266e-02,  1.0835e-01,  6.8649e-02, -3.2458e-02,
          2.6597e-02,  4.8439e-02,  1.7952e-02, -2.8137e-02, -7.0632e-02,
          4.3958e-02, -1.8100e-02, -7.3660e-02,  5.7398e-02,  3.3616e-03,
          2.6109e-02,  3.7306e-02, -6.2157e-02,  7.9770e-03,  3.3713e-02,
          2.6592e-02, -1.2975e-02, -4.4139e-02, -5.8476e-02,  1.1449e-01,
          1.0528e-01, -2.2774e-02, -7.8471e-03,  2.1119e-02, -1.0699e-02,
         -1.9884e-02, -1.9330e-02,  6.0336e-03,  2.9747e-02, -1.3702e-02,
          3.7608e-02, -2.9524e-02, -1.3189e-02, -1.2439e-02,  1.6336e-03,
          5.4166e-02,  2.8675e-02,  8.5352e-02,  5.9350e-02, -1.0524e-01,
         -2.4442e-02, -4.0446e-02,  3.6619e-02,  2.5378e-02, -1.6262e-02,
          2.4679e-02,  1.2487e-02,  6.8267e-02,  1.3713e-02,  4.2463e-02,
         -6.4779e-02, -8.7604e-02, -4.1158e-02,  2.2735e-02,  1.8562e-02,
         -4.9422e-02, -1.3388e-01,  8.0555e-02, -2.2216e-03,  4.0456e-02,
         -3.3659e-02, -5.8466e-02, -5.8378e-02, -3.0675e-02,  4.3992e-02,
         -3.9529e-02,  4.9970e-03,  6.2116e-02,  6.6650e-02,  1.6477e-02,
         -7.1414e-03, -3.8573e-02,  4.3463e-02,  5.2385e-02, -3.0474e-02,
          4.2425e-02, -8.7284e-02, -7.5059e-02,  2.8364e-02,  6.9310e-02,
          8.1196e-02,  2.9004e-02,  1.0965e-01, -1.2753e-02, -8.2168e-02,
         -9.4535e-02,  5.6270e-02,  3.8323e-02,  3.9733e-03, -8.9038e-02,
          4.0479e-03,  3.4712e-03,  5.8842e-03,  2.3096e-03,  1.1015e-01,
         -1.6309e-02,  5.2936e-03, -5.6570e-02, -1.1078e-01,  7.8903e-02,
          5.1394e-02,  8.4073e-02,  5.1063e-02,  2.9976e-02,  6.7440e-02,
         -2.3569e-02,  1.7621e-02, -5.0957e-02, -5.4055e-02, -2.1238e-02,
         -7.3970e-03, -2.1025e-02, -2.8022e-02, -8.5485e-02,  1.9179e-02,
          3.3839e-02, -2.8336e-02, -1.2980e-02, -2.8106e-02, -3.6721e-02,
         -3.7371e-02,  2.2375e-02, -3.8113e-02, -6.3090e-02, -6.7715e-02,
          1.2527e-02, -4.4894e-02,  2.1266e-02, -3.8761e-02,  2.8882e-02,
          3.9691e-02,  8.6715e-02,  3.5905e-02, -1.6445e-02,  3.6563e-02,
         -6.4986e-02,  1.2947e-01,  7.0833e-02,  1.7246e-02,  6.6034e-02,
          3.7206e-02,  8.7160e-02],
        [-2.7849e-02,  2.4233e-02, -1.9502e-02, -4.5036e-02, -4.0589e-02,
         -1.6974e-02, -2.2672e-02,  1.5217e-03,  4.5724e-02,  4.8130e-03,
         -4.2489e-02, -7.2305e-02, -4.4585e-03, -6.8137e-02, -5.9950e-02,
          1.8067e-02,  9.0325e-03,  3.8553e-02, -2.1295e-02, -5.6076e-02,
         -2.4147e-02, -5.1241e-02,  1.6453e-02, -1.4769e-02,  5.4060e-04,
          5.8353e-02, -4.4614e-02, -6.1555e-02, -4.8178e-02, -7.0324e-02,
          3.4876e-02,  5.4612e-02, -6.2139e-02, -1.4220e-02,  1.8243e-02,
          6.5894e-02,  2.9447e-02, -5.0781e-02,  2.5855e-02, -2.2746e-02,
         -4.4699e-02,  3.3483e-02, -2.4196e-02,  5.8453e-02,  1.6098e-02,
         -3.6109e-02,  7.2081e-03,  5.4004e-02,  5.8829e-04, -9.5178e-03,
          2.9056e-02,  2.9415e-02,  3.7328e-02, -2.7486e-02,  6.0476e-03,
          2.8549e-02,  4.0898e-02,  6.5521e-02, -2.1776e-02,  5.5956e-02,
          3.2635e-02,  6.4580e-02, -6.8658e-02,  5.6289e-02,  5.4394e-02,
          3.0282e-02, -2.2768e-02,  4.0208e-02,  9.9676e-02, -5.1106e-02,
          5.7565e-02,  7.0012e-02,  1.1334e-01,  2.1875e-02,  7.8205e-03,
          8.8445e-03,  6.2293e-02,  4.2586e-02, -3.6946e-02, -4.7108e-02,
          8.4422e-03, -5.0471e-02, -6.1083e-02,  1.1139e-02, -2.0501e-02,
         -5.4169e-03,  3.5006e-02, -6.5637e-02, -5.2461e-02, -1.5757e-02,
         -3.2960e-02, -1.2928e-02,  2.7839e-02, -8.4966e-03,  5.9313e-02,
         -2.6990e-02,  2.0887e-02,  1.2572e-03, -4.8785e-03, -3.1025e-02,
          9.0057e-02,  1.7848e-02,  6.6037e-02, -5.0841e-02, -2.8026e-02,
          5.6848e-03, -2.8920e-03, -3.2574e-02,  1.9267e-04,  2.4802e-04,
         -2.2647e-02, -5.3196e-02,  6.6898e-02, -4.3406e-02, -6.5339e-04,
         -6.8718e-02, -8.4359e-03, -3.0936e-02,  5.9002e-02,  1.3670e-02,
         -1.4389e-02,  2.8905e-02, -4.9627e-02,  3.2085e-02, -7.3053e-02,
         -1.3372e-02,  3.1402e-02, -5.7706e-02,  6.8128e-02, -4.1939e-02,
          2.8480e-02,  3.0124e-02, -6.7316e-02,  3.1565e-02, -5.4374e-02,
         -2.6757e-02, -2.2413e-02, -3.9512e-02, -3.1114e-02, -2.1785e-02,
         -2.5187e-02, -4.9850e-02,  4.6871e-02, -6.8884e-03, -4.5091e-02,
          4.0234e-02,  1.8049e-02,  2.9408e-02, -5.3107e-02, -9.5122e-02,
         -4.1946e-02, -4.8398e-03, -1.2430e-02,  7.4269e-02, -1.7290e-02,
          1.2345e-02,  2.1681e-02, -5.6562e-02, -2.4416e-02,  2.2975e-02,
         -5.0983e-02,  2.7379e-02, -1.8485e-02,  7.5363e-02, -4.4146e-02,
          5.5965e-02, -3.0315e-03, -4.2049e-02, -4.7684e-02, -6.5676e-02,
         -1.5757e-02, -6.2731e-02,  3.8697e-02, -1.4793e-02, -3.8489e-02,
         -1.2519e-02,  6.1895e-02,  6.8072e-02,  4.5084e-03,  5.1065e-02,
         -6.1363e-02,  4.4991e-02,  6.8284e-02, -3.7502e-02, -4.4631e-02,
          2.4756e-02, -1.8371e-02, -2.6107e-02, -8.5133e-03, -2.7739e-02,
          7.0918e-02,  7.6813e-03],
        [-1.5107e-02, -1.1965e-02, -5.7519e-02,  4.7401e-02,  6.7036e-02,
          5.6335e-02,  2.4939e-02,  8.0263e-02,  3.9433e-03,  2.7837e-02,
          3.2597e-02,  2.6207e-02, -1.6939e-02,  3.4469e-02,  4.1236e-02,
         -1.2853e-01, -3.7118e-02, -1.8230e-02, -6.5629e-03,  2.9538e-02,
         -1.1127e-02,  6.8528e-03, -5.9918e-02, -3.7207e-02,  7.1108e-03,
          4.1817e-03, -4.2668e-03,  3.9931e-02, -7.3759e-02, -5.4093e-02,
         -6.5103e-02, -1.3141e-02,  6.4064e-02,  2.9128e-02,  4.7307e-05,
          7.4059e-02,  6.7666e-02,  1.2672e-02,  4.2103e-02, -3.2028e-03,
         -2.4799e-02, -4.3023e-02,  6.9244e-02, -1.4671e-03,  6.3580e-02,
         -7.2427e-02,  9.0743e-02,  2.9378e-02,  1.8293e-02, -7.2337e-02,
         -1.8364e-02,  5.9984e-02,  6.6241e-02,  8.6774e-02,  1.9290e-02,
          3.5053e-02,  6.6522e-02, -2.0633e-02, -5.4065e-02, -4.1558e-02,
          9.1733e-02, -1.4338e-02,  6.6002e-03,  6.6601e-02, -2.0225e-02,
         -5.8721e-03, -2.5917e-02,  4.4169e-02,  5.5132e-03, -1.8710e-02,
          6.0119e-03, -4.5771e-02,  5.2119e-02, -5.5841e-03, -3.7915e-02,
          5.7295e-02,  8.6171e-02,  2.3140e-02, -9.1054e-02,  4.7408e-02,
          4.3932e-02, -1.6877e-02,  4.0787e-02,  1.0652e-02,  5.1674e-02,
          4.6950e-02,  4.9582e-02,  6.7517e-02, -6.3670e-02,  5.8152e-02,
          9.3902e-02,  7.2260e-02,  3.2471e-02, -7.4593e-02,  4.3844e-02,
         -9.0180e-02, -1.1828e-02, -6.3880e-02,  3.6495e-02,  6.1398e-02,
          3.8050e-02,  1.3097e-01, -3.5271e-03, -5.1659e-02,  5.1551e-02,
          9.9231e-03,  1.3897e-01,  2.2199e-02,  8.7192e-03, -5.8833e-02,
         -3.2594e-02, -5.0658e-02, -6.3259e-02, -5.5649e-03,  7.7959e-02,
          7.9288e-03,  9.4446e-03, -1.8750e-02, -3.6396e-02,  4.2571e-03,
          3.8486e-02,  1.8838e-02,  6.9056e-02, -3.7315e-02,  2.9836e-02,
         -1.1179e-01, -1.3220e-02, -1.3392e-02,  9.3780e-02,  3.2029e-03,
         -1.1166e-01,  4.1862e-02, -9.7197e-02, -5.3328e-02,  3.7848e-02,
          5.2401e-02,  1.5040e-03, -4.6657e-02,  8.2937e-02,  5.3788e-02,
          6.4863e-03, -1.8398e-03, -1.3891e-02, -1.0175e-01, -1.3003e-02,
         -3.0025e-02, -7.3251e-02,  8.5726e-02,  3.5144e-02, -3.1715e-02,
          3.4352e-02, -8.0854e-02, -9.8085e-03, -4.0235e-02, -7.4443e-02,
          8.5622e-04, -5.5936e-02,  3.6455e-02,  6.9599e-02, -7.3765e-02,
         -3.5989e-02,  1.3632e-02,  5.8111e-02,  1.7841e-02, -6.9025e-03,
         -6.8585e-02,  2.8918e-02, -1.1545e-01, -2.4446e-02, -2.4262e-02,
          9.5111e-02, -2.0774e-02,  1.3603e-03, -3.0139e-02,  3.9432e-02,
         -6.9239e-02,  9.8330e-03,  3.8830e-02,  1.4445e-02,  5.0975e-02,
          5.2482e-03,  6.2839e-02,  2.5841e-02,  6.0796e-02, -9.7068e-02,
          4.1254e-02, -1.3274e-02,  5.8158e-02,  2.2281e-03,  8.7985e-02,
         -5.6630e-04,  8.4469e-03],
        [ 4.8852e-02, -1.4758e-02, -5.9525e-02,  6.9256e-02, -3.1520e-02,
          2.6238e-02,  1.8536e-02, -9.2988e-02, -3.9526e-02, -5.6811e-02,
          1.8383e-02,  5.5245e-02, -5.6857e-02, -3.1057e-02,  3.9707e-02,
         -8.8126e-02,  1.2430e-02,  2.5371e-02,  5.9876e-02,  4.2153e-02,
          3.5053e-02,  3.7125e-02, -5.2402e-02, -4.3949e-02,  2.7686e-02,
         -4.2778e-02, -4.7289e-02, -4.3515e-02,  4.1484e-02, -3.4000e-02,
          9.0349e-02,  3.4899e-02, -1.8837e-02,  2.1551e-02,  2.0252e-02,
         -7.8605e-02, -3.1553e-02, -1.3964e-02, -6.1941e-02, -6.3274e-03,
         -6.3133e-02, -7.1506e-02, -4.9281e-02, -1.3594e-02, -7.8660e-02,
          2.3246e-02, -1.0017e-01, -8.9917e-02, -1.3195e-02,  2.5611e-02,
         -4.3154e-02,  1.7604e-02,  4.1275e-02, -7.6681e-02, -4.6455e-02,
          4.2027e-02, -3.9559e-02, -8.1673e-02,  2.5880e-02, -5.6752e-02,
          3.3118e-02,  9.3060e-02,  3.8464e-02,  1.0181e-02, -4.1230e-02,
         -6.6770e-02, -1.6227e-02,  6.6582e-03,  2.3729e-02,  3.4903e-02,
          5.1887e-02, -5.3811e-02, -6.6709e-02,  4.0121e-03, -2.9100e-02,
          1.4505e-02, -3.6945e-02, -7.2819e-02, -1.0697e-03, -2.1480e-02,
          5.4317e-02, -5.0003e-02, -5.5311e-03,  1.6787e-02, -1.2087e-02,
         -1.3208e-03, -1.9837e-02, -3.7266e-02, -5.9088e-02,  3.9968e-02,
          4.5802e-02, -5.6089e-03, -6.1334e-03,  7.1923e-02, -7.5632e-02,
         -1.3488e-02,  2.6045e-02,  4.0789e-02, -1.8968e-02, -2.6243e-02,
         -9.5584e-03, -8.8634e-02,  1.3056e-02, -7.0239e-02, -7.0068e-03,
         -3.0563e-02, -6.0468e-02,  4.1582e-02, -5.6567e-02,  8.8165e-02,
          2.5900e-02, -1.8314e-02, -3.4494e-02,  3.6977e-02, -1.8431e-02,
          1.0611e-02, -4.5721e-02, -5.9734e-02,  6.7945e-02, -4.1507e-02,
          5.7573e-02,  2.6595e-02, -6.2061e-02, -2.5953e-03,  4.1412e-02,
          3.1254e-04, -6.5445e-02,  1.6923e-02, -2.5855e-02, -4.2574e-02,
         -3.3500e-02, -3.6152e-02, -1.4996e-02, -4.0188e-02,  2.8778e-02,
          7.2051e-02,  3.7417e-02, -3.7306e-02, -1.8042e-02,  1.9780e-02,
          4.8729e-02, -4.4290e-02,  3.0585e-02,  1.3991e-02, -3.3042e-02,
          4.2434e-02, -6.5066e-04, -6.5725e-02,  5.5255e-03, -4.2796e-02,
         -7.3875e-02, -2.7958e-02, -2.2600e-02,  1.8978e-03, -2.3210e-02,
         -5.9422e-02, -3.0235e-02,  4.3747e-02, -3.1546e-02,  5.4582e-02,
         -2.4297e-02, -6.1988e-02,  2.8537e-02,  1.8048e-02,  2.4167e-02,
         -9.0230e-02, -8.4471e-03,  1.3935e-02, -6.1203e-02,  9.1836e-03,
         -2.2613e-02, -7.0552e-02, -6.1605e-02, -9.2881e-03, -5.1902e-03,
          3.0881e-02,  4.7098e-02,  4.0904e-02,  3.3114e-02,  1.2124e-02,
         -8.9481e-02, -2.1411e-02, -6.6383e-02, -1.9486e-02,  3.4065e-03,
         -9.5860e-02, -2.5548e-02, -3.9324e-02, -2.5632e-03,  3.4028e-02,
          2.0478e-02, -2.1177e-02],
        [-3.8097e-02,  3.2666e-02,  7.5479e-04,  1.7520e-02, -3.0136e-02,
          9.9658e-03,  1.3667e-02, -1.8712e-02,  5.0924e-02,  2.7224e-02,
         -4.6424e-02,  6.6125e-02, -2.6962e-02, -6.4794e-02,  1.0526e-02,
         -1.9799e-02,  4.7385e-02, -6.5489e-02, -1.5715e-03, -6.0916e-02,
         -2.6354e-02,  3.6019e-02,  3.9631e-02, -3.0862e-02, -6.0209e-02,
         -1.7722e-03, -6.0230e-02, -1.6457e-02,  3.0491e-02,  1.1044e-02,
         -8.4232e-02,  1.7554e-02, -6.3113e-02,  3.5249e-02, -6.2440e-02,
         -3.4231e-02, -5.1317e-02, -3.1028e-02, -3.3857e-02,  6.0587e-03,
         -8.4594e-02,  1.0667e-03, -3.4852e-02, -4.4581e-02, -7.2523e-02,
         -8.7653e-02,  5.8344e-02,  2.1267e-02, -6.7274e-03,  1.8745e-02,
         -4.0889e-02, -5.2460e-02,  2.4281e-02,  4.4336e-02, -5.7148e-02,
         -4.7362e-02,  5.1194e-02, -7.5232e-02,  1.0238e-02, -7.5018e-04,
         -2.1540e-02, -4.3454e-02, -4.8424e-02,  6.4579e-02, -7.1417e-02,
          3.1636e-02, -4.1896e-02,  5.8394e-02, -2.1709e-02, -7.3299e-02,
          4.7304e-02, -3.4084e-02,  2.3125e-03, -8.9278e-02,  5.1071e-02,
          5.8135e-02,  6.0440e-02,  9.9197e-02,  4.2083e-02,  2.6953e-02,
         -7.6864e-02, -3.2238e-02,  4.3223e-02, -3.0713e-03,  6.4222e-02,
         -5.9608e-03, -6.3077e-02, -4.0908e-02, -3.1671e-02, -3.1336e-02,
         -4.3806e-02,  6.3510e-02,  4.4807e-03,  7.1544e-02,  6.9433e-02,
         -5.4152e-02, -8.5720e-02,  4.8247e-02, -6.3432e-02, -1.6656e-02,
         -1.3131e-02,  2.4242e-02, -5.7916e-02, -8.1054e-02,  2.2326e-03,
          1.8000e-02,  2.0348e-02,  8.4399e-03, -1.3827e-04,  7.3894e-03,
          3.0949e-02, -2.9515e-02,  5.5568e-02, -1.1615e-02, -7.3031e-03,
          5.8486e-02,  5.5185e-02, -5.6472e-02,  1.8064e-02,  1.3107e-02,
         -1.1798e-04,  4.7892e-02, -4.6190e-03,  3.2130e-02,  5.5952e-02,
         -5.2715e-02,  6.1475e-02, -2.2413e-03, -5.6011e-02, -3.4940e-02,
          6.8291e-02, -2.3276e-02,  6.6684e-02, -2.1379e-02, -5.5981e-02,
         -3.5691e-02,  5.5481e-03, -6.6502e-02,  1.5074e-02,  1.2536e-02,
          3.8780e-02, -4.0601e-02,  2.1475e-02,  5.0530e-02,  1.8705e-02,
          4.7219e-02,  2.1019e-02,  1.2218e-03, -4.3030e-02, -3.2435e-02,
          2.0130e-03, -5.9628e-02,  3.1379e-02,  6.0024e-02, -1.9744e-02,
         -3.9608e-02, -2.2171e-02, -4.0816e-02, -4.6101e-02, -4.6494e-02,
         -5.9531e-02, -2.4929e-02, -7.5038e-02,  5.0952e-02, -4.2914e-02,
          6.4019e-02,  5.8194e-02, -3.1668e-02,  2.8679e-02, -6.6401e-02,
          3.9910e-02, -6.3154e-02, -6.0782e-02, -4.8179e-02, -2.2822e-02,
         -4.9814e-02,  4.5317e-02, -7.0289e-02,  6.8228e-02,  5.0867e-02,
         -7.4101e-02, -6.6887e-03,  5.5574e-02,  6.1948e-02, -2.1699e-02,
         -7.2214e-03, -1.0736e-01, -3.3894e-02, -4.2619e-02, -4.7984e-02,
         -4.7635e-02,  5.8845e-02]], device='cuda:0')), ('fc.bias', tensor([-0.0026,  0.0974,  0.0672,  0.0107,  0.0186, -0.0471, -0.0096,  0.0707,
         0.0387, -0.0013,  0.0718,  0.0424, -0.0174,  0.0085,  0.0366, -0.0832,
        -0.0307, -0.0602,  0.0714,  0.0438, -0.0233, -0.0171,  0.0412, -0.0022,
        -0.0011, -0.0038,  0.0130, -0.0678,  0.0508, -0.0446, -0.0273,  0.0099,
        -0.0357,  0.0597, -0.0634, -0.0406, -0.0361, -0.0319, -0.0106,  0.0752,
        -0.0085, -0.0359,  0.0374,  0.0080,  0.0771,  0.0852,  0.0860, -0.0224,
         0.0422,  0.0307,  0.0548, -0.0759, -0.0144, -0.0342, -0.0610,  0.0063,
         0.0600,  0.0562,  0.0901,  0.0745, -0.0087,  0.0366, -0.0629,  0.0022],
       device='cuda:0')), ('pi_fc.weight', tensor([[ 8.2860e-02, -2.4497e-03,  9.7402e-02, -2.3635e-04,  5.0356e-02,
          9.4200e-02, -5.3765e-02, -2.6075e-02, -8.8505e-02, -1.0935e-01,
          1.3253e-01,  3.3498e-02,  1.0131e-01, -7.3345e-02,  1.0579e-01,
         -1.5098e-01, -2.0607e-02,  4.6696e-02,  1.4126e-01,  1.5780e-02,
         -7.4514e-02, -3.1014e-02, -1.5632e-02,  7.5061e-02,  8.4381e-02,
          9.3526e-02, -4.8890e-03,  8.3065e-02,  3.2696e-02,  9.0876e-02,
         -1.1004e-03, -1.0947e-03, -6.6634e-03, -7.9213e-02, -5.6723e-02,
          8.4546e-02, -5.5105e-02, -9.3481e-02, -1.8655e-02, -9.1838e-02,
          1.4959e-01,  2.9212e-02, -4.6165e-03, -1.2645e-01, -1.3802e-02,
          1.0808e-01,  9.3653e-02,  1.8069e-02, -2.3950e-02, -8.7915e-02,
          1.5898e-01,  1.1010e-01,  1.6233e-01,  4.3582e-02, -1.7121e-02,
         -1.2626e-01, -3.5348e-02,  4.8750e-02,  1.0844e-01,  5.0800e-02,
         -9.5314e-02, -2.9198e-02,  7.9103e-02, -5.6890e-02],
        [-9.7651e-02, -1.1851e-02,  5.7801e-02,  6.6752e-02,  4.4698e-02,
         -7.8962e-02,  1.0355e-01,  7.6127e-02,  2.1275e-02,  1.1371e-01,
         -8.1218e-02,  1.2753e-02, -4.0070e-02,  8.3709e-03,  8.7928e-02,
          3.1142e-03,  1.2521e-01, -9.0966e-02, -6.9920e-02,  7.3657e-02,
         -7.3605e-02,  9.0900e-02,  1.2645e-01,  1.2761e-02, -9.3840e-02,
         -7.4934e-02, -1.0484e-01, -1.1551e-01,  5.9376e-03,  4.7524e-02,
          4.4196e-02, -9.3674e-02, -1.0535e-01, -6.0180e-02,  1.1220e-01,
          4.6815e-02, -3.1774e-02, -6.8333e-02,  5.1092e-02, -1.0513e-01,
          6.3656e-03, -6.1870e-02,  1.1671e-01,  3.4675e-02, -1.3617e-02,
          1.6107e-01,  4.0944e-02,  7.9705e-02,  1.5970e-02,  3.7406e-03,
         -5.9138e-02, -6.7939e-02,  7.0107e-02, -1.4023e-01, -1.1235e-01,
          6.4965e-02, -1.1639e-01,  9.2661e-02,  5.9429e-02,  8.9838e-02,
          8.8031e-02,  1.4528e-03,  3.2440e-02, -7.7047e-02],
        [-3.2765e-02, -1.1561e-02,  6.9512e-02,  1.1828e-01,  9.1288e-02,
          2.0311e-02,  9.0418e-02, -5.5845e-02,  4.6353e-02,  1.4017e-01,
         -5.8675e-02, -3.2766e-02, -9.9641e-02, -7.5984e-02, -4.7844e-02,
         -3.9005e-02, -1.1315e-01,  1.3001e-01,  6.7915e-02,  1.1277e-01,
          1.2166e-01, -6.5088e-02, -6.4741e-02,  6.9456e-02, -1.4660e-01,
          7.9277e-02,  1.0565e-01, -1.3141e-01, -3.6708e-02,  8.6067e-02,
          2.6035e-02,  1.0288e-01, -1.0925e-01, -1.1793e-01, -1.0894e-01,
         -1.3677e-01, -2.2644e-02,  9.8002e-02,  8.2226e-02,  3.6127e-02,
          7.8898e-02, -2.3268e-02,  8.5630e-02, -1.9938e-02, -2.4823e-02,
          1.4648e-01,  2.4169e-02, -2.2038e-02, -6.7193e-03, -5.7287e-02,
         -2.5958e-02,  9.2186e-03, -5.2762e-02, -1.3815e-01, -5.2026e-02,
          9.7711e-02, -6.1036e-02,  8.7950e-02, -9.2336e-02,  7.4099e-02,
         -7.9007e-02,  1.2233e-01,  2.0840e-02, -1.5000e-03],
        [ 9.1052e-02, -7.4686e-02,  1.2371e-01, -2.1212e-02, -5.9483e-04,
          5.1498e-02, -8.5501e-02,  1.1314e-01, -1.0106e-01,  1.2399e-03,
         -1.2846e-01, -1.2003e-02, -8.8175e-02, -4.3660e-02,  2.5956e-02,
         -3.0392e-03, -8.3790e-02,  8.3888e-02,  9.3445e-02,  6.7640e-02,
         -1.0476e-01,  4.5109e-02,  9.3846e-02, -9.1277e-02, -2.5167e-02,
         -7.3860e-02,  8.2301e-02, -1.5107e-01, -2.8737e-02, -8.6452e-03,
         -1.4939e-01, -5.2398e-02,  2.6072e-02,  1.0118e-01, -2.6838e-02,
         -3.2207e-02, -2.6559e-02, -4.6661e-02, -1.0256e-01, -1.1897e-01,
          7.9598e-02, -4.8380e-02,  8.8472e-02, -9.3043e-02,  7.1043e-03,
         -6.6119e-02, -2.3238e-02,  2.7017e-02, -1.2052e-01, -7.7092e-02,
          3.7434e-02, -1.2769e-01,  3.7951e-02, -7.3579e-02,  9.1080e-02,
          1.3200e-02,  1.0438e-01, -2.2816e-02, -5.2043e-02,  1.0264e-01,
          9.6851e-02,  3.0090e-02, -1.2663e-01, -9.1385e-02],
        [ 1.6977e-02,  5.5574e-02, -1.8505e-02, -1.7962e-02,  8.2867e-02,
         -1.0073e-01,  6.3232e-02, -5.6682e-02, -1.1949e-01, -6.4358e-02,
         -1.4768e-02, -1.3415e-01, -3.7932e-02,  1.0670e-04,  1.2496e-02,
         -2.4454e-02, -2.1211e-02,  8.7706e-02, -6.0251e-02, -7.8578e-02,
         -7.3837e-02, -2.2034e-02,  3.8308e-02, -3.4161e-03,  8.4244e-02,
         -9.5706e-02,  6.5626e-02, -1.0813e-01,  3.8760e-02, -5.7663e-02,
          5.9284e-02,  1.1785e-01, -1.3136e-01, -9.8526e-02, -2.1345e-02,
         -1.2684e-01,  2.4541e-02,  2.0826e-02, -7.0803e-02,  8.4719e-02,
         -1.1697e-02, -1.0477e-01, -1.4774e-01, -6.7393e-02, -6.9877e-02,
         -1.2144e-02, -1.9004e-02, -1.0574e-01,  2.7833e-04,  5.4436e-02,
         -1.0656e-01, -3.1867e-02, -1.4001e-01,  3.2961e-02, -3.1005e-02,
          1.0371e-02, -9.4370e-02, -4.2129e-02, -1.0377e-01,  4.2951e-02,
          1.0711e-01, -3.5500e-02,  6.8230e-03, -4.0781e-02],
        [ 5.0156e-02,  1.1615e-01,  1.6010e-01, -3.2141e-03,  1.9996e-02,
          2.0995e-02, -1.0637e-01,  4.5744e-02,  7.3300e-02,  3.4598e-02,
         -4.7480e-02, -4.3024e-02,  7.2250e-02, -7.3048e-02, -6.7058e-02,
         -7.5879e-02,  2.5763e-02,  9.8018e-02, -4.1307e-02, -2.8547e-03,
         -4.2648e-02,  1.3075e-01,  4.8196e-02, -2.5753e-02, -3.3580e-02,
          1.1275e-01,  1.5988e-01,  4.4630e-02,  3.3970e-02,  1.1194e-01,
          6.6609e-02,  3.5581e-02, -5.6974e-02, -3.3079e-02, -1.2800e-01,
          1.4853e-02,  5.9895e-03, -2.8317e-02, -6.5075e-02,  1.7297e-01,
          3.1378e-02,  2.3944e-02,  5.4424e-02,  3.3668e-02,  6.6147e-02,
         -8.2151e-02,  1.7679e-02,  1.0961e-01,  1.5618e-01,  2.6641e-02,
          3.5255e-02,  3.4500e-02,  3.0401e-02, -8.8128e-02, -3.0837e-03,
          8.5335e-02, -6.6796e-02,  9.2645e-02,  1.5797e-01, -3.8254e-02,
          1.1642e-01, -3.3595e-03,  2.1860e-02,  6.2620e-02],
        [ 4.6948e-02, -1.0093e-01, -1.4330e-01,  5.0870e-02, -1.3348e-01,
         -2.3583e-02,  9.1482e-02, -1.2690e-01, -1.3584e-01, -3.9906e-02,
          4.2447e-02,  5.3183e-02,  3.1044e-02, -6.4648e-02, -1.6371e-01,
         -1.1479e-01, -7.2669e-02, -4.4395e-02,  6.9893e-02,  4.1989e-02,
         -5.9961e-04,  2.5218e-02,  4.4777e-02,  3.7509e-02,  1.2743e-02,
         -9.3375e-02, -7.0379e-02,  2.1944e-02,  1.3496e-02, -7.3088e-03,
         -1.0214e-01, -8.6957e-02, -6.3311e-02, -2.8441e-02, -9.9331e-02,
         -2.0774e-02, -5.3703e-02,  5.6387e-02, -1.8627e-01, -5.5909e-02,
         -5.6006e-02, -1.0519e-01, -1.5205e-01, -4.7827e-02,  5.8487e-02,
          3.5489e-02, -1.1564e-01, -5.8764e-02,  1.8697e-02,  6.1590e-02,
          7.5142e-02, -6.7863e-02, -1.5886e-01, -1.0558e-01, -4.9801e-02,
         -9.7882e-02, -1.3163e-01,  3.9666e-03, -1.8355e-01, -1.7996e-01,
         -2.5507e-02,  7.5607e-03,  8.3008e-02, -3.8009e-02]], device='cuda:0')), ('pi_fc.bias', tensor([ 0.0719,  0.0487,  0.0295, -0.0332, -0.0820,  0.0755, -0.1591],
       device='cuda:0')), ('v.weight', tensor([[ 0.0973, -0.0238,  0.1268,  0.0149, -0.0762,  0.0492, -0.1448,  0.1199,
         -0.1157, -0.1162, -0.1112,  0.0781,  0.0847,  0.1282,  0.1018,  0.0354,
          0.1285,  0.0369, -0.1086,  0.0322, -0.1097, -0.0720, -0.1109, -0.0833,
          0.0859,  0.0887,  0.1276,  0.0230, -0.1383,  0.1110,  0.0278, -0.0204,
          0.1445,  0.1077, -0.0886, -0.0821, -0.0851,  0.0569,  0.0835,  0.0267,
         -0.0526, -0.0277,  0.0973,  0.1342, -0.0903, -0.1229, -0.0495, -0.0023,
          0.0988,  0.1204, -0.1154, -0.0159,  0.0983, -0.0071,  0.0015,  0.1440,
          0.1638,  0.0992,  0.0426,  0.1327, -0.0096, -0.1032, -0.0186, -0.0041]],
       device='cuda:0')), ('v.bias', tensor([-0.0118], device='cuda:0'))])


connectx_agent = None
initialized = False
def agent(obs, config):
    global connectx_agent
    global initialized
    if not initialized:
        connectx_agent = ConnectXAgent(config, device='cuda')
        connectx_agent.load_policy(state_dict)
        initialized = True
    s = obs['board']
    mark = obs['mark']
    move = int(connectx_agent.choose_move(s, mark))
    valid_moves = legal_moves(np.reshape(s, (config.rows, config.columns)), config)
    if move not in valid_moves:
        valid = valid_moves[0]  
        move = valid
    return move
