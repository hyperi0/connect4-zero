import math
import connectx
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
        if connectx.is_terminal_grid(s, self.config):
            return -connectx.score_game(s, self.config)
        
        # first visit: initialize to neural net's predicted action probs and value
        if s not in self.visited:
            self.visited.append(s)
            pi, v = self.policy.predict(s)
            legal_mask = connectx.legal_moves_mask(s, self.config)
            self.P[s] = [p * mask for p, mask in zip(pi, legal_mask)]
            self.N[s] = [0 for _ in range(self.config.columns)]
            self.Q[s] = [0 for _ in range(self.config.columns)]
            return -v
        
        # choose action with best upper confidence bound
        max_u, best_a = -np.inf, -1
        for a in connectx.legal_moves(s, self.config):
            u = self.u_value(s, a)
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        # calculate state value recursively
        next_s = connectx.drop_piece(s, a, 1, self.config)
        next_s = connectx.reverse_grid(next_s)
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