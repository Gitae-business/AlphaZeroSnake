
import numpy as np
import math

class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0
        self.u_value = 0
        self.p_value = prior_p

    def expand(self, action_priors):
        for action, prob in enumerate(action_priors):
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += 1.0 * (leaf_value - self.q_value) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self.u_value = (c_puct * self.p_value * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.q_value + self.u_value

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, policy_value_fn, num_actions, num_snakes, c_puct=5, n_playout=10000):
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.num_actions = num_actions
        self.num_snakes = num_snakes

    def _playout(self, state, current_player):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            state.update(current_player, action) # This needs to be adapted to how your game state updates

        action_probs, leaf_value = self.policy(state, current_player)
        
        if not state.is_game_over:
            node.expand(action_probs)
        
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, current_player=0):
        for n in range(self.n_playout):
            state_copy = state.copy() # You'll need to implement a copy method for your Board
            self._playout(state_copy, current_player)

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = np.power(visits, 1.0 / temp)
        act_probs /= np.sum(act_probs)

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
