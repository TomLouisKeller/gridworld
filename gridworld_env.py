import gym
from gym import spaces
import numpy as np
from enum import Enum

# ----------------------------------------------------------------------------------------------------------------------- #

class Action(Enum):
    UP = (0, -1, 0)
    RIGHT = (1, 0, 1)
    DOWN = (2, 1, 0)
    LEFT = (3, 0, -1)
    
    def __init__(self, id, row_move, column_move):
        self.id = id
        self.row_move = row_move
        self.column_move = column_move
        
# ----------------------------------------------------------------------------------------------------------------------- #
        
class Gridworld(gym.Env):
    action_space = spaces.Discrete(4)
    observation_space = spaces.Discrete(16)

    def __init__(self):
        self.state = 0
        self.gridworld = np.arange(self.observation_space.n).reshape((4, 4))
        
        # Initiate state transation matrix
        self.state_transition_matrix = np.zeros((self.observation_space.n, self.action_space.n), dtype=int)
        for state in self.gridworld.flat:
            row, column = np.argwhere(self.gridworld == state)[0]
            for action in Action:
                next_row = max(0, min(row + action.row_move, 3))
                next_column = max(0, min(column + action.column_move, 3))
                state_prime = self.gridworld[next_row, next_column]
                self.state_transition_matrix[state, action.id] = state_prime
                
    def step(self, action: int):
        self.state = self.state_transition_matrix[self.state, action]

        if self.state == 15:
            done = True
        else:
            done = False

        return self.state, -1, done, {}

        
    def reset(self):
        self.state = 0
        return self.state
