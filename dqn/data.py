import random
import numpy as np
import torch
from collections import namedtuple, deque


Transition = namedtuple(
    'Transition',(
        'state',                     # current state
        'action',                    # action taken
        'next_state',                # next state
        'next_state_valid_actions',  # valid actions in next state
        'reward'                     # reward received
    )
)


class ReplayMemory(object):
    """Replay memory for DQN
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GameRunner(object):

    def __init__(
        self,
        rule,
        q_network,
        device,
        board_size=3,
        min_value=50,
        max_value=99,
    ):
        self.rule = rule
        self.q_network = q_network
        self.device = device
        self.board_size = board_size
        self.min_value = min_value
        self.max_value = max_value
        #
        self.state = None
        self.total_reward = -1
        self.is_terminal = True

    def reset(self):
        self.state = np.random.randint(
            self.min_value,
            self.max_value,
            size=(self.board_size, self.board_size),
            dtype=np.int32
        )
        self.total_reward = 0
        is_terminal, _ = self.rule.is_terminal(self.state)
        self.is_terminal = is_terminal

    def step(self, epsilon=0.):
        state_tensor = torch.tensor(
            self.state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).flatten(1)
        valid_actions = torch.tensor(
            self.rule.actions(self.state),
            dtype=torch.int64,
            device=self.device
        )
        # select an action
        action = self.q_network.act(
            state_tensor,
            valid_actions,
            epsilon
        )
        # take the action
        next_state, reward, is_terminal = self.rule.step(
            self.state, action.item())
        self.total_reward += reward
        self.is_terminal = is_terminal
        if is_terminal:
            next_state_tensor = None
            next_state_actions = None
        else:
            next_state_tensor = torch.tensor(
                next_state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0).flatten(1)
            next_state_actions = torch.tensor(
                self.rule.actions(next_state),
                dtype=torch.int64,
                device=self.device
            )
        # update state
        self.state = next_state
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        return state_tensor, action, next_state_tensor, next_state_actions, reward
