import numpy as np
from src import interface
from typing import Optional, List, Tuple


class SubtractionRule(interface.Environment):
    """game environment

    used to define the rules of the game, such as:
        - what actions are available given a state
        - result state given a state and an action
        - cost of an action
        - whether a state is terminal and the cost of the state
    """

    def __init__(self,
                 end_game_reward: float = 15.,
                 end_game_penalty: float = 7.):
        self.cost_reward = -end_game_reward
        self.cost_penality = end_game_penalty

    def actions(self,
                state: np.ndarray) -> List:
        """Return a list of available actions given a state.

        can only subtract 1, 2, or 3 from a single row or column
        result state's elements must be non-negative

        Args:
            state: np.ndarray

        Returns:
            actions: List(Tuple(row_col_idx, subtract_value))
                row_col_idx:
                    [0 .. num_rows - 1] for row operations
                    [num_rows .. num_rows + num_cols - 1] for col operations
        """
        actions = []
        # row operations
        min_per_row = np.min(state, axis=1)
        for row_idx, min_value in enumerate(min_per_row):
            for subtract_value in range(1, min(int(min_value), 3) + 1):
                actions.append((row_idx, subtract_value))
        # col operations
        num_rows = state.shape[0]
        min_per_col = np.min(state, axis=0)
        for col_idx, min_value in enumerate(min_per_col):
            for subtract_value in range(1, min(int(min_value), 3) + 1):
                actions.append((col_idx + num_rows, subtract_value))
        return actions

        # NOTE(shawn): vectorized version, only faster for larger board size
        s = np.stack([state[i] for i in range(state.shape[0])] + [state[:, i] for i in range(state.shape[1])])
        m = np.min(s, axis=1)
        b = np.stack([m > i for i in range(3)], axis=1)
        y, x = np.where(b)
        return [k for k in zip(y, x+1)]

    def step(
        self,
        state: np.ndarray,
        action: Tuple
    ) -> Tuple[np.ndarray, float, bool]:
        """Return the next state given a state and an action.

        Args:
            state: np.ndarray
            action: Tuple(row_col_idx, subtract_value)

        Returns:
            next_state: np.ndarray
            cost: float
            is_terminal: bool
        """
        # next state
        state = state.copy()
        num_rows = state.shape[0]
        row_col_idx, subtract_value = action
        if row_col_idx < num_rows:  # row operations
            state[row_col_idx] -= subtract_value
        else:                       # col operations
            state[:, row_col_idx - num_rows] -= subtract_value
        # cost
        cost = subtract_value
        # is_terminal
        is_terminal, terminal_cost = self._is_terminal_and_cost(state)
        cost = cost + terminal_cost
        return state, cost, is_terminal

    def is_terminal(self,
                    state: np.ndarray) -> bool:
        """Return whether a state is terminal.
        """
        is_terminal, _ = self._is_terminal_and_cost(state)
        return is_terminal

    def _is_terminal_and_cost(self, state: np.ndarray) -> Tuple[bool, float]:
        """Check if a state is terminal. If so, return the cost of the state.

        Args:
            state: np.ndarray

        Returns:
            is_end: bool
            cost: float
        """
        # end condition 1:
        #   all the numbers in any row, column, or diagonal become 0
        if np.all(np.diag(state) == 0):             # diag all zeros
            return True, self.cost_reward
        if np.all(np.diag(np.fliplr(state)) == 0):  # flip diag all zeros
            return True, self.cost_reward
        if np.any(np.all(state == 0, axis=1)):      # any row all zeros
            return True, self.cost_reward
        if np.any(np.all(state == 0, axis=0)):      # any col all zeros
            return True, self.cost_reward

        # end condition 2:
        #   every row and column contains the number 0
        all_row_contains_zero = np.all(np.any(state == 0, axis=1))
        all_col_contains_zero = np.all(np.any(state == 0, axis=0))
        if all_row_contains_zero and all_col_contains_zero:
            return True, self.cost_penality

        return False, 0.

    def __str__(self):
        return "SubtractionRule(reward={}, penalty={})".format(
            self.cost_reward, self.cost_penality)


class TwoPlayerSubtractionEnv(interface.Environment):
    """view another player as part of the environment

        used for reinforcement learning
    """

    def __init__(self,
                 against: interface.Agent,
                 rule: interface.Environment = SubtractionRule()):
        self.against = against
        self.rule = rule

    def _encode_action(self, action: Tuple) -> int:
        """Encode an action into an integer.
        """
        row_col_idx, subtract_value = action
        return row_col_idx * 3 + subtract_value - 1

    def _decode_action(self, action: int) -> Tuple:
        """Decode an integer into an action.
        """
        row_col_idx = action // 3
        subtract_value = action % 3 + 1
        return row_col_idx, subtract_value

    def actions(self, state: np.ndarray) -> List:
        """Return a list of available actions given a state.

        Args:
            state: np.ndarray

        Returns:
            actions: List(int)
        """
        return [self._encode_action(action)
                for action in self.rule.actions(state)]

    def step(
        self,
        state: np.ndarray,
        action: int
    ) -> Tuple[np.ndarray, float, bool]:
        """
        """
        # player phase
        action = self._decode_action(action)
        valid_actions = self.rule.actions(state)
        if action not in valid_actions:
            # choose a random action, set cost to 999
            action = valid_actions[np.random.randint(len(valid_actions))]
            state, _, is_terminal = self.rule.step(state, action)
            cost = 999.
        else:
            state, cost, is_terminal = self.rule.step(state, action)
        if is_terminal:
            return state, -cost, is_terminal
        # opponent phase
        oppo_actions = self.rule.actions(state)
        oppo_action = self.against.act(state, oppo_actions)
        state, oppo_cost, is_terminal = self.rule.step(state, oppo_action)
        cost -= oppo_cost
        return state, -cost, is_terminal

    def is_terminal(self, state: np.ndarray) -> bool:
        return self.rule.is_terminal(state)

    def __str__(self):
        return "TwoPlayerSubtractionEnv(against={}, rule={})".format(
            self.against, self.rule)