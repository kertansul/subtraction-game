import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from typing import Tuple
from src import interface, environment


class TwoPlayerSubtractionEnv(py_environment.PyEnvironment):

    def __init__(
        self,
        against: interface.Agent,
        rule: interface.Environment = environment.SubtractionRule(),
        board_size: int = 3,
        min_value: int = 50,
        max_value: int = 99,
    ):
        self.against = against
        self.rule = rule
        self.board_size = board_size
        self.min_value = min_value
        self.max_value = max_value
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=17, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.board_size,self.board_size),
            dtype=np.int32, minimum=0, name='observation')
        self._state = np.random.randint(
            self.min_value, self.max_value,
            size=(self.board_size, self.board_size),
            dtype=np.int32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.random.randint(
            self.min_value, self.max_value,
            size=(self.board_size, self.board_size),
            dtype=np.int32)
        self._episode_ended = False
        return ts.restart(self._state)

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

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        valid_actions = self.rule.actions(self._state)
        valid_actions = [self._encode_action(a) for a in valid_actions]

        # if action not in valid, give penalty and terminate
        if action not in valid_actions:
            self._episode_ended = True
            return ts.termination(self._state, reward=-999)

        # step
        action = self._decode_action(action)
        state, cost, is_terminal = self.rule.step(self._state, action)
        self._state = state

        # if terminal, give reward and terminate
        if is_terminal:
            self._episode_ended = True
            return ts.termination(self._state, reward=-cost)

        # if not terminal, let the other player play
        oppo_actions = self.rule.actions(state)
        oppo_action = self.against.act(state, oppo_actions)
        state, oppo_cost, is_terminal = self.rule.step(state, oppo_action)
        cost -= oppo_cost
        self._state = state
        if is_terminal:
            self._episode_ended = True
            return ts.termination(self._state, reward=-cost)
        return ts.transition(self._state, reward=-cost, discount=1.0)
