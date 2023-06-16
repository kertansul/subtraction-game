from absl import logging
import numpy as np
import os
import time
import json
from src import interface
from typing import List, Tuple


class TwoPlayerGame(interface.GameSimulator):
    """game simulator
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # static
        self.init_state = None
        self.init_cost_p1 = 0
        self.init_cost_p2 = 0
        # dynamic
        self.state = None
        self.cost_p1 = 0
        self.cost_p2 = 0
        self.actions_p1 = []
        self.actions_p2 = []

    def run(
        self,
        agents: List[interface.Agent],
        env: interface.Environment,
        initial_state: np.ndarray,
        initial_costs: List[int],
        save_path: str = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Args:
            agents: List[Agent], list of agents
            env: Environment, game rules
            initial_state: np.ndarray, initial state of the game
            initial_costs: List[int], initial costs of the game
            save_path: str, path to save the game
            verbose: bool, whether to print the game process

        Returns:
            states: np.ndarray, states of the game
            costs: List[int], costs of the game
        """
        # parse
        self.init_state = initial_state.copy()
        self.init_cost_p1 = initial_costs[0]
        self.init_cost_p2 = initial_costs[1]
        self.state = self.init_state.copy()
        self.cost_p1 = self.init_cost_p1
        self.cost_p2 = self.init_cost_p2

        # verbose
        _log_level = logging.INFO if verbose else logging.DEBUG
        logging.log(_log_level, 'env: %s', env)
        logging.log(_log_level, 'player1: %s', agents[0])
        logging.log(_log_level, 'player2: %s', agents[1])
        logging.log(_log_level, 'initial state: %s', self.init_state)
        logging.log(_log_level, 'initial cost p1: %s', self.init_cost_p1)
        logging.log(_log_level, 'initial cost p2: %s', self.init_cost_p2)

        # initial board is end game
        is_terminal = env.is_terminal(self.state)
        if is_terminal:
            logging.log(_log_level, 'game already ended, no need to play')
            return self.state, [self.cost_p1, self.cost_p2]

        # play until game ends
        self.act_times_p1 = []
        self.act_times_p2 = []
        while True:
            # player 1 ply
            _st = time.time()
            actions_p1 = env.actions(self.state)
            action_p1 = agents[0].act(self.state, actions_p1)
            self.act_times_p1.append(time.time() - _st)
            logging.log(_log_level, 'player1 act time: %s', time.time() - _st)
            self.actions_p1.append(action_p1)
            state_nxt, cost, is_terminal = env.step(self.state, action_p1)
            self.state = state_nxt
            self.cost_p1 += cost
            logging.log(_log_level, 'player1 action: %s', action_p1)
            logging.log(_log_level, 'player1 cost: %s', self.cost_p1)
            if is_terminal:
                break
            # player 2 ply
            _st = time.time()
            actions_p2 = env.actions(self.state)
            action_p2 = agents[1].act(self.state, actions_p2)
            self.act_times_p2.append(time.time() - _st)
            logging.log(_log_level, 'player2 act time: %s', time.time() - _st)
            self.actions_p2.append(action_p2)
            state_nxt, cost, is_terminal = env.step(self.state, action_p2)
            self.state = state_nxt
            self.cost_p2 += cost
            logging.log(_log_level, 'player2 action: %s', action_p2)
            logging.log(_log_level, 'player2 cost: %s', self.cost_p2)
            if is_terminal:
                break

        # restore verbosity
        logging.log(_log_level, 'end state: %s', self.state)
        logging.log(_log_level, 'end cost p1: %s', self.cost_p1)
        logging.log(_log_level, 'end cost p2: %s', self.cost_p2)

        # save game
        if save_path is not None:
            self._save(save_path, str(agents[0]), str(agents[1]), str(env))

        return self.state, [self.cost_p1, self.cost_p2]

    def _save(self, out_json_fn, player1: str, player2: str, env: str):
        """save game result to json file

        Args:
            out_json_fn (str): output json file name
        """
        _dir = os.path.dirname(os.path.realpath(out_json_fn))
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        with open(out_json_fn, 'w') as f:
            json.dump({
                'env': env,
                'player1': player1,
                'player2': player2,
                'start_board': self.init_state.tolist(),
                'start_cost_p1': self.init_cost_p1,
                'start_cost_p2': self.init_cost_p2,
                'end_board': self.state.tolist(),
                'end_cost_p1': self.cost_p1,
                'end_cost_p2': self.cost_p2,
                'actions_p1': self.actions_p1,
                'actions_p2': self.actions_p2,
                'act_times_p1': self.act_times_p1,
                'act_times_p2': self.act_times_p2
            }, f)
        return

    def __str__(self):
        return 'TwoPlayerGame()'