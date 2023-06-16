import abc
import numpy as np
from typing import Tuple, List, Union


class Environment(abc.ABC):
    """Interface for Game Rules
    """

    @abc.abstractmethod
    def actions(self, state: np.ndarray) -> List:
        """
        Args:
            state: np.ndarray, current state of the game

        Returns:
            actions: List, available actions for the player
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: np.ndarray, action: Tuple) -> Tuple[np.ndarray, int, bool]:
        """
        Args:
            state: np.ndarray, current state of the game
            action: Tuple, action to take

        Returns:
            next_state: np.ndarray, next state of the game
            cost: int, cost of the action
            is_terminal: bool, whether the game is over
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Args:
            state: np.ndarray, current state of the game

        Returns:
            is_terminal: bool, whether the game is over
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Agent(abc.ABC):
    """Interface for Players
    """

    @abc.abstractmethod
    def act(self, state: np.ndarray, actions: List) -> Tuple:
        """
        Args:
            state: np.ndarray, current state of the game
            actions: List, available actions for the player

        Returns:
            action: Tuple, action to take
        """
        raise NotImplementedError

    def make_your_move_method(self, env: Environment) -> callable:
        """
        Args:
            env: Environment, game rules

        Returns:
            make_your_move: callable, function to make a move
        """
        def make_your_move(state: List) -> Tuple:
            state = np.array(state)
            actions = env.actions(state)
            return self.act(state, actions)
        return make_your_move

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class GameSimulator(abc.ABC):
    """Interface for a single game simulation
    """

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(
        self,
        agents: List[Agent],
        env: Environment,
        initial_state: np.ndarray,
        initial_costs: List[int],
        save_path: str,
        verbose: bool,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Args:
            agents: List[Agent], list of agents
            env: Environment, game rules
            initial_state: np.ndarray, initial state of the game
            initial_costs: List[int], initial costs of the game
            save_path: str, path to save the game process
            verbose: bool, whether to print the game process

        Returns:
            states: np.ndarray, states of the game
            costs: List[int], costs of the game
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Evaluator(abc.ABC):
    """Interface for evalute the strength of a player
    """

    @abc.abstractmethod
    def evaluate(
        self,
        player: Agent,
        num_games: int,
    ) -> float:
        """
        Args:
            player: Agent, player to evaluate
            num_games: int, number of games to play

        Returns:
            score: float, average score of the player
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Estimator(abc.ABC):
    """Interface for heuristic functions
    """

    @abc.abstractmethod
    def estimate(self, state: np.ndarray) -> int:
        """
        Args:
            state: np.ndarray, current state of the game

        Returns:
            min_cost: int, estimated minimum cost of the game
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError