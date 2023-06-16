"""Evaluate player strength by competing with another player
"""
import os
import tqdm
import numpy as np
import random
from typing import Optional, Tuple
import multiprocessing as mp
from src import interface, simulate


class TwoPlayerEvaluator(interface.Evaluator):

    def __init__(
        self,
        board_size: int,
        min_value: int,
        max_value: int,
        against: interface.Agent,
        env: interface.Environment,
        save_dir: Optional[str] = None,
        num_workers: int = 1,
        handicapped: int = 0,
        seed: Optional[int] = None,
    ):
        self.board_size = board_size
        self.min_value = min_value
        self.max_value = max_value
        self.against = against
        self.env = env
        self.save_dir = save_dir
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        self.num_workers = num_workers
        self.handicapped = handicapped
        self.seed = seed or np.random.randint(0, 2**16 - 1)

    def worker(self, idx_seed: Tuple[int, int]):
        """Worker function for multiprocessing
        """
        index, seed = idx_seed
        seed += index
        np.random.seed(seed)
        random.seed(seed)
        game = simulate.TwoPlayerGame()
        _, costs = game.run(
            agents=[self.player, self.against],
            env=self.env,
            initial_state=np.random.randint(
                self.min_value,
                self.max_value,
                size=(self.board_size, self.board_size)),
            initial_costs=[self.handicapped, 0],
            save_path=os.path.join(self.save_dir, 'play_%06d.json' % index) \
                if self.save_dir is not None else None,
            verbose=False
        )
        return index, costs

    def evaluate(
        self,
        player: interface.Agent,
        num_games: int,
    ) -> float:
        win = 0
        self.player = player
        with mp.Pool(self.num_workers) as p:
            iterator = p.imap_unordered(self.worker, zip(range(num_games), [self.seed for _ in range(num_games)]), chunksize=1)
            for _, costs in tqdm.tqdm(iterator, total=num_games):
                # calculate win
                if costs[0] < costs[1]:
                    win += 1
        return win / num_games

    def __str__(self):
        return 'TwoPlayerEvaluator(board_size=%d, min_value=%d, max_value=%d, against=%s, env=%s, num_workers=%d, seed=%d)' % \
            (self.board_size, self.min_value, self.max_value, self.against, self.env, self.num_workers, self.seed)