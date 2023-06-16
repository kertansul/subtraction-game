import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import random
import time
from src import agents, environment, simulate, heuristic

FLAGS = flags.FLAGS

flags.DEFINE_integer('board_size', 3, 'board size')
flags.DEFINE_integer('max_value', 99, 'max value in board')
flags.DEFINE_integer('min_value', 50, 'min value in board')
flags.DEFINE_integer('seed', None, 'random seed')
flags.DEFINE_integer('verbose', 0, 'verbose level')
flags.DEFINE_string('player1', 'RandomSelect()', 'player1')
flags.DEFINE_string('player2', 'RandomSelect()', 'player2')
flags.DEFINE_string('logdir', None, 'log directory')
flags.DEFINE_string('save_fn', None, 'save game to file')
flags.DEFINE_integer('handicapped', 0, 'init cost from [handicapped, 0]')


def main(argv):

    # init environment
    if FLAGS.logdir is not None:
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        logging.get_absl_handler().use_absl_log_file('main.log', FLAGS.logdir)
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        random.seed(FLAGS.seed)

    # select players
    player1 = eval('agents.' + FLAGS.player1)
    player2 = eval('agents.' + FLAGS.player2)
    logging.info('player1: %s', player1)
    logging.info('player2: %s', player2)
    game = simulate.TwoPlayerGame()

    # initialize starting board
    init_board = np.random.randint(
        FLAGS.min_value, FLAGS.max_value, size=(FLAGS.board_size, FLAGS.board_size))
    logging.info('initial board:\n%s', init_board)

    # play game
    _st = time.time()
    game.run(
        agents=[player1, player2],
        env=environment.SubtractionRule(),
        initial_state=init_board,
        initial_costs=[FLAGS.handicapped, 0],
        save_path=FLAGS.save_fn,
        verbose=FLAGS.verbose > 0
    )

    # log results
    logging.info('total time taken: %s', time.time() - _st)
    logging.info('game ends with final state:\n%s', game.state)
    logging.info('player1 cost: %s', game.cost_p1)
    logging.info('player2 cost: %s', game.cost_p2)
    logging.debug('player1 actions: %s', game.actions_p1)
    logging.debug('player2 actions: %s', game.actions_p2)


if __name__ == '__main__':
    app.run(main)