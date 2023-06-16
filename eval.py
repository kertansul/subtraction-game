from absl import app
from absl import flags
from absl import logging
import os
import time
from src import agents, environment, evaluate, heuristic

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_games', 100, 'number of games to play')
flags.DEFINE_integer('board_size', 3, 'board size')
flags.DEFINE_integer('max_value', 99, 'max value in board')
flags.DEFINE_integer('min_value', 50, 'min value in board')
flags.DEFINE_string('player', 'RandomSelect()', 'player')
flags.DEFINE_string('against', 'RandomSelect()', 'player to play against')
flags.DEFINE_string('env', 'SubtractionRule()', 'environment')
flags.DEFINE_string('evaluator', 'TwoPlayerEvaluator', 'evaluator')
flags.DEFINE_string('logdir', None, 'log directory')
flags.DEFINE_integer('num_workers', 1, 'number of workers')
flags.DEFINE_integer('handicapped', 0, 'init cost from [handicapped, 0]')
flags.DEFINE_integer('seed', None, 'random seed')


def main(argv):

    # logging
    _st = time.time()
    if FLAGS.logdir is not None:
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        logging.get_absl_handler().use_absl_log_file('eval', FLAGS.logdir)

    # parse
    player = eval('agents.' + FLAGS.player)
    against = eval('agents.' + FLAGS.against)
    env = eval('environment.' + FLAGS.env)
    logging.info('player: %s', player)
    logging.info('against: %s', against)
    logging.info('env: %s', env)

    # init
    evaluator = eval('evaluate.' + FLAGS.evaluator)
    evaluator = evaluator(
        board_size=FLAGS.board_size,
        min_value=FLAGS.min_value,
        max_value=FLAGS.max_value,
        against=against,
        env=env,
        save_dir=FLAGS.logdir,
        num_workers=FLAGS.num_workers,
        handicapped=FLAGS.handicapped,
        seed=FLAGS.seed,)
    logging.info('evaluator: %s', evaluator)

    # evaluate
    win_rate = evaluator.evaluate(
        player=player,
        num_games=FLAGS.num_games,
    )
    logging.info('win rate: %s', win_rate)
    logging.info('total time: %s', time.time() - _st)


if __name__ == '__main__':
    app.run(main)