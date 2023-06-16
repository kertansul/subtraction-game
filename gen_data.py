from absl import app
from absl import flags
from absl import logging
import numpy as np
import random
import sqlite3
import time
import multiprocessing as mp
import tqdm
from src import agents

FLAGS = flags.FLAGS

flags.DEFINE_integer('board_size', 3, 'board size')
flags.DEFINE_integer('max_value', 10, 'max value in board')
flags.DEFINE_integer('min_value', 0, 'min value in board')
flags.DEFINE_integer('insert_zeros', 0, 'max number of zeros to insert')
flags.DEFINE_string('db_path', None, 'path to database')
flags.DEFINE_integer('max_depth', 30, 'max depth for minimax')
flags.DEFINE_integer('num_workers', 1, 'number of workers')
flags.DEFINE_integer('num_runs', 1, 'number of states to collect')


def worker(index):

    # init player
    player = agents.MinimaxV2(branch2depth_factors={0: 30})
    player.max_depth=FLAGS.max_depth
    player.memory.clear()

    # set seed based on time + index
    np.random.seed(int(time.time()) + index)
    random.seed(int(time.time()) + index)

    # init state
    while True:
        state = np.random.randint(
            FLAGS.min_value, FLAGS.max_value,
            size=(FLAGS.board_size, FLAGS.board_size))
        # random zeros for easier state
        state[np.random.randint(0, FLAGS.board_size, size=FLAGS.insert_zeros),
            np.random.randint(0, FLAGS.board_size, size=FLAGS.insert_zeros)] = 0
        # if number of zeros < FLAGS.insert_zeros, regenerate
        if np.sum(state == 0) < FLAGS.insert_zeros:
            continue
        break
    state_str = str(state.tolist())

    # check if state exists
    #conn = sqlite3.connect(FLAGS.db_path)
    #cur = conn.cursor()
    #cur.execute(
    #    "SELECT * FROM transitions WHERE state=?", (state_str,)
    #)
    #if cur.fetchone() is not None:
    #    return
    _st = time.time()
    value, action = player._minimax(
        state=state,
        from_action="",
        depth=0,
        path_cost=0,
        alpha=-np.inf,
        beta=np.inf,
    )
    conn = sqlite3.connect(FLAGS.db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO transitions VALUES (?, ?, ?, ?)",
        (state_str, value, str(action), time.time() - _st))
    conn.commit()
    conn.close()
    return


def main(argv):

    # init table if not exists
    conn = sqlite3.connect(FLAGS.db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS transitions(state, value, action, process_time)")
    cur.execute("CREATE TABLE IF NOT EXISTS info(board_size, max_value, min_value, insert_zeros, max_depth, num_runs)")
    cur.execute("INSERT INTO info VALUES (?, ?, ?, ?, ?, ?)",
                (FLAGS.board_size, FLAGS.max_value, FLAGS.min_value, FLAGS.insert_zeros, FLAGS.max_depth, FLAGS.num_runs))
    conn.commit()
    conn.close()

    with mp.Pool(FLAGS.num_workers) as p:
        iterator = p.imap_unordered(worker, [i for i in range(FLAGS.num_runs)], chunksize=1)
        for _ in tqdm.tqdm(iterator, total=FLAGS.num_runs):
            pass

if __name__ == '__main__':
    app.run(main)