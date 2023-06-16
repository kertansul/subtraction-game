import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tqdm
import time
from itertools import count
from collections import deque
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src import agents, environment, config_parser
from dqn import data, network

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'config file')


ROT_ACT_MAPPING = [9, 10, 11, 12, 13,
                   14, 15, 16, 17, 6,
                   7, 8, 3, 4, 5,
                   0, 1, 2]


def main(argv):

    # init
    config = config_parser.process_config(FLAGS.config, 'configs/default_dqn_config.yaml')
    config_parser.create_ckpt_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment.TwoPlayerSubtractionEnv(
        rule=eval(config.game.rule),
        against=eval(config.game.against))
    memory_win = data.ReplayMemory(config.data.win_replay_size)
    memory_lose = data.ReplayMemory(config.data.lose_replay_size)
    q = network.QLearning(
        preprocess=config.network.preprocess,
        network_archit=config.network.model,
        gamma=config.dqn.gamma,
        tau=config.dqn.tau)
    optimizer = torch.optim.Adam(
        q.policy_net.parameters(),
        lr=config.optimizer.lr,
        amsgrad=False)
    writer = SummaryWriter(config.checkpoint_dir)

    # load checkpoint
    if config.network.pretrained_weights is not None:
        q.load(config.network.pretrained_weights)

    # fill memory
    logging.info('filling memory...')
    with tqdm.tqdm(total=config.data.win_replay_size) as wbar:
        with tqdm.tqdm(total=config.data.lose_replay_size) as lbar:
            while len(memory_win) < config.data.batch_size or \
                    len(memory_lose) < config.data.batch_size:
                state_np = np.random.randint(config.game.min_value, config.game.max_value, size=(3, 3))
                total_reward = 0
                buffer = []
                for _ in count():
                    valid_actions_np = env.actions(state_np)
                    # convert to torch
                    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).flatten(1)
                    valid_actions = torch.tensor(valid_actions_np, dtype=torch.long, device=device)
                    # select action by pure random
                    action = q.act(state, valid_actions, config.dqn.epsilon.start)
                    # step in environment
                    next_state_np, reward, terminated = env.step(state_np, action.item())
                    total_reward += reward
                    # convert to torch
                    reward = torch.tensor([reward], device=device)
                    if terminated:
                        next_state = None
                        next_state_valid_actions = None
                    else:
                        next_state = torch.tensor(
                            next_state_np, dtype=torch.float32, device=device).unsqueeze(0).flatten(1)
                        next_state_valid_actions = torch.tensor(
                            env.actions(next_state_np), dtype=torch.long, device=device)
                    # save to buffer
                    buffer.append([state, action, next_state, next_state_valid_actions, reward])
                    # move to next state
                    state_np = next_state_np
                    if terminated:
                        break
                if total_reward > 0:
                    for i in range(len(buffer)):
                        state, action, next_state, next_state_valid_actions, reward = buffer[i]
                        memory_win.push(state, action, next_state, next_state_valid_actions, reward)
                        # rotate90
                        for _ in range(3):
                            state = torch.rot90(state.reshape(3, 3)).reshape(1, 9)
                            action = torch.tensor([ROT_ACT_MAPPING[action.item()]], device=device)
                            next_state = None if next_state is None else torch.rot90(next_state.reshape(3, 3)).reshape(1, 9)
                            next_state_valid_actions = None if next_state_valid_actions is None else \
                                torch.tensor([ROT_ACT_MAPPING[a.item()] for a in next_state_valid_actions], device=device)
                            memory_win.push(state, action, next_state, next_state_valid_actions, reward)
                        wbar.update(4)
                else:
                    for i in range(len(buffer)):
                        state, action, next_state, next_state_valid_actions, reward = buffer[i]
                        memory_lose.push(state, action, next_state, next_state_valid_actions, reward)
                        lbar.update(1)

    # main loop
    num_steps = 0
    train_wins = deque(maxlen=100)
    with tqdm.tqdm(total=config.training.num_episodes) as pbar:

        for episode in range(config.training.num_episodes):

            ## data collector, run a game
            buffer = []
            state_np = np.random.randint(config.game.min_value, config.game.max_value, size=(3, 3))
            total_reward = 0
            for _ in count():
                # convert to torch
                state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).flatten(1)
                valid_actions = torch.tensor(env.actions(state_np), dtype=torch.long, device=device)
                # select action by epsilon-greedy policy
                eps_threshold = config.dqn.epsilon.end + (config.dqn.epsilon.start - config.dqn.epsilon.end) * \
                    np.exp(-1. * num_steps / config.dqn.epsilon.decay_steps)
                action = q.act(state, valid_actions, eps_threshold)  # (1, 1)
                num_steps += 1
                # step in environment
                next_state_np, reward, terminated = env.step(state_np, action.item())
                total_reward += reward
                # convert to torch
                reward = torch.tensor([reward], device=device)
                if terminated:
                    next_state = None
                    next_state_valid_actions = None
                else:
                    next_state = torch.tensor(
                        next_state_np, dtype=torch.float32, device=device).unsqueeze(0).flatten(1)
                    next_state_valid_actions = torch.tensor(
                        env.actions(next_state_np), dtype=torch.long, device=device)
                # save to buffer
                buffer.append([state, action, next_state, next_state_valid_actions, reward])
                # move to next state
                state_np = next_state_np
                #
                if terminated:
                    if total_reward > 0:
                        train_wins.append(1)
                    else:
                        train_wins.append(0)
                    break

            ## push into memory
            buffer = buffer[::-1]
            push_size = len(buffer)
            if config.data.only_take_n_from_terminal is not None:
                push_size = min(push_size, config.data.only_take_n_from_terminal)
            for i in range(push_size):
                # normal
                state, action, next_state, next_state_valid_actions, reward = buffer[i]
                if total_reward > 0:
                    memory_win.push(state, action, next_state, next_state_valid_actions, reward)
                    # rotate90
                    for _ in range(3):
                        state = torch.rot90(state.reshape(3, 3)).reshape(1, 9)
                        action = torch.tensor([ROT_ACT_MAPPING[action.item()]], device=device)
                        next_state = None if next_state is None else torch.rot90(next_state.reshape(3, 3)).reshape(1, 9)
                        next_state_valid_actions = None if next_state_valid_actions is None else \
                            torch.tensor([ROT_ACT_MAPPING[a.item()] for a in next_state_valid_actions], device=device)
                        #print(state, action, reward)
                        memory_win.push(state, action, next_state, next_state_valid_actions, reward)
                    # TODO(shawn): flip
                else:
                    memory_lose.push(state, action, next_state, next_state_valid_actions, reward)

            ## update network
            loss_accum = 0
            loss_count = 0
            for _ in range(config.training.num_updates_per_episode):
                transitions_win = memory_win.sample(config.data.batch_size)
                transitions_lose = memory_lose.sample(config.data.batch_size)
                transitions = transitions_win + transitions_lose
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = data.Transition(*zip(*transitions))
                batched_state = torch.cat(batch.state)
                batched_action = torch.cat(batch.action)
                batched_reward = torch.cat(batch.reward)

                # Compute Q(s_t, a)
                state_action_values = q.state_action_values(batched_state, batched_action)

                # Compute V(s_{t+1}) for all next states.
                expected_state_action_values = q.expected_state_action_values(
                    batch.next_state, batch.next_state_valid_actions, batched_reward)

                # loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                loss_accum += loss.item()
                loss_count += 1

                # optimize
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_value_(q.policy_net.parameters(), config.optimizer.clip_grad)
                optimizer.step()

                # soft update target network
                q.update_target_net()

            # write to tensorboard
            if loss_count > 0:
                writer.add_scalar("loss/train", loss_accum / loss_count, episode)

            # evaluate
            if episode % config.evaluation.eval_interval == 0 and episode != 0:
                num_wins = 0
                for _ in tqdm.tqdm(range(config.evaluation.num_episodes)):
                    # init environment
                    state_np = np.random.randint(config.game.min_value, config.game.max_value, size=(3, 3))
                    total_reward = 0
                    sel_actions = []
                    get_rewards = []
                    for _ in count():
                        valid_actions_np = env.actions(state_np)
                        # convert to torch
                        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).flatten(1)
                        valid_actions = torch.tensor(valid_actions_np, dtype=torch.long, device=device)
                        # select action by max Q value
                        action = q.act(state, valid_actions, 0.)  # (1, 1)
                        # step in environment
                        next_state_np, reward, terminated = env.step(state_np, action.item())
                        total_reward += reward
                        #
                        sel_actions.append(action.item())
                        get_rewards.append(reward)
                        # move to next state
                        state_np = next_state_np
                        if terminated:
                            break
                    logging.info('selected actions: {}'.format(sel_actions))
                    logging.info('get rewards: {}'.format(get_rewards))
                    logging.debug('total reward: {}'.format(total_reward))
                    if total_reward > 0:
                        num_wins += 1
                logging.info('win rate: {}\n'.format(num_wins / config.evaluation.num_episodes))
                writer.add_scalar("win_rate", num_wins / config.evaluation.num_episodes, episode)

            # save model
            if episode % config.training.save_interval == 0 and episode != 0:
                q.save(
                    os.path.join(config.checkpoint_dir, f'ckpt_{episode:05d}.pth'))

            # update progress bar
            pbar.set_description(f'loss: {loss_accum/(loss_count+1e-8):.3f}, eps: {eps_threshold:.3f}, win rate: {np.mean(train_wins):.3f}')
            pbar.update()
            writer.flush()
    writer.close()


if __name__ == '__main__':
    app.run(main)