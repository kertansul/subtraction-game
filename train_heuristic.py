import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import sqlite3
import json
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchrl.modules import MLP
from torch.utils.tensorboard import SummaryWriter
from src import config_parser
from dqn.transformer import *
from dqn.network import FeatureExtract, PreNormalization, PostScaling


FLAGS = flags.FLAGS
flags.DEFINE_string('config', None, 'Path to the configuration file.')


class TransitionDataset(Dataset):

    def __init__(self, db_path, test_split=0.1, as_test=False):
        self.db_path = db_path
        # load sql data
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('SELECT * FROM transitions')
        data = cur.fetchall()
        conn.close()
        # convert to torch tensor
        x = [json.loads(datum[0]) for datum in data]
        y = [datum[1] for datum in data]
        # split train and test
        if as_test:
            x = x[:int(len(x) * test_split)]
            y = y[:int(len(y) * test_split)]
        else:
            x = x[int(len(x) * test_split):]
            y = y[int(len(y) * test_split):]
        self.x = torch.tensor(x, dtype=torch.float32)  # (N, 3, 3)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        # load to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        # random rotation
        x = torch.rot90(x, k=np.random.randint(0, 4), dims=(0, 1))
        # random flip
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=(1,))
        return x, self.y[idx]


def main(argv):

    # init environment
    config = config_parser.process_config(FLAGS.config, 'configs/default_config.yaml')
    config_parser.create_ckpt_dir(config)

    # init dataset
    dataset_train = TransitionDataset(
        config.data.db_path, test_split=config.data.eval_split, as_test=False)
    dataset_eval = TransitionDataset(
        config.data.db_path, test_split=config.data.eval_split, as_test=True)
    dataset_train
    dataset_eval
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)

    # init network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocesses = []
    if isinstance(config.network.preprocess, list):
        for preprocess in config.network.preprocess:
            preprocesses.append(eval(preprocess).to(device))
    network = eval(config.network.model).to(device)
    postprocesses = []
    if isinstance(config.network.postprocess, list):
        for postprocess in config.network.postprocess:
            postprocesses.append(eval(postprocess).to(device))
    # load pretrained weight
    if config.network.pretrained_weight is not None:
        network.load_state_dict(torch.load(config.network.pretrained_weight))

    # init optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.optimizer.lr,
        amsgrad=True)

    # init tensorboard
    writer = SummaryWriter(config.checkpoint_dir)

    # init training
    for epoch in range(config.train.num_epochs):
        # train
        losses = []
        with tqdm.tqdm(dataloader_train) as pbar:
            for x, y in dataloader_train:
                for preprocess in preprocesses:
                    x = preprocess(x)
                y_pred = network(x)
                for postprocess in postprocesses:
                    y_pred = postprocess(y_pred)
                criterion = nn.SmoothL1Loss()
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(network.parameters(), config.optimizer.clip_grad)
                optimizer.step()
                losses.append(loss.item())
                pbar.set_description(f'Epoch {epoch} | Loss {losses[-1]:.4f}')
                pbar.update()
        writer.add_scalar('loss/train', np.mean(losses), epoch)
        # eval
        if epoch % config.train.eval_interval == 0 and epoch != 0:
            losses = []
            diffs = []
            with torch.no_grad():
                for x, y in dataloader_eval:
                    for preprocess in preprocesses:
                        x = preprocess(x)
                    y_pred = network(x)
                    for postprocess in postprocesses:
                        y_pred = postprocess(y_pred)
                    diff = torch.abs(y_pred - y)
                    diffs.extend(diff.cpu().numpy().tolist())
                    criterion = nn.SmoothL1Loss()
                    loss = criterion(y_pred, y)
                    losses.append(loss.item())
            writer.add_scalar('loss/eval', np.mean(losses), epoch)
            writer.add_scalar('diff/eval/max', np.max(diffs), epoch)
            writer.add_scalar('diff/eval/mean', np.mean(diffs), epoch)
        # save model
        if epoch % config.train.save_interval == 0 and epoch != 0:
            torch.save(
                network.state_dict(),
                os.path.join(config.checkpoint_dir, f'ckpt_{epoch:06d}.pth'))
        writer.flush()
    writer.close()

if __name__ == '__main__':
    app.run(main)