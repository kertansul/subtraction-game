import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import *
from src import interface, agents
from typing import List


class QLearning(object):

    def __init__(
        self,
        preprocess: str = 'FeatureExtract',
        network_archit: str = 'MLP(out_features=18, num_cells=[256, 256, 256, 256])',
        gamma: float = 0.99,
        tau: float = 0.01,
    ):
        """
        Args:
            preprocess (str): name of the preprocessing function
            network_archit (str): name of the network architecture
            num_inputs (int): number of inputs
            num_outputs (int): number of outputs
            gamma (float): discount factor for future rewards
            tau (float): target network update rate
        """
        self.gamma = gamma
        self.tau = tau
        # init network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = eval(preprocess)().to(device)
        self.prenomalization = PreNormalization().to(device)
        self.postscaling = PostScaling().to(device)
        self.policy_net = eval(network_archit).to(device)
        self.target_net = eval(network_archit).to(device)
        # copy weights from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # counter
        self.steps = 0

    def act(self, state: torch.Tensor, valid_actions: List[torch.Tensor], eps_threshold: float = 0.0):
        """Select an action based on epsilon-greedy policy

        Args:
            state (torch.Tensor): current state (1, n_observations)
            valid_actions (List[torch.Tensor]): available actions (n_valid_actions,)

        Returns:
            torch.Tensor: action taken (1, 1)
        """
        sample = torch.rand(1)
        # exploitation, select action with max Q value
        if sample > eps_threshold:
            with torch.no_grad():
                feat = self.preprocess(self.prenomalization(state))
                # t.max(0) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # NOTE(shawn): can only choose from actions that are available
                #               in the current state
                return valid_actions[self.policy_net(feat)[:, valid_actions].max(1)[1]]

        # exploration, select random action per batch
        else:
            return valid_actions[torch.randint(len(valid_actions), (1,))]

    def act_target(self, state: torch.Tensor, valid_actions: List[torch.Tensor]):
        with torch.no_grad():
            feat = self.preprocess(self.prenomalization(state))
            return valid_actions[self.target_net(feat)[:, valid_actions].max(1)[1]]

    def state_action_values(self, state: torch.Tensor, action: torch.Tensor):
        """Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            columns of actions taken. These are the actions which would've been taken
            for each batch state according to policy_net

        Args:
            state (torch.Tensor): current state (B, n_observations)
            action (torch.Tensor): action taken (B,)

        Returns:
            torch.Tensor: Q(s_t, a) (B, 1)
        """
        feat = self.preprocess(self.prenomalization(state))
        values = self.postscaling(self.policy_net(feat))
        return values.gather(1, action.unsqueeze(1))

    def expected_state_action_values(
        self,
        next_state: List[torch.Tensor],
        next_valid_actions: List[torch.Tensor],
        reward: torch.Tensor
    ):
        """Compute V(s_{t+1}) for all next states.
            Expected values of actions for non_final_next_states are computed based
            on the "older" target_net; selecting their best reward with max(1)[0].
            This is merged based on the mask, such that we'll have either the expected
            state value or 0 in case the state was final.

        Args:
            next_state (List[torch.Tensor]): next state [(n_observations,),]
            non_final_next_valid_actions (List[torch.Tensor]): available actions [(n_valid_actions,),]
            reward (torch.Tensor): reward received (B,)
            NOTE(shawn): if there's no next_state, will be None

        Returns:
            torch.Tensor: V(s_{t+1}) (B,)
        """
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state)),
            device=reward.device,
            dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in next_state if s is not None])
        non_final_next_valid_actions = \
            [a for a in next_valid_actions if a is not None]

        next_state_values = torch.zeros(len(next_state), device=reward.device)
        with torch.no_grad():
            non_final_next_feat = self.preprocess(self.prenomalization(non_final_next_states))
            # remove the actions that are not available in the next state
            target_q = self.postscaling(self.target_net(non_final_next_feat))
            target_q_max = torch.zeros(len(non_final_next_valid_actions), device=reward.device)
            for i in range(len(non_final_next_valid_actions)):
                target_q_max[i] = target_q[i, non_final_next_valid_actions[i]].max(0)[0]
            next_state_values[non_final_mask] = target_q_max

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward
        return expected_state_action_values

    def update_target_net(self):
        """soft update of the target network's weights
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (1 - self.tau) * target_net_state_dict[key] + \
                self.tau * policy_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, path: str):
        """save model weights

        Args:
            path (str): path to save the model weights
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """load model weights

        Args:
            path (str): path to load the model weights
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


class FeatureExtract(nn.Module):

    def __init__(self):
        super(FeatureExtract, self).__init__()
        self.index_list = [
            [0, 3, 6], [0, 4, 8], [0, 5, 7],
            [1, 4, 7], [1, 3, 8], [1, 5, 6],
            [2, 5, 8], [2, 4, 6], [2, 3, 7],
            [0, 1, 2], [3, 4, 5], [6, 7, 8]
        ]

    def forward(self, x):
        x = x.view(-1, 9)  # (B, 9)
        sets = x[:, self.index_list]  # (B, 12, 3)
        min_feat = torch.min(sets, dim=2)[0]  # (B, 12)
        max_feat = torch.max(sets, dim=2)[0]  # (B, 12)
        x = torch.cat([x, min_feat, max_feat], dim=1)  # (B, 33)
        return x


class PreNormalization(nn.Module):

    def __init__(self, shift=-50, scale=100):
        super(PreNormalization, self).__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x):
        x = (x + self.shift) / self.scale
        return x


class PostScaling(nn.Module):

    def __init__(self, scale=20.):
        super(PostScaling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        return x