from absl import logging
import numpy as np
import random
import time
from src import interface, environment, heuristic, simulate
from typing import Tuple, List, Optional


class RandomSelect(interface.Agent):
    """Randomly select an action
    """

    def act(self, state: np.ndarray, actions: List) -> Tuple:
        return random.choice(actions)

    def __str__(self):
        return 'RandomSelect()'


class GreedyMinimumCost(interface.Agent):
    """Randomly select one if there are multiple actions with minimum cost

        this is a simplified version of Minimax(max_depth=1)
    """

    def __init__(
        self,
        env: interface.Environment = environment.SubtractionRule(),
    ):
        self.env = env

    def act(self, state: np.ndarray, actions: List) -> Tuple:
        costs = []
        for action in actions:
            _, cost, _ = self.env.step(state, action)
            costs.append(cost)
        # if there are multiple actions with minimum cost, randomly select one
        min_cost = min(costs)
        min_cost_actions = [action for action, cost in zip(actions, costs) if cost == min_cost]
        return random.choice(min_cost_actions)

    def __str__(self):
        return 'GreedyMinimumCost(env={})'.format(self.env)


class Minimax(interface.Agent):
    """minimax with alpha-beta pruning
    """

    # TODO(shawn): add iterative deepening when time is enough

    class Node:
        def __init__(self, state: np.ndarray, depth: int, action: Tuple = None):
            self.state = state     # state of this node
            self.depth = depth     # depth of this node
            self.action = action   # action to reach this node
            self.value = None      # value (cost backpropagated by minimax) of this node
            self.path_cost = None  # path cost (cost from root to this node) of this node
            self.children = []     # children of this node

    def __init__(
        self,
        max_depth: int = 3,
        heuristic: interface.Estimator = heuristic.Zero(),
        env: interface.Environment = environment.SubtractionRule(),
        epsilon_greedy: float = 0.0,
    ) -> None:
        """
        Args:
            max_depth: max depth to search
            heuristic: heuristic function
            env: game rule
        """
        self.max_depth = max_depth
        self.env = env
        self.heuristic = heuristic
        self.epsilon_greedy = epsilon_greedy

    def _minimax(
        self,
        node: Node,
        max_depth: int,
        alpha: float = -np.inf,
        beta: float = np.inf
    ) -> None:
        """a DFS alpha beta pruning method

            if node.depth is even, it's player's turn (do min)
            if node.depth is odd, it's opponent's turn (do max)

        Args:
            node: current node
            max_depth: max depth to search
            alpha: alpha value
            beta: beta value

        Returns:
            None, value is saved to node.value
        """
        # if terminal node
        if self.env.is_terminal(node.state):
            node.value = node.path_cost
            return
        # if reaches max_depth
        if node.depth == max_depth:
            h = self.heuristic.estimate(node.state)
            # NOTE(shawn): h is in the perspective of player
            #  if the next step is player's turn (node.depth is even), h is positive
            node.value = h if node.depth % 2 == 0 else -h
            node.value += node.path_cost
            return
        # expand children
        for action in self.env.actions(node.state):
            child_state, cost, _ = self.env.step(node.state, action)
            child = self.Node(
                state=child_state,
                depth=node.depth + 1,
                action=action,
            )
            # NOTE(shawn): cost is in the perspective of player
            #  if the next step is player's turn (node.depth is even), cost is positive
            child.path_cost = cost if node.depth % 2 == 0 else -cost
            child.path_cost += node.path_cost
            node.children.append(child)
        # if player's turn, try to minimize cost
        if node.depth % 2 == 0:
            node.value = np.inf
            for child in node.children:
                self._minimax(child, max_depth, alpha, beta)
                node.value = min(node.value, child.value)
                beta = min(beta, node.value)
                if beta < alpha:
                    break
        # if opponent's turn, try to maximize cost
        else:
            node.value = -np.inf
            for child in node.children:
                self._minimax(child, max_depth, alpha, beta)
                node.value = max(node.value, child.value)
                alpha = max(alpha, node.value)
                if beta < alpha:
                    break

    def act(self, state: np.ndarray, actions: List) -> Tuple:
        # epsilon greedy
        if random.random() < self.epsilon_greedy:
            return random.choice(actions)

        # init first layer
        self.root = self.Node(state, depth=0)
        self.root.path_cost = 0
        # start search
        self._minimax(self.root, self.max_depth)
        # return action with min value
        min_value = min([child.value for child in self.root.children])
        min_value_actions = [child.action for child in self.root.children if child.value == min_value]
        return random.choice(min_value_actions)

    def __str__(self):
        return 'Minimax(max_depth={}, heuristic={}, env={}, epsilon_greedy={})'.format(
            self.max_depth, self.heuristic, self.env, self.epsilon_greedy)


class MinimaxV2(interface.Agent):
    """minimax with alpha-beta pruning + map branch factor to max depth
    """

    def __init__(
        self,
        max_computing_nodes: int = 1200000,
        heuristic: interface.Estimator = heuristic.OneStepLookAhead(),
        env: interface.Environment = environment.SubtractionRule(),
        timeout: float = 58.,
    ) -> None:
        """
        Args:
            max_computing_nodes: to estimate max depth
            heuristic: heuristic function
            env: game rule
        """
        self.max_computing_nodes = max_computing_nodes
        self.env = env
        self.heuristic = heuristic
        self.timeout = timeout
        self.act_start_time = None
        self.memory = {}

    def _minimax(
        self,
        state: np.ndarray,
        from_action: str,
        path_cost: int,
        depth: int,
        alpha: float = -np.inf,
        beta: float = np.inf
    ) -> Tuple[int, Tuple[int, int]]:
        """a DFS alpha beta pruning method

            if node.depth is even, it's player's turn (do min)
            if node.depth is odd, it's opponent's turn (do max)

        Args:
            state: current state
            path_cost: path cost (cost from root to this node)
            depth: current depth
            alpha: alpha value
            beta: beta value

        Returns:
            value: value of this node
            move: action to reach this node
        """
        # check if in memory
        #if depth == 2 and from_action in self.memory:
        #    return self.memory[from_action], None
        # if terminal node
        # TODO(shawn): refactor environment API
        is_terminal, cost_terminal = self.env._is_terminal_and_cost(state)
        if is_terminal:
            if depth % 2 != 0:
                return path_cost + cost_terminal, None
            else:
                return path_cost - cost_terminal, None
        # if reaches max_depth
        if depth == self.max_depth:
            h = self.heuristic.estimate(state)
            # NOTE(shawn): h is in the perspective of player
            #  if the next step is player's turn (node.depth is even), h is positive
            value = h if depth % 2 == 0 else -h
            value += path_cost
            return value, None
        # if player's turn, try to minimize cost
        if depth % 2 == 0:
            value = np.inf
            for action in self.env.actions(state):
                # take a step
                row_col_idx, subtract_val = action
                if row_col_idx < 3:
                    state[row_col_idx, :] -= subtract_val
                else:
                    state[:, row_col_idx - 3] -= subtract_val
                # go deeper if still have time
                if time.time() - self.act_start_time < self.timeout:
                    child_v, _ = self._minimax(
                        state=state,
                        from_action=from_action + str(action),
                        path_cost=path_cost + subtract_val,
                        depth=depth + 1,
                        alpha=alpha,
                        beta=beta,
                    )
                else:
                    child_v = self.heuristic.estimate(state)
                    logging.warning('run out of time during minimax, depth: {}, action: {}'.format(depth, action))
                # save to memory if depth == 0
                if depth == 0:
                    self.memory[str(action)] = child_v
                # undo step
                if row_col_idx < 3:
                    state[row_col_idx, :] += subtract_val
                else:
                    state[:, row_col_idx - 3] += subtract_val
                # update value
                if child_v < value:
                    value, move = child_v, action
                    beta = min(beta, value)
                if value <= alpha:
                    break
            return value, move
        # if opponent's turn, try to maximize cost
        else:
            value = -np.inf
            for action in self.env.actions(state):
                # take a step
                row_col_idx, subtract_val = action
                if row_col_idx < 3:
                    state[row_col_idx, :] -= subtract_val
                else:
                    state[:, row_col_idx - 3] -= subtract_val
                # go deeper if still have time
                if time.time() - self.act_start_time < self.timeout:
                    child_v, _ = self._minimax(
                        state=state,
                        from_action=from_action + str(action),
                        path_cost=path_cost - subtract_val,
                        depth=depth + 1,
                        alpha=alpha,
                        beta=beta,
                    )
                else:
                    child_v = self.heuristic.estimate(state)
                    logging.warning('run out of time during minimax, depth: {}, action: {}'.format(depth, action))
                # save to memory if depth == 1
                #if depth == 1 and not self.env.is_terminal(state):
                #    # NOTE(shawn): sub_val_x here is to compensate the path_cost when being swapped
                #    sub_val_first = int(from_action[1:-1].split(',')[1])
                #    sub_val_second = action[1]
                #    self.memory[str(action) + from_action] = child_v + (sub_val_second - sub_val_first) * 2
                # XXX(shawn): debug
                #if depth == 1:
                #    if from_action + str(action) in self.memory:
                #        if self.memory[from_action + str(action)] != child_v:
                #            print('from_action: {}, action: {}, value: {}'.format(from_action, action, self.memory[from_action + str(action)]))
                #            print('from_action: {}, action: {}, value: {}'.format(from_action, action, child_v))
                #            print('state: {}'.format(state))
                # undo step
                if row_col_idx < 3:
                    state[row_col_idx, :] += subtract_val
                else:
                    state[:, row_col_idx - 3] += subtract_val
                # update value
                if child_v > value:
                    value, move = child_v, action
                    alpha = max(alpha, value)
                if value >= beta:
                    break
            return value, move

    def act(self, state: np.ndarray, actions: List) -> Tuple:
        # init timer
        self.act_start_time = time.time()

        # estimate max depth
        num_branches = len(actions)
        self.max_depth = min(8,
            max(3,int(np.log(self.max_computing_nodes) / (np.log(num_branches) + 1e-6))))

        # start search
        self.memory.clear()
        _, action_mx = self._minimax(
            state=state,
            from_action="",
            path_cost=0,
            depth=0,
            alpha=-np.inf,
            beta=np.inf,
        )
        return action_mx

        # TODO(shawn): select the action that subtracts to the min element in state
        # get all actions with min value from memory
        values = [self.memory[str(action)] for action in actions]
        min_value_actions = [action for action, value in zip(actions, values) if value == min(values)]
        # select the action that subtracts to the min element in state
        min_state_value = [np.min(self.env.step(state, action)[0], axis=1 if action[0] < 3 else 0)[action[0] % 3] for action in min_value_actions]
        #return min_value_actions[np.argmin(min_state_value)]

    def __str__(self):
        return 'MinimaxV2(max_computing_nodes={}, heuristic={}, env={}, timeout={})'.format(
            self.max_computing_nodes, self.heuristic, self.env, self.timeout)


class MonteCarloTreeSearch(interface.Agent):
    """Monte Carlo Tree Search
    """

    class Node(object):
        def __init__(
            self,
            state: np.ndarray,
            parent: 'Node' = None,
            action: Tuple = None,
            depth: int = 0,
            cost_player: int = 0,
            cost_opponent: int = 0,
        ):
            # static
            self.state = state
            self.parent = parent
            self.action = action
            self.depth = depth
            self.cost_player = cost_player
            self.cost_opponent = cost_opponent
            # dynamic
            self.children = []
            self.wins = 0  # in player's perspective
            self.visits = 0

    def __init__(
        self,
        max_num_simulations: int = 1000,
        max_time_seconds: float = 58.,
        coeffi_explore: float = np.sqrt(2),
        player: interface.Agent = RandomSelect(),
        env: interface.Environment = environment.SubtractionRule(),
        game: interface.GameSimulator = simulate.TwoPlayerGame(),
        handicapped: int = 0,
        run_only_when_min_val_less_than: int = -1,
    ):
        self.player = player
        self.env = env
        self.game = game
        self.max_num_simulations = max_num_simulations
        self.max_time_seconds = max_time_seconds
        self.coeffi_explore = coeffi_explore
        self.root = None
        self.run_condition = run_only_when_min_val_less_than
        self.handicapped = handicapped
        # MCTS tree
        self.hashmap = {}

    def act(self, state: np.ndarray, actions: List) -> Tuple:

        # debug
        print('state: {}'.format(state))


        _st = time.time()

        # init root
        state_hash = str(state)
        #if state_hash not in self.hashmap:
        if True:
            root = self.Node(state, action=(-1,-1), cost_player=self.handicapped, cost_opponent=0)
            #self.hashmap[state_hash] = root
        else:
            root = self.hashmap[state_hash]
            print(state_hash)
            print('wins:', root.wins)
            print('visits:', root.visits)
            print('state:', root.state)

        # NOTE(shawn): replace with min cost select if you doesn't believe in MCTS
        if self.run_condition > 0 and np.min(state) > self.run_condition:
            '''
            # find minimum cost
            min_cost = np.min([child.cost_player for child in root.children])
            # find most visited child with minimum cost
            min_cost_children = [child for child in root.children if child.cost_player == min_cost]
            child_most_visited = min_cost_children[np.argmax([child.visits for child in min_cost_children])]

            # debug print
            for child in child_most_visited.children:
                print('\tstate: {}, action: {}, visits: {}, wins: {}'.format(child.state, child.action, child.visits, child.wins))

            return child_most_visited.action
            '''
            # random select action with minimum cost
            costs = []
            for action in actions:
                _, cost, _ = self.env.step(state, action)
                costs.append(cost)
            min_cost = min(costs)
            min_cost_actions = [action for action, cost in zip(actions, costs) if cost == min_cost]
            return random.choice(min_cost_actions)

        # start search
        num_simulations = 0
        while num_simulations < self.max_num_simulations and time.time() - _st < self.max_time_seconds:
            leaf = self._selection(root)
            child = self._expansion(leaf)
            win = self._simulation(child)
            self._backpropagation(child, win)
            num_simulations += 1

        # debug
        for child in root.children:
            print('action: {}, visits: {}, wins: {}, depth: {}'.format(child.action, child.visits, child.wins, child.depth))


        # return action with max wins
        child_robust = root.children[np.argmax([child.visits for child in root.children])]
        #child_max = root.children[np.argmax([child.wins for child in root.children])]
        return child_robust.action

    def _selection(self, x: Node) -> Node:
        """select a node to expand
        """
        while len(x.children) > 0:  # loop until reaches leaf
            # calculate UCB1
            ucb1s = []
            for child in x.children:
                if child.visits == 0:
                    ucb1 = np.inf if x.depth % 2 == 0 else -np.inf
                else:
                    exploit = child.wins / child.visits
                    explore = np.sqrt(np.log(x.visits) / child.visits)
                    ucb1 = exploit + self.coeffi_explore * explore if x.depth % 2 == 0 \
                        else exploit - self.coeffi_explore * explore
                ucb1s.append(ucb1)

            # debug
            #print('depth:', x.depth)
            #print('wins:', [child.wins for child in x.children])
            #print('visits:', [child.visits for child in x.children])
            #print('ucb1s:', ucb1s)

            # select child with max/min UCB1
            if x.depth % 2 == 0:
                x = x.children[np.argmax(ucb1s)]  # if player's turn, select child with max ucb1
            else:
                x = x.children[np.argmin(ucb1s)]  # if opponent's turn, select child with min ucb1
        return x

    def _expansion(self, x: Node) -> Node:
        """expand a node
        """
        if x.visits != 0:
            # expand and add children to tree
            for action in self.env.actions(x.state):
                child_state, cost, _ = self.env.step(x.state, action)
                _n = self.Node(
                    state=child_state,
                    parent=x,
                    action=action,
                    depth=x.depth + 1,
                    cost_player=x.cost_player + (cost if x.depth % 2 == 0 else 0),
                    cost_opponent=x.cost_opponent + (cost if x.depth % 2 != 0 else 0),
                )
                #self.hashmap[str(child_state)] = _n  # add to hashmap
                x.children.append(_n)
            # return itself if no children
            if len(x.children) == 0:
                return x
            # random select a child
            x = random.choice(x.children)
        return x

    def _simulation(self, x: Node) -> int:
        """run simulation
        """
        self.game.reset()
        _, costs = self.game.run(
            agents=[self.player, self.player],
            env=self.env,
            initial_state=x.state,
            initial_costs=\
                (x.cost_player, x.cost_opponent) if x.depth % 2 == 0 \
                else (x.cost_opponent, x.cost_player)
        )
        cost_player = costs[0] if x.depth % 2 == 0 else costs[1]
        cost_opponent = costs[1] if x.depth % 2 == 0 else costs[0]
        win = 1 if cost_player < cost_opponent else -1
        return win

    def _backpropagation(self, x: Node, win: int):
        """backpropagation
        """
        while x is not None:
            x.wins += win
            x.visits += 1
            x = x.parent

    def __str__(self):
        return 'MonteCarloTreeSearch(player={}, env={}, game={}, num_simulations={}, time_limit={}, coeffi={}, run_condition={}, handicapped={})'.format(
            self.player, self.env, self.game,
            self.max_num_simulations, self.max_time_seconds,
            self.coeffi_explore, str(self.run_condition), self.handicapped)