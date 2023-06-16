import numpy as np
from src import interface


class Zero(interface.Estimator):

    def estimate(self, state: np.ndarray) -> int:
        return 0

    def __str__(self):
        return 'Zero()'


def two_zeros_check_for_3x3(
    state: np.ndarray,
    cost_reward: float,
    cost_penalty: float
) -> float:
    WIN_VECTOR_INDICES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # row
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # column
        [0, 4, 8], [2, 4, 6]]             # diag
    IMPOSSIBLE_LARGE_INT = 99999

    s_flatten = state.reshape(9)
    win_vectors = np.stack([s_flatten[np.array(idx)] for idx in WIN_VECTOR_INDICES])  # (8, 3)
    num_zeros_per_vector = np.sum(win_vectors == 0, axis=1)  # (8,)
    num_distinct_action_directions_left = np.sum(num_zeros_per_vector[:6] == 0)
    # NOTE(shawn): to find the min value excluding 0, we set 0 to a large number
    win_vectors[win_vectors == 0] = IMPOSSIBLE_LARGE_INT
    min_indices_per_vector = np.argmin(win_vectors, axis=1)  # (8,)
    for row_idx, num_zero in enumerate(num_zeros_per_vector):
        # early continue if num_zero != 2 (either cannot take a move or unknown)
        if num_zero != 2:
            continue
        col_idx = min_indices_per_vector[row_idx]  # the column index of the value that's left
        min_val = win_vectors[row_idx][col_idx]    # the value that's left
        # if win vector is column or diag -> check row
        if row_idx not in [0, 1, 2]:
            # if there's a zero in the row, we can't take a move
            if num_zeros_per_vector[col_idx] > 0:
                pass
            # if there's a smaller value in the row, we can't subtract all the way to min_val
            elif min_val != win_vectors[col_idx][min_indices_per_vector[col_idx]]:
                pass
            # if we can do it in one step
            elif min_val < 4:
                return cost_reward + min_val  # reward - action_cost
            # if there's only one distinct action direction left, reduces to a 1x1 game
            elif num_distinct_action_directions_left == 1:
                if min_val > cost_penalty - cost_reward:
                    return 0.
                if min_val % 4 == 0:  # cannot win
                    # minimize action_cost by selecting 1
                    return min_val // 4
                else:
                    # to compensate oppo select 1 + reward + last step
                    return (min_val // 4) * 3 + cost_reward + (min_val % 4)
        # if win vector is row or diag -> check column
        if row_idx not in [3, 4, 5]:
            # if there's a zero in the column, we can't take a move
            if num_zeros_per_vector[col_idx+3] > 0:
                pass
            # if there's a smaller value in the column, we can't subtract all the way to min_val
            elif min_val != win_vectors[col_idx+3][min_indices_per_vector[col_idx+3]]:
                pass
            # if we can do it in one step
            elif min_val < 4:
                return cost_reward + min_val  # reward - action_cost
            # if there's only one distinct action direction left, reduces to a 1x1 game
            elif num_distinct_action_directions_left == 1:
                if min_val > cost_penalty - cost_reward:
                    return 0.
                if min_val % 4 == 0:  # cannot win
                    # minimize action_cost by selecting 1
                    return min_val // 4
                else:
                    # to compensate oppo select 1 + reward + last step
                    return (min_val // 4) * 3 + cost_reward + (min_val % 4)
    return 0.  # set heuristic as 0 for unknown cases

class OneStepLookAhead(interface.Estimator):

    def __init__(self, reward: float = 15., penalty: float = 7.):
        self.cost_reward = -reward
        self.cost_penalty = penalty

    def estimate(self, state: np.ndarray) -> float:
        return two_zeros_check_for_3x3(state, self.cost_reward, self.cost_penalty)

    def __str__(self):
        return 'OneStepLookAhead()'