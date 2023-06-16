import random
import time

SIZE = 3

def random_init(board):
    random.seed()
    for i in range(SIZE):
        for j in range(SIZE):
            board[i][j] = random.randint(50, 99)

def given_init():
    given = [[4, 2, 3],
             [2, 3, 4],
             [3, 4, 2]]
    return given

def print_board(board):
    for i in range(SIZE):
        for j in range(SIZE):
            print(board[i][j], end=" ")
        print()

def check_valid(board, row_or_col, subtract):
    min_val = float('inf')

    if subtract <= 0 or subtract > 3: # subtract not 1, 2 or 3
        return False

    if row_or_col >= 0 and row_or_col < SIZE: # row
        for i in range(SIZE):
            min_val = min(board[row_or_col][i], min_val)
        if min_val == 0:
            return False
        elif min_val == 1 and subtract > 1:
            return False
        elif min_val == 2 and subtract > 2:
            return False
        return True
    elif row_or_col >= SIZE and row_or_col < SIZE*2: # col
        row_or_col -= SIZE
        for i in range(SIZE):
            min_val = min(board[i][row_or_col], min_val)
        if min_val == 0:
            return False
        elif min_val == 1 and subtract > 1:
            return False
        elif min_val == 2 and subtract > 2:
            return False
        return True
    else:
        return False

def board_subtract(board, row_or_col, subtract):
    if row_or_col >= 0 and row_or_col < SIZE: # row
        for i in range(SIZE):
            board[row_or_col][i] -= subtract
    elif row_or_col >= SIZE and row_or_col < SIZE*2: # col
        row_or_col -= SIZE
        for i in range(SIZE):
            board[i][row_or_col] -= subtract

def check_diagonal(board):
    diagonal1 = True
    diagonal2 = True
    for i in range(SIZE):
        diagonal1 &= (board[i][i] == 0)
        diagonal2 &= (board[i][SIZE - i - 1] == 0)
    return diagonal1 or diagonal2

def check_row(board, row): # check num in row are all 0
    for i in range(SIZE):
        if board[row][i] != 0:
            return False
    return True

def check_col(board, col):
    for i in range(SIZE):
        if board[i][col] != 0:
            return False
    return True

def check_game_end(board, dead_end):
    game_end = False
    row_end = True
    col_end = True

    # end condition 1 : all the numbers in any row, column, or diagonal become 0
    for i in range(SIZE):
        game_end |= check_row(board, i)
        game_end |= check_col(board, i)
    game_end |= check_diagonal(board)

    # end condition 2 : every row or column contains the number 0
    if not game_end:
        for i in range(SIZE):
            row_exist_zero = False
            col_exist_zero = False
            for j in range(SIZE):
                if board[i][j] == 0:
                    row_exist_zero = True
                if board[j][i] == 0:
                    col_exist_zero = True
            if not row_exist_zero:
                row_end = False
            if not col_exist_zero:
                col_end = False
        dead_end = row_end and col_end
        game_end |= dead_end

    return game_end, dead_end

'''
def make_your_move(board):
    #**********
    # TODO HERE:
    # 1.You CANNOT directly modify "board" in this function
    # 2.just return "row_or_col" and "subtract"
    # 3.You can call check_valid() if you want
    # row_or_col: 1st_row, 2nd_row, 3rd_row -> 0, 1, 2 ; 1st_col, 2nd_col, 3rd_col -> 3, 4, 5
    # subtract: number to subtract, should be 1, 2 or 3 (don't forget restriction!)
    #**********

    row_or_col = 3
    subtract = 1

    # row_or_col = int(input())
    # subtract = int(input())

    return row_or_col, subtract
'''
import sys
sys.path.append('./')  # TODO(shawn): replace this path with the path to "subtraction-game" folder
from src import agents, environment, heuristic

make_your_move = agents.MinimaxV2(
  max_computing_nodes=1200000,             # to estimate max_depth, use 1.2M or 1M
  heuristic=heuristic.OneStepLookAhead(),  # heuristic function for non-terminal leaf states
  env=environment.SubtractionRule(),       # game rule known by agent (to provide legal moves)
  timeout=58.,                             # to prevent timeout
).make_your_move_method(
  env=environment.SubtractionRule()        # game rule known by environment (to provide legal moves)
)


def opponent_move(board):

    #row_or_col = int(input())
    #subtract = int(input())

    valid = False
    while(not valid):
        row_or_col = random.randint(0, 5)
        subtract = random.randint(1, 3)
        valid = check_valid(board, row_or_col, subtract)

    return row_or_col, subtract

if __name__ == "__main__":

    board = [[0 for i in range(SIZE)] for j in range(SIZE)]
    player = 0 # player 0 goes first
    total_cost = [0, 0] # total cost for each player
    your_turn = True
    game_end = False
    dead_end = False
    reward = 15
    penalty = 7

    print("Board initialization…\n")

    random_init(board)   # initialize board with positive integers
    #board = given_init() # initialize board with given intergers (for testing only)

    while not game_end:
        print("Current board:")
        print_board(board)
        print("\nPlayer", player, "'s turn:\n")
        #input("Press any key to continue…")
        row_or_col = 0  # row1, row2, row3 -> 0, 1, 2 ; col1, col2, col3 -> 3, 4, 5
        subtract = 0  # number to subtract

        if your_turn:
            start = time.time()
            row_or_col, subtract = make_your_move(board)
            end = time.time()

            print("Time:", end - start, "seconds\n")
        else:
            #start = time.time()
            row_or_col, subtract = opponent_move(board)
            #end = time.time()

            #print("Time:", end - start, "seconds\n")

        print("Player", player, "'s move: row_or_col:", row_or_col, "subtract:", subtract, "\n")

        print("Valid checking...", end="")
        valid = check_valid(board, row_or_col, subtract)  # legal move checking
        if valid: print("The move is vaild.")
        else: print("The move is invalid, game over.")
        #input("Press any key to continue…")

        board_copy = [row[:] for row in board]  # make a copy of the board
        board_subtract(board_copy, row_or_col, subtract)

        # update the board
        board = board_copy

        # update player's total cost
        total_cost[player] += subtract

        print("Player 0 total cost:", total_cost[0])
        print("Player 1 total cost:", total_cost[1])
        print("--------------------------------------\n")

        # check if game has ended
        game_end, dead_end= check_game_end(board, dead_end)

        if not game_end:
            # switch to other player
            your_turn = not your_turn
            player = (player + 1) % 2

    print("Final board:")
    print_board(board)
    if not dead_end:
        print("Player", player, "ends with a diagonal/row/col of 0's!")
        total_cost[player] -= reward
    else:
        print("Player", player, "ends with a dead end!")
        total_cost[player] += penalty

    print("Player 0 total cost:", total_cost[0])
    print("Player 1 total cost:", total_cost[1])
    input("Press Enter to exit…")