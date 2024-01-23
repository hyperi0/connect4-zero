import random
import numpy as np

### Generate new agent function using model
def build_agent(model):
    def _function(obs, config):
        # Use the best model to select a column
        col, _ = model.predict(np.array(obs['board']).reshape(1, config.rows, config.columns))
        # Check if selected column is valid
        is_valid = (obs['board'][int(col)] == 0)
        # If not valid, select random move. 
        if is_valid:
            return int(col)
        else:
            return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

    return _function

# Get new board with given piece dropped
def drop_piece(grid, col, mark, config):
    next_grid = np.asarray(grid)
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return tuple(map(tuple, next_grid))

def legal_moves(grid, config):
    return [col for col in range(config.columns) if grid[0][col] == 0]

def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(-1) == config.inarow

def is_terminal_grid(grid, config):
    return check_winner(grid, config) != 0

def check_window_winner(window, config):
    if window.count(1) == config.inarow:
        return 1
    if window.count(-1) == config.inarow:
        return -1
    else:
        raise Exception("Winner not found")
    
def check_winner(grid, config):
    np_grid = np.asarray(grid)
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(np_grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return check_window_winner(window, config)
    # Check for draw 
    if list(np_grid[0, :]).count(0) == 0:
        return 3
    
    return 0

def score_game(grid, config):
    np_grid = np.asarray(grid)
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(np_grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(np_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                winner = check_window_winner(window, config)
                return 1 if winner == 1 else -1
    # Check for draw 
    if list(np_grid[0, :]).count(0) == 0:
        return 0
    
    return 0

def empty_grid(config):
    return tuple(map(tuple, np.zeros((config.rows, config.columns), dtype=int)))

def legal_moves_mask(grid, config):
    moves = legal_moves(grid, config)
    mask = [1 if i in moves else 0 for i in range(config.columns)]
    return mask

def reverse_grid(grid):
    return tuple(tuple(-piece for piece in row) for row in grid)