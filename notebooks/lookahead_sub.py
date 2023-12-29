def my_agent(obs, config):
    
    import random
    import numpy as np
    
    N_STEPS = 3

    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    def check_window(window, n_disks, mark, config):
        return window.count(mark) == n_disks and window.count(0) == config.inarow - n_disks

    def count_windows(grid, n_disks, mark, config):
        n_windows = 0

        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - config.inarow + 1):
                window = list(grid[row, col: col + config.inarow])
                if check_window(window, n_disks, mark, config):
                    n_windows += 1
        
        # vertical
        for row in range(config.rows - config.inarow + 1):
            for col in range(config.columns):
                window = list(grid[row: row + config.inarow, col])
                if check_window(window, n_disks, mark, config):
                    n_windows += 1

        # positive diagonal
        for row in range(config.rows - config.inarow + 1):
            for col in range(config.columns - config.inarow + 1):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, n_disks, mark, config):
                    n_windows += 1

        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - config.inarow + 1):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, n_disks, mark, config):
                    n_windows += 1
            
        return n_windows

    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
        return score

    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
                
        return False

    def alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config):
        isTerminal = is_terminal_node(node, config)
        if depth == 0 or isTerminal:
            return get_heuristic(node, mark, config)

        valid_moves = [col for col in range(config.columns) if node[0, col] == 0]

        if maximizingPlayer:
            value = -np.Inf
            for move in valid_moves:
                child = drop_piece(node, move, mark, config)
                value = max(value, alphabeta(child, depth-1, alpha, beta, False, mark, config))
                if value > beta:
                    break
                alpha = max(alpha, value)
            return value
        else:
            value = np.Inf
            for move in valid_moves:
                child = drop_piece(node, move, mark%2+1, config)
                value = min(value, alphabeta(child, depth-1, alpha, beta, True, mark, config))
                if value < alpha:
                    break
                beta = min(beta, value)
            return value

    # Uses minimax with alphabeta pruning to calculate value of dropping piece in selected column
    def score_move_alphabeta(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark, config)
        return score


    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    scores = dict(zip(valid_moves, [score_move_alphabeta(grid, move, obs.mark, config, N_STEPS)
                                    for move in valid_moves]))
    max_score = max(scores.values())
    max_moves = [key for key in scores.keys() if scores[key] == max_score]
    move = random.choice(max_moves)
    
    return move