import random
import sys


# Difference between our movements and the opponent
def heuristic_simple(game, player):
    # Getting legal moves for our player
    own_moves = len(game.get_legal_moves(player))
    # Getting legal moves for the opponent
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Subtracting opponents moves from our moves
    return float(own_moves - opp_moves)


# Adding weight of 2 to our moves
def heuristic_weighted(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Applying the weight to our moves
    return float(own_moves * 2 - opp_moves)

# Dynamic weighting, the weights values change through the game
def heuristic_moves_to_board(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Getting board size
    board_size = game.height * game.width
    # Getting the weight from the current state of the game
    moves_to_board = game.move_count / board_size
    # Applying the new weight to our weighted model
    return float((own_moves * moves_to_board * 2 - opp_moves))


# Pondering movements and adding the blank spaces to the equation
def heuristic_weighted_with_board(game, player):
    # Getting blank spaces 
    blank_spaces = len(game.get_blank_spaces())
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Descending ponderating
    return float(own_moves * 3 - opp_moves * 2 + blank_spaces * 1)


# Using different strategies in different states of the game
def heuristic_defensive_to_offensive(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    board_size = game.height * game.width
    moves_to_board = game.move_count / board_size

    #Validating the current sate (first half or second half)
    if moves_to_board <= 0.5:
        # Playing defensive
        return float(own_moves * 2 - opp_moves)
    else:
        # Playing offensive
        return float(own_moves - opp_moves * 2)


# Using different strategies in different states of the game
def heuristic_offensive_to_defensive(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    board_size = game.height * game.width
    moves_to_board = game.move_count / board_size

    #Validating the current sate (first half or second half)
    if moves_to_board > 0.5:
        # Playing defensive
        return float(own_moves * 2 - opp_moves)
    else:
        # Playing offensive
        return float(own_moves - opp_moves * 2)


# Chasing the opponent
def heuristic_blocking_opponent(game, player):
    # Getting player moves
    play_moves = game.get_legal_moves(player)
    # Getting opponent moves
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    # Saving the intersected moves betwen player and opponent
    same_moves = len(list(set(play_moves) & set(opp_moves)))
    # Calculating the length of the moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Applying the intersected moves to our offensive model
    return float(own_moves - opp_moves * 2 + same_moves)


# Array of the methods, it's necesary when running from terminal
# (automatised_tester.sh)
heuristic = {
    'heuristic_simple': heuristic_simple,
    'heuristic_weighted': heuristic_weighted,
    'heuristic_moves_to_board': heuristic_moves_to_board,
    'heuristic_weighted_with_board': heuristic_weighted_with_board,
    'heuristic_defensive_to_offensive': heuristic_defensive_to_offensive,
    'heuristic_offensive_to_defensive': heuristic_offensive_to_defensive,
    'heuristic_blocking_opponent': heuristic_blocking_opponent
}

# Timeout exception
class Timeout(Exception):
    pass


# Returning the value of the current heuristic to test
def custom_score(game, player):
    # If loose, return - infinity
    if game.is_loser(player):
        return float("-inf")

    # If win return infinity
    if game.is_winner(player):
        return float("inf")

    return heuristic_moves_to_board(game, player)
    # Uncomment the line below when running from terminal (automatised)
    #return heuristic[sys.argv[1]](game, player)



# Class of the Agent and Archenemy
class CustomPlayer:

    # Default constructor
    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=5.):
        # Depth of the search
        self.search_depth = search_depth
        # Flag indicating the use of Iterative Deepening algorithm
        self.iterative = iterative
        # The score returned by the heuristic
        self.score = score_fn
        # Algorithm to use (minimax or alpha-beta pruning)
        self.method = method
        # Time counters and threshold
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        # Variables to use as alpha and beta
        self.BETA = float("inf")
        self.ALFA = float("-inf")
        # Move to use when lose
        self.surrender_move = (-1, -1)

    def get_move(self, game, legal_moves, time_left):

        # Time counter
        self.time_left = time_left

        # Validating legal moves
        if not legal_moves:
            # If not surrender
            return self.surrender_move

        # Initialising the values of best move as any available moves 
        # and ALFA as -infinity
        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_score = self.ALFA

        try:
            # Selecting the desired algorithm
            if self.method is 'minimax':
                # Choosing minimax
                search = self.minimax
            else:
                # Choosing alpha-beta pruning
                search = self.alphabeta

            # If using Iterative Deepening
            if self.iterative:
                # Initialising Depth
                search_depth = 1

                while 1:
                    # Getting score and next move
                    score, next_move = search(game, depth=search_depth, maximizing_player=True)
                    # Validating if the new score is better than the old one
                    if (score, next_move) > (best_score, best_move):
                        # If so, assign the new values to the best scores and move
                        (best_score, best_move) = (score, next_move)

                    # Increasing depth in 1
                    search_depth += 1

            # If not ID, just perform the search in the fixed depth
            else:
                score, next_move = search(game, self.search_depth)

                # Validating if the new score is better than the old one
                if (score, next_move) > (best_score, best_move):
                    (best_score, best_move) = (score, next_move)

        except Timeout:
            # Returning best move until now
            return best_move

        return best_move

        # Return the best move from the last completed search iteration

    # Minimax algorithm
    def minimax(self, game, depth, maximizing_player=True):
        # Initialising our best move
        best_move = self.surrender_move

        # Assigning the best score based on if the player is a maximiser player or minimiser player
        best_score = self.ALFA if maximizing_player else self.BETA
        optimizer = max if maximizing_player else min

        if self.time_left() < self.TIMER_THRESHOLD: raise Timeout()

        # Validating depth and returning the heuristic score if leaf node reached
        if depth is 0: return self.score(game, self), best_move

        for move in game.get_legal_moves():

            # Recursive call on next move to get the score
            score, _ = self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)
            # Getting the best score, depending the actual player (minimiser, maximiser)
            best_score, best_move = optimizer((best_score, best_move), (score, move))

        return best_score, best_move


    # Alpha-beta pruning algorithm
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        # Throwing exception if we run out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            # Timeout exception
            raise Timeout()

        # Initialising our best move
        best_move = self.surrender_move
        # Initialising our best score, if maximiser then alpha, if minimiser the beta
        best_score = alpha if maximizing_player else beta

        # Throwin exception if we run out of time
        if self.time_left() < self.TIMER_THRESHOLD: raise Timeout()

        # Validating depth and returning the heuristic score if leaf node reached
        if depth is 0: return self.score(game, self), best_move

        # Validating for current player in algorithm
        if maximizing_player:

            # Iterating through the number of available moves
            for move in game.get_legal_moves():

                # Recursive call on next move to get the score
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, best_score, beta, not maximizing_player)

                # Validating if the new score is better (greater) than the older 
                if score > best_score:
                    # Asigning new score
                    best_score, best_move = score, move

                # Validating if alfa >= beta, then prune
                if best_score >= beta:
                    return (best_score, best_move)

        # Minimiser Player
        else:
            # Iterating through the number of available moves
            for move in game.get_legal_moves():
                # Getting score
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, best_score, not maximizing_player)

                # Validating if the new score is better (lesser) than the older
                if score < best_score:
                    # Asigning new score
                    best_score, best_move = score, move

                # Validating if alfa >= beta, then prune
                if best_score <= alpha:
                    return (best_score, best_move)

        return best_score, best_move
