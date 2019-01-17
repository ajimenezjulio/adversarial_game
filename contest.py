import itertools
import random
import warnings

from collections import namedtuple

from isolation import Board
from sample_players import RandomPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from Agent import CustomPlayer
from Agent import custom_score

# Number of matches against each opponent
NUM_MATCHES = 5
# Time before timeout
TIME_LIMIT = 300

# Timeout warning
TIMEOUT_WARNING = "Timeout warning, exceeded time limit"

# Creating Agent tuple, one class, two fields
Agent = namedtuple("Agent", ["player", "name"])

# This function play two matches, one with player 1 moving first
# and the oher with player 2 moving first
def play_match(player1, player2):
    # Counters
    num_wins = {player1: 0, player2: 0}
    num_timeouts = {player1: 0, player2: 0}
    num_invalid_moves = {player1: 0, player2: 0}

    # Defining games
    games = [Board(player1, player2), Board(player2, player1)]

    # initialize both games with a random move and response
    for _ in range(2):
        move = random.choice(games[0].get_legal_moves())
        games[0].apply_move(move)
        games[1].apply_move(move)

    # Playing both games
    for game in games:
        winner, _, termination = game.play(time_limit=TIME_LIMIT)

        # Increasing counter if win
        if player1 == winner:
            num_wins[player1] += 1

            if termination == "timeout":
                # Timeout exception
                num_timeouts[player2] += 1
            else:
                # Invalid move
                num_invalid_moves[player2] += 1

        # Increasing counter if win
        elif player2 == winner:
            num_wins[player2] += 1

            if termination == "timeout":
                # Timeout exception
                num_timeouts[player1] += 1
            else:
                # Invalid move
                num_invalid_moves[player1] += 1

    # Throwing a warning if exists
    if sum(num_timeouts.values()) != 0:
        warnings.warn(TIMEOUT_WARNING)

    # Returning number of wins for each player
    return num_wins[player1], num_wins[player2]


# This functions plays all the matches for one round
def play_round(agents, num_matches):
    # Taking the first agent
    agent_1 = agents[-1]

    # Counters
    wins = 0.
    total = 0.

    print("\nPlaying Matches:")
    print("---------------------------------------------------")

    for idx, agent_2 in enumerate(agents[:-1]):

        # Initialising counters
        counts = {agent_1.player: 0., agent_2.player: 0.}
        # Getting names of the players
        names = [agent_1.name, agent_2.name]
        # Comment this line if running from console
        print("  Match {}: {!s:^11} vs {!s:^11}".format(idx + 1, *names), end=' ')

        # Playing 20 matches
        for p1, p2 in itertools.permutations((agent_1.player, agent_2.player)):
            for _ in range(num_matches):
                score_1, score_2 = play_match(p1, p2)
                counts[p1] += score_1
                counts[p2] += score_2
                total += score_1 + score_2

        wins += counts[agent_1.player]

        print("\tResult: {} to {}".format(int(counts[agent_1.player]),
                                          int(counts[agent_2.player])))
    # Returning the overall percentage
    return 100. * wins / total


def main():

    # Array of the opponents heuristics
    HEURISTICS = [("Null", null_score),
                  ("Open", open_move_score),
                  ("Improved", improved_score)]

    # Alpha-beta pruning opponent settings
    AB_ARGS = {"search_depth": 5, "method": 'alphabeta', "iterative": False}
    # Minimax opponent settings
    MM_ARGS = {"search_depth": 3, "method": 'minimax', "iterative": False}
    # Player and Archenemy settings
    CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

    # Creating the minimax opponents
    mm_agents = [Agent(CustomPlayer(score_fn=h, **MM_ARGS),
                       "MM_" + name) for name, h in HEURISTICS]
    # Creating alpha-beta pruning opponents
    ab_agents = [Agent(CustomPlayer(score_fn=h, **AB_ARGS),
                       "AB_" + name) for name, h in HEURISTICS]
    # Creating the random opponents
    random_agents = [Agent(RandomPlayer(), "Random")]


    # Creating the Player and its Archenemy
    test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), "Archenemy"),
                   Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "Player")]

    # Printing results status in console
    for agentUT in test_agents:
        print("")
        print("***************************************************")
        print("{:^25}".format("Evaluating: " + agentUT.name))
        print("***************************************************")

        agents = random_agents + mm_agents + ab_agents + [agentUT]
        win_ratio = play_round(agents, NUM_MATCHES)

        print("\n\nResults:")
        print("---------------------------------------------------")
        print("{!s:<15}{:>10.2f}%".format(agentUT.name, win_ratio))


if __name__ == "__main__":
    main()