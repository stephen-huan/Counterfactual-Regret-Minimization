ROCK, PAPER, SCISSORS, ACTIONS = range(4)
actions = ["rock", "paper", "scissors"]
opp_strategy = [0.4, 0.3, 0.3]

def util(opp_action: int) -> list:
    """ Calculates the util list for an opponent action. """
    action_util = [0]*ACTIONS

    action_util[opp_action] = 0
    action_util[(opp_action + 1) % ACTIONS] = 1
    action_util[(opp_action - 1) % ACTIONS] = -1

    return action_util
