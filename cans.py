"""
TJHSST homecoming 2019

Organizers give a certain reward to the person who offers the highest amount.
Each person offers a certain amount of money and they lose what they put in each round.
Organizers are trying to max number of cans so they can’t have the reward too high
or it’d be more efficient to buy cans with the reward money.

Reformulate each HUM as a player.

Class council (CC) has x dollars and they partition y into buying cans and x - y into reward.

It makes no sense to put in more than x - y in because then you're guaranteed to lose money
even if you win which is worse than doing nothing.
Thanks to the can quantum continuous variable (money) -> discrete variable (cans);
cans can only be emitted in discrete packets called quanta, or cans.
so we have abstracted the problem by bounding money between 0 and x - y in steps of 1 can.

Action index itself is the action (how many cans to put in, even for CC).
"""
import regret

# total amount of money available to CC
MONEY = 100
# price per can
CAN = 1
ACTIONS = MONEY//CAN + 1
# patch over the sublibrary
regret.ACTIONS = ACTIONS
# number of players excluding CC
PLAYERS = 1

import cfr
cfr.regret.game.actions = list(range(ACTIONS + 1))
cfr.regret.game.util = list()

def util(info: list, history: str) -> float:
    """ Returns the utility of a terminal state, None if state is not terminal. """
    cc = len(history) == 0
    if not cc:
        history[1]
        history[0]

def list_util(opp_actions: int, cc: bool=False) -> list:
    """ Calculates utility (units of money) given all other opponent's actions. """
    action_util = [0]*ACTIONS
    if not cc:
        large = max(opp_actions)
        # the number of cans put in is losing, lose it all
        for i in range(large):
            action_util[i] = -CAN*i
        # on tie split reward between number of ties
        action_util[large] = (MONEY - CAN*opp_actions[0])/opp_actions.count(large) - CAN*large
        # win reward
        for i in range(large + 1, ACTIONS):
            action_util[i] = MONEY - CAN*opp_actions[0] - CAN*i
    else:
        # regret spending any amount of money on the reward
        for i in range(ACTIONS):
            action_util[i] = MONEY - CAN*i
    return action_util

def train(iterations: int) -> list:
    """ Calculates the Nash equilibrium. """
    # CC is first player (zero index)
    players = [regret.Regret() for player in range(PLAYERS + 1)]

    for i in range(iterations):
        actions = [regret.get_action(p.get_strategy()) for p in players]
        for j, p in enumerate(players):
            action_util = list_util(actions, j == 0)
            p.regret(action_util, action_util[j])

    # all other players symmetric, only need one strategy
    return players[0].get_average_strategy(), players[1].get_average_strategy()

if __name__ == "__main__":
    iterations = 10**5
    cc, hum = train(iterations)
    for strat in [cc, hum]:
        for a in range(ACTIONS):
            print(f"{round(strat[a], 3): <5} {a}")
