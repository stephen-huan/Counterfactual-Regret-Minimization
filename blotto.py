# number of soldiers, number of battlefields
S, N = 5, 3

def possible_actions(curr: str="", cum_sum: int=0, poss: set=set()) -> list:
    """ Recursively calculates the list of possible actions. """
    if len(curr) == N:
        poss.add(curr)
        return

    for i in range(S + 1):
        add = min(i, S - cum_sum)
        if len(curr) == N - 1:
            add = S - cum_sum
        possible_actions(curr + str(add), cum_sum + add, poss)

    return sorted(list(poss))

def compare(action1: str, action2: str) -> int:
    """ Evalutes the number of battlefields won from player 1's perspective. """
    return sum(int(action1[i]) > int(action2[i]) for i in range(N))

def result(action1: str, action2: str) -> int:
    """ Returns the final loss for the game. """
    cap1, cap2 = compare(action1, action2), compare(action2, action1)
    if cap1 == cap2:
        return 0
    return 1 if cap1 > cap2 else -1

def util(opp_action: int) -> list:
    """ Calculates the util list for an opponent action. """
    return [result(a, actions[opp_action]) for a in actions]

actions = possible_actions()
ACTIONS = len(actions)
