# total number of dice in play
D = 2

def s(n: int, r: int) -> int:
    """ Computes the strength of a claim n x r. """
    if r != 1:
        return 5*n + r + (n >> 1) - 7
    return 11*n - 6 if n <= (D >> 1) else 5*D + n - 1

def get_player(history: list) -> int:
    """ Returns the current player given history. """
    return sum(history) % 2

def last(history: list):
    """ Returns the index of the last move, excluding dudo.
    the last play must be the highest index true value in the claim history. """
    l = [i for i in range(ACTIONS - 1) if history[i]]
    return l[-1] if len(l) > 0 else -1

def util(rolls: list, history: list) -> float:
    """ Returns the util of a terminal state. """
    # the player being doubted, NOT the player calling doubt
    player = get_player(history)

    # dudo has been called
    if history[DUDO]:
        l = last(history)
        n, r = num(l), rank(l)
        t = rolls.count(1) + (rolls.count(r) if r != 1 else 0)
        # guessed exactly right
        if n == t:
            return 1
            return 1 if player == 0 else -1
        return -abs(n - t)
        return (1 if player == 0 else -1)*(-abs(n - t))

def num(action: int) -> int:
    """ Returns the num of an action. """
    return action//6 + 1

def rank(action: int) -> int:
    """ Returns the rank of an action. """
    return ((action + 1) % 6) + 1

def format_history(history: list) -> str:
    """ Takes the history of claims and turns it into a string. """
    return "".join([f"{num(c)}x{rank(c)} " if history[c] else "" for c in range(ACTIONS - 1)]).strip()

def hash_info_set(roll: int, history: list) -> int:
    """ Converts an information set to an unique integer for hashing. """
    n = roll
    for a in range(ACTIONS - 2, -1, -1):
        n <<= 1
        n += history[a]
    return n

claims = sorted([(n, r) for n in range(1, D + 1) for r in range(1, 7)], key=lambda c: s(*c))
actions = ["x".join(map(str, claim)) for claim in claims] + ["dudo"]
ACTIONS = len(actions)
DUDO = ACTIONS - 1

for i, claim in enumerate(claims):
    assert claim == (num(i), rank(i))
