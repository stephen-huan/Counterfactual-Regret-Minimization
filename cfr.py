import random
import regret

ACTIONS = regret.ACTIONS
nodes = {}

# TODO: graph dist(node.get_average_strategy(), analytical nash equilibrium) via KL divergence (has to be D(nash || avg strat) bc zeroes)

class Node(regret.Regret):
    """ Information set node class definition. """

    def __init__(self, info_set: str):
        super().__init__()
        self.info_set = info_set

    def __str__(self):
        """ Gets the information set string representation. """
        return f"{self.info_set: <3}: {list(map(lambda x: round(x, 3), self.get_average_strategy()))}"

def cfr(cards: list, history: str, p0: float, p1: float) -> float:
    """ Counterfactual regret minimzation iteration. """
    player = len(history) % 2

    # Return payoff for terminal states
    util = regret.game.util(cards, history)
    if util is not None:
        return util

    info_set = str(cards[player]) + history

    # Get information set node or create it if nonexistant
    if info_set not in nodes:
        nodes[info_set] = Node(info_set)
    node = nodes[info_set]

    # For each action, recursively call cfr with additional history and probability
    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = [0]*ACTIONS
    node_util = 0

    for a in range(ACTIONS):
        next_history = history + regret.game.actions[a]
        # negative because next call's value is from the opponent's perspective
        util[a] = -(cfr(cards, next_history, p0*strategy[a], p1) if player == 0 else \
                    cfr(cards, next_history, p0, p1*strategy[a]))
        node_util += strategy[a]*util[a]

    # For each action, compute and accumulate counterfactual regret
    node.regret(util, node_util, p1 if player == 0 else p0)

    return node_util

@regret.store
def train(iterations: int) -> float:
    """ Calculates the Nash equilibrium. """
    cards = list(range(1, 4))
    util = 0
    for i in range(iterations):
        random.shuffle(cards)
        util += cfr(cards, "", 1, 1)

    return nodes, util

def play(first: int=random.randint(0, 1)) -> float:
    """ Has a human play againt the computer. """
    cards = list(range(1, 4))
    random.shuffle(cards)

    print(f"Your card is {cards[first]}")
    p = [lambda i: regret.get_move(), lambda i: regret.get_action(nodes[i].get_average_strategy())]
    if first == 1:
        p = p[::-1]

    history = ""
    turn = 0
    while regret.game.util(cards, history) is None:
        move = regret.game.actions[p[turn](str(cards[turn]) + history)]
        if turn != first:
            print(f"Computer plays {move}")
        history += move
        turn ^= 1

    print(f"Computer had card {cards[first ^ 1]}")
    return (1 if len(history) % 2 == first else -1)*regret.game.util(cards, history)

if __name__ == "__main__":
    random.seed(7)
    iterations = 10**6

    _, util = train(iterations)
    print(f"Average game value: {util/iterations:.3f}")
    for n in sorted(map(str, nodes.values())):
        print(n)

    # print(play(1))
    # regret.game_session(play)
