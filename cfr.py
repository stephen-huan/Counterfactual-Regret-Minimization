# main file for the counterfactual regret minimization algorithm
import random
import cache, play
"""
0 - first player
1 - second player
a particular action is represented as an int

backend methods specific to a certain game
should implement:
    - ACTIONS: the number of actions
    - actions: the string interpretation of a given action
    -    util: the util for each possible action given an opponent action
"""
import games.kuhn as game

ACTIONS = game.ACTIONS
# set the cache name to the name of the game
cache.NAME = game.__name__.split(".")[-1]

def get_action(strategy: list) -> int:
    """ Gets a random action according to the mixed-strategy distribution. """
    r = random.random()
    a = cum_prob = 0
    while a < ACTIONS - 1:
        cum_prob += strategy[a]
        if r < cum_prob:
            break
        a += 1
    return a

# expose methods
play.game, play.get_action, play.ACTIONS = game, get_action, ACTIONS

class Regret:

    """ Represents a strategy. """

    def __init__(self):
        self.regret_sum = [0]*ACTIONS
        self.strategy = [0]*ACTIONS
        self.strategy_sum =[0]*ACTIONS

    def get_strategy(self, realization_weight: float=1) -> list:
        """ Gets the current mixed strategy through regret-matching.
            TODO: try softmax instead of positive scaling """
        self.strategy = [self.regret_sum[a] if self.regret_sum[a] > 0 else 0 for a in range(ACTIONS)]
        norm_sum = sum(self.strategy)

        for a in range(ACTIONS):
            if norm_sum > 0:
                self.strategy[a] /= norm_sum
            else:
                # uniform strategy
                self.strategy[a] = 1/ACTIONS
            self.strategy_sum[a] += realization_weight*self.strategy[a]

        return self.strategy

    def get_average_strategy(self) -> list:
        """ Gets the average mixed strategy across all training iterations. """
        norm_sum = sum(self.strategy_sum)
        return [self.strategy_sum[a]/norm_sum if norm_sum > 0 else 1/ACTIONS for a in range(ACTIONS)]

    def regret(self, action_util: list, my_util: float, realization_weight: float=1) -> None:
        """ Updates regret based upon an utility list and the current utility. """
        # Accumulate action regrets
        for a in range(ACTIONS):
            regret = action_util[a] - my_util
            self.regret_sum[a] += realization_weight*regret

    # @cache.graph
    # @cache.cache(overwrite=False)
    def train(self, iters: int, graph: bool=False) -> list:
        """ Trains the CFR minimization again a known opponenet strategy. """
        if graph: rtn = []

        for i in range(iters):
            # Compute action utilities
            action_util = game.util(get_action(game.OPP_STRATEGY))
            # Get regret-matched mixed-strategy actions
            self.regret(action_util, action_util[get_action(self.get_strategy())])

            if graph: rtn.append(self.get_average_strategy())

        return rtn if graph else self.get_average_strategy()

# @cache.graph
# @cache.cache(overwrite=True)
def train_normal(iters: int, graph: bool=False) -> list:
    """ Calculates the Nash equilibrium for a normal form game. """
    if graph: rtn = []
    p1, p2 = Regret(), Regret()

    for i in range(iters):
        a1 = get_action(p1.get_strategy())
        a2 = get_action(p2.get_strategy())

        p1.regret(game.util(a2), game.util(a2)[a1])
        p2.regret(game.util(a1), game.util(a1)[a2])

        if graph: rtn.append(p1.get_average_strategy())

    return rtn if graph else (p1.get_average_strategy(), p2.get_average_strategy())

### non-normal-form games

nodes = {}

class Node(Regret):

    """ Information set node class definition. """

    def __init__(self, info_set: list) -> None:
        super().__init__()
        self.info_set = info_set

    def __str__(self) -> str:
        """ Gets the information set string representation. """
        return f"{self.info_set: <3}: {list(map(lambda x: round(x, 3), self.get_average_strategy()))}"

def cfr(n, info: list, history: list=[], p0: float=1, p1: float=1) -> float:
    """ Counterfactual regret minimzation iteration. """
    player = len(history) % 2

    # Return payoff for terminal states
    util = game.util(info, history)
    if util is not None:
        return util

    info_set = [str(info[player])] + history

    # Get information set node or create it if nonexistant
    repr = " ".join(info_set)
    if repr not in nodes:
        nodes[repr] = Node(repr)
    node = nodes[repr]

    # For each action, recursively call cfr with additional history and probability
    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = [0]*ACTIONS
    node_util = 0

    for a in range(ACTIONS):
        next_history = history + [game.actions[a]]
        # negative because next call's value is from the opponent's perspective
        util[a] = -(cfr(info, next_history, p0*strategy[a], p1) if player == 0 else \
                    cfr(info, next_history, p0, p1*strategy[a]))
        node_util += strategy[a]*util[a]

    # For each action, compute and accumulate counterfactual regret
    node.regret(util, node_util, p1 if player == 0 else p0)

    return node_util

@cache.cache(overwrite=False)
def train(iterations: int) -> float:
    """ Calculates the Nash equilibrium. """
    cards = list(range(1, 4))
    util = 0
    for i in range(iterations):
        random.shuffle(cards)
        util += cfr(cards)

    return nodes, util

def play_kuhn(first: int=random.randint(0, 1)) -> float:
    """ Has a human play againt the computer. """
    cards = list(range(1, 4))
    random.shuffle(cards)

    print(f"Your card is {cards[first]}")
    p = [lambda i: play.get_move(), lambda i: get_action(nodes[i].get_average_strategy())]
    if first == 1:
        p = p[::-1]

    history = ""
    turn = 0
    while game.util(cards, history) is None:
        move = game.actions[p[turn](str(cards[turn]) + history)]
        if turn != first:
            print(f"Computer plays {move}")
        history += " " + move 
        turn ^= 1

    print(f"Computer had card {cards[first ^ 1]}")
    return (1 if len(history) % 2 == first else -1)*game.util(cards, history)

if __name__ == "__main__":
    # random.seed(7)
    iters = 10**6

    ### normal form games
    # # train against fixed opponent
    # r = Regret()
    # print(r.train(iters))
    #
    # # find Nash equilibrium
    # strategy, _ = train_normal(iters)
    # print(strategy, _)
    # print(play.format_strategy(strategy))
    #
    # play.game_session(strategy)

    nodes, util = train(iters)

    print(f"Average game value: {util/iters:.3f}")
    for n in sorted(map(str, nodes.values())):
        print(n)

    play_kuhn()
