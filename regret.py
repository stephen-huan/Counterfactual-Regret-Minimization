import random, functools, pickle, os
try:
    from scipy.stats import entropy
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Scientific libraries not found, some functionality may be broken...")
# backend methods specific to a certain game
import blotto as game

ACTIONS = game.ACTIONS

# TODO: try softmax instead of positive scaling

def get_name(f) -> str:
    """ Makes a filename for a given function. """
    return f"{game.__name__}_{f.__qualname__}.pickle"

def store(f):
    """ Stores the output of a function in a pickle file. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        val = tuple(f(*args, **kwargs))
        if not os.path.exists(get_name(f)):
            with open(get_name(f), "wb") as fi:
                pickle.dump(val[0] if isinstance(val, tuple) else val, fi)
        return val

    return wrapper

def graph(f):
    """ Graphs the output of the function. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        plt.plot(list(map(lambda x: entropy(load(f), x), f(*args, graph=True, **kwargs))))
        plt.title("Loss over time")
        plt.ylabel("KL divergence between Nash equilibrium and strategy")
        plt.xlabel("Time (iterations)")
        plt.show()
        exit()

    return wrapper

def load(f):
    """ Gets the cached file given a function. """
    with open(get_name(f), "rb") as f:
        return pickle.load(f)

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


class Regret:


    def __init__(self):
        self.regret_sum = [0]*ACTIONS
        self.strategy = [0]*ACTIONS
        self.strategy_sum =[0]*ACTIONS

    def get_strategy(self, realization_weight: float=1) -> list:
        """ Gets the current mixed strategy through regret-matching. """
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

    # @graph
    @store
    def train(self, iterations: int, graph: bool=False) -> list:
        """ Trains the CFR minimization. """
        for i in range(iterations):
            # Compute action utilities
            action_util = game.util(get_action(game.opp_strategy))
            # Get regret-matched mixed-strategy actions
            self.regret(action_util, action_util[get_action(self.get_strategy())])

            if graph: yield self.get_average_strategy()

        yield self.get_average_strategy()

@graph
@store
def train(iterations: int, graph: bool=False) -> list:
    """ Calculates the Nash equilibrium. """
    p1, p2 = Regret(), Regret()

    for i in range(iterations):
        a1 = get_action(p1.get_strategy())
        a2 = get_action(p2.get_strategy())

        p1.regret(game.util(a2), game.util(a2)[a1])
        p2.regret(game.util(a1), game.util(a1)[a2])

        if graph: yield p1.get_average_strategy()

    yield p1.get_average_strategy()
    yield p2.get_average_strategy()

def get_move() -> int:
    """ Prompts the user to make a move. """
    while True:
        try:
            move = game.actions.index(input("Your move? "))
        except ValueError:
            print("Not a valid move.")
        else:
            return move

def play(strategy: list) -> float:
    """ Has a human play against the computer. """
    move = get_move()
    opp = get_action(strategy)
    print(f"Computer plays: {game.actions[opp]}")
    return game.util(opp)[move]

def game_session(f=lambda: play(strategy)) -> None:
    """ Plays games over and over again. """
    util = games = 0
    try:
        while True:
            result = f()
            if result == 0:
                print("Tie\n")
            else:
                sign = result//abs(result)
                print(f"{['', 'You', 'Computer'][sign]} win{['', '', 's'][sign]} +{abs(result)}\n")
            games += 1
            util += result
    except KeyboardInterrupt:
        if games > 0:
            print(f"\nAverage util: {util}/{games} = {util/games:.2f}")
        else:
            print()

if __name__ == "__main__":
    random.seed(7)
    iterations = 10**3

    # cfr = Regret()
    # cfr.train(iterations)
    # print(cfr.get_average_strategy())

    strategy, _ = train(iterations)
    for a in range(ACTIONS):
        print(f"{round(strategy[a], 3): <5} {game.actions[a]}")

    # game_session()
