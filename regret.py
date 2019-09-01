import random
# backend methods specific to a certain game
import kuhn as game

ACTIONS = game.ACTIONS

# TODO: try softmax instead of positive scaling

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

    def regret(self, my_action: int, opp_action: int) -> None:
        """ Updates regret based upon my action and an opponent move. """
        # Compute action utilities
        action_util = game.util(opp_action)

        # Accumulate action regrets
        for a in range(ACTIONS):
            self.regret_sum[a] += action_util[a] - action_util[my_action]

    def train(self, iterations: int) -> None:
        """ Trains the CFR minimization. """
        for i in range(iterations):
            # Get regret-matched mixed-strategy actions
            self.regret(get_action(self.get_strategy()), get_action(game.opp_strategy))

def train(iterations: int) -> list:
    """ Calculates the Nash equilibrium. """
    p1, p2 = Regret(), Regret()

    for i in range(iterations):
        a1 = get_action(p1.get_strategy())
        a2 = get_action(p2.get_strategy())

        p1.regret(a1, a2)
        p2.regret(a2, a1)

    return p1.get_average_strategy(), p2.get_average_strategy()

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
    iterations = 10**5

    # cfr = Regret()
    # cfr.train(iterations)
    # print(cfr.get_average_strategy())

    strategy, _ = train(iterations)
    for a in range(ACTIONS):
        print(f"{round(strategy[a], 3): <5} {game.actions[a]}")

    game_session()
