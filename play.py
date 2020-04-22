# library for user interactions
import random

def format_strategy(strategy: list) -> str:
    """ Formats a strategy. """
    print("\nOptimal strategy:\n" + "-"*10)
    s = []
    for a in range(game.ACTIONS):
        s.append(f"{round(strategy[a], 3): <5} {game.actions[a]}")
    return "\n".join(s)

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

def game_session(strategy: list) -> None:
    """ Plays games over and over again. """
    f = lambda: play(strategy)
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
