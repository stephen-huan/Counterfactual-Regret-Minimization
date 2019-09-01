PASS, BET, ACTIONS = range(3)
actions = ["p", "b"]

def util(cards: list, history: str) -> float:
    """ Returns the utility of a terminal state, None if state is not terminal. """
    plays = len(history)
    player = plays % 2
    opp = player ^ 1

    if plays > 1:
        terminal_pass = history[-1] == "p"
        double_bet = history[-2:] == "bb"
        player_card_higher = cards[player] > cards[opp]
        if terminal_pass:
            return (1 if player_card_higher else -1) if history == "pp" else 1
        elif double_bet:
            return 2 if player_card_higher else -2
