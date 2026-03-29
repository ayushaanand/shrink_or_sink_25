from chain_reaction import ChainReactionGame

# Pick which bots to run by changing these imports
import claude_bot as bot1
import ayush_bot as bot0
import time

def print_board(state):
    for row in state:
        parts = []
        for owner, count in row:
            if owner is None or count == 0:
                parts.append("__")
            else:
                parts.append(f"{owner}{count}")
        print(" ".join(parts))


def main():
    rows, cols = 12, 8 
    max_turns = 1000

    game = ChainReactionGame(rows=rows, cols=cols)
    bots = {0: bot0.get_move, 1: bot1.get_move}

    for turn in range(max_turns):
        player = turn % 2
        
        try:
            start_time = time.time()
            move = bots[player](game.get_state(), player)
            if time.time() - start_time > 1.0:
                raise Exception("Bot took too long to make a move (exceeded 1000ms)")
            print(f"Turn {turn + 1}: Player {player} plays {move}")
            
            game.apply_move(player, move)
            print_board(game.get_state())
            
        except Exception as e:
            winner = 1 - player
            print(f"Player {player} forfeits (crashed or made invalid move): {e}")
            print(f"Winner: Player {winner}")
            return

        winner = game.check_winner()
        if winner is not None:
            print("Final board:")
            print_board(game.get_state())
            print(f"Winner: Player {winner}")
            return

    print("Reached max_turns without a winner.")
    print("Final board:")
    state = game.get_state()
    print_board(state)
    
    # Tiebreaker: Player with the most cells owned wins
    counts = {0: 0, 1: 0}
    for row in state:
        for owner, orb_count in row:
            if owner in (0, 1) and orb_count > 0:
                counts[owner] += 1
                
    print(f"Cell counts - Player 0: {counts[0]}, Player 1: {counts[1]}")
    if counts[0] > counts[1]:
        print("Winner: Player 0 (Tie-breaker by most cells)")
    elif counts[1] > counts[0]:
        print("Winner: Player 1 (Tie-breaker by most cells)")
    else:
        print("Result: Absolute Tie!")

if __name__ == "__main__":
    main()
