import os
import sys
from ayush_bot import get_move, simulate_cascade, COLS, ROWS

def print_board(own, orb):
    # Print column headers
    print("\n   " + " ".join([str(c) for c in range(COLS)]))
    for r in range(ROWS):
        row_str = f"{r:2} "
        for c in range(COLS):
            i = r * COLS + c
            if own[i] == -1:
                row_str += ". "
            elif own[i] == 0:
                # Red color for Player 0
                row_str += f"\033[91m{orb[i]}\033[0m " 
            elif own[i] == 1:
                # Green color for Player 1
                row_str += f"\033[92m{orb[i]}\033[0m " 
        print(row_str)
    print("\n")

def main():
    print("=" * 50)
    print("CHAIN REACTION - MCTS HUMAN TEST LOOP")
    print("=" * 50)
    print("You can visually copy the BOT'S moves and play them directly")
    print("against another AI or random opponent online, and then type")
    print("their response back into this terminal!\n")
    
    own = [-1] * (ROWS * COLS)
    orb = [0] * (ROWS * COLS)
    
    try:
        player_id = int(input("What Player ID is the Bot playing as? (0=Red, 1=Green) [0/1]: ").strip())
        if player_id not in [0, 1]: raise ValueError
    except:
        print("Invalid input. Defaulting Bot to Player 0.")
        player_id = 0
        
    opp_id = 1 - player_id
    current_turn = 0 # 0 strictly always starts first in Chain Reaction
    
    while True:
        # Re-construct the official competition 2D state matrix to perfectly mimic testing environment
        state_matrix = [[(None, 0) for _ in range(COLS)] for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                i = r * COLS + c
                if own[i] != -1:
                    state_matrix[r][c] = (own[i], orb[i])
                    
        print_board(own, orb)
        
        if current_turn == player_id:
            print(f"--> BOT (Player {player_id}) is thinking... (up to 0.98s)")
            move = get_move(state_matrix, player_id)
            print(f">>> BOT PLAYS: {move} <<<")
            print(">> Go play this move on the opposing GUI!")
            move_i = move[0] * COLS + move[1]
            simulate_cascade(own, orb, move_i, player_id)
        else:
            try:
                print(f"--> OPPONENT (Player {opp_id}) Turn")
                inp = input("What did the opponent do? (Enter 'row col', e.g., '10 5'): ")
                if not inp.strip(): continue
                r_str, c_str = inp.strip().split()
                r, c = int(r_str), int(c_str)
                move_i = r * COLS + c
                
                if own[move_i] == player_id:
                    print("\n[!] INVALID: Opponent cannot play on the bot's tiles! Try again.\n")
                    continue
                simulate_cascade(own, orb, move_i, opp_id)
            except Exception as e:
                print("\n[!] Invalid format. Please enter 'row col' with a space in between.\n")
                continue
                
        # Chain Reaction Rule: Check if a player has been totally eliminated
        c0, c1 = 0, 0
        for o in own:
            if o == 0: c0 += 1
            elif o == 1: c1 += 1
            
        if c0 == 0 and c1 > 1:
            print_board(own, orb)
            print("!!! PLAYER 1 (Green) WINS BY ELIMINATION !!!")
            break
        if c1 == 0 and c0 > 1:
            print_board(own, orb)
            print("!!! PLAYER 0 (Red) WINS BY ELIMINATION !!!")
            break
            
        current_turn = 1 - current_turn

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest Terminated.")
        sys.exit(0)
