import tkinter as tk
from chain_reaction import ChainReactionGame
import dummy_bot as bot0
import random_bot as bot1
import time

CELL_SIZE = 60
COLORS = {0: "#ff4d4d", 1: "#4d4dff"} # Red and Blue

class ChainReactionGUI:
    def __init__(self, master, rows=12, cols=8):
        self.master = master
        self.master.title("Chain Reaction - Bot Arena")
        self.master.configure(bg="#1a1a1a")
        
        self.game = ChainReactionGame(rows, cols)
        
        self.canvas = tk.Canvas(master, width=cols*CELL_SIZE, height=rows*CELL_SIZE, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(padx=20, pady=20)
        
        self.status_label = tk.Label(master, text="Player 0's Turn (Red)", fg=COLORS[0], bg="#1a1a1a", font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=(0, 20))
        
        self.turn = 0
        
        # Change a player to `None` if you want to click and play manually!
        # Example: self.bots = {0: None, 1: bot1.get_move} 
        self.bots = {0: bot0.get_move, 1: bot1.get_move} 
        
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.draw_board()
        self.master.after(500, self.play_next_turn) # Start game loop

    def draw_board(self):
        self.canvas.delete("all")
        state = self.game.get_state()
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                
                # Draw grid cell
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#333333", width=2)
                
                owner, count = state[r][c]
                if count > 0:
                    self.draw_orbs(x0, y0, count, COLORS[owner])

    def draw_orbs(self, x, y, count, color):
        cx, cy = x + CELL_SIZE/2, y + CELL_SIZE/2
        r_orb = 8
        positions = []
        if count == 1:
            positions = [(0, 0)]
        elif count == 2:
            positions = [(-12, 0), (12, 0)]
        elif count == 3:
            positions = [(0, -12), (-10, 10), (10, 10)]
        else:
            positions = [(-10, -10), (10, -10), (-10, 10), (10, 10)]
            
        for dx, dy in positions[:count]:
            self.canvas.create_oval(
                cx+dx-r_orb, cy+dy-r_orb, 
                cx+dx+r_orb, cy+dy+r_orb, 
                fill=color, outline=color
            )

    def on_click(self, event):
        player = self.turn % 2
        # Ignore manual clicks if it is a bot's turn
        if self.bots[player] is not None:
            return 
            
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        
        self.apply_move(player, (r, c))
        
    def apply_move(self, player, move, from_bot=False):
        try:
            self.game.apply_move(player, move)
            self.turn += 1
            self.draw_board()
            
            winner = self.game.check_winner()
            if winner is not None:
                winner_color = "Red" if winner == 0 else "Blue"
                self.status_label.config(text=f"Game Over! {winner_color} Wins!", fg=COLORS[winner])
            else:
                next_player = self.turn % 2
                next_color_name = "Red" if next_player == 0 else "Blue"
                self.status_label.config(text=f"Player {next_player}'s Turn ({next_color_name})", fg=COLORS[next_player])
                
                # If next player is a bot, schedule its move automatically
                if self.bots[next_player] is not None:
                    # Adjust bot v bot speed - 500 = 0.5 seconds between bot moves 
                    self.master.after(500, self.play_next_turn)
                    
        except Exception as e:
            if from_bot:
                raise e
            else:
                print("Invalid move or error:", e)

    def play_next_turn(self):
        player = self.turn % 2
        winner = self.game.check_winner()
        if winner is not None:
            return
            
        bot_func = self.bots[player]
        if bot_func is not None:
            try:
                start_time = time.time()
                move = bot_func(self.game.get_state(), player)
                if time.time() - start_time > 1.0:
                    raise Exception("Bot took too long to make a move (exceeded 1000ms)")
                self.apply_move(player, move, from_bot=True)
            except Exception as e:
                # Bot crashed or made an illegal move
                winner_color = "Blue" if player == 0 else "Red"
                self.status_label.config(text=f"Bot {player} crashed! {winner_color} Wins!", fg=COLORS[1 - player])
                print(f"Error from Bot {player}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChainReactionGUI(root)
    root.mainloop()
