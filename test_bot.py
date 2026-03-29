import time
from ayush_bot import get_move

# 1. Simulate the exact dummy logic from the competition
ROWS = 12
COLS = 8

# Create a blank 12x8 board exactly as the hackathon server provides
# Each cell is a tuple: (owner_id, orb_count). Empty = (None, 0)
dummy_state = [[(None, 0) for _ in range(COLS)] for _ in range(ROWS)]

# Let's forcefully populate some cells to simulate a mid-game state
dummy_state[2][3] = (0, 2)  # Player 0 has 2 orbs at (2,3)
dummy_state[5][5] = (1, 3)  # Player 1 has 3 orbs (critical mass edge) at (5,5)
dummy_state[0][0] = (0, 1)  # Player 0 grabbed top-left corner
dummy_state[11][7] = (1, 1) # Player 1 grabbed bottom-right corner

print("Firing up MCTS Bot on simulated 12x8 board...")

# 2. Test Player 0
start_time = time.time()
move = get_move(dummy_state, player_id=0)
execution_time = time.time() - start_time

print("===" * 15)
print(f"Player 0 (You) Chose Move: {move}")
print(f"Execution Clock: {execution_time:.4f} seconds (Must be < 1.0s)")
print("===" * 15)

# If you want to visually verify the Python output format:
assert isinstance(move, tuple) and len(move) == 2, "FAIL: Output is not a (row, col) tuple!"
print("Status: I/O format is PASSED. Ready for upload!")
