import time
import ayush_bot

state = [[(None, 0) for _ in range(8)] for _ in range(12)]
# Set up some dummy board state with > 2 mass
state[0][0] = (0, 1)
state[1][1] = (1, 1)
state[2][2] = (0, 1)
state[3][3] = (1, 1)

print("Testing ayush_bot.get_move(0)")
t0 = time.time()
move = ayush_bot.get_move(state, 0)
t1 = time.time()
print(f"Chosen move: {move}, took {t1 - t0:.3f}s")
