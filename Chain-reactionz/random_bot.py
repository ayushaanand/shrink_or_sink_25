import random

def get_move(state, player_id):
    """
    Picks a random valid move (useful for testing).
    """
    moves = []
    rows = len(state)
    cols = len(state[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):
            owner, count = state[r][c]
            if owner is None or owner == player_id:
                moves.append((r, c))

    if not moves:
        raise RuntimeError("No valid moves found")
    return random.choice(moves)

