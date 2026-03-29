def get_move(state, player_id):
    """
    Very simple template bot.

    The engine will call:
        move = get_move(state, player_id)
    where:
    - state is a deep-copied board:
      state[row][col] == (owner, orb_count)
    - player_id is your id (0 or 1)

    You must return a move as (row, col).
    Valid moves are cells that are empty (owner is None) or owned by you.
    """
    rows = len(state)
    cols = len(state[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):
            owner, _count = state[r][c]
            if owner is None or owner == player_id:
                return (r, c)
