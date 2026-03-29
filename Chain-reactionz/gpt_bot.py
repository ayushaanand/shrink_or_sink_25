import copy

def get_neighbors(r, c, rows, cols):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def critical_mass(r, c, rows, cols):
    count = 4
    if r == 0 or r == rows-1:
        count -= 1
    if c == 0 or c == cols-1:
        count -= 1
    return count

def simulate_move(state, move, player_id):
    """
    Simulate full chain reaction after placing one orb
    """
    rows, cols = len(state), len(state[0])
    new_state = copy.deepcopy(state)

    r, c = move
    owner, count = new_state[r][c]

    new_state[r][c] = (player_id, count + 1)

    queue = [(r, c)]

    while queue:
        r, c = queue.pop(0)
        owner, count = new_state[r][c]

        if count >= critical_mass(r, c, rows, cols):
            new_state[r][c] = (None, 0)

            for nr, nc in get_neighbors(r, c, rows, cols):
                n_owner, n_count = new_state[nr][nc]
                new_state[nr][nc] = (player_id, n_count + 1)
                queue.append((nr, nc))

    return new_state

def evaluate(state, player_id):
    """
    Heuristic evaluation:
    + orbs advantage
    + control
    - risky cells
    """
    rows, cols = len(state), len(state[0])
    score = 0

    for r in range(rows):
        for c in range(cols):
            owner, count = state[r][c]
            if owner is None:
                continue

            cm = critical_mass(r, c, rows, cols)

            if owner == player_id:
                score += 5 + count
                if count == cm - 1:
                    score += 10  # ready to explode
            else:
                score -= 5 + count
                if count == cm - 1:
                    score -= 12  # opponent danger

    return score

def get_move(state, player_id):
    rows = len(state)
    cols = len(state[0]) if rows else 0

    best_score = float('-inf')
    best_move = None

    for r in range(rows):
        for c in range(cols):
            owner, _ = state[r][c]

            if owner is None or owner == player_id:
                new_state = simulate_move(state, (r, c), player_id)
                score = evaluate(new_state, player_id)

                if score > best_score:
                    best_score = score
                    best_move = (r, c)

    return best_move