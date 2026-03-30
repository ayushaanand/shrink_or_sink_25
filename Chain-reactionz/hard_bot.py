import random

def get_move(state, player_id):
    rows = len(state)
    cols = len(state[0])
    opponent_id = 1 - player_id

    def get_neighbors(r, c):
        res = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                res.append((nr, nc))
        return res

    def get_critical_mass(r, c):
        # In the original Java, this is 'GetAtomFlag(crvector, 1)'
        return len(get_neighbors(r, c))

    def score_move(r, c):
        owner, count = state[r][c]
        crit_mass = get_critical_mass(r, c)
        
        # Base logic from the Java code:
        # i3 = atom count + 1 (the count after we move there)
        new_count = count + 1
        is_exploding = new_count >= crit_mass
        
        # d2 = ((i3 / crit_mass) * 8.0)
        score = (new_count / crit_mass) * 8.0
        
        if is_exploding:
            score += 120.0 # Huge bonus for triggering a chain

        for nr, nc in get_neighbors(r, c):
            n_owner, n_count = state[nr][nc]
            n_crit = get_critical_mass(nr, nc)
            
            # Is neighbor about to explode? (Java's z2 logic)
            n_is_critical = (n_count == n_crit - 1)

            if n_owner is None:
                score += 1.0 if is_exploding else 0.2
            elif n_owner == player_id:
                score += 2.0 if is_exploding else 0.6
                if n_is_critical:
                    score += 0.8
            else: # Opponent owner
                # score += (z ? 10.0 : 1.2) + (z ? n_count * 2.0 : 0.4 * n_count)
                score += (10.0 if is_exploding else 1.2)
                score += (n_count * 2.0 if is_exploding else 0.4 * n_count)
                if n_is_critical:
                    score += 6.0 if is_exploding else 1.0
        
        # Java final adjustment: (6.0 - crit_mass) * 0.8 
        # This prioritizes corners (2 neighbors) over centers (4 neighbors)
        score += (6.0 - crit_mass) * 0.8
        return score

    scored_moves = []
    for r in range(rows):
        for c in range(cols):
            owner, _ = state[r][c]
            if owner is None or owner == player_id:
                # Add a tiny bit of RNG to break ties, just like the OG code
                move_score = score_move(r, c) + (random.random() * 0.01)
                scored_moves.append(((r, c), move_score))

    # Sort by score descending
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    if not scored_moves:
        return None

    # Hard Mode Selection Logic:
    # It picks from the top moves that are within a certain threshold of the best score
    best_score = scored_moves[0][1]
    threshold = max(3.0, 0.08 * best_score)
    
    candidates = [m for m, s in scored_moves if (best_score - s) <= threshold]
    
    return random.choice(candidates)