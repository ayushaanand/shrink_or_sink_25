def get_move(state, player_id):
    rows = len(state)
    cols = len(state[0])
    
    opponent_id = 1 - player_id
    best_move = None
    max_score = -float('inf')

    def get_critical_mass(r, c):
        # Corners: 2, Edges: 3, Middle: 4
        neighbors = 0
        if r > 0: neighbors += 1
        if r < rows - 1: neighbors += 1
        if c > 0: neighbors += 1
        if c < cols - 1: neighbors += 1
        return neighbors

    for r in range(rows):
        for c in range(cols):
            owner, count = state[r][c]
            
            # Check if move is valid
            if owner is None or owner == player_id:
                score = 0
                critical_mass = get_critical_mass(r, c)
                
                # 1. Prioritize lower critical mass (Corners > Edges > Center)
                if critical_mass == 2:
                    score += 30
                elif critical_mass == 3:
                    score += 15
                
                # 2. Prefer cells closer to exploding
                # If it's about to pop, it's a high-pressure move
                score += (count * 5)
                
                # 3. Criticality Check: Is an opponent nearby?
                # If we can pop a cell next to an opponent's cell, that's huge.
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        opp_owner, opp_count = state[nr][nc]
                        if opp_owner == opponent_id:
                            # If we are one away from popping next to them
                            if count == critical_mass - 1:
                                score += 20
                            # Vulnerability: if they are one away from popping next to our move
                            opp_crit = get_critical_mass(nr, nc)
                            if opp_count == opp_crit - 1:
                                score -= 25

                # 4. Tie-breaking with slight randomness or position
                if score > max_score:
                    max_score = score
                    best_move = (r, c)

    return best_move if best_move else (0, 0)