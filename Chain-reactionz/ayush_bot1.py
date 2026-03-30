import math
import time
import random
import operator
import gc

COLS = 8
ROWS = 12
N = ROWS * COLS

# PRECOMPUTE BOARD GEOMETRY AND SYMMETRY MAPPINGS
ADJ = []
CRITICAL = []
FLIP_H = [0] * N
FLIP_V = [0] * N
ROT_180 = [0] * N

for r in range(ROWS):
    for c in range(COLS):
        i = r * COLS + c
        
        # 1. Adjacencies
        neighbors = []
        if r > 0: neighbors.append((r - 1) * COLS + c)
        if r < ROWS - 1: neighbors.append((r + 1) * COLS + c)
        if c > 0: neighbors.append(r * COLS + c - 1)
        if c < COLS - 1: neighbors.append(r * COLS + c + 1)
        ADJ.append(neighbors)
        CRITICAL.append(len(neighbors))
        
        # 2. Symmetries (12x8)
        # Horizontal Flip: Same row, inverted column (7 - c)
        FLIP_H[i] = r * COLS + (COLS - 1 - c)
        
        # Vertical Flip: Inverted row (11 - r), same column
        FLIP_V[i] = (ROWS - 1 - r) * COLS + c
        
        # 180 Rotation: Invert row and column
        ROT_180[i] = (ROWS - 1 - r) * COLS + (COLS - 1 - c)

# ── Positional heatmap ──────────────────────────────────────────────────────
# POS_VAL[i] = 1 / (1 + dist_to_nearest_row_edge + dist_to_nearest_col_edge)
# Peaks at 1.0 on corners, decays smoothly toward 0.11 at board centre.
# Precomputed once — O(1) lookup during hot MCTS path.
POS_VAL = [0.0] * N
for _r in range(ROWS):
    for _c in range(COLS):
        _dr = min(_r, ROWS - 1 - _r)   # 0 at top/bottom rows, grows inward
        _dc = min(_c, COLS - 1 - _c)   # 0 at left/right cols, grows inward
        POS_VAL[_r * COLS + _c] = 1.0 / (1.0 + _dr + _dc)

IDX_H = [FLIP_H[i] for i in range(N)] + [FLIP_H[i] + N for i in range(N)]
IDX_V = [FLIP_V[i] for i in range(N)] + [FLIP_V[i] + N for i in range(N)]
IDX_180 = [ROT_180[i] for i in range(N)] + [ROT_180[i] + N for i in range(N)]

GET_H = operator.itemgetter(*IDX_H)
GET_V = operator.itemgetter(*IDX_V)
GET_180 = operator.itemgetter(*IDX_180)

def get_canonical_hash(own, orb):
    """
    Computes all 4 Dihedral Geometric states.
    Returns the statistically 'minimum' hash among the 4 so all 4 realities map to exactly 1 memory ID natively.
    """
    state_tuple = tuple(own) + tuple(orb)
    return min(
        hash(state_tuple),
        hash(GET_H(state_tuple)),
        hash(GET_V(state_tuple)),
        hash(GET_180(state_tuple))
    )


class MCTSNode:
    __slots__ = ('move', 'parent', 'owner', 'orbs', 'player_id', 'stats', 'untried_moves', 'children', 'c0', 'c1')
    
    def __init__(self, move, parent, owner, orbs, player_id, global_TT, c0, c1):
        self.move = move
        self.parent = parent
        self.owner = owner
        self.orbs = orbs
        self.player_id = player_id
        
        self.children = []
        
        # Hashing logic to merge identical mirrored topologies
        canonical_state = get_canonical_hash(owner, orbs)
        if canonical_state not in global_TT:
            global_TT[canonical_state] = [0.0, 0] # [Wins, Visits] Shared Memory Core Pointer
        self.stats = global_TT[canonical_state]
        
        # Store cell counts directly — eliminates O(N) scan at every expansion
        self.c0 = c0
        self.c1 = c1
        
        # ── Candidate Move Pruning ──────────────────────────────────────────────
        # Score every valid move by strategic relevance.
        # Capping at top-15 collapses branching factor from ~63 → 15, unlocking
        # 3-4 ply search with the same iteration budget vs 1 ply before.
        opp_id = 1 - player_id
        all_moves = []
        for i in range(N):
            if owner[i] != player_id and owner[i] != -1:
                continue  # opponent's cell, skip
            adj_opp = sum(1 for a in ADJ[i] if owner[a] == opp_id)
            adj_all = len(ADJ[i])
            # Exclude completely trapped cells: exploding hands orbs to all opponents
            if adj_opp == adj_all:
                continue
            # ── Strategic Score: Territory > Mass in opening
            # 1. POS_VAL: Corner/Edge heatmap (1.0 -> 0.11)
            # 2. (owner == -1): Massive boost (+1.5) to spread and take new territory
            # 3. Near-criticality: (+orbs/crit) for immediate tactical pressure
            # 4. Encirclement: (+0.8 * adj_opp) to respond to enemy build-ups
            is_empty = (owner[i] == -1)
            score = POS_VAL[i] + (1.5 if is_empty else 0.0) + (orbs[i] / CRITICAL[i]) + 0.8 * adj_opp
            all_moves.append((score, i))
        
        if not all_moves:
            # Last resort: allow any own/empty move to avoid getting stuck
            all_moves = [(0.0, i) for i in range(N) if owner[i] == player_id or owner[i] == -1]
        
        all_moves.sort(reverse=True)
        # Keep top-15; Fisher-Yates below handles random sampling order
        self.untried_moves = [i for _, i in all_moves[:15]]

def simulate_cascade(own, orb, move_i, player_id, c0, c1):
    old_owner = own[move_i]
    if old_owner != player_id:
        if old_owner == 0: c0 -= 1
        elif old_owner == 1: c1 -= 1
        if player_id == 0: c0 += 1
        elif player_id == 1: c1 += 1
        
    own[move_i] = player_id
    orb[move_i] += 1
    
    if orb[move_i] < CRITICAL[move_i]:
        return None, c0, c1
        
    queue = [move_i]
    q_head = 0
    
    while q_head < len(queue):
        curr = queue[q_head]
        q_head += 1
        
        c_orb = orb[curr]
        c_crit = CRITICAL[curr]
        
        if c_orb >= c_crit:
            c_owner = own[curr]
            orb[curr] = c_orb - c_crit
            if orb[curr] == 0:
                own[curr] = -1
                if c_owner == 0: c0 -= 1
                elif c_owner == 1: c1 -= 1
                
            for adj in ADJ[curr]:
                adj_owner = own[adj]
                if adj_owner != c_owner:
                    if adj_owner == 0: c0 -= 1
                    elif adj_owner == 1: c1 -= 1
                    if c_owner == 0: c0 += 1
                    elif c_owner == 1: c1 += 1
                    own[adj] = c_owner
                    
                orb[adj] += 1
                if orb[adj] == CRITICAL[adj]: 
                    queue.append(adj)
                    
            if orb[curr] >= c_crit:
                queue.append(curr)
                
        if c0 == 0 and c1 > 1: return 1, c0, c1
        if c1 == 0 and c0 > 1: return 0, c0, c1
        
    if c0 == 0 and c1 > 1: return 1, c0, c1
    if c1 == 0 and c0 > 1: return 0, c0, c1
    return None, c0, c1

def get_move(state, player_id):
    start_time = time.time()
    gc.disable()  # Prevent cyclic GC pauses from firing mid-MCTS
    try:
        return _mcts(state, player_id, start_time)
    finally:
        gc.enable()
        gc.collect()  # Clean up MCTSNode cycles after safely returning

def _mcts(state, player_id, start_time):
    
    own = [-1] * N
    orb = [0] * N
    
    total_raw_mass = 0
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            owner, orbs = state[r][c]
            if owner is not None:
                own[i] = owner
                orb[i] = orbs
                total_raw_mass += orbs
                
    # Opening: grab any available corner immediately before running MCTS.
    # Corners need only 2 orbs to explode — the highest-leverage opening plays.
    CORNERS = [0, COLS-1, (ROWS-1)*COLS, (ROWS-1)*COLS + COLS-1]
    # Early Game Book: Claim all corners and edges before starting expensive MCTS.
    # We now strictly check for UNOWNED cells to prevent clustering in Turn 1 corner.
    if total_raw_mass <= 12:
        # First priority: Empty corners
        for c_idx in CORNERS:
            if own[c_idx] == -1: return (c_idx // COLS, c_idx % COLS)
        # Second priority: Empty edges (adjacent to corners)
        EDGES = [1, COLS-2, (ROWS-2)*COLS, (ROWS-1)*COLS+1, (ROWS-1)*COLS+COLS-2]
        for e_idx in EDGES:
            if e_idx < N and own[e_idx] == -1: return (e_idx // COLS, e_idx % COLS)
                
    global_TT = {}
    
    # Compute c0/c1 once at root — all child nodes inherit from simulate_cascade return
    root_c0, root_c1 = 0, 0
    for o in own:
        if o == 0: root_c0 += 1
        elif o == 1: root_c1 += 1
        
    root = MCTSNode(None, None, own, orb, player_id, global_TT, root_c0, root_c1)
    
    # No sigmoid — it promoted clustering in early game (wrong direction).

    while time.time() - start_time < 0.800:
        node = root
        
        # 1. SELECTION 
        while not node.untried_moves and node.children:
            best_score = float('-inf')
            best_candidates = []
            for c in node.children:
                # Direct pointer read `c.stats[0]` and `c.stats[1]`
                c_wins, c_visits = c.stats
                if c_visits == 0:
                    score = 10000.0 # Mathematically forces exploration over any tested node
                else:
                    n_visits = node.stats[1] if node.stats[1] > 0 else 1
                    score = (c_wins / c_visits) + 1.414 * math.sqrt(math.log(n_visits) / c_visits)
                    
                if score > best_score + 1e-6:
                    best_score = score
                    best_candidates = [c]
                elif abs(score - best_score) <= 1e-6:
                    best_candidates.append(c)
                    
            node = random.choice(best_candidates)
            
        # 2. EXPANSION 
        if node.untried_moves:
            idx = random.randrange(len(node.untried_moves))
            m = node.untried_moves[idx]
            # O(1) swap-and-pop lazy Fisher-Yates
            node.untried_moves[idx] = node.untried_moves[-1]
            node.untried_moves.pop()
            
            new_own = list(node.owner)
            new_orb = list(node.orbs)
            
            # Pass parent's counts directly — no O(N) scan needed
            winner, new_c0, new_c1 = simulate_cascade(new_own, new_orb, m, node.player_id, node.c0, node.c1)
            node = MCTSNode(m, node, new_own, new_orb, 1 - node.player_id, global_TT, new_c0, new_c1)
            node.parent.children.append(node)
        else:
            winner = None
            
        # 3. ROLLOUT 
        sim_own = list(node.owner)
        sim_orb = list(node.orbs)
        sim_player = node.player_id
        
        # Inherit counts directly from node — no O(N) scan needed
        sim_c0, sim_c1 = node.c0, node.c1
            
        if winner is None:
            for _ in range(8):
                if time.time() - start_time > 0.800: break
                
                # Rollout move: prefer near-critical cells adj to opponents.
                # Filter out completely trapped cells (all neighbours are opponents)
                # — exploding those hands free orbs to the enemy.
                max_score = -1.0
                hot_moves = []
                sim_opp = 1 - sim_player
                for i in range(N):
                    if sim_own[i] == sim_opp:
                        continue
                    adj_opp = sum(1 for a in ADJ[i] if sim_own[a] == sim_opp)
                    # Skip completely trapped cells
                    if adj_opp == len(ADJ[i]):
                        continue
                    score = sim_orb[i] + 0.5 * adj_opp
                    if score > max_score + 1e-9:
                        max_score = score
                        hot_moves = [i]
                    elif abs(score - max_score) < 1e-9:
                        hot_moves.append(i)
                            
                if not hot_moves:
                    winner = 1 - sim_player
                    break
                    
                rm = random.choice(hot_moves)
                winner, sim_c0, sim_c1 = simulate_cascade(sim_own, sim_orb, rm, sim_player, sim_c0, sim_c1)
                if winner is not None:
                    break
                sim_player = 1 - sim_player
                
        # 4. EVALUATION & BACKPROPAGATION
        if winner is None:
            my_mass, opp_mass = 0, 0
            my_cells, opp_cells = 0, 0
            threat = 0.0
            opportunity = 0.0
            
            for i in range(N):
                o = sim_own[i]
                if o == -1:
                    continue
                orbs_i = sim_orb[i]
                danger = orbs_i / CRITICAL[i]  # 0.0=empty to 1.0=at critical
                if o == player_id:
                    my_mass += orbs_i
                    my_cells += 1
                    # How much threat does my near-critical cell have vs opponents?
                    for adj in ADJ[i]:
                        if sim_own[adj] == 1 - player_id:
                            opportunity += danger
                else:
                    opp_mass += orbs_i
                    opp_cells += 1
                    # How much does this opponent near-critical cell threaten my cells?
                    for adj in ADJ[i]:
                        if sim_own[adj] == player_id:
                            threat += danger
                            
            # H_weighted: Weigh existing orbs AND potential territory
            # Added a penalty for being surrounded and a bonus for controlling corners
            my_weighted  = sum((sim_orb[i] + 0.5) * POS_VAL[i] for i in range(N) if sim_own[i] == player_id)
            opp_weighted = sum((sim_orb[i] + 0.5) * POS_VAL[i] for i in range(N) if sim_own[i] == 1 - player_id)
            H_weighted    = 1.0 if my_weighted  > opp_weighted  else (0.5 if my_weighted  == opp_weighted  else 0.0)
            
            H_spread      = 1.0 if my_cells > opp_cells else (0.5 if my_cells == opp_cells else 0.0)
            max_pos       = max(opportunity + threat, 1e-9)
            H_positional  = (opportunity - threat) / max_pos
            H_positional  = (H_positional + 1.0) * 0.5
            
            # Weighted Mass and Positional Sensitivity are key to winning high-level games
            vic_score = 0.50 * H_positional + 0.35 * H_weighted + 0.15 * H_spread
        else:
            vic_score = 1.0 if winner == player_id else 0.0
            
        curr = node
        while curr is not None:
            curr.stats[1] += 1
            if curr.parent is not None:
                if curr.parent.player_id == player_id:
                    curr.stats[0] += vic_score
                else:
                    curr.stats[0] += (1.0 - vic_score)
            curr = curr.parent

    if not root.children:
        vm = [i for i in range(N) if own[i] == player_id or own[i] == -1]
        # Heuristic: pick cell closest to critical with most opponent neighbors
        def _score(i):
            return (orb[i] / CRITICAL[i]) * sum(1 for a in ADJ[i] if own[a] == 1 - player_id)
        best_i = max(vm, key=_score) if vm else 0
        return (best_i // COLS, best_i % COLS)
        
    max_visits = max(c.stats[1] for c in root.children)
    best_candidates = [c for c in root.children if c.stats[1] == max_visits]
    best_child = random.choice(best_candidates)
    
    best_i = best_child.move
    return (best_i // COLS, best_i % COLS)
