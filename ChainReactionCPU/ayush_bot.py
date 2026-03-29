import math
import time
import random
import operator

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

# Precompute index arrays for true C-tuple itemgetter mappings
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
    __slots__ = ('move', 'parent', 'owner', 'orbs', 'player_id', 'stats', 'untried_moves', 'children')
    
    def __init__(self, move, parent, owner, orbs, player_id, global_TT):
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
        
        # Valid physical moves for expansion
        self.untried_moves = [i for i in range(N) if owner[i] == player_id or owner[i] == -1]

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
                
    if total_raw_mass <= 2:
        corners = [0, COLS-1, (ROWS-1)*COLS, (ROWS-1)*COLS + COLS-1]
        for c_idx in corners:
            if own[c_idx] == -1 or own[c_idx] == player_id:
                return (c_idx // COLS, c_idx % COLS)
                
    global_TT = {}
    root = MCTSNode(None, None, own, orb, player_id, global_TT)
    
    # Pre-compute Sigmoid weight W once from true root board state
    # S = board saturation index (0=empty, 1=full). Drives early(cluster) vs late(spread) pivot.
    _S = total_raw_mass / 248.0
    _exp = -15.0 * (_S - 0.55)
    if _exp > 50:   _W = 0.0
    elif _exp < -50: _W = 1.0
    else:            _W = 1.0 / (1.0 + math.exp(_exp))

    while time.time() - start_time < 0.930:
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
            
            # Recalculate c0, c1 precisely at branch point
            e_c0, e_c1 = 0, 0
            for o in new_own:
                if o == 0: e_c0 += 1
                elif o == 1: e_c1 += 1
                
            winner, _, _ = simulate_cascade(new_own, new_orb, m, node.player_id, e_c0, e_c1)
            node = MCTSNode(m, node, new_own, new_orb, 1 - node.player_id, global_TT)
            node.parent.children.append(node)
        else:
            winner = None
            
        # 3. ROLLOUT 
        sim_own = list(node.owner)
        sim_orb = list(node.orbs)
        sim_player = node.player_id
        
        # Track initial rollout state natively
        sim_c0, sim_c1 = 0, 0
        for o in sim_own:
            if o == 0: sim_c0 += 1
            elif o == 1: sim_c1 += 1
            
        if winner is None:
            for _ in range(8):
                if time.time() - start_time > 0.930: break
                
                # Natively bypass 'vm' filtering logic
                max_orb = -1
                hot_moves = []
                for i in range(N):
                    if sim_own[i] != 1 - sim_player:
                        o = sim_orb[i]
                        if o > max_orb:
                            max_orb = o
                            hot_moves = [i]
                        elif o == max_orb:
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
            total_mass = 0
            
            for i, o in enumerate(sim_own):
                if o != -1:
                    m = sim_orb[i]
                    total_mass += m
                    if o == player_id:
                        my_mass += m
                        my_cells += 1
                    else:
                        opp_mass += m
                        opp_cells += 1
                        
            H_cluster = 1.0 if my_mass > opp_mass else (0.5 if my_mass == opp_mass else 0.0)
            H_spread  = 1.0 if my_cells > opp_cells else (0.5 if my_cells == opp_cells else 0.0)
            
            vic_score = (1.0 - _W) * H_cluster + _W * H_spread
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
        best_i = random.choice(vm) if vm else 0
        return (best_i // COLS, best_i % COLS)
        
    max_visits = max(c.stats[1] for c in root.children)
    best_candidates = [c for c in root.children if c.stats[1] == max_visits]
    best_child = random.choice(best_candidates)
    
    best_i = best_child.move
    return (best_i // COLS, best_i % COLS)
