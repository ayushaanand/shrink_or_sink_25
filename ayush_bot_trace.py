import math
import time
import random

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

def get_canonical_hash(own, orb):
    """
    Computes all 4 Dihedral Geometric states.
    Returns the statistically 'minimum' hash among the 4 so all 4 realities map to exactly 1 memory ID natively.
    """
    h_orig = hash(tuple(own) + tuple(orb))
    
    # Generate the 3 mirrored variants natively
    own_h = [own[FLIP_H[i]] for i in range(N)]
    orb_h = [orb[FLIP_H[i]] for i in range(N)]
    h_h = hash(tuple(own_h) + tuple(orb_h))
    
    own_v = [own[FLIP_V[i]] for i in range(N)]
    orb_v = [orb[FLIP_V[i]] for i in range(N)]
    h_v = hash(tuple(own_v) + tuple(orb_v))
    
    own_180 = [own[ROT_180[i]] for i in range(N)]
    orb_180 = [orb[ROT_180[i]] for i in range(N)]
    h_180 = hash(tuple(own_180) + tuple(orb_180))
    
    return min(h_orig, h_h, h_v, h_180)

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
        moves = [i for i in range(N) if owner[i] == player_id or owner[i] == -1]
        random.shuffle(moves)
        self.untried_moves = moves

def simulate_cascade(own, orb, move_i, player_id):
    own[move_i] = player_id
    orb[move_i] += 1
    
    if orb[move_i] < CRITICAL[move_i]:
        return None
        
    queue = [move_i]
    q_head = 0
    
    while q_head < len(queue):
        if q_head % 10000 == 0: print('q_head', q_head, 'len', len(queue))
        curr = queue[q_head]
        q_head += 1
        
        c_orb = orb[curr]
        c_crit = CRITICAL[curr]
        
        if c_orb >= c_crit:
            c_owner = own[curr]
            orb[curr] = c_orb - c_crit
            if orb[curr] == 0:
                own[curr] = -1
                
            for adj in ADJ[curr]:
                own[adj] = c_owner
                orb[adj] += 1
                if orb[adj] == CRITICAL[adj]: 
                    queue.append(adj)
                    
    c0 = 0
    c1 = 0
    for o in own:
        if o == 0: c0 += 1
        elif o == 1: c1 += 1
        
    if c0 == 0 and c1 > 1: return 1
    if c1 == 0 and c0 > 1: return 0
    return None

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
    
    while time.time() - start_time < 0.985:
        print('Entering Outer MCTS Loop')
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
            m = node.untried_moves.pop()
            new_own = list(node.owner)
            new_orb = list(node.orbs)
            winner = simulate_cascade(new_own, new_orb, m, node.player_id)
            node = MCTSNode(m, node, new_own, new_orb, 1 - node.player_id, global_TT)
            node.parent.children.append(node)
        else:
            winner = None
            
        # 3. ROLLOUT 
        sim_own = list(node.owner)
        sim_orb = list(node.orbs)
        sim_player = node.player_id
        
        if winner is None:
            for _ in range(8):
                print('Entering Rollout')
                # Micro-timer explicit failsafe (Break mid-rollout without corrupting physics)
                if time.time() - start_time > 0.985: break
                
                vm = [i for i in range(N) if sim_own[i] == sim_player or sim_own[i] == -1]
                if not vm:
                    winner = 1 - sim_player
                    break
                rm = random.choice(vm)
                winner = simulate_cascade(sim_own, sim_orb, rm, sim_player)
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
            
            S = total_mass / 248.0
            
            exponent = -15.0 * (S - 0.55)
            if exponent > 50:
                W = 0.0
            elif exponent < -50:
                W = 1.0
            else:
                W = 1.0 / (1.0 + math.exp(exponent))
                
            vic_score = (1.0 - W) * H_cluster + W * H_spread
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
