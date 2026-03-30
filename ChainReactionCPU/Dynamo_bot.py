import math
import time
import random
import operator
import gc
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
#  Chain Reaction Optimized Bot — 12 × 8 board
#  Architecture: Iterative-deepening Minimax + α-β + Transposition Table (TT)
#  Optimizations: Zobrist Hashing, Symmetry Mapping, Chebyshev Heatmap
# ──────────────────────────────────────────────────────────────────────────────

COLS = 8
ROWS = 12
N = ROWS * COLS
TIMEOUT = 0.85  # Optimized to squeeze depth while leaving 150ms buffer
MAX_DEPTH = 10   # Deep ceiling; ID will break on timeout anyway

# ── 1. Static Geometry & Symmetry ──────────────────────────────────────────────

ADJ = []
CRITICAL = []
S_FLIP_H = [0] * N
S_FLIP_V = [0] * N
S_ROT_180 = [0] * N

for r in range(ROWS):
    for c in range(COLS):
        i = r * COLS + c
        
        # neighbors
        neighbors = []
        if r > 0: neighbors.append((r - 1) * COLS + c)
        if r < ROWS - 1: neighbors.append((r + 1) * COLS + c)
        if c > 0: neighbors.append(r * COLS + c - 1)
        if c < COLS - 1: neighbors.append(r * COLS + c + 1)
        ADJ.append(tuple(neighbors))
        CRITICAL.append(len(neighbors))
        
        # Dihedral Symmetries for 12x8
        S_FLIP_H[i] = r * COLS + (COLS - 1 - c)
        S_FLIP_V[i] = (ROWS - 1 - r) * COLS + c
        S_ROT_180[i] = (ROWS - 1 - r) * COLS + (COLS - 1 - c)

# ── 2. Positional Heatmap (Chebyshev Distance) ────────────────────────────────

POS_VAL = [0.0] * N
for r in range(ROWS):
    for c in range(COLS):
        # Distance to nearest edge in both directions
        dr = min(r, ROWS - 1 - r)
        dc = min(c, COLS - 1 - c)
        # Smooth decay from 1.0 (corners) down to center
        POS_VAL[r * COLS + c] = 1.0 / (1.0 + dr + dc)

# ── 3. Zobrist Hashing ─────────────────────────────────────────────────────────

# 96 cells * 3 owners (-1, 0, 1) * 8 max_orbs
# Use a static dictionary for fast lookup
ZOBRIST = {}
random.seed(42) # Deterministic for debugging
for i in range(N):
    for owner in [-1, 0, 1]:
        for count in range(9): # Max orbs rarely exceeds 8
            ZOBRIST[(i, owner, count)] = random.getrandbits(63)

def get_hash(owners, orbs):
    h = 0
    for i in range(N):
        h ^= ZOBRIST[(i, owners[i], orbs[i])]
    return h

# Pre-prepare itemgetters for symmetry canonicalization
sym_idx_h = S_FLIP_H + [idx + N for idx in S_FLIP_H]
sym_idx_v = S_FLIP_V + [idx + N for idx in S_FLIP_V]
sym_idx_180 = S_ROT_180 + [idx + N for idx in S_ROT_180]

get_h = operator.itemgetter(*sym_idx_h)
get_v = operator.itemgetter(*sym_idx_v)
get_180 = operator.itemgetter(*sym_idx_180)

def get_canonical_hash(owners, orbs):
    """Returns the minimum hash among the 4 symmetry variations."""
    state = tuple(owners) + tuple(orbs)
    h0 = hash(state)
    h1 = hash(get_h(state))
    h2 = hash(get_v(state))
    h3 = hash(get_180(state))
    return min(h0, h1, h2, h3)

# ── 4. Global Transposition Table ──────────────────────────────────────────────

# Format: {hash: (value, depth, flag, best_move)}
# Flags: 0 = EXACT, 1 = ALPHA (upper bound), 2 = BETA (lower bound)
global_TT = {}

def get_move(state, player_id):
    gc.disable() # Speed up by avoiding GC pauses mid-search
    rows = ROWS
    cols = COLS
    opponent = 1 - player_id
    deadline = time.time() + TIMEOUT

    # Convert state to linear arrays for performance
    owners = []
    orbs = []
    p0_count = 0
    p1_count = 0
    for row in state:
        for cell in row:
            own, cnt = cell
            owners.append(own if own is not None else -1)
            orbs.append(cnt)
            if own == 0: p0_count += 1
            elif own == 1: p1_count += 1

    # ── Move Simulation (Optimized) ───────────────────────────────────────────

    def apply_move(s_owners, s_orbs, move_idx, player, p0, p1):
        own = list(s_owners)
        orb = list(s_orbs)
        
        # Initial placement
        old_owner = own[move_idx]
        orb[move_idx] += 1
        if old_owner != player:
            own[move_idx] = player
            if player == 0: p0 += 1; p1 -= (1 if old_owner == 1 else 0)
            else: p1 += 1; p0 -= (1 if old_owner == 0 else 0)

        if orb[move_idx] < CRITICAL[move_idx]:
            return own, orb, p0, p1

        queue = deque([move_idx])
        while queue:
            # Termination: If any player has 0 pieces, cascade stops
            if (p0 + p1) > 1 and (p0 == 0 or p1 == 0):
                break
                
            curr = queue.popleft()
            if orb[curr] < CRITICAL[curr]:
                continue
            
            # Explode
            crit = CRITICAL[curr]
            orb[curr] -= crit
            if orb[curr] == 0:
                own[curr] = -1
                if player == 0: p0 -= 1
                else: p1 -= 1
                
            for adj in ADJ[curr]:
                adj_owner = own[adj]
                orb[adj] += 1
                if adj_owner != player:
                    own[adj] = player
                    if player == 0: p0 += 1; p1 -= (1 if adj_owner == 1 else 0)
                    else: p1 += 1; p0 -= (1 if adj_owner == 0 else 0)
                
                if orb[adj] >= CRITICAL[adj]:
                    queue.append(adj)
            
            # If still has enough orbs, re-queue
            if orb[curr] >= CRITICAL[curr]:
                queue.append(curr)
                
        return own, orb, p0, p1

    # ── Heuristic Eval ───────────────────────────────────────────────────────

    def evaluate(s_owners, s_orbs, p0, p1):
        if player_id == 0:
            if p1 == 0 and p0 > 0: return 100000
            if p0 == 0 and p1 > 0: return -100000
        else:
            if p0 == 0 and p1 > 0: return 100000
            if p1 == 0 and p0 > 0: return -100000

        score = 0
        for i in range(N):
            own = s_owners[i]
            if own == -1: continue
            
            cnt = s_orbs[i]
            crit = CRITICAL[i]
            
            # Base value: positional leverage + mass
            val = (POS_VAL[i] * 5) + cnt
            
            # Criticality bonus: threatening to explode
            if cnt == crit - 1:
                val += 2.0
                
            # Encirclement penalty (approx)
            adj_opp = sum(1 for a in ADJ[i] if s_owners[a] != -1 and s_owners[a] != own)
            if adj_opp >= 2:
                val -= (1.5 * adj_opp)

            if own == player_id: score += val
            else: score -= val
            
        return score

    # ── Search ───────────────────────────────────────────────────────────────

    timed_out = [False]

    def minimax(s_owners, s_orbs, depth, alpha, beta, maximizing, p0, p1):
        if time.time() > deadline:
            timed_out[0] = True
            return evaluate(s_owners, s_orbs, p0, p1), None

        # Win condition check
        if (p0 + p1) > 1:
            if p1 == 0: return (100000 if player_id == 0 else -100000), None
            if p0 == 0: return (100000 if player_id == 1 else -100000), None

        if depth == 0:
            return evaluate(s_owners, s_orbs, p0, p1), None

        # 1. TT Lookup
        h = get_canonical_hash(s_owners, s_orbs)
        if h in global_TT:
            tt_val, tt_depth, tt_flag, tt_move = global_TT[h]
            if tt_depth >= depth:
                if tt_flag == 0: return tt_val, tt_move
                elif tt_flag == 1: alpha = max(alpha, tt_val)
                elif tt_flag == 2: beta = min(beta, tt_val)
                if alpha >= beta: return tt_val, tt_move

        curr_player = player_id if maximizing else opponent
        
        # 2. Move Ordering / Candidate Generation
        valid_moves = []
        for i in range(N):
            if s_owners[i] == -1 or s_owners[i] == curr_player:
                # Strategic scoring for ordering
                m_score = POS_VAL[i] + (s_orbs[i] / CRITICAL[i])
                valid_moves.append((m_score, i))
        
        valid_moves.sort(reverse=True)
        # Pruning the branching factor to stay stable
        valid_moves = [m[1] for m in valid_moves[:15]]
        
        if not valid_moves:
            return evaluate(s_owners, s_orbs, p0, p1), None

        best_move = valid_moves[0]
        if maximizing:
            best_val = -1000000
            for move in valid_moves:
                n_owners, n_orbs, np0, np1 = apply_move(s_owners, s_orbs, move, curr_player, p0, p1)
                res_val, _ = minimax(n_owners, n_orbs, depth - 1, alpha, beta, False, np0, np1)
                
                if res_val > best_val:
                    best_val = res_val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha: break
        else:
            best_val = 1000000
            for move in valid_moves:
                n_owners, n_orbs, np0, np1 = apply_move(s_owners, s_orbs, move, curr_player, p0, p1)
                res_val, _ = minimax(n_owners, n_orbs, depth - 1, alpha, beta, True, np0, np1)
                
                if res_val < best_val:
                    best_val = res_val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha: break

        # 3. TT Store
        new_flag = 0 # EXACT
        if best_val <= alpha: new_flag = 1 # ALPHA
        elif best_val >= beta: new_flag = 2 # BETA
        global_TT[h] = (best_val, depth, new_flag, best_move)

        return best_val, best_move

    # ── Iterative Deepening ──────────────────────────────────────────────────

    final_move = None
    # Quick Greedy fallback
    all_initial = [(POS_VAL[i] + (orbs[i] / CRITICAL[i]), i) for i in range(N) if owners[i] == -1 or owners[i] == player_id]
    all_initial.sort(reverse=True)
    final_move = all_initial[0][1]

    try:
        for d in range(1, MAX_DEPTH + 1):
            if time.time() > deadline: break
            val, move = minimax(owners, orbs, d, -1000000, 1000000, True, p0_count, p1_count)
            if not timed_out[0] and move is not None:
                final_move = move
    finally:
        gc.enable()

    # Convert linear index back to (r, c)
    return (final_move // cols, final_move % cols)
