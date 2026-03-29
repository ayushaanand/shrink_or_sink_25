import time
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
#  Chain Reaction Bot  —  12 × 8 board
#  Iterative-deepening minimax + α-β + candidate pruning + hard timeout
# ──────────────────────────────────────────────────────────────────────────────

TIMEOUT   = 0.8   # seconds per move
MAX_CANDS = 15    # candidate moves considered at each node
MAX_DEPTH = 4     # ceiling for iterative deepening

def get_move(state, player_id):
    rows     = len(state)
    cols     = len(state[0]) if rows else 0
    opponent = 1 - player_id
    deadline = time.time() + TIMEOUT

    # ── geometry (cached) ─────────────────────────────────────────────────────

    _nbrs = {}
    def neighbors(r, c):
        if (r, c) not in _nbrs:
            out = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    out.append((nr, nc))
            _nbrs[(r, c)] = out
        return _nbrs[(r, c)]

    def cm(r, c):
        return len(neighbors(r, c))

    # ── move simulation ────────────────────────────────────────────────────────

    def apply_move(board, r, c, player):
        """Place one orb at (r,c) for `player`, then cascade explosions."""
        b = [list(row) for row in board]   # shallow copy; tuples are replaced not mutated

        o, cnt = b[r][c]
        b[r][c] = (player, cnt + 1)

        queue = deque()
        if b[r][c][1] >= cm(r, c):
            queue.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            _, cnt2 = b[cr][cc]
            cmass   = cm(cr, cc)
            if cnt2 < cmass:
                continue
            b[cr][cc] = (None, 0)
            for nr, nc in neighbors(cr, cc):
                _, nc2 = b[nr][nc]
                b[nr][nc] = (player, nc2 + 1)
                if b[nr][nc][1] >= cm(nr, nc):
                    queue.append((nr, nc))

        return b

    # ── board queries ──────────────────────────────────────────────────────────

    def has_pieces(board, p):
        return any(board[r][c][0] == p for r in range(rows) for c in range(cols))

    def all_valid(board, p):
        return [
            (r, c)
            for r in range(rows) for c in range(cols)
            if board[r][c][0] is None or board[r][c][0] == p
        ]

    # ── cheap move scoring (for candidate pruning) ────────────────────────────

    def score_move(board, r, c, p):
        opp    = 1 - p
        _, cnt = board[r][c]
        cmass  = cm(r, c)
        s      = 0

        # About to explode?
        if cnt + 1 >= cmass:
            s += 30
            for nr, nc in neighbors(r, c):
                if board[nr][nc][0] == opp:
                    s += 8            # captures an opponent cell
                no, nc2 = board[nr][nc]
                if no == p and nc2 + 1 >= cm(nr, nc):
                    s += 4            # triggers a friendly chain

        s += cnt * 2                  # prefer already-loaded cells
        s += (4 - cmass) * 3          # prefer corners/edges (lower CM)
        return s

    def candidates(board, p, limit=MAX_CANDS):
        moves = all_valid(board, p)
        if len(moves) <= limit:
            return moves
        moves.sort(key=lambda m: score_move(board, m[0], m[1], p), reverse=True)
        return moves[:limit]

    # ── heuristic evaluation ──────────────────────────────────────────────────

    def evaluate(board, pov):
        opp  = 1 - pov
        my_p = has_pieces(board, pov)
        op_p = has_pieces(board, opp)
        if not op_p and my_p:
            return 100_000
        if not my_p and op_p:
            return -100_000

        my_sc = opp_sc = 0.0
        for r in range(rows):
            for c in range(cols):
                owner, cnt = board[r][c]
                if cnt == 0:
                    continue
                cmass = cm(r, c)
                val   = cnt + (cnt / cmass) * 4 + (4 - cmass) * 0.5
                if cnt == cmass - 1:
                    val += 2.0
                if owner == pov:
                    my_sc += val
                elif owner == opp:
                    opp_sc += val

        return my_sc - opp_sc

    # ── minimax + α-β ─────────────────────────────────────────────────────────

    timed_out = [False]

    def minimax(board, depth, alpha, beta, maximizing, cur, moves_made):
        if timed_out[0]:
            return evaluate(board, player_id), None

        if moves_made >= 2:
            mp = has_pieces(board, player_id)
            op = has_pieces(board, opponent)
            if not op and mp:
                return  100_000, None
            if not mp and op:
                return -100_000, None

        if depth == 0:
            return evaluate(board, player_id), None

        opp_cur = 1 - cur
        moves   = candidates(board, cur)
        if not moves:
            return evaluate(board, player_id), None

        best_move = moves[0]

        if maximizing:
            best_val = float('-inf')
            for r, c in moves:
                if time.time() > deadline:
                    timed_out[0] = True
                    break
                nb  = apply_move(board, r, c, cur)
                val, _ = minimax(nb, depth - 1, alpha, beta, False, opp_cur, moves_made + 1)
                if val > best_val:
                    best_val, best_move = val, (r, c)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best_val, best_move
        else:
            best_val = float('inf')
            for r, c in moves:
                if time.time() > deadline:
                    timed_out[0] = True
                    break
                nb  = apply_move(board, r, c, cur)
                val, _ = minimax(nb, depth - 1, alpha, beta, True, opp_cur, moves_made + 1)
                if val < best_val:
                    best_val, best_move = val, (r, c)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best_val, best_move

    # ── iterative deepening ───────────────────────────────────────────────────

    pieces_on_board = sum(
        1 for r in range(rows) for c in range(cols) if state[r][c][0] is not None
    )

    best_move = None

    for depth in range(1, MAX_DEPTH + 1):
        if time.time() > deadline:
            break
        timed_out[0] = False
        _, move = minimax(
            state,
            depth=depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing=True,
            cur=player_id,
            moves_made=pieces_on_board,
        )
        if move is not None and not timed_out[0]:
            best_move = move        # only commit a fully-searched result

    # ── fallback: greedy single-ply if time ran out before depth 1 ───────────
    if best_move is None:
        moves = all_valid(state, player_id)
        if not moves:
            return (0, 0)
        best_move = max(moves, key=lambda m: score_move(state, m[0], m[1], player_id))

    return best_move