from collections import deque
from copy import deepcopy


class ChainReactionGame:
    """
    Simple Chain Reaction game engine (2-player).

    Board cells store (owner, orb_count):
    - owner is None if the cell is empty, otherwise it's an integer player id.
    - orb_count is the number of orbs in the cell (0..).

    Rules:
    - Players take turns placing 1 orb into an empty cell or a cell they already own.
    - Every cell has a "critical mass" (capacity) based on its position:
        * Corners: 2 orbs
        * Edges: 3 orbs
        * Middle: 4 orbs
    - When a cell reaches its critical mass, it explodes!
      It shoots 1 orb into each of its directly adjacent (orthogonal) neighbors.
    - If the exploding cell's orbs land in a cell owned by the opponent, 
      the exploding player takes ownership of that cell and the orbs inside it.
    - Explosions can push neighboring cells past their capacity, 
      triggering a cascading chain reaction!
    - Winner: After both players have played at least once,
      a player immediately wins if the opponent has 0 owned cells left.
    """

    def __init__(self, rows=12, cols=8):
        self.rows = rows
        self.cols = cols
        self.board = [[(None, 0) for _ in range(cols)] for _ in range(rows)]
        self.moves_played = {0: 0, 1: 0}

    def get_state(self):
        return deepcopy(self.board)

    def get_valid_moves(self, player_id):
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                owner, _count = self.board[r][c]
                if owner is None or owner == player_id:
                    moves.append((r, c))
        return moves

    def apply_move(self, player_id, move):

        if not (isinstance(move, tuple) and len(move) == 2):
            raise ValueError("move must be a (row, col) tuple")

        r, c = move
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError("move is out of bounds")

        owner, count = self.board[r][c]
        if owner is not None and owner != player_id:
            raise ValueError("invalid move: cell is owned by opponent")

        # Place orb- ownership becomes the moving player
        self.board[r][c] = (player_id, count + 1)
        self.moves_played[player_id] += 1

        # Resolve chain reactions
        q = deque()
        if self.board[r][c][1] >= self.capacity(r, c):
            q.append((r, c))

        while q:
            cr, cc = q.popleft()
            cur_owner, cur_count = self.board[cr][cc]

            # It's possible this cell no longer needs to explode
            if cur_count < self.capacity(cr, cc) :
                continue

            exploding_owner = cur_owner
            if self.check_winner() is not None:
                # Board reaches win state, can stop chain reactions
                break

            # Explosion- cell loses capacity orbs
            cap = self.capacity(cr, cc)
            remaining = cur_count - cap
            if remaining > 0:
                self.board[cr][cc] = (exploding_owner, remaining)
                if remaining >= cap:
                    q.append((cr, cc))
            else:
                self.board[cr][cc] = (None, 0)

            # Distribute 1 orb to each neighbor and set ownership to exploding owner
            for nr, nc in self.neighbors(cr, cc):
                n_owner, n_count = self.board[nr][nc]
                self.board[nr][nc] = (exploding_owner, n_count + 1)
                if self.board[nr][nc][1] == self.capacity(nr, nc):
                    q.append((nr, nc))

    def check_winner(self):
        """
        Returns:
        - 0 if player 0 has won
        - 1 if player 1 has won
        - None if no winner yet
        """
        if self.moves_played[0] == 0 or self.moves_played[1] == 0:
            return None

        counts = {0: 0, 1: 0}
        for r in range(self.rows):
            for c in range(self.cols):
                owner, orb_count = self.board[r][c]
                if owner in (0, 1) and orb_count > 0:
                    counts[owner] += 1

        if counts[0] == 0 and counts[1] > 0:
            return 1
        if counts[1] == 0 and counts[0] > 0:
            return 0
        return None

    def capacity(self, r, c):
        # corner: 2, edge: 3, middle: 4
        is_top = r == 0
        is_bottom = r == self.rows - 1
        is_left = c == 0
        is_right = c == self.cols - 1

        if (is_top and is_left) or (is_top and is_right) or (is_bottom and is_left) or (is_bottom and is_right):
            return 2
        if is_top or is_bottom or is_left or is_right:
            return 3
        return 4

    def neighbors(self, r, c):
        res = []
        if r - 1 >= 0:
            res.append((r - 1, c))
        if r + 1 < self.rows:
            res.append((r + 1, c))
        if c - 1 >= 0:
            res.append((r, c - 1))
        if c + 1 < self.cols:
            res.append((r, c + 1))
        return res

