# main.py (安全版: 盤面コピーで探索 / 返却前に満杯チェック)
from typing import List, Tuple, Optional
from framework import Alg3D, Board  # 本番用

Coord = Tuple[int, int, int]  # (x, y, z)

# ---------------- 基本ユーティリティ ----------------
def clone(board: Board) -> Board:
    # 4x4x4 固定なので手書きコピーが最速&安全
    return [[row[:] for row in plane] for plane in board]

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(4):
        if board[z][y][x] == 0:
            return z
    return None

def column_has_space(board: Board, x: int, y: int) -> bool:
    return board[3][y][x] == 0

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def legal_cols(board: Board):
    for x in range(4):
        for y in range(4):
            if column_has_space(board, x, y):
                yield (x, y)

def count_bits(board: Board) -> int:
    c = 0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                c += 1 if board[z][y][x] != 0 else 0
    return c

# ---------------- 勝ち筋（76本） ----------------
def all_win_lines() -> List[List[Coord]]:
    L: List[List[Coord]] = []
    rng = range(4)
    # 軸方向
    for y in rng:
        for z in rng:
            L.append([(x, y, z) for x in rng])  # X軸
    for x in rng:
        for z in rng:
            L.append([(x, y, z) for y in rng])  # Y軸
    for x in rng:
        for y in rng:
            L.append([(x, y, z) for z in rng])  # Z軸
    # 平面斜め
    for z in rng:
        L.append([(i, i, z) for i in rng])
        L.append([(i, 3 - i, z) for i in rng])
    for y in rng:
        L.append([(i, y, i) for i in rng])
        L.append([(i, y, 3 - i) for i in rng])
    for x in rng:
        L.append([(x, i, i) for i in rng])
        L.append([(x, i, 3 - i) for i in rng])
    # 空間対角
    L.append([(i, i, i) for i in rng])
    L.append([(i, i, 3 - i) for i in rng])
    L.append([(i, 3 - i, i) for i in rng])
    L.append([(3 - i, i, i) for i in rng])
    return L

WIN_LINES = all_win_lines()

def is_win_for(board: Board, player: int) -> bool:
    for line in WIN_LINES:
        ok = True
        for (x, y, z) in line:
            if board[z][y][x] != player:
                ok = False
                break
        if ok:
            return True
    return False

# ---------------- 即勝ち / ダブルスレット（コピー盤で判定） ----------------
def immediate_winning_moves(board: Board, player: int):
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        b2 = clone(board)
        if place_inplace(b2, x, y, player) is not None and is_win_for(b2, player):
            res.append((x, y))
    return res

def creates_double_threat(board: Board, player: int):
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        b2 = clone(board)
        if place_inplace(b2, x, y, player) is None:
            continue
        wins_next = immediate_winning_moves(b2, player)  # ここもコピー盤
        if len(wins_next) >= 2:
            res.append((x, y))
    return res

CENTER_COLS = [(1, 1), (1, 2), (2, 1), (2, 2)]
PREFERRED_COLS = CENTER_COLS + [
    (1, 0), (2, 0), (0, 1), (3, 1), (0, 2), (3, 2), (1, 3), (2, 3)
] + [(0, 0), (0, 3), (3, 0), (3, 3)]

# ---------------- 軽量評価＆探索（コピー盤） ----------------
def eval_position(board: Board, me: int, opp: int) -> int:
    score = 0
    for line in WIN_LINES:
        myc = 0
        opc = 0
        for (x, y, z) in line:
            v = board[z][y][x]
            if v == me:
                myc += 1
            elif v == opp:
                opc += 1
        if myc and opc:
            continue
        if opc == 0:
            score += (1000 if myc == 3 else 50 if myc == 2 else 5 if myc == 1 else 1)
        elif myc == 0:
            score -= (900 if opc == 3 else 45 if opc == 2 else 4 if opc == 1 else 1)
    for (x, y) in CENTER_COLS:
        if board[3][y][x] == 0:
            score += 1
    return score

def negamax(board: Board, depth: int, alpha: int, beta: int, me: int, opp: int):
    # 一手勝ち（コピー盤で安全判定）
    wins = immediate_winning_moves(board, me)
    if wins:
        return 10_000 + depth, wins[0]
    if depth == 0:
        return eval_position(board, me, opp), None

    moves = list(legal_cols(board))
    opp_wins = set(immediate_winning_moves(board, opp))

    def move_key(xy: Tuple[int, int]):
        in_pref = xy in PREFERRED_COLS
        pref_idx = PREFERRED_COLS.index(xy) if in_pref else 999
        is_block = xy in opp_wins
        return (not is_block, not in_pref, pref_idx)

    moves.sort(key=move_key)

    best_move: Optional[Tuple[int, int]] = None
    for (x, y) in moves:
        b2 = clone(board)
        if place_inplace(b2, x, y, me) is None:
            continue
        val, _ = negamax(b2, depth - 1, -beta, -alpha, opp, me)
        val = -val
        if val > alpha:
            alpha = val
            best_move = (x, y)
            if alpha >= beta:
                break
    return alpha, best_move

# ---------------- メインAI ----------------
class MyAI(Alg3D):
    def get_move(
        self,
        board: Board,                 # board[z][y][x]
        player: int,                  # 1:先手, 2:後手
        last_move: Tuple[int, int, int]
    ) -> Tuple[int, int]:
        me = player
        opp = 1 if player == 2 else 2
        move_count = count_bits(board)

        # 1) 初手ブック（先手）：中央寄り
        if me == 1 and move_count == 0:
            for (x, y) in CENTER_COLS:
                if column_has_space(board, x, y):
                    return (x, y)
            for (x, y) in PREFERRED_COLS:
                if column_has_space(board, x, y):
                    return (x, y)

        # 2) 即勝ち
        wins = immediate_winning_moves(board, me)
        if wins:
            x, y = wins[0]
            # 念のため返却前チェック
            if column_has_space(board, x, y):
                return (x, y)

        # 3) 相手の即勝ちブロック
        opp_wins = immediate_winning_moves(board, opp)
        if opp_wins:
            opp_wins.sort(key=lambda xy: (
                xy not in PREFERRED_COLS,
                PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999
            ))
            for (x, y) in opp_wins:
                if column_has_space(board, x, y):
                    return (x, y)

        # 4) ダブルスレット
        dts = creates_double_threat(board, me)
        if dts:
            dts.sort(key=lambda xy: (
                xy not in PREFERRED_COLS,
                PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999
            ))
            x, y = dts[0]
            if column_has_space(board, x, y):
                return (x, y)

        # 5) Negamax + αβ（深さ調整）
        DEPTH = 5 if move_count >= 8 else 4
        _, best = negamax(board, DEPTH, -10**9, 10**9, me, opp)
        if best is not None and column_has_space(board, best[0], best[1]):
            return best

        # 6) フォールバック（中央優先 → 最初の合法手）
        for (x, y) in PREFERRED_COLS:
            if column_has_space(board, x, y):
                return (x, y)
        for (x, y) in legal_cols(board):
            return (x, y)

        # 盤が完全に埋まっているはずだが、安全策
        return (0, 0)
