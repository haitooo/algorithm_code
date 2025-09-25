# main.py
# 提出仕様に準拠: Python 3.9互換 / 標準ライブラリのみ / get_move(board, player, last_move)
from typing import List, Tuple, Optional
from framework import Alg3D, Board  # 本番用（local_driverと同じシグネチャに揃える）

Coord = Tuple[int, int, int]  # (x, y, z)

# --- 盤面は board[z][y][x] に注意（local_driverと同じ） ---
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(4):
        if board[z][y][x] == 0:
            return z
    return None

def place(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def unplace(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

def legal_cols(board: Board):
    for x in range(4):
        for y in range(4):
            if board[3][y][x] == 0:  # 最上段が埋まっていなければ合法（重力あり）
                yield (x, y)

def count_bits(board: Board) -> int:
    c = 0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                if board[z][y][x] != 0:
                    c += 1
    return c

# --- 4x4x4 の全勝ち筋（76本）を列挙（座標は(x,y,z)だが参照は board[z][y][x]） ---
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
    # XY 平面の斜め（各 z）
    for z in rng:
        L.append([(i, i, z) for i in rng])
        L.append([(i, 3 - i, z) for i in rng])
    # XZ 平面の斜め（各 y）
    for y in rng:
        L.append([(i, y, i) for i in rng])
        L.append([(i, y, 3 - i) for i in rng])
    # YZ 平面の斜め（各 x）
    for x in rng:
        L.append([(x, i, i) for i in rng])
        L.append([(x, i, 3 - i) for i in rng])
    # 空間対角（4本）
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

def immediate_winning_moves(board: Board, player: int):
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        z = place(board, x, y, player)
        if z is None:
            continue
        if is_win_for(board, player):
            res.append((x, y))
        unplace(board, x, y, z)
    return res

def creates_double_threat(board: Board, player: int):
    # 着手後、「次の自分の手」で勝てる合法着手が2つ以上になる手
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        z = place(board, x, y, player)
        if z is None:
            continue
        wins_next = immediate_winning_moves(board, player)
        if len(wins_next) >= 2:
            res.append((x, y))
        unplace(board, x, y, z)
    return res

CENTER_COLS = [(1, 1), (1, 2), (2, 1), (2, 2)]
PREFERRED_COLS = CENTER_COLS + [
    (1, 0), (2, 0), (0, 1), (3, 1), (0, 2), (3, 2), (1, 3), (2, 3)
] + [(0, 0), (0, 3), (3, 0), (3, 3)]

def eval_position(board: Board, me: int, opp: int) -> int:
    # ライン単位での静的評価（自専有:+ / 相手専有:-）。軽量で高速。
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
            continue  # 相殺ライン
        if opc == 0:
            if myc == 3:
                score += 1000
            elif myc == 2:
                score += 50
            elif myc == 1:
                score += 5
            else:
                score += 1
        elif myc == 0:
            if opc == 3:
                score -= 900
            elif opc == 2:
                score -= 45
            elif opc == 1:
                score -= 4
            else:
                score -= 1
    # 空きのある中央列を微加点（盤面の詰まりを緩く反映）
    for (x, y) in CENTER_COLS:
        if board[3][y][x] == 0:
            score += 1
    return score

def negamax(board: Board, depth: int, alpha: int, beta: int, me: int, opp: int):
    # 一手勝ちの早見。ここで見つかれば最大評価で返す
    wins = immediate_winning_moves(board, me)
    if wins:
        return 10_000 + depth, wins[0]
    if depth == 0:
        return eval_position(board, me, opp), None

    # 着手順序：相手の即勝受け → 中央優先 → その他
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
        z = place(board, x, y, me)
        if z is None:
            continue
        val, _ = negamax(board, depth - 1, -beta, -alpha, opp, me)
        val = -val
        unplace(board, x, y, z)
        if val > alpha:
            alpha = val
            best_move = (x, y)
            if alpha >= beta:
                break
    return alpha, best_move

class MyAI(Alg3D):
    def get_move(
        self,
        board: Board,                 # board[z][y][x]
        player: int,                  # 1:先手(黒), 2:後手(白)
        last_move: Tuple[int, int, int]  # 直前手 (x,y,z) or (None, None, None)
    ) -> Tuple[int, int]:
        me = player
        opp = 1 if player == 2 else 2
        move_count = count_bits(board)

        # 1) 初手ブック（先手のみ）：中央寄りの列を最優先
        if me == 1 and move_count == 0:
            for (x, y) in CENTER_COLS:
                if board[3][y][x] == 0:
                    return (x, y)
            for (x, y) in PREFERRED_COLS:
                if board[3][y][x] == 0:
                    return (x, y)

        # 2) 即勝ち
        wins = immediate_winning_moves(board, me)
        if wins:
            return wins[0]

        # 3) 相手の即勝ちをブロック
        opp_wins = immediate_winning_moves(board, opp)
        if opp_wins:
            opp_wins.sort(key=lambda xy: (
                xy not in PREFERRED_COLS,
                PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999
            ))
            for (x, y) in opp_wins:
                if board[3][y][x] == 0:
                    return (x, y)

        # 4) ダブルスレット作成
        dts = creates_double_threat(board, me)
        if dts:
            dts.sort(key=lambda xy: (
                xy not in PREFERRED_COLS,
                PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999
            ))
            return dts[0]

        # 5) 浅い探索（Negamax + αβ）
        # 盤の進行度で深さを調整（高速・安定運用）
        DEPTH = 5 if move_count >= 8 else 4
        val, best = negamax(board, DEPTH, -10**9, 10**9, me, opp)
        if best is not None:
            return best

        # 6) フォールバック（中央優先 → 任意の合法手）
        for (x, y) in PREFERRED_COLS:
            if board[3][y][x] == 0:
                return (x, y)
        for (x, y) in legal_cols(board):
            return (x, y)

        # 盤が完全に埋まっているケース（安全策）
        return (0, 0)
