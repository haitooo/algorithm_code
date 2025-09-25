# main.py
# 4x4x4 立体四目並べ AI（定石3手 + ブロック修正版）

from typing import List, Tuple, Optional
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4

WIN_SCORE      = 1_000_000
DOUBLE_THREAT  = 5_000     # 自分が次に勝てる手が2つ以上: 加点 / 相手なら減点
TPOINT_BONUS   = 150        # t点(z==3) 浮きリーチのボーナス
CENTER_BONUS   = 5          # 位置ボーナス（序盤寄与）
DEPTH_DEFAULT  = 3

# ---------------- 基本ユーティリティ ----------------
def clone(board: Board) -> Board:
    return [[row[:] for row in plane] for plane in board]

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    ms: List[Coord2] = []
    top = SIZE - 1
    for y in range(SIZE):
        for x in range(SIZE):
            if board[top][y][x] == 0:
                ms.append((x, y))
    return ms

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def undo_place(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

def stones_count(board: Board) -> int:
    c = 0
    for z in range(SIZE):
        for y in range(SIZE):
            for x in range(SIZE):
                if board[z][y][x] != 0:
                    c += 1
    return c

# --- 全勝ち筋（76本） ---
def generate_lines() -> List[List[Coord3]]:
    lines: List[List[Coord3]] = []
    # x直線 / y直線 / z直線
    for z in range(SIZE):
        for y in range(SIZE):
            lines.append([(x, y, z) for x in range(SIZE)])
    for z in range(SIZE):
        for x in range(SIZE):
            lines.append([(x, y, z) for y in range(SIZE)])
    for y in range(SIZE):
        for x in range(SIZE):
            lines.append([(x, y, z) for z in range(SIZE)])
    # 各面内の斜め
    for z in range(SIZE):
        lines.append([(i, i, z) for i in range(SIZE)])
        lines.append([(i, SIZE-1-i, z) for i in range(SIZE)])
    for y in range(SIZE):
        lines.append([(i, y, i) for i in range(SIZE)])
        lines.append([(i, y, SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        lines.append([(x, i, i) for i in range(SIZE)])
        lines.append([(x, i, SIZE-1-i) for i in range(SIZE)])
    # 空間対角4本
    lines.append([(i, i, i) for i in range(SIZE)])
    lines.append([(i, i, SIZE-1-i) for i in range(SIZE)])
    lines.append([(i, SIZE-1-i, i) for i in range(SIZE)])
    lines.append([(SIZE-1-i, i, i) for i in range(SIZE)])
    return lines

ALL_LINES = generate_lines()

# ある着手で勝つか
def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None:
        return False
    win = check_win_at(board, player, (x, y, z))
    undo_place(board, x, y, z)
    return win

# last を含むラインのみチェック（安全＆軽量）
def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    lx, ly, lz = last
    for line in ALL_LINES:
        includes_last = False
        for (x, y, z) in line:
            if x == lx and y == ly and z == lz:
                includes_last = True
                break
        if includes_last and all(board[z][y][x] == player for (x, y, z) in line):
            return True
    return False

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    return [(x, y) for (x, y) in valid_xy_moves(board) if is_winning_after(board, player, x, y)]

def count_immediate_wins(board: Board, player: int) -> int:
    return len(immediate_winning_squares(board, player))

# --- 評価関数 ---
def eval_board(board: Board, me: int, ply: int) -> int:
    you = 3 - me

    # 終局
    # （全ライン走査だが 76 本 × 4 で常に十分軽い）
    if any(all(board[z][y][x] == me  for (x,y,z) in line) for line in ALL_LINES):
        return WIN_SCORE - ply
    if any(all(board[z][y][x] == you for (x,y,z) in line) for line in ALL_LINES):
        return -WIN_SCORE + ply

    score = 0

    # パターン認識
    table = [0, 1, 10, 120]   # 3 連は大きめ。t点でさらに上乗せ
    for line in ALL_LINES:
        cnt_me = cnt_you = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == me:
                cnt_me += 1
            elif v == you:
                cnt_you += 1
            else:
                empties.append((x,y,z))

        if cnt_you == 0:
            score += table[cnt_me]
            if cnt_me == 3:
                e = empties[0]
                x, y, z = e
                ze = lowest_empty_z(board, x, y)
                if ze is not None and ze == z and z == 3:
                    score += TPOINT_BONUS
        if cnt_me == 0:
            score -= table[cnt_you]
            if cnt_you == 3:
                e = empties[0]
                x, y, z = e
                ze = lowest_empty_z(board, x, y)
                if ze is not None and ze == z and z == 3:
                    score -= TPOINT_BONUS

    # 位置ボーナス（中心寄り）
    top = SIZE - 1
    for y in range(SIZE):
        for x in range(SIZE):
            if board[top][y][x] == 0:
                cx = abs(1.5 - x)
                cy = abs(1.5 - y)
                score += int(CENTER_BONUS * (1.5 - (cx + cy) / 2))

    # ダブルリーチ加点/減点（即勝ち手の本数で判定）
    my_wins  = count_immediate_wins(board, me)
    you_wins = count_immediate_wins(board, you)
    if my_wins  >= 2: score += DOUBLE_THREAT
    if you_wins >= 2: score -= DOUBLE_THREAT

    return score

# --- 着手並べ替え（即勝ち / ダブルリーチ / ブロック / その他） ---
def order_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    wins = []
    double_threats = []
    blocks = []
    rest = []
    you = 3 - me

    # 現在の相手即勝ち数（ブロック判定の基準）
    before_threats = count_immediate_wins(board, you)

    for (x, y) in moves:
        # 自分の即勝ち
        if is_winning_after(board, me, x, y):
            wins.append((x, y))
            continue

        z = place_inplace(board, x, y, me)
        # 自分のダブルリーチ生成？
        my_wins_after = count_immediate_wins(board, me)
        # 相手の即勝ち数が減る？（ブロック判定を前後比較で）
        opp_after = count_immediate_wins(board, you)
        undo_place(board, x, y, z)

        if my_wins_after >= 2:
            double_threats.append((x, y))
        elif before_threats > 0 and opp_after < before_threats:
            blocks.append((x, y))
        else:
            rest.append((x, y))

    return wins + double_threats + blocks + rest

# --- ミニマックス + αβ ---
def minimax(board: Board, depth: int, alpha: int, beta: int, me: int, turn: int, ply: int) -> int:
    moves = valid_xy_moves(board)

    if depth == 0 or not moves:
        return eval_board(board, me, ply)

    # 手番側の即勝ちは即スコア（勝敗確定）
    for (x, y) in moves:
        if is_winning_after(board, turn, x, y):
            return WIN_SCORE - ply if turn == me else -WIN_SCORE + ply

    # 並べ替え
    moves = order_moves(board, turn, moves)

    if turn == me:
        best = -10**9
        for (x, y) in moves:
            z = place_inplace(board, x, y, turn)
            val = minimax(board, depth-1, alpha, beta, me, 3-turn, ply+1)
            undo_place(board, x, y, z)
            if val > best:
                best = val
            if best > alpha: alpha = best
            if alpha >= beta: break
        return best
    else:
        best = 10**9
        for (x, y) in moves:
            z = place_inplace(board, x, y, turn)
            val = minimax(board, depth-1, alpha, beta, me, 3-turn, ply+1)
            undo_place(board, x, y, z)
            if val < best:
                best = val
            if best < beta: beta = best
            if alpha >= beta: break
        return best

# --- 定石（最初の3手） ---
def opening_move(board: Board, player: int) -> Optional[Coord2]:
    """ゲーム全体の手数が 0,1,2 のいずれかなら定石を返す。置けない時は None。"""
    n = stones_count(board)

    # 第1手（先手）
    if n == 0 and player == 1:
        return (1, 1) if lowest_empty_z(board, 1, 1) is not None else None

    # 第2手（後手）
    if n == 1 and player == 2:
        # 中央対称をまず狙う → (2,2) がベストに近い
        for (x, y) in [(2, 2), (2, 1), (1, 2), (1, 0), (0, 1), (2, 3), (3, 2)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
        return None

    # 第3手（先手2手目）
    if n == 2 and player == 1:
        # 相手が(2,2)など中心寄りに来た場合、(2,1)→(1,2)→(2,0)→(0,2)の優先で展開
        for (x, y) in [(2, 1), (1, 2), (2, 0), (0, 2), (1, 0), (0, 1), (2, 3), (3, 2)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
        return None

    return None

# --- ルートでの最善選択 ---
def choose_best(board: Board, me: int, depth: int) -> Coord2:
    # 0) 定石（最初の3手）
    mv = opening_move(board, me)
    if mv is not None:
        return mv

    # 1) 自分の即勝ち
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちブロック：相手の即勝ち手を列挙し、その数を最小化する手を選ぶ
    you = 3 - me
    opp_wins_now = immediate_winning_squares(board, you)
    if opp_wins_now:
        best_move = None
        best_after = 10**9
        for (x, y) in valid_xy_moves(board):
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            undo_place(board, x, y, z)
            # 最優先：after==0（完全ブロック）。次点：after が最小。
            if after < best_after:
                best_after = after
                best_move = (x, y)
                if after == 0:
                    break
        if best_move is not None:
            return best_move

    # 3) ダブルリーチ作成手を優先
    cand = []
    for (x, y) in valid_xy_moves(board):
        z = place_inplace(board, x, y, me)
        k = count_immediate_wins(board, me)
        undo_place(board, x, y, z)
        if k >= 2:
            cand.append((x, y))
    if cand:
        return cand[0]

    # 4) 通常探索（ミニマックス+αβ）
    best_val = -10**9
    best_move = None
    moves = order_moves(board, me, valid_xy_moves(board))
    for (x, y) in moves:
        z = place_inplace(board, x, y, me)
        val = minimax(board, depth-1, -10**9, 10**9, me, 3-me, 1)
        undo_place(board, x, y, z)
        if val > best_val:
            best_val = val
            best_move = (x, y)

    # 5) フォールバック：中心寄り
    if best_move is None:
        cx_order = sorted(valid_xy_moves(board), key=lambda p: (abs(1.5-p[0]) + abs(1.5-p[1])))
        best_move = cx_order[0]
    return best_move

# --- エンジン ---
class MyAI(Alg3D):
    def __init__(self, depth: int = DEPTH_DEFAULT):
        self.depth = depth

    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player, self.depth)

AI = MyAI()
