# main.py
# 4x4x4 立体四目並べ AI（10秒制限対応版）
# - 3手定石
# - 反復深化 + 時間/ノード制限
# - Move ordering + ビーム（上位Kのみ深掘り）
# - 即勝ち/即ブロック/ダブルリーチの優先
from typing import List, Tuple, Optional, Dict
import time

from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4

WIN_SCORE      = 1_000_000
DOUBLE_THREAT  = 5_000
TPOINT_BONUS   = 150
CENTER_BONUS   = 5

# ---- タイム/ノード制御 ----
TIME_BUDGET_SEC   = 9.5       # 呼び出し側が10秒なら安全マージン
NODE_BUDGET       = 180_000   # 上限ノード（安全ネット）

# 盤面状況で深さとビーム幅を調整
def depth_and_beam(stones: int) -> Tuple[int, int]:
    # stones: 置かれている石の総数
    if stones <= 4:         # 序盤（分岐大）→浅く・広く
        return 3, 10
    if stones <= 24:        # 中盤
        return 2, 8
    else:                   # 終盤（分岐減）→少し深く
        return 3, 12

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

# last を含むラインのみチェック（軽量）
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
    if any(all(board[z][y][x] == me  for (x,y,z) in line) for line in ALL_LINES):
        return WIN_SCORE - ply
    if any(all(board[z][y][x] == you for (x,y,z) in line) for line in ALL_LINES):
        return -WIN_SCORE + ply

    score = 0
    table = [0, 1, 10, 120]   # 3連は大きめ。t点でさらに上乗せ

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

    # ダブルリーチ加点/減点
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

    before_threats = count_immediate_wins(board, you)

    for (x, y) in moves:
        # 自分の即勝ち
        if is_winning_after(board, me, x, y):
            wins.append((x, y))
            continue

        z = place_inplace(board, x, y, me)
        my_wins_after = count_immediate_wins(board, me)
        opp_after = count_immediate_wins(board, you)
        undo_place(board, x, y, z)

        if my_wins_after >= 2:
            double_threats.append((x, y))
        elif before_threats > 0 and opp_after < before_threats:
            blocks.append((x, y))
        else:
            rest.append((x, y))

    return wins + double_threats + blocks + rest

# ===== 反復深化 + 時間/ノード制御 αβ =====
class Searcher:
    def __init__(self, me: int, max_depth: int, beam_width: int, deadline: float, node_budget: int):
        self.me = me
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.deadline = deadline
        self.node_budget = node_budget
        self.nodes = 0
        self.tt: Dict[Tuple, Tuple[int,int,int]] = {}
        # key -> (depth_remaining, value, flag)
        # flag: 0=exact, -1=upperbound, +1=lowerbound

    def time_up(self) -> bool:
        return (self.nodes >= self.node_budget) or (time.perf_counter() >= self.deadline)

    def key_of(self, board: Board, turn: int) -> Tuple:
        # 盤面をタプル化（Zobristほど速くないが十分軽量）
        return (turn, tuple(tuple(tuple(board[z][y][x] for x in range(SIZE)) for y in range(SIZE)) for z in range(SIZE)))

    def minimax(self, board: Board, depth: int, alpha: int, beta: int, turn: int, ply: int) -> int:
        if self.time_up():
            return eval_board(board, self.me, ply)  # 打ち切り評価

        self.nodes += 1
        moves = valid_xy_moves(board)

        if depth == 0 or not moves:
            return eval_board(board, self.me, ply)

        # 即勝ち刈り取り
        for (x, y) in moves:
            if is_winning_after(board, turn, x, y):
                return WIN_SCORE - ply if turn == self.me else -WIN_SCORE + ply

        # TT
        key = self.key_of(board, turn)
        entry = self.tt.get(key)
        if entry and entry[0] >= depth:
            val, flag = entry[1], entry[2]
            if flag == 0:
                return val
            if flag == -1 and val <= alpha:
                return val
            if flag == +1 and val >= beta:
                return val

        # 並べ替え + ビーム
        ordered = order_moves(board, turn, moves)[:self.beam_width]

        if turn == self.me:
            best = -10**9
            a0 = alpha
            for (x, y) in ordered:
                z = place_inplace(board, x, y, turn)
                val = self.minimax(board, depth-1, alpha, beta, 3-turn, ply+1)
                undo_place(board, x, y, z)
                if val > best: best = val
                if best > alpha: alpha = best
                if alpha >= beta: break
            # TT 保存
            flag = 0
            if best <= a0: flag = -1
            elif best >= beta: flag = +1
            self.tt[key] = (depth, best, flag)
            return best
        else:
            best = 10**9
            b0 = beta
            for (x, y) in ordered:
                z = place_inplace(board, x, y, turn)
                val = self.minimax(board, depth-1, alpha, beta, 3-turn, ply+1)
                undo_place(board, x, y, z)
                if val < best: best = val
                if best < beta: beta = best
                if alpha >= beta: break
            # TT 保存
            flag = 0
            if best >= b0: flag = +1
            elif best <= alpha: flag = -1
            self.tt[key] = (depth, best, flag)
            return best

# --- 定石（最初の3手） ---
def opening_move(board: Board, player: int) -> Optional[Coord2]:
    n = stones_count(board)
    if n == 0 and player == 1:
        return (1, 1) if lowest_empty_z(board, 1, 1) is not None else None
    if n == 1 and player == 2:
        for (x, y) in [(2, 2), (2, 1), (1, 2), (1, 0), (0, 1), (2, 3), (3, 2)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
    if n == 2 and player == 1:
        for (x, y) in [(2, 1), (1, 2), (2, 0), (0, 2), (1, 0), (0, 1), (2, 3), (3, 2)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
    return None

# --- ルートでの最善選択（反復深化） ---
def choose_best(board: Board, me: int) -> Coord2:
    # 0) 定石
    mv = opening_move(board, me)
    if mv is not None:
        return mv

    # 1) 自分の即勝ち
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちブロック：即勝ち手の数を最小化
    you = 3 - me
    opp_wins_now = immediate_winning_squares(board, you)
    if opp_wins_now:
        best_move = None
        best_after = 10**9
        for (x, y) in valid_xy_moves(board):
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            undo_place(board, x, y, z)
            if after < best_after:
                best_after = after
                best_move = (x, y)
                if after == 0:
                    break
        if best_move is not None:
            return best_move

    # 3) ダブルリーチ作成手
    cand = []
    for (x, y) in valid_xy_moves(board):
        z = place_inplace(board, x, y, me)
        k = count_immediate_wins(board, me)
        undo_place(board, x, y, z)
        if k >= 2:
            cand.append((x, y))
    if cand:
        return cand[0]

    # 4) 反復深化（時間/ノード制限）
    stones = stones_count(board)
    max_depth, beam = depth_and_beam(stones)
    deadline = time.perf_counter() + TIME_BUDGET_SEC
    search = Searcher(me, max_depth, beam, deadline, NODE_BUDGET)

    root_moves = order_moves(board, me, valid_xy_moves(board))
    root_moves = root_moves[:beam]  # ルートでもビーム

    best_move = root_moves[0] if root_moves else (1, 1)
    best_val  = -10**9

    # 深さ1→max_depth と段階的に深く（間に合った最後の結果を採用）
    for depth in range(1, max_depth + 1):
        if search.time_up():
            break
        local_best_move = best_move
        local_best_val  = -10**9
        alpha, beta = -10**9, 10**9

        for (x, y) in root_moves:
            if search.time_up():
                break
            z = place_inplace(board, x, y, me)
            val = search.minimax(board, depth-1, alpha, beta, 3-me, 1)
            undo_place(board, x, y, z)
            if val > local_best_val:
                local_best_val = val
                local_best_move = (x, y)

        # 深さ depth を完了できたら更新
        if not search.time_up():
            best_move = local_best_move
            best_val  = local_best_val
        else:
            break

    return best_move

# --- エンジン ---
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player)

AI = MyAI()
