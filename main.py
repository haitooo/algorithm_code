# main.py
# 4x4x4 立体四目並べ AI（角優先・確実ブロック・安全フィルタ／10秒内軽量版）
# 依存: framework.Alg3D, Board  (get_move(board, player, last_move) -> (x,y))

from typing import List, Tuple, Optional
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# --- 時間予算（保険。今回は軽量なので通常は超えません） ---
TIME_BUDGET_SEC = 9.5

# --- 評価重み（軽量ヒューリスティクス） ---
WIN_SCORE      = 1_000_000
W_MY_O3        = 800    # 自分の open-3（リーチ）
W_MY_O2        = 220    # 自分の open-2（二連）
W_OPP_O3       = 900    # 相手の open-3 を嫌う
W_DOUBLE_TH    = 5000   # 自分の即勝てる手が2つ以上
W_CENTER2D     = 4      # 2D中心寄りの微加点
W_LAYER_CENTER = 40     # z=1,2 の中央4

CORNERS = [(0,0),(3,0),(0,3),(3,3)]
CENTERS_2D = {(1,1),(2,1),(1,2),(2,2)}
EDGES = [(1,0),(2,0),(0,1),(3,1),(0,2),(3,2),(1,3),(2,3)]

# ---------------- 基本ユーティリティ ----------------
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    # 重力: z=0(最下)→z=3(最上)
    for z in range(SIZE):
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    ms: List[Coord2] = []
    top = SIZE-1
    for y in range(SIZE):
        for x in range(SIZE):
            if board[top][y][x] == 0:
                ms.append((x,y))
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
    return sum(1 for z in range(SIZE) for y in range(SIZE) for x in range(SIZE) if board[z][y][x] != 0)

# ---------------- 勝ち筋（76本） ----------------
def generate_lines() -> List[List[Coord3]]:
    L: List[List[Coord3]] = []
    # x / y / z 直線
    for z in range(SIZE):
        for y in range(SIZE):
            L.append([(x,y,z) for x in range(SIZE)])
    for z in range(SIZE):
        for x in range(SIZE):
            L.append([(x,y,z) for y in range(SIZE)])
    for y in range(SIZE):
        for x in range(SIZE):
            L.append([(x,y,z) for z in range(SIZE)])
    # 各面の斜め
    for z in range(SIZE):
        L.append([(i,i,z) for i in range(SIZE)])
        L.append([(i,SIZE-1-i,z) for i in range(SIZE)])
    for y in range(SIZE):
        L.append([(i,y,i) for i in range(SIZE)])
        L.append([(i,y,SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        L.append([(x,i,i) for i in range(SIZE)])
        L.append([(x,i,SIZE-1-i) for i in range(SIZE)])
    # 空間対角 4 本
    L.append([(i,i,i) for i in range(SIZE)])
    L.append([(i,i,SIZE-1-i) for i in range(SIZE)])
    L.append([(i,SIZE-1-i,i) for i in range(SIZE)])
    L.append([(SIZE-1-i,i,i) for i in range(SIZE)])
    return L

ALL_LINES = generate_lines()

# ---------------- 終局・即勝ち ----------------
def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    lx,ly,lz = last
    for line in ALL_LINES:
        if any((x==lx and y==ly and z==lz) for (x,y,z) in line):
            if all(board[z][y][x] == player for (x,y,z) in line):
                return True
    return False

def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None:
        return False
    ok = check_win_at(board, player, (x,y,z))
    undo_place(board, x, y, z)
    return ok

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    return [(x,y) for (x,y) in valid_xy_moves(board) if is_winning_after(board, player, x, y)]

# ---------------- open-2 / open-3 カウント ----------------
def lowest_ok(board: Board, e: Coord3) -> bool:
    x,y,z = e
    ze = lowest_empty_z(board, x, y)
    return (ze is not None) and (ze == z)

def count_open2_open3(board: Board, player: int) -> Tuple[int,int]:
    you = 3 - player
    o2 = o3 = 0
    for line in ALL_LINES:
        pm = py = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == player: pm += 1
            elif v == you:  py += 1
            else: empties.append((x,y,z))
        if py != 0:  # 相手石混入ラインは対象外
            continue
        if pm == 2 and len(empties) == 2 and lowest_ok(board, empties[0]) and lowest_ok(board, empties[1]):
            o2 += 1
        elif pm == 3 and len(empties) == 1 and lowest_ok(board, empties[0]):
            o3 += 1
    return o2, o3

# ---------------- 危険度チェック ----------------
def opponent_immediate_wins_after(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    z = place_inplace(board, x, y, me)
    cnt = len(immediate_winning_squares(board, you))
    undo_place(board, x, y, z)
    return cnt

# ---------------- 序盤の角優先（安全フィルタ付き） ----------------
def corner_first_opening(board: Board, me: int) -> Optional[Coord2]:
    n = stones_count(board)
    moves = valid_xy_moves(board)
    if not moves:
        return None

    # 先手初手：角を最優先（安全なら）
    if n == 0 and me == 1:
        for (x,y) in CORNERS:
            if (x,y) in moves and opponent_immediate_wins_after(board, me, x, y) == 0:
                return (x,y)
        # 角が危険または埋まっている → 2D中央へ
        for (x,y) in [(1,1),(2,1),(1,2),(2,2)]:
            if (x,y) in moves and opponent_immediate_wins_after(board, me, x, y) == 0:
                return (x,y)
        return None

    # 後手2手目：相手が角→対角／相手が中央寄り→中央
    if n == 1 and me == 2:
        # 相手の最初の石の (x,y) を探す
        opx = opy = None
        for y in range(SIZE):
            for x in range(SIZE):
                for z in range(SIZE):
                    if board[z][y][x] == 1:
                        opx, opy = x, y
                        break
                if opx is not None: break
            if opx is not None: break
        if opx is not None:
            # 角には対角
            if (opx,opy) in CORNERS:
                dx,dy = (3-opx, 3-opy)
                if (dx,dy) in moves and opponent_immediate_wins_after(board, me, dx, dy) == 0:
                    return (dx,dy)
            # 中央寄りには中央4
            if (opx,opy) in CENTERS_2D:
                for (cx,cy) in [(1,1),(2,1),(1,2),(2,2)]:
                    if (cx,cy) in moves and opponent_immediate_wins_after(board, me, cx, cy) == 0:
                        return (cx,cy)
    # 序盤（総手数 <= 6 手）: 空いていれば角を優先（安全フィルタ）
    if n <= 6:
        for (x,y) in CORNERS:
            if (x,y) in moves and opponent_immediate_wins_after(board, me, x, y) == 0:
                return (x,y)
    return None

# ---------------- 1手仮置きの軽量評価 ----------------
def move_score(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    # 自分の即勝ちは最大
    if is_winning_after(board, me, x, y):
        return WIN_SCORE

    z = place_inplace(board, x, y, me)

    my_o2, my_o3 = count_open2_open3(board, me)
    opp_o2, opp_o3 = count_open2_open3(board, you)

    # 2D中心寄り・中層中央4 の微ボーナス
    center = 0
    if (x,y) in CENTERS_2D and z in (1,2):
        center += W_LAYER_CENTER
    cx = abs(1.5 - x); cy = abs(1.5 - y)
    center += int(W_CENTER2D * (1.5 - (cx+cy)/2))

    # ダブルリーチ（即勝ち手の本数）
    my_kill = len(immediate_winning_squares(board, me))

    undo_place(board, x, y, z)

    score  = 0
    score += W_MY_O2 * my_o2 + W_MY_O3 * my_o3 + (W_DOUBLE_TH if my_kill >= 2 else 0)
    score -= W_OPP_O3 * opp_o3
    score += center
    return score

# ---------------- ルート選択（安全フィルタ込み） ----------------
def choose_best(board: Board, me: int) -> Coord2:
    start = time.perf_counter()
    moves = valid_xy_moves(board)
    if not moves:
        return (0,0)

    you = 3 - me

    # 0) 自分の即勝ち
    for (x,y) in moves:
        if is_winning_after(board, me, x, y):
            return (x,y)

    # 1) 相手の即勝ちブロック（自分が即勝ちできない場合は最優先）
    opp_now = immediate_winning_squares(board, you)
    if opp_now:
        best = None
        best_after = 10**9
        best_sc = -10**9
        for (x,y) in moves:
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            sc = move_score(board, me, x, y)
            undo_place(board, x, y, z)
            if after < best_after or (after == best_after and sc > best_sc):
                best_after, best_sc = after, sc
                best = (x,y)
                if after == 0:
                    return best
        return best if best is not None else moves[0]

    # 2) 角優先の序盤定石（安全フィルタ付き）
    mv = corner_first_opening(board, me)
    if mv is not None:
        return mv

    # 3) 危険手フィルタ：自分が打った直後に相手の即勝ちが発生する案は却下
    safe_moves: List[Coord2] = []
    for (x,y) in moves:
        if opponent_immediate_wins_after(board, me, x, y) == 0:
            safe_moves.append((x,y))
    cand = safe_moves if safe_moves else moves  # 全滅なら最小被害へ

    # 4) 軽量評価で最大化
    best = cand[0]
    best_sc = -10**9
    for (x,y) in cand:
        sc = move_score(board, me, x, y)
        if sc > best_sc:
            best_sc = sc
            best = (x,y)

    # 時間安全ガード
    if time.perf_counter() - start > TIME_BUDGET_SEC:
        return best
    return best

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        try:
            mv = choose_best(board, player)
            # 念のため、常に合法手を返す
            moves = valid_xy_moves(board)
            return mv if mv in moves else (moves[0] if moves else (0,0))
        except Exception:
            moves = valid_xy_moves(board)
            return moves[0] if moves else (0,0)

AI = MyAI()
