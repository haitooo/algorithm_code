# main.py
# 4x4x4 立体四目並べ AI（即負けブロック徹底 & 危険手却下 版）
# 優先度:
#  1) 自分の即勝ち
#  2) 相手の即勝ちブロック（先手でも最優先 / after==0 を最重視）
#  3) 初手/応手の定石（先手=角 / 後手=角なら対角，中央寄りなら中央）
#  4) 危険手フィルタ（自手後に相手の即勝ち or リーチが出る案は却下）
#  5) 軽量評価（先手=攻め, 後手=守り）
#
# 重力: z=0 が最下層。lowest_empty_z は z=0→3 の順で探索＝下から置く。

from typing import List, Tuple, Optional
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4

# 評価重み
WIN_SCORE        = 1_000_000
W_DOUBLE_THREAT  = 6000
W_MY_OPEN3       = 800
W_MY_OPEN2       = 220
W_OPP_OPEN3      = 900
W_OPP_DTHREAT    = 7000
W_CENTER_LAYER   = 120
W_CORNER         = 80
W_EDGE           = 40
W_CENTER_2D      = 6

CENTERS_2D = {(1,1), (2,1), (1,2), (2,2)}
CORNERS_2D = {(0,0), (3,0), (0,3), (3,3)}
EDGES_2D   = {(1,0),(2,0),(0,1),(3,1),(0,2),(3,2),(1,3),(2,3)}

# ---------------- 基本ユーティリティ ----------------
def clone(board: Board) -> Board:
    return [[row[:] for row in plane] for plane in board]

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    # 重力: 下(z=0)から上(z=3)へ詰める
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
    return sum(1 for z in range(SIZE) for y in range(SIZE) for x in range(SIZE) if board[z][y][x] != 0)

# --- 勝ち筋（76本） ---
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
    # 各面の斜め
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

# --- 勝ち判定／即勝ち手 ---
def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    lx, ly, lz = last
    for line in ALL_LINES:
        if any((x==lx and y==ly and z==lz) for (x,y,z) in line):
            if all(board[z][y][x] == player for (x,y,z) in line):
                return True
    return False

def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None:
        return False
    win = check_win_at(board, player, (x, y, z))
    undo_place(board, x, y, z)
    return win

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    return [(x, y) for (x, y) in valid_xy_moves(board) if is_winning_after(board, player, x, y)]

# --- open2/open3 と ダブルリーチ ---
def is_legal_empty(board: Board, e: Coord3) -> bool:
    x, y, z = e
    ze = lowest_empty_z(board, x, y)
    return (ze is not None) and (ze == z)

def count_open2_open3(board: Board, player: int) -> Tuple[int, int]:
    """(open2数, open3数) 自分石のみライン・合法空セルで成立。"""
    you = 3 - player
    o2 = o3 = 0
    for line in ALL_LINES:
        me_cnt = you_cnt = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == player: me_cnt += 1
            elif v == you:  you_cnt += 1
            else: empties.append((x,y,z))
        if you_cnt != 0:
            continue
        if me_cnt == 2 and len(empties) == 2:
            if is_legal_empty(board, empties[0]) and is_legal_empty(board, empties[1]):
                o2 += 1
        elif me_cnt == 3 and len(empties) == 1:
            if is_legal_empty(board, empties[0]):
                o3 += 1
    return o2, o3

def count_double_threat(board: Board, player: int) -> int:
    """即勝ち手の数（>=2でダブルリーチ扱い）。"""
    return len(immediate_winning_squares(board, player))

# ---------------- 定石（初手/応手） ----------------
def opening_first_move(board: Board, player: int) -> Optional[Coord2]:
    n = stones_count(board)
    if n == 0 and player == 1:
        for (x,y) in [(0,0),(3,3),(0,3),(3,0)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
    if n == 1 and player == 2:
        # 先手(=1)の最初の石の (x,y) を検出（zは問わない）
        opx = opy = None
        for y in range(SIZE):
            for x in range(SIZE):
                for z in range(SIZE):
                    if board[z][y][x] == 1:
                        opx, opy = x, y
                        break
                if opx is not None:
                    break
            if opx is not None:
                break
        if opx is not None:
            # 角なら対角
            if (opx, opy) in CORNERS_2D:
                dx, dy = (3-opx, 3-opy)
                if lowest_empty_z(board, dx, dy) is not None:
                    return (dx, dy)
            # 中央寄りなら中央4マスから取得
            if (opx, opy) in CENTERS_2D:
                for (cx, cy) in [(1,1),(2,1),(1,2),(2,2)]:
                    if lowest_empty_z(board, cx, cy) is not None:
                        return (cx, cy)
            # それ以外: 中央 > 角 > 辺
            for (cx, cy) in [(1,1),(2,1),(1,2),(2,2)]:
                if lowest_empty_z(board, cx, cy) is not None:
                    return (cx, cy)
            for (x2,y2) in [(0,0),(3,3),(0,3),(3,0)]:
                if lowest_empty_z(board, x2, y2) is not None:
                    return (x2, y2)
            for (x2,y2) in EDGES_2D:
                if lowest_empty_z(board, x2, y2) is not None:
                    return (x2, y2)
    return None

# ---------------- 層/位置ボーナス ----------------
def layer_position_bonus(board: Board, x: int, y: int, z: int) -> int:
    sc = 0
    if z in (1,2) and (x, y) in CENTERS_2D:
        sc += W_CENTER_LAYER
    if z in (0,3):
        if (x, y) in CORNERS_2D: sc += W_CORNER
        elif (x, y) in EDGES_2D: sc += W_EDGE
    cx = abs(1.5 - x); cy = abs(1.5 - y)
    sc += int(W_CENTER_2D * (1.5 - (cx + cy) / 2))
    return sc

# ---------------- 危険判定（自手後） ----------------
def opponent_immediate_wins_after(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    z = place_inplace(board, x, y, me)
    cnt = len(immediate_winning_squares(board, you))
    undo_place(board, x, y, z)
    return cnt

def opponent_open3_after(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    z = place_inplace(board, x, y, me)
    _, o3 = count_open2_open3(board, you)
    undo_place(board, x, y, z)
    return o3

# ---------------- 軽量評価（先攻=攻め / 後攻=守り） ----------------
def move_score(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    if is_winning_after(board, me, x, y):
        return WIN_SCORE

    z = place_inplace(board, x, y, me)

    my_o2, my_o3 = count_open2_open3(board, me)
    my_dth = count_double_threat(board, me)

    opp_o2, opp_o3 = count_open2_open3(board, you)
    opp_dth = count_double_threat(board, you)

    pos_bonus = layer_position_bonus(board, x, y, z)

    undo_place(board, x, y, z)

    if me == 1:  # 先手=攻め
        score = (
            pos_bonus
            + W_MY_OPEN2 * my_o2
            + W_MY_OPEN3 * my_o3
            + (W_DOUBLE_THREAT if my_dth >= 2 else 0)
            - (W_OPP_OPEN3 // 2) * opp_o3
            - (W_OPP_DTHREAT // 2 if opp_dth >= 2 else 0)
        )
    else:        # 後手=守り
        score = (
            pos_bonus
            + (W_MY_OPEN2 // 2) * my_o2
            + (W_MY_OPEN3 // 2) * my_o3
            + ((W_DOUBLE_THREAT // 2) if my_dth >= 2 else 0)
            - W_OPP_OPEN3 * opp_o3
            - (W_OPP_DTHREAT if opp_dth >= 2 else 0)
        )
    return score

# ---------------- ルート選択 ----------------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves:
        return (0, 0)  # 盤面が満杯の保険

    you = 3 - me

    # 1) 自分の即勝ち
    for (x, y) in moves:
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちブロック（先手でも最優先）
    opp_now = immediate_winning_squares(board, you)
    if opp_now:
        best_move = None
        best_after = 10**9
        best_score = -10**9
        for (x, y) in moves:
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            sc = move_score(board, me, x, y)
            undo_place(board, x, y, z)
            # 最優先: after==0（相手の即勝ちを全消し）
            if after < best_after or (after == best_after and sc > best_score):
                best_after, best_score = after, sc
                best_move = (x, y)
                if after == 0:
                    return best_move
        # どうしても0にできない場合は最小 after を選ぶ
        return best_move if best_move is not None else moves[0]

    # 3) 初手/応手の定石（ただし危険手は使わない）
    mv = opening_first_move(board, me)
    if mv is not None:
        x, y = mv
        if is_winning_after(board, me, x, y):
            return mv
        if opponent_immediate_wins_after(board, me, x, y) == 0 and opponent_open3_after(board, me, x, y) == 0:
            return mv
        # 定石が危険ならスキップして通常選択へ

    # 4) 危険手フィルタ
    safe_moves: List[Coord2] = []
    for (x, y) in moves:
        # 自分即勝ちはすでに捕捉済みなのでここでは純粋に危険判定
        if opponent_immediate_wins_after(board, me, x, y) > 0:
            continue  # “打ったら即負け”になる案は却下
        if opponent_open3_after(board, me, x, y) > 0:
            continue  # リーチを与える案も原則却下
        safe_moves.append((x, y))

    target_moves = safe_moves
    if not target_moves:
        # 全滅なら被害最小の案（即負け発生数 → open-3発生数 → 自分評価）
        best = moves[0]
        best_key = (10**9, 10**9, -10**9)  # (opp_immediate, opp_open3, -self_score) を最小化
        for (x, y) in moves:
            oi = opponent_immediate_wins_after(board, me, x, y)
            o3 = opponent_open3_after(board, me, x, y)
            sc = move_score(board, me, x, y)
            key = (oi, o3, -sc)
            if key < best_key:
                best_key = key
                best = (x, y)
        return best

    # 5) 軽量評価でベスト
    best = target_moves[0]
    best_sc = -10**9
    for (x, y) in target_moves:
        sc = move_score(board, me, x, y)
        if sc > best_sc:
            best_sc = sc
            best = (x, y)
    return best

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player)

AI = MyAI()
