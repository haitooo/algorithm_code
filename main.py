# main.py
# 4x4x4 立体四目並べ AI
# 仕様（ユーザー疑似コードを忠実にPython化）:
# 1) 自分の即勝ちがあれば即それを打つ
# 2) 相手の即勝ちがあれば最優先でブロック（自分の即勝ちがない場合）
# 3) 角 {0,4,D,G} の 1層目(z==0) に置けるなら、その中から置く
# 4) 中央 {6,7,A,B} の 2〜3層(z==1 or 2) に置けるなら、その中から置く
# 5) 辺 {1,2,5,9,8,C,E,F} で 2〜3層に置け、かつその列(行/列)の両端角のどちらかに自分石があるなら、そこに置く
# 6) {6,7,A,B} で置けるところがあれば、そこに置く
# 7) 以上で決まらなければ合法手のうち適当（中央寄り）を返す
#
# 盤面のラベル対応（ユーザー指定の上面ビュー）:
#   y=0: 0 1 2 4
#   y=1: 5 6 7 8
#   y=2: 9 A B C
#   y=3: D E F G
# → 角: 0=(0,0), 4=(3,0), D=(0,3), G=(3,3)
# → 中央: 6=(1,1), 7=(2,1), A=(1,2), B=(2,2)
# → 辺: 1,2,5,9,8,C,E,F（行・列の両端角とペア）

from typing import List, Tuple, Optional
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)
SIZE = 4

# ---------- 基本ユーティリティ ----------
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    """重力: z=0(最下)から順に空きを探す。置けなければ None。"""
    for z in range(SIZE):
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    """上面(z=3)が空の列(x,y)を合法手として返す。"""
    res: List[Coord2] = []
    top = SIZE - 1
    for y in range(SIZE):
        for x in range(SIZE):
            if board[top][y][x] == 0:
                res.append((x, y))
    return res

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def undo_place(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

# ---------- 勝ち筋（76本） ----------
def generate_lines() -> List[List[Coord3]]:
    L: List[List[Coord3]] = []
    # x直線 / y直線 / z直線
    for z in range(SIZE):
        for y in range(SIZE):
            L.append([(x, y, z) for x in range(SIZE)])
    for z in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for y in range(SIZE)])
    for y in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for z in range(SIZE)])
    # 各面の斜め
    for z in range(SIZE):
        L.append([(i, i, z) for i in range(SIZE)])
        L.append([(i, SIZE-1-i, z) for i in range(SIZE)])
    for y in range(SIZE):
        L.append([(i, y, i) for i in range(SIZE)])
        L.append([(i, y, SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        L.append([(x, i, i) for i in range(SIZE)])
        L.append([(x, i, SIZE-1-i) for i in range(SIZE)])
    # 空間対角 4 本
    L.append([(i, i, i) for i in range(SIZE)])
    L.append([(i, i, SIZE-1-i) for i in range(SIZE)])
    L.append([(i, SIZE-1-i, i) for i in range(SIZE)])
    L.append([(SIZE-1-i, i, i) for i in range(SIZE)])
    return L

ALL_LINES = generate_lines()

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
    ok = check_win_at(board, player, (x, y, z))
    undo_place(board, x, y, z)
    return ok

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    """そのプレイヤーが今置いて勝てる(x,y)の一覧（自分手番として評価）。"""
    res: List[Coord2] = []
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, player, x, y):
            res.append((x, y))
    return res

# ---------- ラベル→座標 ----------
# ユーザー指定の上面ラベル表:
# y=0: 0 1 2 4
# y=1: 5 6 7 8
# y=2: 9 A B C
# y=3: D E F G
LABEL_GRID = [
    ["0","1","2","4"],
    ["5","6","7","8"],
    ["9","A","B","C"],
    ["D","E","F","G"],
]
LABEL_POS: dict[str, Coord2] = {LABEL_GRID[y][x]:(x,y) for y in range(4) for x in range(4)}

# 角・中央・辺の集合
CORNERS = [LABEL_POS["0"], LABEL_POS["4"], LABEL_POS["D"], LABEL_POS["G"]]
CENTERS = [LABEL_POS["6"], LABEL_POS["7"], LABEL_POS["A"], LABEL_POS["B"]]
EDGES   = [LABEL_POS[k] for k in ["1","2","5","9","8","C","E","F"]]

def row_col_corners_for_edge(x: int, y: int) -> List[Coord2]:
    """辺セル (x,y) に対応する『その列(行/列)の角』2点を返す。"""
    res: List[Coord2] = []
    if y == 0:      # 上辺 → 行の両端角 = (0,0),(3,0)
        res.append((0,0)); res.append((3,0))
    if y == 3:      # 下辺 → (0,3),(3,3)
        res.append((0,3)); res.append((3,3))
    if x == 0:      # 左辺 → 列の両端角 = (0,0),(0,3)
        res.append((0,0)); res.append((0,3))
    if x == 3:      # 右辺 → (3,0),(3,3)
        res.append((3,0)); res.append((3,3))
    # 重複除去
    uniq = []
    for p in res:
        if p not in uniq:
            uniq.append(p)
    return uniq

def column_has_my_stone(board: Board, me: int, x: int, y: int) -> bool:
    """(x,y) 縦列（z=0..3）のどこかに自分石があるか。"""
    for z in range(SIZE):
        if board[z][y][x] == me:
            return True
    return False

# ---------- ルール実装（choose_best） ----------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves:
        return (0, 0)

    you = 3 - me

    # 1) 自分の即勝ち
    for (x, y) in moves:
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちを最優先でブロック（※自分の即勝ちが無いとき）
    opp_now = immediate_winning_squares(board, you)
    if opp_now:
        # after==0 を最優先
        best = None
        best_after = 10**9
        for (x, y) in moves:
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            undo_place(board, x, y, z)
            if after == 0:
                return (x, y)
            if after < best_after:
                best_after = after
                best = (x, y)
        return best if best is not None else moves[0]

    # 3) 角 {0,4,D,G} の 1層目に置けるなら
    for (x, y) in CORNERS:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z == 0:
                return (x, y)

    # 4) 中央 {6,7,A,B} の 2〜3層に置けるなら
    for (x, y) in CENTERS:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                return (x, y)

    # 5) 辺 {1,2,5,9,8,C,E,F}:
    #    2〜3層に置けて、かつその行/列の両端角のどちらかの縦列に自分石があるなら置く
    for (x, y) in EDGES:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                corners = row_col_corners_for_edge(x, y)
                if any(column_has_my_stone(board, me, cx, cy) for (cx, cy) in corners):
                    return (x, y)

    # 6) 中央 {6,7,A,B} がどれか置けるなら（層は問わず）
    for (x, y) in CENTERS:
        if (x, y) in moves:
            return (x, y)

    # 7) フォールバック：中央寄り > その他
    moves_sorted = sorted(moves, key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
    return moves_sorted[0] if moves_sorted else moves[0]

# ---------- エンジン ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        # 例外時も必ず合法手を返す保険
        try:
            mv = choose_best(board, player)
            return mv if mv in valid_xy_moves(board) else (valid_xy_moves(board)[0] if valid_xy_moves(board) else (0, 0))
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0, 0)

AI = MyAI()
