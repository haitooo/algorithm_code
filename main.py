# main.py
# 4x4x4 立体四目並べ AI
# 仕様:
# - 最優先: 即勝ち / 即負けブロック
# - 初手定石: 先手は角。後手は相手が角→対角 / 相手が中央寄り→中央を取る
# - 層/位置の定石:
#     * z=1,2（2層,3層）は中央4マス(1,1),(2,1),(1,2),(2,2)を積極的に
#     * z=0,3（最下/最上層）は角優先→次に辺
# - 先手は攻め（ダブルリーチ＝即勝ち手二つ以上を重視）、後手は守り（相手脅威の最小化）
# - 重い探索はしない（1手仮置き評価＋簡易指向）→ 10秒制限対応

from typing import List, Tuple, Optional
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4

# 評価重み（軽量ヒューリスティクス）
WIN_SCORE        = 1_000_000
W_DOUBLE_THREAT  = 6_000   # 自分の即勝ち手が2つ以上
W_MY_OPEN3       = 800     # 自分のリーチ(3連+合法空1)
W_MY_OPEN2       = 220     # 自分の二連(2連+合法空2) → ダブルチェイン狙い
W_OPP_OPEN3      = 900     # 相手リーチは大きく減点（防御）
W_OPP_DTHREAT    = 7_000   # 相手ダブルリーチはさらに大きく減点
W_CENTER_LAYER   = 120     # z=1,2の中央4マスボーナス
W_CORNER         = 80      # z=0,3 の角
W_EDGE           = 40      # z=0,3 の辺
W_CENTER_2D      = 6       # 2Dの中央寄り（軽めタイブレーク）

# 位置集合
CENTERS_2D = {(1,1), (2,1), (1,2), (2,2)}
CORNERS_2D = {(0,0), (3,0), (0,3), (3,3)}
EDGES_2D   = {(1,0), (2,0), (0,1), (3,1), (0,2), (3,2), (1,3), (2,3)}  # 角以外の外周

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
    return sum(1 for z in range(SIZE) for y in range(SIZE) for x in range(SIZE) if board[z][y][x] != 0)

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
        hit = False
        for (x, y, z) in line:
            if x == lx and y == ly and z == lz:
                hit = True
                break
        if hit and all(board[z][y][x] == player for (x, y, z) in line):
            return True
    return False

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    return [(x, y) for (x, y) in valid_xy_moves(board) if is_winning_after(board, player, x, y)]

# ---------------- ライン評価（open2/open3 & ダブルリーチ） ----------------
def is_legal_empty(board: Board, e: Coord3) -> bool:
    x, y, z = e
    ze = lowest_empty_z(board, x, y)
    return (ze is not None) and (ze == z)

def count_open2_open3(board: Board, player: int) -> Tuple[int, int]:
    """(open2数, open3数) 自分石のみのラインに限定。合法空セルで成立。"""
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
    """即勝ち手の数（>=2 でダブルリーチとみなす）"""
    return len(immediate_winning_squares(board, player))

# ---------------- 定石（初手/応手 & 層・位置優先） ----------------
def is_centerish(x: int, y: int) -> bool:
    return (x, y) in CENTERS_2D

def opening_first_move(board: Board, player: int) -> Optional[Coord2]:
    """初手: 先手は角、後手は相手に応じて対角/中央"""
    n = stones_count(board)
    if n == 0 and player == 1:
        # 置ける角に置く（(0,0)優先→他角）
        for (x,y) in [(0,0), (3,3), (0,3), (3,0)]:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
    if n == 1 and player == 2:
        # 相手の着手を読む
        for y in range(SIZE):
            for x in range(SIZE):
                if board[0][y][x] != 0:  # 最下層z=0に相手がいるはず
                    opx, opy = x, y
                    # 相手が角→対角
                    if (opx, opy) in CORNERS_2D:
                        dx, dy = (3-opx, 3-opy)
                        if lowest_empty_z(board, dx, dy) is not None:
                            return (dx, dy)
                    # 相手が中央寄り→中央を取る
                    if is_centerish(opx, opy):
                        for (cx, cy) in [(1,1),(2,1),(1,2),(2,2)]:
                            if lowest_empty_z(board, cx, cy) is not None:
                                return (cx, cy)
                    # それ以外→中央 > 角 > 辺の順で
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

def layer_position_bonus(board: Board, x: int, y: int, z: int) -> int:
    """層/位置の定石ボーナス"""
    sc = 0
    if z in (1, 2) and (x, y) in CENTERS_2D:
        sc += W_CENTER_LAYER
    if z in (0, 3):
        if (x, y) in CORNERS_2D: sc += W_CORNER
        elif (x, y) in EDGES_2D: sc += W_EDGE
    # 2D中心寄りの微調整（常時）
    cx = abs(1.5 - x); cy = abs(1.5 - y)
    sc += int(W_CENTER_2D * (1.5 - (cx + cy) / 2))
    return sc

# ---------------- 1手仮置き評価（攻め/守りの指向を反映） ----------------
def move_score(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me

    # 即勝ちは別ハンドリングだが保険で巨大加点
    if is_winning_after(board, me, x, y):
        return WIN_SCORE

    z = place_inplace(board, x, y, me)

    # 自分の open2/open3 とダブルリーチ
    my_o2, my_o3 = count_open2_open3(board, me)
    my_dth = count_double_threat(board, me)

    # 相手の open3 とダブルリーチ（この手でどの程度減らせたかを間接的に反映）
    opp_o2, opp_o3 = count_open2_open3(board, you)
    opp_dth = count_double_threat(board, you)

    # 位置（層/座標）ボーナス
    pos_bonus = layer_position_bonus(board, x, y, z)

    undo_place(board, x, y, z)

    # 先手＝攻め寄り、後手＝守り寄りの配合
    # me==1 → 攻め強化：自分指標を重く、相手指標は軽めに抑止
    # me==2 → 守り強化：相手指標の抑止を重く
    if me == 1:
        score = (
            pos_bonus
            + W_MY_OPEN2 * my_o2
            + W_MY_OPEN3 * my_o3
            + (W_DOUBLE_THREAT if my_dth >= 2 else 0)
            - (W_OPP_OPEN3 // 2) * opp_o3
            - (W_OPP_DTHREAT // 2 if opp_dth >= 2 else 0)
        )
    else:
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
    # 0) 初手/応手の定石
    mv = opening_first_move(board, me)
    if mv is not None:
        return mv

    # 1) 自分の即勝ち
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちブロック（全列挙→完全に消す手を最優先）
    you = 3 - me
    opp_wins = immediate_winning_squares(board, you)
    if opp_wins:
        best_move = None
        best_after = 1_000_000
        best_score = -1_000_000
        for (bx, by) in valid_xy_moves(board):
            z = place_inplace(board, bx, by, me)
            after = len(immediate_winning_squares(board, you))
            sc = move_score(board, me, bx, by)  # ブロックの中でより良い手
            undo_place(board, bx, by, z)
            # afterが最小（まず0を狙う）、同率ならスコアで比較
            if after < best_after or (after == best_after and sc > best_score):
                best_after = after
                best_score = sc
                best_move = (bx, by)
                if after == 0:
                    # 完全ブロックが見つかったら即採用
                    return best_move
        if best_move is not None:
            return best_move

    # 3) 一手仮置き評価でベスト（先手は攻め寄り / 後手は守り寄り）
    best = None
    best_sc = -10**9
    for (x, y) in valid_xy_moves(board):
        sc = move_score(board, me, x, y)
        if sc > best_sc:
            best_sc = sc
            best = (x, y)

    # 4) 念のためのフォールバック（中央寄り優先）
    if best is None:
        cx_order = sorted(valid_xy_moves(board), key=lambda p: (abs(1.5-p[0]) + abs(1.5-p[1])))
        best = cx_order[0]
    return best

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player)

AI = MyAI()
