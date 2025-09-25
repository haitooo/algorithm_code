# main.py
# 4x4x4 立体四目並べ AI（軽量リセット版）
# 方針:
#  - 即勝ち、即負けブロック、序盤定石は従来どおり
#  - 以降は「自分の二連(open-2)を最大化」する貪欲選択でダブルチェインを狙う
#  - 重い全探索はしない（1手だけ仮置き評価）

from typing import List, Tuple, Optional

from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4

# 軽量ヒューリスティック定数
WIN_SCORE       = 1_000_000
OPEN2_WEIGHT    = 200          # 自分の open-2 の重み（ダブルチェイン狙いの主眼）
OPEN3_WEIGHT    = 800          # 自分の open-3（リーチ）重み（即勝ちの一歩手前）
BLOCK_OPEN3_W   = 700          # 相手の open-3 を減らす効果（ブロック価値）
CENTER_BONUS    = 4            # 中央寄り
TPOINT_BONUS    = 60           # t点(z==3) の二連/三連に少し上乗せ

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
        # last を含むラインだけ確認
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

# ---------------- open-2 / open-3 計数（重力を考慮） ----------------
def line_status(board: Board, player: int, line: List[Coord3]) -> Tuple[int,int,List[Coord3]]:
    """そのライン内の (自分石数, 相手石数, 空きセル一覧) を返す"""
    you = 3 - player
    pm = py = 0
    empties: List[Coord3] = []
    for (x,y,z) in line:
        v = board[z][y][x]
        if v == player: pm += 1
        elif v == you:  py += 1
        else: empties.append((x,y,z))
    return pm, py, empties

def is_legal_empty(board: Board, e: Coord3) -> bool:
    x, y, z = e
    ze = lowest_empty_z(board, x, y)
    return (ze is not None) and (ze == z)

def count_open2_open3(board: Board, player: int) -> Tuple[int, int, int, int]:
    """
    戻り値: (open2数, t点open2数, open3数, t点open3数)
      open2: 自分2石+空2、相手石0、かつ空2マスが合法位置（重力OK）
      open3: 自分3石+空1、相手石0、かつ空1マスが合法位置
    """
    open2 = open2_t = open3 = open3_t = 0
    for line in ALL_LINES:
        me, you, empties = line_status(board, player, line)
        if you != 0:
            continue
        if me == 2 and len(empties) == 2:
            # 2つの空きが両方合法に打てる列か
            legal = is_legal_empty(board, empties[0]) and is_legal_empty(board, empties[1])
            if legal:
                open2 += 1
                # t点（いずれかが z==3）を少し評価
                if empties[0][2] == 3 or empties[1][2] == 3:
                    open2_t += 1
        elif me == 3 and len(empties) == 1:
            if is_legal_empty(board, empties[0]):
                open3 += 1
                if empties[0][2] == 3:
                    open3_t += 1
    return open2, open2_t, open3, open3_t

# ---------------- 定石（最初の3手：そのまま） ----------------
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

# ---------------- 1手仮置き評価（ダブルチェイン重視） ----------------
def move_score(board: Board, me: int, x: int, y: int) -> int:
    """
    1手だけ仮置きして、二連(open-2)の“数”を最優先で評価。
    タイブレークに open-3（自分優先・相手妨害）、中心寄り、t点ボーナス。
    """
    you = 3 - me

    # 即勝ちは別ハンドリングだが、ここでも一応巨大加点（保険）
    if is_winning_after(board, me, x, y):
        return WIN_SCORE

    z = place_inplace(board, x, y, me)

    # 自分の open-2 / open-3
    my_o2, my_o2_t, my_o3, my_o3_t = count_open2_open3(board, me)
    # 相手の open-3（こちらの着手により減っていれば、その分を加点として扱う）
    opp_o2, opp_o2_t, opp_o3, opp_o3_t = count_open2_open3(board, you)

    undo_place(board, x, y, z)

    # 中央ボーナス（軽め）
    cx = abs(1.5 - x)
    cy = abs(1.5 - y)
    center = int(CENTER_BONUS * (1.5 - (cx + cy) / 2))

    # 総合スコア
    score  = 0
    score += OPEN2_WEIGHT * my_o2 + TPOINT_BONUS * my_o2_t
    score += OPEN3_WEIGHT * my_o3 + (TPOINT_BONUS // 2) * my_o3_t
    score += center
    # 相手 open-3 が多い局面は危険 → 減点（=ブロック価値）
    score -= BLOCK_OPEN3_W * opp_o3
    score -= (TPOINT_BONUS // 2) * opp_o3_t

    return score

# ---------------- ルート選択 ----------------
def choose_best(board: Board, me: int) -> Coord2:
    # 0) 定石
    mv = opening_move(board, me)
    if mv is not None:
        return mv

    # 1) 自分の即勝ち
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, me, x, y):
            return (x, y)

    # 2) 相手の即勝ちブロック（相手の1手勝ち手があれば最優先で消す）
    you = 3 - me
    opp_wins = immediate_winning_squares(board, you)
    if opp_wins:
        # 複数ある場合は、ブロックしたうえで自分のスコアが最大になる手を選ぶ
        best_move = None
        best_val  = -10**9
        for (bx, by) in opp_wins:
            # ブロック可能か確認（その列が合法に置ける）
            if lowest_empty_z(board, bx, by) is None:
                continue
            val = move_score(board, me, bx, by)
            if val > best_val:
                best_val = val
                best_move = (bx, by)
        if best_move is not None:
            return best_move

    # 3) ダブルチェイン狙い：open-2 の“数”を最大化する手を選ぶ
    best = None
    best_sc = -10**9
    for (x, y) in valid_xy_moves(board):
        sc = move_score(board, me, x, y)
        if sc > best_sc:
            best_sc = sc
            best = (x, y)

    # 4) 念のためのフォールバック（中央寄り）
    if best is None:
        cx_order = sorted(valid_xy_moves(board), key=lambda p: (abs(1.5-p[0]) + abs(1.5-p[1])))
        best = cx_order[0]
    return best

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player)

AI = MyAI()
