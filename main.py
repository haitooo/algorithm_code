# main.py
# 4x4x4 立体四目並べ AI（全76ライン精査＋“側面ターゲット形”＋逆ミッキー優先）
from typing import List, Tuple, Optional, Dict, Set
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)
SIZE = 4

# ---------- 基本ユーティリティ ----------
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):              # 重力: z=0(最下)→上
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    res: List[Coord2] = []
    top = SIZE - 1
    for y in range(SIZE):
        for x in range(SIZE):
            if board[top][y][x] == 0:
                res.append((x, y))
    return res

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None: return None
    board[z][y][x] = player
    return z

def undo_place(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

# ---------- 勝ち筋（76本） ----------
def generate_lines() -> List[List[Coord3]]:
    L: List[List[Coord3]] = []
    for z in range(SIZE):
        for y in range(SIZE):
            L.append([(x, y, z) for x in range(SIZE)])
    for z in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for y in range(SIZE)])
    for y in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for z in range(SIZE)])
    for z in range(SIZE):
        L.append([(i, i, z) for i in range(SIZE)])
        L.append([(i, SIZE-1-i, z) for i in range(SIZE)])
    for y in range(SIZE):
        L.append([(i, y, i) for i in range(SIZE)])
        L.append([(i, y, SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        L.append([(x, i, i) for i in range(SIZE)])
        L.append([(x, i, SIZE-1-i) for i in range(SIZE)])
    L.append([(i, i, i) for i in range(SIZE)])
    L.append([(i, i, SIZE-1-i) for i in range(SIZE)])
    L.append([(i, SIZE-1-i, i) for i in range(SIZE)])
    L.append([(SIZE-1-i, i, i) for i in range(SIZE)])
    return L

ALL_LINES = generate_lines()

# ---------- 全76ラインの即勝ち列挙（重力考慮） ----------
def line_immediate_winning_moves(board: Board, player: int) -> List[Coord2]:
    you = 3 - player
    wins: Set[Coord2] = set()
    for line in ALL_LINES:
        cnt_me = cnt_you = 0
        empty: Optional[Coord3] = None
        for (x, y, z) in line:
            v = board[z][y][x]
            if v == player: cnt_me += 1
            elif v == you:  cnt_you += 1
            else:           empty = (x, y, z)
        if cnt_you == 0 and cnt_me == 3 and empty is not None:
            ex, ey, ez = empty
            ze = lowest_empty_z(board, ex, ey)
            if ze is not None and ze == ez:
                wins.add((ex, ey))
    return list(wins)

def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None: return False
    won = False
    for line in ALL_LINES:
        if all(board[z0][y0][x0] == player for (x0, y0, z0) in line):
            won = True; break
    undo_place(board, x, y, z)
    return won

# ---------- ラベル→座標 ----------
LABEL_GRID = [
    ["0","1","2","4"],
    ["5","6","7","8"],
    ["9","A","B","C"],
    ["D","E","F","G"],
]
LABEL_POS: Dict[str, Coord2] = {LABEL_GRID[y][x]:(x,y) for y in range(4) for x in range(4)}
CORNERS = [LABEL_POS["0"], LABEL_POS["4"], LABEL_POS["D"], LABEL_POS["G"]]
CENTERS = [LABEL_POS["6"], LABEL_POS["7"], LABEL_POS["A"], LABEL_POS["B"]]
EDGES   = [LABEL_POS[k] for k in ["1","2","5","9","8","C","E","F"]]

def row_col_corners_for_edge(x: int, y: int) -> List[Coord2]:
    cs: List[Coord2] = []
    if y == 0: cs += [(0,0), (3,0)]
    if y == 3: cs += [(0,3), (3,3)]
    if x == 0: cs += [(0,0), (0,3)]
    if x == 3: cs += [(3,0), (3,3)]
    uniq: List[Coord2] = []
    for p in cs:
        if p not in uniq:
            uniq.append(p)
    return uniq

def column_has_my_stone(board: Board, me: int, x: int, y: int) -> bool:
    for z in range(SIZE):
        if board[z][y][x] == me:
            return True
    return False

# ---------- 側面ターゲット形のスコア ----------
def sideview_required_cells_for_row(y_fixed: int) -> List[Coord3]:
    # 横（y固定）: 必須6マス
    return [(0, y_fixed, 0), (3, y_fixed, 0), (1, y_fixed, 1), (2, y_fixed, 1), (1, y_fixed, 2), (2, y_fixed, 2)]

def sideview_required_cells_for_col(x_fixed: int) -> List[Coord3]:
    # 横（x固定）: 必須6マス
    return [(x_fixed, 0, 0), (x_fixed, 3, 0), (x_fixed, 1, 1), (x_fixed, 2, 1), (x_fixed, 1, 2), (x_fixed, 2, 2)]

def sideview_pattern_score_after_move(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    z = place_inplace(board, x, y, me)
    if y in (0, 3):
        req = sideview_required_cells_for_row(y)
    else:
        req = sideview_required_cells_for_col(x)
    opp_block = any(board[zz][yy][xx] == you for (xx, yy, zz) in req)
    score = 0 if opp_block else sum(1 for (xx, yy, zz) in req if board[zz][yy][xx] == me)
    undo_place(board, x, y, z)
    return score

# ---------- ルール実装（choose_best） ----------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves:
        return (0, 0)
    you = 3 - me

    # 1) 自分の即勝ち
    my_wins_now = line_immediate_winning_moves(board, me)
    if my_wins_now:
        for mv in moves:
            if mv in my_wins_now:
                return mv
        return my_wins_now[0]

    # 2) 相手の即勝ちブロック
    opp_wins_now = line_immediate_winning_moves(board, you)
    if opp_wins_now:
        best = None; best_after = 10**9
        for (x, y) in moves:
            z = place_inplace(board, x, y, me)
            after = len(line_immediate_winning_moves(board, you))
            undo_place(board, x, y, z)
            if after == 0:
                return (x, y)
            if after < best_after:
                best_after = after; best = (x, y)
        return best if best is not None else moves[0]

    # 2.5) 逆ミッキー型（ダブルスレット作成）最優先
    #  置いた直後に「自分の即勝ち手」が2本以上 & 相手の即勝ちが0 なら採用。
    best_rm_move: Optional[Coord2] = None
    best_dt = 0
    best_tie = -1
    for (x, y) in moves:
        z = place_inplace(board, x, y, me)
        my_next = len(line_immediate_winning_moves(board, me))
        opp_next = len(line_immediate_winning_moves(board, you))
        # タイブレークに“側面ターゲット形スコア”を使用（特に辺で効く）
        tie = sideview_pattern_score_after_move(board, me, x, y) if (x, y) in EDGES else 0
        undo_place(board, x, y, z)
        if my_next >= 2 and opp_next == 0:
            if my_next > best_dt or (my_next == best_dt and tie > best_tie):
                best_dt = my_next
                best_tie = tie
                best_rm_move = (x, y)
    if best_rm_move is not None:
        return best_rm_move

    # 3) 角 {0,4,D,G} の 1層目
    for (x, y) in CORNERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) == 0:
            return (x, y)

    # 4) 中央 {6,7,A,B} の 2〜3層
    for (x, y) in CENTERS:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                return (x, y)

    # 5) 辺 {1,2,5,9,8,C,E,F}（2〜3層 かつ 行/列の角いずれかに自分石）→
    #    “側面ターゲット形”の pattern_score を最大化して選ぶ
    edge_cands: List[Coord2] = []
    for (x, y) in EDGES:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                corners = row_col_corners_for_edge(x, y)
                if any(column_has_my_stone(board, me, cx, cy) for (cx, cy) in corners):
                    edge_cands.append((x, y))
    if edge_cands:
        best_mv = edge_cands[0]
        best_score = -1
        for (x, y) in edge_cands:
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_score:
                best_score = sc
                best_mv = (x, y)
        return best_mv

    # 6) 中央 {6,7,A,B} が置けるなら（層は問わず）
    for (x, y) in CENTERS:
        if (x, y) in moves:
            return (x, y)

    # 7) フォールバック：中央寄り
    moves_sorted = sorted(moves, key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
    return moves_sorted[0] if moves_sorted else moves[0]

# ---------- エンジン ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        try:
            mv = choose_best(board, player)
            ms = valid_xy_moves(board)
            return mv if mv in ms else (ms[0] if ms else (0, 0))
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0, 0)

AI = MyAI()
