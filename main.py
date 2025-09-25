# main.py
# 4x4x4 立体四目並べ AI（堅牢ブロック版：ライン∩総当たり + 最終ガード + 二重仮置き修正）
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

def _center_sorted(moves: List[Coord2]) -> List[Coord2]:
    return sorted(moves, key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))

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
    # 空間対角
    L.append([(i, i, i) for i in range(SIZE)])
    L.append([(i, i, SIZE-1-i) for i in range(SIZE)])
    L.append([(i, SIZE-1-i, i) for i in range(SIZE)])
    L.append([(SIZE-1-i, i, i) for i in range(SIZE)])
    return L

ALL_LINES = generate_lines()

# 各マスが属するライン（最後の一手だけを見るための索引）
LINES_THROUGH: Dict[Coord3, List[int]] = {}
for li, line in enumerate(ALL_LINES):
    for (x,y,z) in line:
        LINES_THROUGH.setdefault((x,y,z), []).append(li)

# ---------- 即勝ち検出 ----------
def line_immediate_winning_moves(board: Board, player: int) -> List[Coord2]:
    """全76ライン走査での即勝ち候補（重力OKのみ）"""
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
    """最後に置いたマスを含むラインでのみ勝ち判定（誤検出防止）"""
    z = place_inplace(board, x, y, player)
    if z is None: return False
    win = False
    for li in LINES_THROUGH[(x,y,z)]:
        line = ALL_LINES[li]
        if all(board[zz][yy][xx] == player for (xx,yy,zz) in line):
            win = True; break
    undo_place(board, x, y, z)
    return win

def immediate_winning_squares_try(board: Board, player: int) -> List[Coord2]:
    """合法手総当たりの即勝ち列挙（堅牢）"""
    res: List[Coord2] = []
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, player, x, y):
            res.append((x, y))
    return res

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

# ---------- 側面ターゲット形（評価） ----------
def sideview_required_cells_for_row(y_fixed: int) -> List[Coord3]:
    return [(0, y_fixed, 0), (3, y_fixed, 0),
            (1, y_fixed, 1), (2, y_fixed, 1),
            (1, y_fixed, 2), (2, y_fixed, 2)]

def sideview_required_cells_for_col(x_fixed: int) -> List[Coord3]:
    return [(x_fixed, 0, 0), (x_fixed, 3, 0),
            (x_fixed, 1, 1), (x_fixed, 2, 1),
            (x_fixed, 1, 2), (x_fixed, 2, 2)]

def sideview_pattern_score_after_move(board: Board, me: int, x: int, y: int) -> int:
    """まだ置いていない前提で (x,y) に仮置きして評価"""
    you = 3 - me
    z = place_inplace(board, x, y, me)
    if z is None:
        return 0
    if y in (0, 3): req = sideview_required_cells_for_row(y)
    else:           req = sideview_required_cells_for_col(x)
    opp_block = any(board[zz][yy][xx] == you for (xx, yy, zz) in req)
    score = 0 if opp_block else sum(1 for (xx, yy, zz) in req if board[zz][yy][xx] == me)
    undo_place(board, x, y, z)
    return score

def sideview_pattern_score_on_board(board: Board, me: int, x: int, y: int) -> int:
    """既に (x,y) に自石を仮置き“済み”の盤面を評価（追加の仮置きはしない）"""
    you = 3 - me
    if y in (0, 3): req = sideview_required_cells_for_row(y)
    else:           req = sideview_required_cells_for_col(x)
    if any(board[zz][yy][xx] == you for (xx, yy, zz) in req):
        return 0
    return sum(1 for (xx, yy, zz) in req if board[zz][yy][xx] == me)

# ---------- 逆ミッキー“完成”検出 → サイド直行 ----------
def is_reverse_mickey_row_complete(board: Board, me: int, y: int) -> bool:
    for (x,y0,z) in sideview_required_cells_for_row(y):
        if board[z][y0][x] != me: return False
    return True

def is_reverse_mickey_col_complete(board: Board, me: int, x: int) -> bool:
    for (x0,y,z) in sideview_required_cells_for_col(x):
        if board[z][y][x0] != me: return False
    return True

def find_reverse_mickey_side_move(board: Board, me: int) -> Optional[Coord2]:
    candidates: List[Coord2] = []
    for y in range(SIZE):
        if is_reverse_mickey_row_complete(board, me, y):
            for sx in (0, 3):
                z = lowest_empty_z(board, sx, y)
                if z in (1, 2):
                    candidates.append((sx, y))
    for x in range(SIZE):
        if is_reverse_mickey_col_complete(board, me, x):
            for sy in (0, 3):
                z = lowest_empty_z(board, x, sy)
                if z in (1, 2):
                    candidates.append((x, sy))
    if not candidates: return None
    best_mv = candidates[0]; best_dt = -1
    for (x, y) in candidates:
        z = place_inplace(board, x, y, me)
        my_dt = len(immediate_winning_squares_try(board, me))  # 総当たり版で堅牢化
        undo_place(board, x, y, z)
        if my_dt > best_dt:
            best_dt = my_dt; best_mv = (x, y)
    return best_mv

def _score_direct_block(board: Board, me: int, mv: Coord2, opp_wins_now: List[Coord2]) -> Tuple[int,int,int]:
    """ダイレクトブロック候補の優先度: (ブロック数, 自分のDT, 側面形スコア)"""
    you = 3 - me
    x, y = mv
    z = place_inplace(board, x, y, me)
    if z is None: return (0, 0, 0)

    before = len(opp_wins_now)
    after  = len(immediate_winning_squares_try(board, you))
    blocked = max(0, before - after)

    my_next = len(immediate_winning_squares_try(board, me))
    tie_sv  = sideview_pattern_score_on_board(board, me, x, y) if (x, y) in EDGES else 0

    undo_place(board, x, y, z)
    return (blocked, my_next, tie_sv)

# ---------- ブロック安全化の“最終ガード” ----------
def force_block_guard(board: Board, me: int, chosen: Coord2) -> Coord2:
    """返す直前に再確認：相手の即勝ちがあるのに塞いでいなければ強制ブロック。"""
    you = 3 - me
    my_now = immediate_winning_squares_try(board, me)
    if my_now:
        return chosen
    opp_set = set(line_immediate_winning_moves(board, you)) | set(immediate_winning_squares_try(board, you))
    if not opp_set:
        return chosen
    if chosen in opp_set:
        return chosen
    # ダイレクトに塞げるなら差し替え
    moves = valid_xy_moves(board)
    direct = [mv for mv in moves if mv in opp_set]
    if direct:
        scored = [(_score_direct_block(board, me, mv, list(opp_set)), mv) for mv in direct]
        scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
        return scored[0][1]
    # ダイレクト不可 → after==0 を最優先、なければ after 最小化
    best = None; best_after = 10**9
    for (x, y) in _center_sorted(moves):
        z = place_inplace(board, x, y, me)
        if z is None: continue
        after = len(immediate_winning_squares_try(board, you))
        undo_place(board, x, y, z)
        if after == 0: return (x, y)
        if after < best_after:
            best_after = after; best = (x, y)
    return best if best is not None else (_center_sorted(moves)[0] if moves else (0,0))

# ---------- 手選択（優先順位ロジック） ----------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0, 0)
    you = 3 - me

    # 1) 自分の即勝ち（厳密）
    my_wins_now = immediate_winning_squares_try(board, me)
    if my_wins_now:
        for mv in moves:
            if mv in my_wins_now:
                return mv
        return _center_sorted(my_wins_now)[0]

    # 2) 相手の即勝ちブロック（ダイレクト最優先）
    opp_wins_now = list(set(immediate_winning_squares_try(board, you)) | set(line_immediate_winning_moves(board, you)))
    if opp_wins_now:
        direct = [mv for mv in opp_wins_now if mv in moves]
        if direct:
            scored = [(_score_direct_block(board, me, mv, opp_wins_now), mv) for mv in direct]
            scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
            return scored[0][1]
        # ダイレクト不可 → after==0 > after最小
        best = None; best_after = 10**9
        for (x, y) in _center_sorted(moves):
            z = place_inplace(board, x, y, me)
            if z is None: continue
            after = len(immediate_winning_squares_try(board, you))
            undo_place(board, x, y, z)
            if after == 0: return (x, y)
            if after < best_after:
                best_after = after; best = (x, y)
        return best if best is not None else _center_sorted(moves)[0]

    # 2.4) 逆ミッキー“完成”→サイド直行
    mv_rm = find_reverse_mickey_side_move(board, me)
    if mv_rm is not None:
        return mv_rm

    # 2.5) 逆ミッキー“作成”狙い（DT）— 仮置き済み評価は on_board を使用
    best_rm_move: Optional[Coord2] = None
    best_dt = 0; best_tie = -1
    for (x, y) in moves:
        z = place_inplace(board, x, y, me)
        if z is None: continue
        my_next = len(immediate_winning_squares_try(board, me))
        opp_next = len(immediate_winning_squares_try(board, you))
        tie = sideview_pattern_score_on_board(board, me, x, y) if (x, y) in EDGES else 0
        undo_place(board, x, y, z)
        if my_next >= 2 and opp_next == 0:
            if my_next > best_dt or (my_next == best_dt and tie > best_tie):
                best_dt = my_next; best_tie = tie; best_rm_move = (x, y)
    if best_rm_move is not None:
        return best_rm_move

    # 3) 角1層
    for (x, y) in CORNERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) == 0:
            return (x, y)

    # 4) 中央 2〜3層
    for (x, y) in CENTERS:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                return (x, y)

    # 5) 辺（2〜3層 & 行/列角に自石）→ 側面ターゲット形最大化（未仮置きなので after_move を使用）
    edge_cands: List[Coord2] = []
    for (x, y) in EDGES:
        if (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z in (1, 2):
                corners = row_col_corners_for_edge(x, y)
                if any(column_has_my_stone(board, me, cx, cy) for (cx, cy) in corners):
                    edge_cands.append((x, y))
    if edge_cands:
        best_mv = edge_cands[0]; best_score = -1
        for (x, y) in edge_cands:
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_score:
                best_score = sc; best_mv = (x, y)
        return best_mv

    # 6) 中央（層問わず）
    for (x, y) in CENTERS:
        if (x, y) in moves:
            return (x, y)

    # 7) 中央寄りフォールバック
    return _center_sorted(moves)[0]

# ---------- エンジン（最終ガード付き） ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        try:
            mv = choose_best(board, player)
            mv = force_block_guard(board, player, mv)   # ★最後の安全弁：ブロック必須なら差し替え
            ms = valid_xy_moves(board)
            return mv if mv in ms else (ms[0] if ms else (0, 0))
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0, 0)

AI = MyAI()
