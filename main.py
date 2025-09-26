# 4x4x4 立体四目並べ AI（F型/DT押し切り + 角強制 + 近接応手 + 安全フィルタ強化）
# 変更点サマリ
# - フィニッシュ最上位: 置いた直後に自分の即勝ち手(DT)が2本以上になる“押し切り手”を最優先（F型含む汎用フィニッシャ）
# - 角ポリシー強化: 先手はz=0角に“必ず”行く。後手は相手角に重ねず他のz=0角を最優先
# - 近接応手: 初期に相手が内側に置いた場合、相手の前手に“距離最小”の一層マスを優先
# - 即勝/即ブロック > 予防ブロック(≤2手到達) > 角 > 中央/縦クランプ > 脅威読み(反復深化)
# - 安全フィルタ強化(safe_filter_moves): t支え/自殺/相手DT誘発/相手即勝増加の抑止

from typing import List, Tuple, Optional, Dict, Set
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)
SIZE = 4

# ===== 時間設定 =====
TIME_BUDGET_SEC = 9.5
TS_MAX_DEPTH    = 4
THREAT_BEAM     = 12

# ---------- 基本 ----------
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):
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

def _center_sorted(moves: List[Coord2]) -> List[Coord2]:
    return sorted(moves, key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))

def manhattan(p: Coord2, q: Coord2) -> int:
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def count_placed(board: Board) -> int:
    cnt = 0
    for z in range(SIZE):
        for y in range(SIZE):
            for x in range(SIZE):
                if board[z][y][x] != 0: cnt += 1
    return cnt

# ---------- 76ライン ----------
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
LINES_THROUGH: Dict[Coord3, List[int]] = {}
for li, line in enumerate(ALL_LINES):
    for (x,y,z) in line:
        LINES_THROUGH.setdefault((x,y,z), []).append(li)

# ---------- 即勝ち ----------
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
    win = False
    for li in LINES_THROUGH[(x,y,z)]:
        line = ALL_LINES[li]
        if all(board[zz][yy][xx] == player for (xx,yy,zz) in line):
            win = True; break
    undo_place(board, x, y, z)
    return win

def immediate_winning_squares_try(board: Board, player: int) -> List[Coord2]:
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
        if p not in uniq: uniq.append(p)
    return uniq

def column_has_my_stone(board: Board, me: int, x: int, y: int) -> bool:
    for z in range(SIZE):
        if board[z][y][x] == me:
            return True
    return False

# ---------- 側面ターゲット形（評価/逆ミッキー完成→サイド直行） ----------
def sideview_required_cells_for_row(y_fixed: int) -> List[Coord3]:
    return [(0, y_fixed, 0), (3, y_fixed, 0),
            (1, y_fixed, 1), (2, y_fixed, 1),
            (1, y_fixed, 2), (2, y_fixed, 2)]

def sideview_required_cells_for_col(x_fixed: int) -> List[Coord3]:
    return [(x_fixed, 0, 0), (x_fixed, 3, 0),
            (x_fixed, 1, 1), (x_fixed, 2, 1),
            (x_fixed, 1, 2), (x_fixed, 2, 2)]

def sideview_pattern_score_on_board(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    req = sideview_required_cells_for_row(y) if y in (0,3) else sideview_required_cells_for_col(x)
    if any(board[zz][yy][xx] == you for (xx,yy,zz) in req):
        return 0
    return sum(1 for (xx,yy,zz) in req if board[zz][yy][xx] == me)

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
                if lowest_empty_z(board, sx, y) in (1,2):
                    candidates.append((sx, y))
    for x in range(SIZE):
        if is_reverse_mickey_col_complete(board, me, x):
            for sy in (0, 3):
                if lowest_empty_z(board, x, sy) in (1,2):
                    candidates.append((x, sy))
    if not candidates: return None
    best_mv = candidates[0]; best_dt = -1
    for (x, y) in candidates:
        z = place_inplace(board, x, y, me)
        my_dt = len(immediate_winning_squares_try(board, me))
        undo_place(board, x, y, z)
        if my_dt > best_dt:
            best_dt = my_dt; best_mv = (x, y)
    return best_mv

# ---------- 自殺・t支え・DT誘発の禁止 ----------
def is_t_support_move(board: Board, me: int, x: int, y: int) -> bool:
    # 置くと相手が同列z==3に即勝できる“支え”になる手
    you = 3 - me
    z = lowest_empty_z(board, x, y)
    if z != 2: return False
    z2 = place_inplace(board, x, y, me)
    if z2 is None: return True
    opens_tpoint_win = is_winning_after(board, you, x, y)
    undo_place(board, x, y, z2)
    return opens_tpoint_win

def is_suicide_move(board: Board, me: int, x: int, y: int) -> bool:
    # 自分が勝てないのに相手の即勝を増やす手
    you = 3 - me
    if is_winning_after(board, me, x, y):  # 自分の即勝ならOK
        return False
    before = len(immediate_winning_squares_try(board, you))
    z = place_inplace(board, x, y, me)
    if z is None: return True
    after = len(immediate_winning_squares_try(board, you))
    undo_place(board, x, y, z)
    return after > before

def _max_opp_dt_after_reply(board: Board, me: int, x: int, y: int, beam: int = 10) -> int:
    # こちらが(x,y)に置いた後、相手手番でのDT最大本数を概算
    you = 3 - me
    z0 = place_inplace(board, x, y, me)
    if z0 is None: return 99
    moves = valid_xy_moves(board)
    wins_opp = set(immediate_winning_squares_try(board, you))
    cands: Set[Coord2] = set(wins_opp) | set(_center_sorted(moves[:6]))
    worst = 0
    for (ox, oy) in list(cands)[:beam]:
        z1 = place_inplace(board, ox, oy, you)
        if z1 is None: continue
        dt = len(immediate_winning_squares_try(board, you))
        undo_place(board, ox, oy, z1)
        worst = max(worst, dt)
        if worst >= 2: break
    undo_place(board, x, y, z0)
    return worst

def safe_filter_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    """危険排除: t支え/自殺/相手DT誘発>=2 を除外。全滅なら元のmovesを返す"""
    you = 3 - me
    base_opp_now = len(immediate_winning_squares_try(board, you))
    safe: List[Coord2] = []
    for (x, y) in moves:
        if is_t_support_move(board, me, x, y): continue
        if is_winning_after(board, me, x, y):  # 勝ち手は許容
            safe.append((x,y)); continue
        z = place_inplace(board, x, y, me)
        if z is None: continue
        opp_now = len(immediate_winning_squares_try(board, you))
        bad = (opp_now > base_opp_now) or (_max_opp_dt_after_reply(board, me, x, y) >= 2)
        undo_place(board, x, y, z)
        if bad: continue
        safe.append((x, y))
    return safe if safe else moves

# ---------- “DT押し切り”フィニッシャ（F型含む汎用） ----------
def double_threat_finisher(board: Board, me: int) -> Optional[Coord2]:
    """置いた直後に自分の即勝が2本以上(相手の即勝0)になる手を最優先。"""
    you = 3 - me
    best: Optional[Tuple[int, Coord2]] = None
    for (x, y) in valid_xy_moves(board):
        z = place_inplace(board, x, y, me)
        if z is None: continue
        my_dt  = len(immediate_winning_squares_try(board, me))
        opp_dt = len(immediate_winning_squares_try(board, you))
        undo_place(board, x, y, z)
        if my_dt >= 2 and opp_dt == 0:
            if best is None or my_dt > best[0]:
                best = (my_dt, (x, y))
    return None if best is None else best[1]

# ---------- 予防ブロック（≤2手到達脅威） ----------
def collect_reachable_threats(board: Board, player: int, steps: int) -> Set[Coord2]:
    """“あと≤steps手で三並び+空1になる着点”の列(x,y)集合を概算"""
    you = 3 - player
    res: Set[Coord2] = set()
    for line in ALL_LINES:
        cnt_p = cnt_y = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == player: cnt_p += 1
            elif v == you:  cnt_y += 1
            else: empties.append((x,y,z))
        # 自分の石優勢/相手の遮断なし で “埋めやすい”ラインだけ見る
        if cnt_y == 0 and len(empties) <= steps+1:
            for (x,y,z) in empties:
                if lowest_empty_z(board, x, y) == z:
                    res.add((x,y))
    return res

# ---------- 縦列関連（クランプ/自圧作り） ----------
def vertical_profile(board: Board, me: int, x: int, y: int) -> Tuple[int,int]:
    you = 3 - me
    m = o = 0
    for z in range(SIZE):
        v = board[z][y][x]
        if v == me: m += 1
        elif v == you: o += 1
    return (m,o)

def opponent_focus_columns(board: Board, me: int) -> List[Coord2]:
    you = 3 - me
    stats: List[Tuple[int,int,Coord2]] = []
    for y in range(SIZE):
        for x in range(SIZE):
            c = sum(1 for z in range(SIZE) if board[z][y][x] == you)
            if c>0:
                center = int(abs(1.5-x)+abs(1.5-y))
                stats.append((c, -center, (x,y)))
    stats.sort(reverse=True)
    return [p for _,__,p in stats]

def urgent_vertical_clamp(board: Board, me: int) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    focus = opponent_focus_columns(board, me)
    rank = {p:i for i,p in enumerate(focus)}
    pr: List[Tuple[int,int,int,Coord2]] = []
    for (x,y) in moves:
        z = lowest_empty_z(board, x, y)
        m,o = vertical_profile(board, me, x, y)
        if o>=2 and z==2:   pr.append((2, -rank.get((x,y),99), -int(abs(1.5-x)+abs(1.5-y)), (x,y)))
        elif o==1 and z==1: pr.append((1, -rank.get((x,y),99), -int(abs(1.5-x)+abs(1.5-y)), (x,y)))
    if not pr: return None
    pr.sort(reverse=True)
    cands = [pr[0][3]]
    cands = safe_filter_moves(board, me, cands)
    return cands[0] if cands else None

def build_own_vertical_pressure(board: Board, me: int) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    cands: List[Coord2] = []
    for (x,y) in moves:
        if lowest_empty_z(board, x, y) == 1:
            m,o = vertical_profile(board, me, x, y)
            if m>=1 and o==0: cands.append((x,y))
    cands = safe_filter_moves(board, me, cands)
    cands.sort(key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))
    return cands[0] if cands else None

# ---------- ダイレクトブロック ----------
def _score_direct_block(board: Board, me: int, mv: Coord2, opp_wins_now: List[Coord2]) -> Tuple[int,int,int]:
    you = 3 - me
    x,y = mv
    z = place_inplace(board, x, y, me)
    if z is None: return (0,0,0)
    before = len(opp_wins_now)
    after  = len(immediate_winning_squares_try(board, you))
    blocked = max(0, before - after)
    my_next = len(immediate_winning_squares_try(board, me))
    tie_sv  = sideview_pattern_score_on_board(board, me, x, y) if (x,y) in EDGES else 0
    undo_place(board, x, y, z)
    return (blocked, my_next, tie_sv)

def force_block_guard(board: Board, me: int, chosen: Coord2) -> Coord2:
    you = 3 - me
    if immediate_winning_squares_try(board, me):  # 自分が即勝ちならそのまま
        return chosen
    opp_set = set(line_immediate_winning_moves(board, you)) | set(immediate_winning_squares_try(board, you))
    if not opp_set or chosen in opp_set:
        return chosen
    moves = valid_xy_moves(board)
    direct = [mv for mv in moves if mv in opp_set]
    if direct:
        scored = [(_score_direct_block(board, me, mv, list(opp_set)), mv) for mv in direct]
        scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
        return scored[0][1]
    best = None; best_after = 10**9
    for (x,y) in _center_sorted(moves):
        z = place_inplace(board, x, y, me)
        if z is None: continue
        after = len(immediate_winning_squares_try(board, you))
        undo_place(board, x, y, z)
        if after == 0: return (x,y)
        if after < best_after:
            best_after = after; best = (x,y)
    return best if best else (_center_sorted(moves)[0] if moves else (0,0))

# ---------- Threat-space Negamax（脅威限定・反復深化） ----------
def _tactical_eval(board: Board, me: int) -> int:
    you = 3 - me
    my_now  = len(immediate_winning_squares_try(board, me))
    opp_now = len(immediate_winning_squares_try(board, you))
    score = 10_000*(my_now - opp_now)
    if my_now >= 2:  score += 1500
    if opp_now >= 2: score -= 2000
    # 軽い縦圧
    v = 0
    for y in range(SIZE):
        for x in range(SIZE):
            m,o = vertical_profile(board, me, x, y)
            if m>=2 and o==0: v += 3
    return score + v

def _tactical_candidates(board: Board, turn: int, me: int) -> List[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return []
    you = 3 - turn
    wins_t = set(immediate_winning_squares_try(board, turn))
    wins_o = set(immediate_winning_squares_try(board, you))
    c: Set[Coord2] = set(wins_t) | set(wins_o)
    # 中央/辺の2-3層
    for (x,y) in CENTERS+EDGES:
        if (x,y) in moves and lowest_empty_z(board, x, y) in (1,2): c.add((x,y))
    # 逆ミッキー完成のサイド
    mv_rm = find_reverse_mickey_side_move(board, turn)
    if mv_rm and mv_rm in moves: c.add(mv_rm)
    # 少なければ中心寄せで補完
    if len(c) < 6:
        for mv in _center_sorted(moves):
            c.add(mv)
            if len(c) >= 8: break
    ordered = list(c)
    def key(mv: Coord2):
        x,y = mv
        s1 = 3 if mv in wins_t else 0
        s2 = 2 if mv in wins_o else 0
        s3 = -int(abs(1.5-x)+abs(1.5-y))
        return (s1,s2,s3)
    ordered.sort(key=key, reverse=True)
    return ordered[:THREAT_BEAM]

def _negamax(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float) -> int:
    if time.perf_counter() >= deadline or depth == 0:
        return _tactical_eval(board, me)
    moves = _tactical_candidates(board, turn, me)
    if not moves:
        return _tactical_eval(board, me)
    you = 3 - turn
    if immediate_winning_squares_try(board, turn):
        return 100_000 - (5 - depth)
    best = -10**9
    for (x,y) in moves:
        if time.perf_counter() >= deadline: break
        z = place_inplace(board, x, y, turn)
        if z is None: continue
        if is_t_support_move(board, turn, x, y):
            undo_place(board, x, y, z); continue
        sc = -_negamax(board, depth-1, -beta, -alpha, you, me, deadline)
        undo_place(board, x, y, z)
        if sc > best: best = sc
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def threat_space_best_move_iterative(board: Board, me: int, deadline: float) -> Optional[Coord2]:
    root = _tactical_candidates(board, me, me)
    if not root: return None
    root = safe_filter_moves(board, me, root)
    focus = opponent_focus_columns(board, me)
    extra = 1 if focus and len(focus) <= 4 else 0
    max_d = min(TS_MAX_DEPTH + extra, 5)
    best_mv = root[0]; you = 3 - me
    for d in range(2, max_d+1):
        if time.perf_counter() >= deadline: break
        best_sc = -10**9
        for (x,y) in root:
            if time.perf_counter() >= deadline: break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -_negamax(board, d-1, -10**9, 10**9, you, me, deadline)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        if best_mv in root:
            root = [best_mv] + [mv for mv in root if mv != best_mv]
    return best_mv

# ---------- 序盤ユーティリティ ----------
def first_layer_corner_moves(board: Board) -> List[Coord2]:
    return [(x,y) for (x,y) in CORNERS if lowest_empty_z(board, x, y) == 0]

def nearest_first_layer_to(board: Board, target: Coord2) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    one_layer = [(x,y) for (x,y) in moves if lowest_empty_z(board, x, y) == 0]
    if not one_layer: return None
    one_layer.sort(key=lambda p: (manhattan(p, target), abs(1.5-p[0])+abs(1.5-p[1])))
    return one_layer[0]

# ---------- 手選択 ----------
def choose_best(board: Board, me: int, deadline: float, last_move: Coord3) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0,0)
    you = 3 - me
    total = count_placed(board)

    # A) 自即勝
    my_wins = immediate_winning_squares_try(board, me)
    if my_wins:
        for mv in moves:
            if mv in my_wins: return mv
        return _center_sorted(my_wins)[0]

    # B) 相手即勝ブロック
    opp_wins = list(set(immediate_winning_squares_try(board, you)) |
                    set(line_immediate_winning_moves(board, you)))
    if opp_wins:
        direct = [mv for mv in opp_wins if mv in moves]
        if direct:
            scored = [(_score_direct_block(board, me, mv, opp_wins), mv) for mv in direct]
            scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
            return scored[0][1]
        # ダイレクト不可→ after==0 優先、次点 after 最小
        best = None; best_after = 10**9
        for (x,y) in _center_sorted(moves):
            z = place_inplace(board, x, y, me)
            if z is None: continue
            after = len(immediate_winning_squares_try(board, you))
            undo_place(board, x, y, z)
            if after == 0: return (x,y)
            if after < best_after: best_after, best = after, (x,y)
        return best if best else _center_sorted(moves)[0]

    # C) “DT押し切り”（F型含む）— ここを最上位の攻め筋に
    mv_dt = double_threat_finisher(board, me)
    if mv_dt is not None:
        return mv_dt

    # D) 予防ブロック（≤2手到達脅威の列を先んじて塞ぐ）
    opp2 = collect_reachable_threats(board, you, 2)
    if opp2:
        pre = [(x,y) for (x,y) in moves if (x,y) in opp2 and lowest_empty_z(board, x, y) in (1,2)]
        pre = safe_filter_moves(board, me, pre)
        if pre:
            pre.sort(key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))
            return pre[0]

    # E) 角（z==0）強制
    corner0 = first_layer_corner_moves(board)
    if corner0:
        # 後手で相手が直前に角z==0なら、別角を優先
        if me == 2:
            lx,ly,lz = last_move
            if (lx,ly) in CORNERS and lz == 0:
                others = [(x,y) for (x,y) in corner0 if (x,y)!=(lx,ly)]
                if others:
                    cands = safe_filter_moves(board, me, others) or others
                    cands.sort(key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))
                    return cands[0]
        cands = safe_filter_moves(board, me, corner0) or corner0
        cands.sort(key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))
        return cands[0]

    # F) 序盤“近接応手”: 相手が内側に置いたら、相手着手に最も近い一層マスへ
    if total <= 4:
        lx,ly,lz = last_move
        if (lx,ly) not in CORNERS and lz == 0:
            near = nearest_first_layer_to(board, (lx,ly))
            if near:
                near_safe = safe_filter_moves(board, me, [near])
                if near_safe: return near_safe[0]
                return near

    # G) 中央クランプ（センターz∈{1,2}）/ 縦列クランプ / 自縦圧
    clamp = choose_center_clamp_move(board, me)
    if clamp is not None: return clamp
    vblock = urgent_vertical_clamp(board, me)
    if vblock is not None: return vblock
    vbuild = build_own_vertical_pressure(board, me)
    if vbuild is not None: return vbuild

    # H) 脅威限定 反復深化
    mv_ts = threat_space_best_move_iterative(board, me, deadline)
    if mv_ts is not None: return mv_ts

    # I) 逆ミッキー完成→サイド直行（安全）
    mv_rm = find_reverse_mickey_side_move(board, me)
    if mv_rm is not None and mv_rm in moves:
        if not is_suicide_move(board, me, mv_rm[0], mv_rm[1]) and not is_t_support_move(board, me, mv_rm[0], mv_rm[1]):
            return mv_rm

    # J) 逆ミッキー作成狙い（DT化）
    best_move = None; best_dt = -1; best_tie = -1
    cand = safe_filter_moves(board, me, moves)
    for (x,y) in cand:
        z = place_inplace(board, x, y, me)
        if z is None: continue
        my_next = len(immediate_winning_squares_try(board, me))
        tie = sideview_pattern_score_on_board(board, me, x, y) if (x,y) in EDGES else 0
        undo_place(board, x, y, z)
        if my_next > best_dt or (my_next == best_dt and tie > best_tie):
            best_dt, best_tie, best_move = my_next, tie, (x,y)
    if best_move is not None:
        return best_move

    # K) 中央/辺ヒューリスティック（安全）
    for (x,y) in CENTERS:
        if (x,y) in moves and lowest_empty_z(board, x, y) in (1,2):
            if (x,y) in safe_filter_moves(board, me, [(x,y)]): return (x,y)

    edge = []
    for (x,y) in EDGES:
        if (x,y) in moves and lowest_empty_z(board, x, y) in (1,2):
            if any(column_has_my_stone(board, me, cx, cy) for (cx,cy) in row_col_corners_for_edge(x,y)):
                edge.append((x,y))
    edge = safe_filter_moves(board, me, edge)
    if edge:
        edge.sort(key=lambda p: -sideview_pattern_score_on_board(board, me, p[0], p[1]))
        return edge[0]

    for (x,y) in CENTERS:
        if (x,y) in moves and (x,y) in safe_filter_moves(board, me, [(x,y)]): return (x,y)

    fb = safe_filter_moves(board, me, _center_sorted(moves))
    return fb[0]

# ---------- 中央クランプ ----------
def choose_center_clamp_move(board: Board, me: int) -> Optional[Coord2]:
    you = 3 - me
    moves = valid_xy_moves(board)
    if not moves: return None
    cands: List[Coord2] = []
    for (x,y) in CENTERS:
        if (x,y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.append((x,y))
    if not cands: return None
    cands = safe_filter_moves(board, me, cands) or cands
    base = len(immediate_winning_squares_try(board, you))
    scored = []
    for (x,y) in cands:
        z = place_inplace(board, x, y, me)
        aft = len(immediate_winning_squares_try(board, you))
        undo_place(board, x, y, z)
        delta = base - aft
        center_bias = -int(abs(1.5-x)+abs(1.5-y))
        scored.append(((delta, center_bias), (x,y)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]

# ---------- エンジン ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        deadline = time.perf_counter() + TIME_BUDGET_SEC
        try:
            mv = choose_best(board, player, deadline, last_move)
            mv = force_block_guard(board, player, mv)
            ms = valid_xy_moves(board)
            return mv if mv in ms else (ms[0] if ms else (0,0))
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0,0)

AI = MyAI()
