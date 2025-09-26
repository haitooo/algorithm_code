# main.py
# 4x4x4 立体四目並べ AI
# - 角(z==0)積極取得（自分の底角が2以下なら優先、先手はペア角最優先）
# - アンチ重ね（相手底角の上に重ねるのは原則禁止：自即勝/直ブロックのみ例外）
# - 脅威空間IDS + 軽量AB + 全76ライン評価
# - 直ブロック/間接ブロックの強化、時間ガード強化
#
# 禁止API未使用（I/O/並列/open/eval/exec/…全て未使用）

from typing import List, Tuple, Optional, Dict, Set
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# ---- 探索予算（時間ガード付きで動的に縮小）----
TIME_BUDGET_SEC = 9.2
TS_MAX_DEPTH    = 4      # 脅威空間 IDS 深さ（状況で縮む）
THREAT_BEAM     = 14     # 候補束（状況で縮む）
AB_MAX_DEPTH    = 2      # 低深度AB（保険）

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

def _time_up(deadline: float) -> bool:
    return time.perf_counter() >= deadline

def _beam_by_time(deadline: float, base: int) -> int:
    left = max(0.0, deadline - time.perf_counter())
    if left < 0.35: return max(6, base // 3)
    if left < 0.9:  return max(8, base // 2)
    return base

def count_placed(board: Board) -> int:
    c = 0
    for z in range(SIZE):
        for y in range(SIZE):
            for x in range(SIZE):
                if board[z][y][x] != 0:
                    c += 1
    return c

# ---------- 全76ライン ----------
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
LINES_THROUGH: Dict[Coord3, List[int]] = {}
for li, line in enumerate(ALL_LINES):
    for (x,y,z) in line:
        LINES_THROUGH.setdefault((x,y,z), []).append(li)

# ---------- 即勝 ----------
def immediate_winning_squares_try(board: Board, player: int) -> List[Coord2]:
    wins: List[Coord2] = []
    you = 3 - player
    for line in ALL_LINES:
        cnt_me = cnt_you = 0
        empty: Optional[Coord3] = None
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == player: cnt_me += 1
            elif v == you:  cnt_you += 1
            else:           empty = (x,y,z)
        if cnt_me == 3 and cnt_you == 0 and empty is not None:
            ex, ey, ez = empty
            ze = lowest_empty_z(board, ex, ey)
            if ze is not None and ze == ez:
                wins.append((ex, ey))
    return wins

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
    ok = False
    for li in LINES_THROUGH[(x,y,z)]:
        line = ALL_LINES[li]
        if all(board[zz][yy][xx] == player for (xx,yy,zz) in line):
            ok = True; break
    undo_place(board, x, y, z)
    return ok

# ---------- ラベル/分類 ----------
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

def first_layer_corner_moves(board: Board) -> List[Coord2]:
    mv: List[Coord2] = []
    for (x,y) in CORNERS:
        if lowest_empty_z(board, x, y) == 0:
            mv.append((x,y))
    return mv

# ---------- 角・辺補助 ----------
def row_col_corners_for_edge(x: int, y: int) -> List[Coord2]:
    cs: List[Coord2] = []
    if y == 0: cs += [(0,0),(3,0)]
    if y == 3: cs += [(0,3),(3,3)]
    if x == 0: cs += [(0,0),(0,3)]
    if x == 3: cs += [(3,0),(3,3)]
    uniq: List[Coord2] = []
    for p in cs:
        if p not in uniq: uniq.append(p)
    return uniq

def column_has_my_stone(board: Board, me: int, x: int, y: int) -> bool:
    for z in range(SIZE):
        if board[z][y][x] == me:
            return True
    return False

def my_corner_z0(board: Board, me: int) -> List[Coord2]:
    res: List[Coord2] = []
    for (x,y) in CORNERS:
        if board[0][y][x] == me:
            res.append((x,y))
    return res

def opp_corner_z0(board: Board, me: int) -> List[Coord2]:
    you = 3 - me
    res: List[Coord2] = []
    for (x,y) in CORNERS:
        if board[0][y][x] == you:
            res.append((x,y))
    return res

def same_edge_pair_corners(c: Coord2) -> List[Coord2]:
    x,y = c
    pairs: List[Coord2] = []
    if y == 0: pairs.append((3,0))
    if y == 3: pairs.append((0,3))
    if x == 0: pairs.append((0,3))
    if x == 3: pairs.append((3,0))
    uniq: List[Coord2] = []
    for p in pairs:
        if p not in uniq and p != c:
            uniq.append(p)
    return uniq

def is_cap_on_opponent_corner(board: Board, me: int, x: int, y: int) -> bool:
    """相手の底角( z==0 )の上に重ねる手か？"""
    if (x,y) not in CORNERS: return False
    you = 3 - me
    z = lowest_empty_z(board, x, y)
    if z is None or z == 0: return False
    return board[0][y][x] == you

# ---------- 側面テンプレ ----------
def sideview_required_cells_for_row(y_fixed: int) -> List[Coord3]:
    return [(0, y_fixed, 0), (3, y_fixed, 0),
            (1, y_fixed, 1), (2, y_fixed, 1),
            (1, y_fixed, 2), (2, y_fixed, 2)]

def sideview_required_cells_for_col(x_fixed: int) -> List[Coord3]:
    return [(x_fixed, 0, 0), (x_fixed, 3, 0),
            (x_fixed, 1, 1), (x_fixed, 2, 1),
            (x_fixed, 1, 2), (x_fixed, 2, 2)]

def sideview_pattern_score_after_move(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    z = place_inplace(board, x, y, me)
    if z is None: return 0
    req = sideview_required_cells_for_row(y) if y in (0,3) else sideview_required_cells_for_col(x)
    bad = any(board[zz][yy][xx] == you for (xx,yy,zz) in req)
    sc = 0 if bad else sum(1 for (xx,yy,zz) in req if board[zz][yy][xx] == me)
    undo_place(board, x, y, z)
    return sc

# ---------- “今の”即勝 合併検出 ----------
def _opp_immediate_now_union(board: Board, player: int) -> Set[Coord2]:
    return set(immediate_winning_squares_try(board, player)) | set(line_immediate_winning_moves(board, player))

# ---------- 自殺・t支え ----------
def is_t_support_move(board: Board, me: int, x: int, y: int) -> bool:
    z = lowest_empty_z(board, x, y)
    if z != 2:
        return False
    z2 = place_inplace(board, x, y, me)
    if z2 is None:
        return True
    opens = is_winning_after(board, 3-me, x, y)
    undo_place(board, x, y, z2)
    return opens

def is_suicide_move(board: Board, me: int, x: int, y: int) -> bool:
    if is_winning_after(board, me, x, y): return False
    z = place_inplace(board, x, y, me)
    if z is None: return True
    opp_dts = len(immediate_winning_squares_try(board, 3-me))
    undo_place(board, x, y, z)
    return opp_dts >= 2

def safe_filter_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    safe: List[Coord2] = []
    for (x,y) in moves:
        try:
            if is_suicide_move(board, me, x, y):   continue
            if is_t_support_move(board, me, x, y): continue
            safe.append((x,y))
        except Exception:
            safe.append((x,y))
    return safe if safe else moves

# ---------- ブロック ----------
def _score_direct_block(board: Board, me: int, mv: Coord2, opp_wins_now: List[Coord2]) -> Tuple[int,int,int]:
    x,y = mv
    z = place_inplace(board, x, y, me)
    if z is None: return (0,0,0)
    before = len(opp_wins_now)
    after  = len(immediate_winning_squares_try(board, 3-me))
    blocked = max(0, before - after)
    my_next = len(immediate_winning_squares_try(board, me))
    tie = sideview_pattern_score_after_move(board, me, x, y) if (x, y) in EDGES else 0
    undo_place(board, x, y, z)
    return (blocked, my_next, tie)

# --- 1手先の相手即勝“最大値”最小化 ---
def _max_opp_immediate_after_reply(board: Board, me: int, x: int, y: int, deadline: float, beam: int) -> int:
    try:
        you = 3 - me
        z0 = place_inplace(board, x, y, me)
        if z0 is None:
            return 99
        moves = valid_xy_moves(board)
        opp_now = _opp_immediate_now_union(board, you)
        cand: List[Coord2] = [mv for mv in moves if mv in opp_now]
        if len(cand) < beam:
            for mv in _center_sorted(moves):
                if mv not in cand:
                    cand.append(mv)
                if len(cand) >= beam:
                    break
        worst = 0
        for (ox, oy) in cand:
            if _time_up(deadline): break
            z1 = place_inplace(board, ox, oy, you)
            if z1 is None: continue
            cnt = len(_opp_immediate_now_union(board, you))
            if cnt > worst: worst = cnt
            undo_place(board, ox, oy, z1)
            if worst >= 2: break
        undo_place(board, x, y, z0)
        return worst
    except Exception:
        return 2

def _choose_by_minimax_opp_after(board: Board, me: int, moves: List[Coord2], deadline: float, base_beam: int) -> Optional[Coord2]:
    if not moves: return None
    beam = _beam_by_time(deadline, base_beam)
    best_mv = moves[0]; best_key = (999, 9.9)
    for (x, y) in moves:
        if _time_up(deadline): break
        try:
            w = _max_opp_immediate_after_reply(board, me, x, y, deadline, beam)
            key = (w, abs(1.5 - x) + abs(1.5 - y))
        except Exception:
            key = (999, abs(1.5 - x) + abs(1.5 - y))
        if key < best_key:
            best_mv = (x, y); best_key = key
            if key[0] == 0: break
    return best_mv

def force_block_guard(board: Board, me: int, chosen: Coord2, deadline: float) -> Coord2:
    try:
        if _time_up(deadline): return chosen
        my_now = immediate_winning_squares_try(board, me)
        if my_now: return chosen
    except Exception:
        pass

    opp_now = _opp_immediate_now_union(board, 3-me)
    if not opp_now: return chosen
    if chosen in opp_now: return chosen

    moves = valid_xy_moves(board)
    direct = [mv for mv in moves if mv in opp_now]
    if direct:
        best = _choose_by_minimax_opp_after(board, me, direct, deadline, 12)
        if best is not None:
            return best

    best_mv = None; best_key = (999, 9.9, 9.9)
    for (x, y) in _center_sorted(moves):
        if _time_up(deadline): break
        z = place_inplace(board, x, y, me)
        if z is None: continue
        try:
            after = len(_opp_immediate_now_union(board, 3-me))
            worst = _max_opp_immediate_after_reply(board, me, x, y, deadline, _beam_by_time(deadline, 10))
            key = (after, worst, abs(1.5 - x) + abs(1.5 - y))
        finally:
            undo_place(board, x, y, z)
        if key < best_key:
            best_key = key; best_mv = (x, y)
        if best_key[0] == 0 and best_key[1] == 0:
            break
    if best_mv is not None:
        return best_mv
    return _center_sorted(moves)[0] if moves else (0,0)

# ---------- 脅威空間 IDS ----------
def vertical_profile(board: Board, player: int, x: int, y: int) -> Tuple[int,int]:
    you = 3 - player
    a=b=0
    for z in range(SIZE):
        v = board[z][y][x]
        if v==player: a+=1
        elif v==you:  b+=1
    return a,b

def _tactical_eval(board: Board, me: int) -> int:
    you = 3 - me
    my_now  = len(immediate_winning_squares_try(board, me))
    opp_now = len(immediate_winning_squares_try(board, you))
    score = 10000*(my_now - opp_now)
    if my_now >= 2:  score += 1500
    if opp_now >= 2: score -= 2000
    for y in range(SIZE):
        for x in range(SIZE):
            mc, yc = vertical_profile(board, me, x, y)
            if mc >= 2 and yc == 0: score += 3
            s = 3 - int(abs(1.5-x)+abs(1.5-y))
            for z in range(SIZE):
                v = board[z][y][x]
                if v==me: score += s
                elif v==you: score -= s
    return score

def _tactical_candidates(board: Board, turn: int, me: int, deadline: float) -> List[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return []
    if _time_up(deadline): return _center_sorted(moves)[:6]
    you = 3 - turn
    wins_turn = set(immediate_winning_squares_try(board, turn))
    wins_opp  = set(immediate_winning_squares_try(board, you))
    cands: Set[Coord2] = set()
    cands |= wins_turn
    cands |= wins_opp
    for (x, y) in CENTERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.add((x,y))
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.add((x,y))
    if len(cands) < 6:
        for mv in _center_sorted(moves):
            cands.add(mv)
            if len(cands) >= _beam_by_time(deadline, THREAT_BEAM): break
    ordered = list(cands)
    def _score(m: Coord2) -> Tuple[int,int,int]:
        x,y = m
        s1 = 2 if m in wins_turn else 0
        s2 = 1 if m in wins_opp  else 0
        s3 = -int(abs(1.5-x)+abs(1.5-y))
        return (s1,s2,s3)
    ordered.sort(key=_score, reverse=True)
    return ordered[:_beam_by_time(deadline, THREAT_BEAM)]

def _negamax_threat(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float) -> int:
    if _time_up(deadline) or depth == 0:
        return _tactical_eval(board, me)
    wins = immediate_winning_squares_try(board, turn)
    if wins:
        return 100000 - (5 - depth)
    moves = _tactical_candidates(board, turn, me, deadline)
    if not moves:
        return _tactical_eval(board, me)
    best = -10**9
    nxt = 3 - turn
    for (x, y) in moves:
        if _time_up(deadline): break
        z = place_inplace(board, x, y, turn)
        if z is None or is_t_support_move(board, turn, x, y):
            if z is not None: undo_place(board, x, y, z)
            continue
        sc = -_negamax_threat(board, depth-1, -beta, -alpha, nxt, me, deadline)
        undo_place(board, x, y, z)
        if sc > best: best = sc
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def threat_space_best_move_iterative(board: Board, me: int, deadline: float) -> Optional[Coord2]:
    root = _tactical_candidates(board, me, me, deadline)
    if not root: return None
    root = safe_filter_moves(board, me, root)
    best_mv: Optional[Coord2] = root[0]
    opp = 3 - me
    for d in range(2, TS_MAX_DEPTH+1):
        if _time_up(deadline): break
        best_sc = -10**9
        for (x, y) in root:
            if _time_up(deadline): break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -_negamax_threat(board, d-1, -10**9, 10**9, opp, me, deadline)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        if best_mv in root:
            root = [best_mv] + [m for m in root if m != best_mv]
    return best_mv

# ---------- 全局評価 ----------
def evaluate_full(board: Board, me: int) -> int:
    you = 3 - me
    my_th = opp_th = 0
    sc = 0
    for line in ALL_LINES:
        vals = [board[z][y][x] for (x,y,z) in line]
        mc, oc = vals.count(me), vals.count(you)
        ec = 4 - mc - oc
        if oc == 0:
            if mc == 4: return 10**9
            if mc == 3 and ec == 1: sc += 300; my_th += 1
            elif mc == 2 and ec == 2: sc += 20
            elif mc == 1 and ec == 3: sc += 2
        elif mc == 0:
            if oc == 4: return -10**9
            if oc == 3 and ec == 1: sc -= 300; opp_th += 1
            elif oc == 2 and ec == 2: sc -= 20
            elif oc == 1 and ec == 3: sc -= 2
    if my_th >= 2:  sc += 5000
    if opp_th >= 2: sc -= 5000
    for y in range(SIZE):
        for x in range(SIZE):
            w = 3 - int(abs(1.5-x)+abs(1.5-y))
            for z in range(SIZE):
                v = board[z][y][x]
                if v==me: sc += w
                elif v==you: sc -= w
    return sc

# ---------- 軽量フルαβ（保険） ----------
def _order_moves_global(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    wins = set(immediate_winning_squares_try(board, me))
    blocks = set(immediate_winning_squares_try(board, 3-me))
    def key(m: Coord2) -> Tuple[int,int,int]:
        x,y = m
        s1 = 3 if m in wins else 0
        s2 = 2 if m in blocks else 0
        s3 = -int(abs(1.5-x)+abs(1.5-y))
        return (s1,s2,s3)
    return sorted(moves, key=key, reverse=True)

def alphabeta_global(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float) -> int:
    if _time_up(deadline) or depth == 0:
        return evaluate_full(board, me)
    wins = immediate_winning_squares_try(board, turn)
    if wins:
        return 200000 - (5 - depth)
    moves = _order_moves_global(board, turn, valid_xy_moves(board))
    if not moves:
        return evaluate_full(board, me)
    best = -10**9
    nxt = 3 - turn
    for (x,y) in moves:
        if _time_up(deadline): break
        z = place_inplace(board, x, y, turn)
        if z is None:
            continue
        sc = -alphabeta_global(board, depth-1, -10**9, 10**9, nxt, me, deadline)
        undo_place(board, x, y, z)
        if sc > best: best = sc
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def ab_fallback_move(board: Board, me: int, deadline: float) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    moves = _order_moves_global(board, me, moves)
    best_mv = moves[0]; best_sc = -10**9
    for d in range(2, AB_MAX_DEPTH+1):
        if _time_up(deadline): break
        for (x,y) in moves:
            if _time_up(deadline): break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -alphabeta_global(board, d-1, -10**9, 10**9, 3-me, me, deadline)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        moves = [best_mv] + [m for m in moves if m != best_mv]
    return best_mv

# ---------- 緊急縦クランプ ----------
def opponent_focus_columns(board: Board, me: int) -> List[Coord2]:
    you = 3 - me
    stats: List[Tuple[int, int, Coord2]] = []
    for y in range(SIZE):
        for x in range(SIZE):
            c_you = 0
            for z in range(SIZE):
                if board[z][y][x] == you:
                    c_you += 1
            if c_you > 0:
                center = int(abs(1.5 - x) + abs(1.5 - y))
                stats.append((c_you, -center, (x, y)))
    stats.sort(reverse=True)
    return [p for _,__,p in stats]

def urgent_vertical_clamp(board: Board, me: int, deadline: float) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    you = 3 - me

    # z==3 直前で相手3個→即勝を最優先で塞ぐ
    for (x, y) in moves:
        if _time_up(deadline): break
        z = lowest_empty_z(board, x, y)
        if z == 3:
            cnt_you = sum(1 for zz in range(3) if board[zz][y][x] == you)
            if cnt_you == 3:
                if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                    return (x, y)

    # z==2/1 での縦圧抑止
    focus = opponent_focus_columns(board, me)
    focus_rank: Dict[Coord2, int] = {p:i for i,p in enumerate(focus)}
    prio: List[Tuple[int, int, int, Coord2]] = []
    for (x, y) in moves:
        if _time_up(deadline): break
        z = lowest_empty_z(board, x, y)
        if z is None: continue
        me_c, you_c = vertical_profile(board, me, x, y)
        if you_c >= 2 and z == 2:
            if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                prio.append((2, -focus_rank.get((x,y), 99), -int(abs(1.5-x)+abs(1.5-y)), (x, y)))
        elif you_c == 1 and z == 1:
            if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                prio.append((1, -focus_rank.get((x,y), 99), -int(abs(1.5-x)+abs(1.5-y)), (x, y)))
    if not prio:
        return None
    prio.sort(reverse=True)
    return prio[0][3]

# ---------- 手選択（角アグレッシブ化を強化） ----------
def choose_best(board: Board, me: int, deadline: float, last_move: Coord3) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0,0)

    total = count_placed(board)
    opp_corner_bases = set(opp_corner_z0(board, me))
    my_bases = my_corner_z0(board, me)

    def _anti_cap_filter(cands: List[Coord2]) -> List[Coord2]:
        # 相手底角の上に重ねる手は原則除外（自即勝/直ブロックはこの前段で返る）
        res: List[Coord2] = []
        for (x,y) in cands:
            if (x,y) in opp_corner_bases and is_cap_on_opponent_corner(board, me, x, y):
                continue
            res.append((x,y))
        return res if res else cands

    # 1) 自即勝
    my_now = immediate_winning_squares_try(board, me)
    if my_now:
        for mv in moves:
            if mv in my_now: return mv
        return _center_sorted(my_now)[0]

    # 2) 相手即勝ブロック（直/間接）
    opp_now = _opp_immediate_now_union(board, 3-me)
    if opp_now:
        direct = [mv for mv in moves if mv in opp_now]
        if direct:
            best = _choose_by_minimax_opp_after(board, me, direct, deadline, 12)
            if best is not None:
                return best
        # 間接ブロック
        best_mv = None; best_key = (999, 9.9, 9.9)
        for (x, y) in _center_sorted(moves):
            if _time_up(deadline): break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            try:
                after = len(_opp_immediate_now_union(board, 3-me))
                worst = _max_opp_immediate_after_reply(board, me, x, y, deadline, _beam_by_time(deadline, 10))
                key = (after, worst, abs(1.5 - x) + abs(1.5 - y))
            finally:
                undo_place(board, x, y, z)
            if key < best_key:
                best_key = key; best_mv = (x, y)
            if best_key[0] == 0 and best_key[1] == 0:
                break
        if best_mv is not None:
            return best_mv

    # 2.5) 緊急縦クランプ
    vb = urgent_vertical_clamp(board, me, deadline)
    if vb is not None:
        return vb

    # 3) ★角(z==0)アグレッシブ：自分の底角が「2以下」かつ z==0角が空いていれば最優先
    corner0 = first_layer_corner_moves(board)
    if corner0 and len(my_bases) <= 2:
        if me == 1:
            # 先手: まずペア角（同じ辺）を最優先
            drive: List[Coord2] = []
            for b in my_bases:
                for t in same_edge_pair_corners(b):
                    if t in corner0:
                        drive.append(t)
            if drive:
                cands = safe_filter_moves(board, me, drive) or drive
                best = _choose_by_minimax_opp_after(board, me, cands, deadline, 12)
                if best is not None: return best
                return _center_sorted(cands)[0]
            # ペアが作れない場合でも、空き底角を最優先（アンチ重ね適用）
            cands = _anti_cap_filter(corner0)
            cands = safe_filter_moves(board, me, cands) or cands
            best = _choose_by_minimax_opp_after(board, me, cands, deadline, 12)
            if best is not None: return best
            return _center_sorted(cands)[0]
        else:
            # 後手: 相手底角に重ねず、別の空き底角を最優先
            try:
                lx, ly, lz = last_move
                prefer = [mv for mv in corner0 if mv != (lx,ly)]
                if prefer:
                    cands = _anti_cap_filter(prefer)
                else:
                    cands = _anti_cap_filter(corner0)
            except Exception:
                cands = _anti_cap_filter(corner0)
            cands = safe_filter_moves(board, me, cands) or cands
            best = _choose_by_minimax_opp_after(board, me, cands, deadline, 12)
            if best is not None: return best
            return _center_sorted(cands)[0]

    # 4) 中央クランプ（z∈{1,2}）
    center_cands = [(x,y) for (x,y) in CENTERS
                    if (x,y) in moves and lowest_empty_z(board,x,y) in (1,2)]
    center_cands = safe_filter_moves(board, me, center_cands)
    center_cands = _anti_cap_filter(center_cands)
    if center_cands:
        if _time_up(deadline): return _center_sorted(center_cands)[0]
        base = len(_opp_immediate_now_union(board, 3-me))
        scored = []
        for (x,y) in center_cands:
            if _time_up(deadline): break
            z = place_inplace(board, x, y, me)
            aft = len(_opp_immediate_now_union(board, 3-me))
            undo_place(board, x, y, z)
            delta = base - aft
            cbias = -int(abs(1.5-x)+abs(1.5-y))
            scored.append(((delta, cbias), (x,y)))
        scored.sort(key=lambda t: t[0], reverse=True)
        cands = [mv for _,mv in scored] if scored else center_cands
        best = _choose_by_minimax_opp_after(board, me, cands, deadline, 12)
        if best is not None: return best
        return cands[0]

    # 5) 辺(2,3層)で側面形
    edge_cands: List[Coord2] = []
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            rc = row_col_corners_for_edge(x, y)
            if any(column_has_my_stone(board, me, cx, cy) for (cx,cy) in rc):
                edge_cands.append((x, y))
    edge_cands = safe_filter_moves(board, me, edge_cands)
    edge_cands = _anti_cap_filter(edge_cands)
    if edge_cands:
        if _time_up(deadline): return _center_sorted(edge_cands)[0]
        best_mv = edge_cands[0]; best_sc = -1
        for (x, y) in edge_cands:
            if _time_up(deadline): break
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        return best_mv

    # 6) 脅威空間 IDS
    if not _time_up(deadline):
        mv_ts = threat_space_best_move_iterative(board, me, deadline)
        if mv_ts is not None:
            return mv_ts

    # 7) 保険: 軽量フルαβ
    if not _time_up(deadline):
        mv_ab = ab_fallback_move(board, me, deadline)
        if mv_ab is not None:
            return mv_ab

    # 8) 最終フォールバック
    cands = safe_filter_moves(board, me, moves)
    cands = _anti_cap_filter(cands)
    best = _choose_by_minimax_opp_after(board, me, cands, deadline, 12)
    if best is not None:
        return best
    return _center_sorted(cands)[0]

# ---------- Hard Block Gate（最後の安全弁） ----------
def hard_block_gate(board: Board, me: int, proposed: Coord2, deadline: float) -> Coord2:
    legal = valid_xy_moves(board)
    if not legal:
        return proposed
    if proposed not in legal:
        return _center_sorted(legal)[0]

    try:
        if _time_up(deadline):
            return proposed

        # 自即勝は通す
        my_now = _opp_immediate_now_union(board, me)
        if proposed in my_now:
            return proposed

        # 置いた後の相手即勝“最大値”が大なら差し替え
        cur_worst = _max_opp_immediate_after_reply(board, me, proposed[0], proposed[1], deadline, _beam_by_time(deadline, 12))
        if cur_worst >= 2:
            safe = _choose_by_minimax_opp_after(board, me, legal, deadline, 12)
            if safe is not None:
                proposed = safe

        # 直ブロックの見落としを再確認
        opp_now = _opp_immediate_now_union(board, 3 - me)
        if opp_now:
            direct = [mv for mv in legal if mv in opp_now]
            if direct:
                best = _choose_by_minimax_opp_after(board, me, direct, deadline, 12)
                if best is not None:
                    return best

        # アンチ重ね：より安全な“非重ね”手に置換できるなら置換
        better: List[Coord2] = []
        for (x, y) in legal:
            if _time_up(deadline): break
            if is_cap_on_opponent_corner(board, me, x, y):
                continue
            w = _max_opp_immediate_after_reply(board, me, x, y, deadline, _beam_by_time(deadline, 10))
            if w < cur_worst:
                better.append((x, y))
        if better:
            alt = _choose_by_minimax_opp_after(board, me, better, deadline, 10)
            if alt is not None:
                proposed = alt

    except Exception:
        return _center_sorted(legal)[0]

    return proposed

# ---------- エンジン ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        deadline = time.perf_counter() + TIME_BUDGET_SEC
        try:
            mv = choose_best(board, player, deadline, last_move)
            mv = force_block_guard(board, player, mv, deadline)
            mv = hard_block_gate(board, player, mv, deadline)
            legal = valid_xy_moves(board)
            return mv if mv in legal else (legal[0] if legal else (0,0))
        except Exception:
            legal = valid_xy_moves(board)
            return _center_sorted(legal)[0] if legal else (0,0)

AI = MyAI()
