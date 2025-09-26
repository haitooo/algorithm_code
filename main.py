# main.py
# 4x4x4 立体四目並べ AI — 角(z==0)強制優先 + 脅威空間(IDS) + 全76ライン評価 + 低深度AB
# 方針:
#  1) 即勝ち > 即ブロック（直/間接）> 角z==0（先手は必ず角 / 後手は重ねず別角）> 中央/縦圧 > 脅威限定探索(IDS)
#  2) それでも不明なら、低深度のフルαβ(2〜3ply)で「ダブルリーチ/偶奇/中心」重視の評価で最終判断
#  3) すべて安全フィルタを通し、失敗時も必ず合法手を返す
#
# 禁止API: 一切未使用（外部I/Oや乱数も不使用）。標準libの time と typing のみ。

from typing import List, Tuple, Optional, Dict, Set
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# ---- 探索予算 ----
TIME_BUDGET_SEC = 9.5       # 全体持ち時間
TS_MAX_DEPTH    = 4         # 脅威空間 IDS 深さ
THREAT_BEAM     = 12        # 脅威候補束
AB_MAX_DEPTH    = 3         # 最終保険のミニαβの最大深さ（軽量）

# ---------- 基本ユーティリティ ----------
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

# ---------- 全76ライン ----------
def generate_lines() -> List[List[Coord3]]:
    L: List[List[Coord3]] = []
    # 平行3軸
    for z in range(SIZE):
        for y in range(SIZE):
            L.append([(x, y, z) for x in range(SIZE)])
    for z in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for y in range(SIZE)])
    for y in range(SIZE):
        for x in range(SIZE):
            L.append([(x, y, z) for z in range(SIZE)])
    # 平面斜め
    for z in range(SIZE):
        L.append([(i, i, z) for i in range(SIZE)])
        L.append([(i, SIZE-1-i, z) for i in range(SIZE)])
    for y in range(SIZE):
        L.append([(i, y, i) for i in range(SIZE)])
        L.append([(i, y, SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        L.append([(x, i, i) for i in range(SIZE)])
        L.append([(x, i, SIZE-1-i) for i in range(SIZE)])
    # 3D斜め(4本)
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

# ---------- 即勝ち検出 ----------
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

# ---------- 座標ラベル ----------
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

# ---------- 側面テンプレ（逆ミッキー絡み） ----------
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

# ---------- 安全系（自殺・t支え・DT誘発） ----------
def is_t_support_move(board: Board, me: int, x: int, y: int) -> bool:
    # z==2 で置くと相手のその列勝ち(=支え)になる危険
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
    # 置いた直後に相手のDT>=2を許すなら危険。ただし自即勝は除外。
    if is_winning_after(board, me, x, y): return False
    z = place_inplace(board, x, y, me)
    if z is None: return True
    opp_dts = len(immediate_winning_squares_try(board, 3-me))
    undo_place(board, x, y, z)
    return opp_dts >= 2

def safe_filter_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    safe: List[Coord2] = []
    for (x,y) in moves:
        if is_suicide_move(board, me, x, y):   continue
        if is_t_support_move(board, me, x, y): continue
        safe.append((x,y))
    return safe if safe else moves

# ---------- 角周りの補助 ----------
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

# ---------- 即ブロック ----------
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

def force_block_guard(board: Board, me: int, chosen: Coord2) -> Coord2:
    my_now = immediate_winning_squares_try(board, me)
    if my_now: return chosen
    opp_set = set(immediate_winning_squares_try(board, 3-me))
    if not opp_set: return chosen
    if chosen in opp_set: return chosen
    moves = valid_xy_moves(board)
    direct = [mv for mv in moves if mv in opp_set]
    if direct:
        scored = [(_score_direct_block(board, me, mv, list(opp_set)), mv) for mv in direct]
        scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
        return scored[0][1]
    # 間接ブロック：相手の即勝ち数を最小化
    best = None; best_after = 10**9
    for (x, y) in _center_sorted(moves):
        z = place_inplace(board, x, y, me)
        if z is None: continue
        after = len(immediate_winning_squares_try(board, 3-me))
        undo_place(board, x, y, z)
        if after == 0: return (x, y)
        if after < best_after:
            best_after = after; best = (x, y)
    return best if best is not None else (_center_sorted(moves)[0] if moves else (0,0))

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
    if my_now >= 2:  score += 1500      # 自DT
    if opp_now >= 2: score -= 2000      # 相手DT
    # 中央/縦圧
    for y in range(SIZE):
        for x in range(SIZE):
            mc, yc = vertical_profile(board, me, x, y)
            if mc >= 2 and yc == 0: score += 3
            # 中央寄り配置を微加点
            s = 3 - int(abs(1.5-x)+abs(1.5-y))
            for z in range(SIZE):
                v = board[z][y][x]
                if v==me: score += s
                elif v==you: score -= s
    return score

def _tactical_candidates(board: Board, turn: int, me: int) -> List[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return []
    you = 3 - turn
    wins_turn = set(immediate_winning_squares_try(board, turn))
    wins_opp  = set(immediate_winning_squares_try(board, you))
    cands: Set[Coord2] = set()
    cands |= wins_turn
    cands |= wins_opp
    # 中央/辺(2,3層)を軽く混ぜる
    for (x, y) in CENTERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.add((x,y))
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.add((x,y))
    # 少ないなら中心順に補完
    if len(cands) < 6:
        for mv in _center_sorted(moves):
            cands.add(mv)
            if len(cands) >= 8: break
    ordered = list(cands)
    def _score(m: Coord2) -> Tuple[int,int,int]:
        x,y = m
        s1 = 2 if m in wins_turn else 0
        s2 = 1 if m in wins_opp  else 0
        s3 = -int(abs(1.5-x)+abs(1.5-y))
        return (s1,s2,s3)
    ordered.sort(key=_score, reverse=True)
    return ordered[:THREAT_BEAM]

def _negamax_threat(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float) -> int:
    if time.perf_counter() >= deadline or depth == 0:
        return _tactical_eval(board, me)
    wins = immediate_winning_squares_try(board, turn)
    if wins:
        return 100000 - (5 - depth)
    moves = _tactical_candidates(board, turn, me)
    if not moves:
        return _tactical_eval(board, me)
    best = -10**9
    nxt = 3 - turn
    for (x, y) in moves:
        if time.perf_counter() >= deadline: break
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
    root = _tactical_candidates(board, me, me)
    if not root: return None
    root = safe_filter_moves(board, me, root)
    best_mv: Optional[Coord2] = root[0]
    opp = 3 - me
    max_d = TS_MAX_DEPTH
    for d in range(2, max_d+1):
        if time.perf_counter() >= deadline: break
        best_sc = -10**9
        for (x, y) in root:
            if time.perf_counter() >= deadline: break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -_negamax_threat(board, d-1, -10**9, 10**9, opp, me, deadline)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        # PV を先頭に
        if best_mv in root:
            root = [best_mv] + [m for m in root if m != best_mv]
    return best_mv

# ---------- 全局評価（76ライン・ダブルリーチ重視） ----------
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
    # 中央寄り
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
    if time.perf_counter() >= deadline or depth == 0:
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
        if time.perf_counter() >= deadline: break
        z = place_inplace(board, x, y, turn)
        if z is None:
            continue
        sc = -alphabeta_global(board, depth-1, -beta, -alpha, nxt, me, deadline)
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
        if time.perf_counter() >= deadline: break
        for (x,y) in moves:
            if time.perf_counter() >= deadline: break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -alphabeta_global(board, d-1, -10**9, 10**9, 3-me, me, deadline)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        # PV先頭化
        moves = [best_mv] + [m for m in moves if m != best_mv]
    return best_mv

# ---------- メイン手選択 ----------
def choose_best(board: Board, me: int, deadline: float, last_move: Coord3) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0,0)

    # 1) 自即勝
    my_now = immediate_winning_squares_try(board, me)
    if my_now:
        for mv in moves:
            if mv in my_now: return mv
        return _center_sorted(my_now)[0]

    # 2) 相手即勝ブロック
    opp_now = immediate_winning_squares_try(board, 3-me)
    if opp_now:
        direct = [mv for mv in moves if mv in opp_now]
        if direct:
            scored = [(_score_direct_block(board, me, mv, opp_now), mv) for mv in direct]
            scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]), reverse=True)
            return scored[0][1]
        # 間接
        best = None; best_after = 10**9
        for (x, y) in _center_sorted(moves):
            z = place_inplace(board, x, y, me)
            if z is None: continue
            after = len(immediate_winning_squares_try(board, 3-me))
            undo_place(board, x, y, z)
            if after == 0: return (x, y)
            if after < best_after:
                best_after = after; best = (x, y)
        if best is not None: return best

    # 3) 角(z==0)強制優先
    corner0 = first_layer_corner_moves(board)
    if corner0:
        # 後手は「重ねず別角」優先
        if me == 2:
            lx, ly, lz = last_move
            if (lx,ly) in CORNERS and lz == 0:
                others = [(x, y) for (x, y) in corner0 if (x, y) != (lx, ly)]
                if others:
                    cands = safe_filter_moves(board, me, others) or others
                    return sorted(cands, key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))[0]
        cands = safe_filter_moves(board, me, corner0) or corner0
        return sorted(cands, key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))[0]

    # 4) 中央クランプ（z∈{1,2}）
    center_cands = [(x,y) for (x,y) in CENTERS
                    if (x,y) in moves and lowest_empty_z(board,x,y) in (1,2)]
    center_cands = safe_filter_moves(board, me, center_cands)
    if center_cands:
        # 相手の即勝ち数削減度でソート
        base = len(immediate_winning_squares_try(board, 3-me))
        scored = []
        for (x,y) in center_cands:
            z = place_inplace(board, x, y, me)
            aft = len(immediate_winning_squares_try(board, 3-me))
            undo_place(board, x, y, z)
            delta = base - aft
            cbias = -int(abs(1.5-x)+abs(1.5-y))
            scored.append(((delta, cbias), (x,y)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    # 5) 辺(2,3層)で側面形を拡大（自列に角があるエッジ優先）
    edge_cands: List[Coord2] = []
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            rc = row_col_corners_for_edge(x, y)
            if any(column_has_my_stone(board, me, cx, cy) for (cx,cy) in rc):
                edge_cands.append((x, y))
    edge_cands = safe_filter_moves(board, me, edge_cands)
    if edge_cands:
        best_mv = edge_cands[0]; best_sc = -1
        for (x, y) in edge_cands:
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        return best_mv

    # 6) 脅威空間 IDS
    mv_ts = threat_space_best_move_iterative(board, me, deadline)
    if mv_ts is not None:
        return mv_ts

    # 7) 保険: 軽量フルαβ（評価=76ライン+DT重視）
    mv_ab = ab_fallback_move(board, me, deadline)
    if mv_ab is not None:
        return mv_ab

    # 8) 最終フォールバック：中央寄りの安全手
    cands = safe_filter_moves(board, me, moves)
    return _center_sorted(cands)[0]

# ---------- エンジン ----------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        deadline = time.perf_counter() + TIME_BUDGET_SEC
        try:
            mv = choose_best(board, player, deadline, last_move)
            mv = force_block_guard(board, player, mv)
            legal = valid_xy_moves(board)
            return mv if mv in legal else (legal[0] if legal else (0,0))
        except Exception:
            legal = valid_xy_moves(board)
            return legal[0] if legal else (0,0)

AI = MyAI()
