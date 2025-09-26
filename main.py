# main.py
# 4x4x4 立体四目並べ AI（省メモリ版）
# - 即勝ち検出を16bitマスク化（集合/リスト大量生成を撤廃）
# - ビーム幅/深さ/候補束を時間＆ノード数で自動縮退（NODE_BUDGET）
# - Corner Claim Override（角主張の上書き）/ 別角優先 / ペア角優先は維持・強化
# - 直ブロック > 角主張 > そのほか、の優先順位を堅持
#
# 禁止API未使用（標準lib: time, typing） / I/O・並列・eval等ゼロ

from typing import List, Tuple, Optional, Dict
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# ---- 探索・資源設定 ----
TIME_BUDGET_SEC = 9.0
TS_MAX_DEPTH    = 4       # IDS(脅威空間)の最大深さ（時間/ノードで自動縮退）
THREAT_BEAM     = 14      # 脅威候補束の基準（時間/ノードで自動縮退）
AB_MAX_DEPTH    = 2       # 軽量ABの深さ上限
NODE_BUDGET     = 120_000 # 1手あたりの概算ノード上限（超えたら即縮退）

# ---- グローバル補助（前計算） ----
def _center_key(x: int, y: int) -> int:
    # 中央寄りほど大 ⇒ 並べ替えで使う
    return 6 - int(abs(1.5 - x) + abs(1.5 - y))

# 76ラインの生成（固定）
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

ALL_LINES: List[List[Coord3]] = generate_lines()
LINES_THROUGH: Dict[Coord3, List[int]] = {}
for li, line in enumerate(ALL_LINES):
    for (x,y,z) in line:
        LINES_THROUGH.setdefault((x,y,z), []).append(li)

# 位置→ビット（16マス）
def idx16(x: int, y: int) -> int: return y*4 + x
def bit_at(x: int, y: int) -> int: return 1 << (y*4 + x)

# 角/中心/辺
CORNERS: List[Coord2] = [(0,0),(3,0),(0,3),(3,3)]
CENTERS: List[Coord2] = [(1,1),(2,1),(1,2),(2,2)]
EDGES  : List[Coord2] = [(1,0),(2,0),(0,1),(3,1),(0,2),(3,2),(1,3),(2,3)]

# ---- 基本ユーティリティ ----
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    top = 3
    res: List[Coord2] = []
    for y in range(4):
        row = board[top][y]
        # x=0..3 を見る（16固定で軽量）
        if row[0] == 0: res.append((0,y))
        if row[1] == 0: res.append((1,y))
        if row[2] == 0: res.append((2,y))
        if row[3] == 0: res.append((3,y))
    return res

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None: return None
    board[z][y][x] = player
    return z

def undo_place(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

def _time_up(deadline: float) -> bool:
    return time.perf_counter() >= deadline

# ---- ノードカウンタ（時間/資源で縮退） ----
class Budget:
    __slots__ = ("nodes",)
    def __init__(self) -> None:
        self.nodes = 0

    def tick(self, k: int = 1) -> bool:
        self.nodes += k
        return self.nodes >= NODE_BUDGET

# ---- 即勝マスク（16bit） ----
def immediate_mask(board: Board, player: int) -> int:
    """player が今すぐ勝てる (x,y) の集合を 16bit マスクで返す"""
    you = 3 - player
    mask = 0
    for line in ALL_LINES:
        cm = cy = 0
        empty: Optional[Coord3] = None
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == player: cm += 1
            elif v == you:  cy += 1
            else:           empty = (x,y,z)
        if cm == 3 and cy == 0 and empty is not None:
            ex,ey,ez = empty
            if lowest_empty_z(board, ex, ey) == ez:
                mask |= bit_at(ex,ey)
    return mask

def mask_has(mask: int, x: int, y: int) -> bool:
    return (mask >> idx16(x,y)) & 1 == 1

def mask_count(mask: int) -> int:
    return mask.bit_count()

# ---- 危険判定（省メモリ） ----
def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None: return False
    ok = False
    for li in LINES_THROUGH[(x,y,z)]:
        line = ALL_LINES[li]
        good = True
        for (xx,yy,zz) in line:
            if board[zz][yy][xx] != player:
                good = False
                break
        if good: ok = True; break
    undo_place(board, x, y, z)
    return ok

def is_t_support_move(board: Board, me: int, x: int, y: int) -> bool:
    # z==2 に置くと相手が z==3 で即勝になる“支え”を回避
    z = lowest_empty_z(board, x, y)
    if z != 2: return False
    z2 = place_inplace(board, x, y, me)
    if z2 is None: return True
    opens = is_winning_after(board, 3-me, x, y)
    undo_place(board, x, y, z2)
    return opens

def is_suicide_move(board: Board, me: int, x: int, y: int) -> bool:
    if is_winning_after(board, me, x, y): return False
    z = place_inplace(board, x, y, me)
    if z is None: return True
    opp_now = immediate_mask(board, 3-me)
    undo_place(board, x, y, z)
    return mask_count(opp_now) >= 2

def safe_filter_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    if not moves: return moves
    safe: List[Coord2] = []
    for (x,y) in moves:
        if is_suicide_move(board, me, x, y):   continue
        if is_t_support_move(board, me, x, y): continue
        safe.append((x,y))
    return safe if safe else moves

# ---- 補助（角/ペア角/重ね判定） ----
def my_corner_z0(board: Board, me: int) -> List[Coord2]:
    res: List[Coord2] = []
    for (x,y) in CORNERS:
        if board[0][y][x] == me:
            res.append((x,y))
    return res

def first_layer_corner_moves(board: Board) -> List[Coord2]:
    mv: List[Coord2] = []
    for (x,y) in CORNERS:
        if lowest_empty_z(board, x, y) == 0:
            mv.append((x,y))
    return mv

def same_edge_pair_targets(c: Coord2) -> List[Coord2]:
    x,y = c
    pairs: List[Coord2] = []
    if y == 0: pairs.append((3,0))
    if y == 3: pairs.append((0,3))
    if x == 0: pairs.append((0,3))
    if x == 3: pairs.append((3,0))
    # uniq
    r: List[Coord2] = []
    for p in pairs:
        if p != c and p not in r:
            r.append(p)
    return r

def is_cap_on_opponent_corner(board: Board, me: int, x: int, y: int) -> bool:
    if (x,y) not in CORNERS: return False
    z = lowest_empty_z(board, x, y)
    if z is None or z == 0: return False
    return board[0][y][x] == (3 - me)

# ---- 相手最善1手後の“相手即勝数の最大値”を最小化 ----
def _max_opp_immediate_after_reply(board: Board, me: int, x: int, y: int, deadline: float, bud: Budget, beam: int) -> int:
    if _time_up(deadline): return 2
    z0 = place_inplace(board, x, y, me)
    if z0 is None: return 99
    you = 3 - me
    legal = valid_xy_moves(board)
    opp_now = immediate_mask(board, you)
    # 優先: 相手の即勝手 / 不足なら中央寄りで補完
    cands: List[Coord2] = []
    if opp_now:
        for (ox,oy) in legal:
            if mask_has(opp_now, ox, oy):
                cands.append((ox,oy))
    if len(cands) < beam:
        # 中央寄り補完
        legal_sorted = sorted(legal, key=lambda p: _center_key(p[0],p[1]), reverse=True)
        for mv in legal_sorted:
            if mv not in cands:
                cands.append(mv)
            if len(cands) >= beam: break

    worst = 0
    for (ox,oy) in cands:
        if _time_up(deadline): break
        if bud.tick(): break
        z1 = place_inplace(board, ox, oy, you)
        if z1 is None:
            continue
        cnt = mask_count(immediate_mask(board, you))
        if cnt > worst: worst = cnt
        undo_place(board, ox, oy, z1)
        if worst >= 2: break
    undo_place(board, x, y, z0)
    return worst

def _choose_by_minimax_opp_after(board: Board, me: int, moves: List[Coord2], deadline: float, bud: Budget, base_beam: int) -> Optional[Coord2]:
    if not moves: return None
    # 残時間/ノードでビーム縮退
    left = max(0.0, deadline - time.perf_counter())
    beam = base_beam
    if left < 0.35: beam = max(6, base_beam//3)
    elif left < 0.9: beam = max(8, base_beam//2)
    if bud.nodes > NODE_BUDGET*0.7: beam = max(6, beam//2)

    best_mv = moves[0]; best_key = (999, -1)  # (worstの小さい順 / 中央寄り大きい順)
    for (x,y) in moves:
        if _time_up(deadline): break
        w = _max_opp_immediate_after_reply(board, me, x, y, deadline, bud, beam)
        key = (w, _center_key(x,y))
        if key < best_key:
            best_key = key; best_mv = (x,y)
            if w == 0: break
    return best_mv

# ---- 縦/評価などの補助 ----
def vertical_profile(board: Board, player: int, x: int, y: int) -> Tuple[int,int]:
    you = 3 - player
    a=b=0
    col0,col1,col2,col3 = board[0][y][x], board[1][y][x], board[2][y][x], board[3][y][x]
    if   col0 == player: a+=1
    elif col0 == you:    b+=1
    if   col1 == player: a+=1
    elif col1 == you:    b+=1
    if   col2 == player: a+=1
    elif col2 == you:    b+=1
    if   col3 == player: a+=1
    elif col3 == you:    b+=1
    return a,b

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
    bad = False
    s = 0
    for (xx,yy,zz) in req:
        v = board[zz][yy][xx]
        if v == you:
            bad = True; break
        if v == me:
            s += 1
    undo_place(board, x, y, z)
    return 0 if bad else s

# ---- 直ブロックガード（省メモリ版） ----
def force_block_guard(board: Board, me: int, chosen: Coord2, deadline: float, bud: Budget) -> Coord2:
    # 自即勝は通す
    my_now = immediate_mask(board, me)
    if my_now and mask_has(my_now, chosen[0], chosen[1]):
        return chosen

    opp_now = immediate_mask(board, 3-me)
    if opp_now == 0:
        return chosen
    # chosen が直ブロックなら採用
    if mask_has(opp_now, chosen[0], chosen[1]):
        return chosen

    # 直ブロック候補から“最悪即勝数”最小を選ぶ
    legal = valid_xy_moves(board)
    direct: List[Coord2] = [(x,y) for (x,y) in legal if mask_has(opp_now, x, y)]
    if direct:
        best = _choose_by_minimax_opp_after(board, me, direct, deadline, bud, 12)
        return best if best is not None else direct[0]

    # 間接ブロック（控えめ）：after / worst / 中央寄り の順で最小
    best_mv = None; best_key = (999, 999, -1)
    for (x,y) in sorted(legal, key=lambda p: _center_key(p[0],p[1]), reverse=True):
        if _time_up(deadline): break
        if bud.tick(): break
        z = place_inplace(board, x, y, me)
        if z is None:
            continue
        try:
            after = mask_count(immediate_mask(board, 3-me))
            worst = _max_opp_immediate_after_reply(board, me, x, y, deadline, bud, 10)
            key = (after, worst, _center_key(x,y))
        finally:
            undo_place(board, x, y, z)
        if key < best_key:
            best_key = key; best_mv = (x,y)
        if best_key[0] == 0 and best_key[1] == 0:
            break
    return best_mv if best_mv is not None else (legal[0] if legal else (0,0))

# ---- Corner Claim Override（角主張を上書き適用） ----
def corner_claim_override(board: Board, me: int, last_move: Coord3, deadline: float, bud: Budget) -> Optional[Coord2]:
    # 直ブロックが要らないときのみ
    if immediate_mask(board, 3-me):
        return None
    corner0 = first_layer_corner_moves(board)
    if not corner0:
        return None

    legal = valid_xy_moves(board)
    my_bases = my_corner_z0(board, me)

    # 候補列構築：先手=ペア角>他角 / 後手=別角>他角（重ね回避）
    cands: List[Coord2] = []
    if me == 1:
        if len(my_bases) <= 2:
            for b in my_bases:
                for t in same_edge_pair_targets(b):
                    if t in corner0 and t not in cands:
                        cands.append(t)
        for t in corner0:
            if t not in cands:
                cands.append(t)
    else:
        try:
            lx,ly,lz = last_move
            if (lx,ly) in CORNERS and lz == 0:
                for t in corner0:
                    if t != (lx,ly) and t not in cands:
                        cands.append(t)
        except Exception:
            pass
        for t in corner0:
            if t not in cands:
                cands.append(t)

    cands = [mv for mv in cands if mv in legal]
    cands = safe_filter_moves(board, me, cands) or cands
    if not cands:
        return None

    best = _choose_by_minimax_opp_after(board, me, cands, deadline, bud, 12)
    if best is None:
        return None
    # 安全条件：worst<=1
    w = _max_opp_immediate_after_reply(board, me, best[0], best[1], deadline, bud, 10)
    return best if w <= 1 else None

# ---- IDS（脅威空間） ----
def _tactical_eval(board: Board, me: int) -> int:
    you = 3 - me
    my_now  = mask_count(immediate_mask(board, me))
    opp_now = mask_count(immediate_mask(board, you))
    score = 10000*(my_now - opp_now)
    if my_now >= 2:  score += 1500
    if opp_now >= 2: score -= 2000
    # 縦圧＋中央寄り
    for y in range(4):
        for x in range(4):
            mc,yc = vertical_profile(board, me, x, y)
            if mc >= 2 and yc == 0: score += 3
            w = _center_key(x,y)
            for z in range(4):
                v = board[z][y][x]
                if v == me: score += w
                elif v == you: score -= w
    return score

def _tactical_candidates(board: Board, turn: int, me: int, deadline: float, bud: Budget) -> List[Coord2]:
    legal = valid_xy_moves(board)
    if not legal: return []
    if _time_up(deadline): return sorted(legal, key=lambda p: _center_key(p[0],p[1]), reverse=True)[:6]
    you = 3 - turn
    wins_t = immediate_mask(board, turn)
    wins_o = immediate_mask(board, you)
    cands: List[Coord2] = []
    # 即勝/即ブロック
    if wins_t:
        for (x,y) in legal:
            if mask_has(wins_t, x, y):
                cands.append((x,y))
    if wins_o:
        for (x,y) in legal:
            if mask_has(wins_o, x, y) and (x,y) not in cands:
                cands.append((x,y))
    # 中央/辺(2,3層)
    for (x,y) in CENTERS:
        if (x,y) in legal and lowest_empty_z(board, x, y) in (1,2) and (x,y) not in cands:
            cands.append((x,y))
    for (x,y) in EDGES:
        if (x,y) in legal and lowest_empty_z(board, x, y) in (1,2) and (x,y) not in cands:
            cands.append((x,y))
    # 不足なら中央寄り補完（省メモリ）
    need = THREAT_BEAM
    left = max(0.0, deadline - time.perf_counter())
    if left < 0.35: need = max(6, THREAT_BEAM//3)
    elif left < 0.9: need = max(8, THREAT_BEAM//2)
    if bud.nodes > NODE_BUDGET*0.7: need = max(6, need//2)

    if len(cands) < need:
        for mv in sorted(legal, key=lambda p: _center_key(p[0],p[1]), reverse=True):
            if mv not in cands:
                cands.append(mv)
            if len(cands) >= need: break
    return cands[:need]

def _negamax_threat(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float, bud: Budget) -> int:
    if _time_up(deadline) or depth == 0 or bud.tick():
        return _tactical_eval(board, me)
    if immediate_mask(board, turn):
        return 100000 - (5 - depth)
    moves = _tactical_candidates(board, turn, me, deadline, bud)
    if not moves:
        return _tactical_eval(board, me)
    best = -10**9
    nxt = 3 - turn
    for (x,y) in moves:
        if _time_up(deadline) or bud.tick(): break
        z = place_inplace(board, x, y, turn)
        if z is None or is_t_support_move(board, turn, x, y):
            if z is not None: undo_place(board, x, y, z)
            continue
        sc = -_negamax_threat(board, depth-1, -beta, -alpha, nxt, me, deadline, bud)
        undo_place(board, x, y, z)
        if sc > best: best = sc
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def threat_space_best_move_iterative(board: Board, me: int, deadline: float, bud: Budget) -> Optional[Coord2]:
    root = _tactical_candidates(board, me, me, deadline, bud)
    if not root: return None
    root = safe_filter_moves(board, me, root)
    best_mv: Optional[Coord2] = root[0]
    opp = 3 - me
    max_d = TS_MAX_DEPTH
    # 時間/ノードで深さ縮退
    left = max(0.0, deadline - time.perf_counter())
    if left < 0.35: max_d = 2
    elif left < 0.9: max_d = min(max_d, 3)
    if bud.nodes > NODE_BUDGET*0.6: max_d = min(max_d, 3)
    for d in range(2, max_d+1):
        if _time_up(deadline): break
        best_sc = -10**9
        for (x,y) in root:
            if _time_up(deadline) or bud.tick(): break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -_negamax_threat(board, d-1, -10**9, 10**9, opp, me, deadline, bud)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x,y)
        # PV先頭化
        if best_mv in root:
            root = [best_mv] + [m for m in root if m != best_mv]
        if bud.nodes > NODE_BUDGET*0.85: break
    return best_mv

# ---- 全局評価 / 軽量AB（省メモリ） ----
def evaluate_full(board: Board, me: int) -> int:
    you = 3 - me
    my_th = opp_th = 0
    sc = 0
    for line in ALL_LINES:
        mc = oc = 0
        empty = 0
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == me: mc += 1
            elif v == you: oc += 1
            else: empty += 1
        if oc == 0:
            if mc == 4: return 10**9
            if mc == 3 and empty == 1: sc += 300; my_th += 1
            elif mc == 2 and empty == 2: sc += 20
            elif mc == 1 and empty == 3: sc += 2
        elif mc == 0:
            if oc == 4: return -10**9
            if oc == 3 and empty == 1: sc -= 300; opp_th += 1
            elif oc == 2 and empty == 2: sc -= 20
            elif oc == 1 and empty == 3: sc -= 2
    if my_th >= 2:  sc += 5000
    if opp_th >= 2: sc -= 5000
    for y in range(4):
        for x in range(4):
            w = _center_key(x,y)
            for z in range(4):
                v = board[z][y][x]
                if v == me: sc += w
                elif v == you: sc -= w
    return sc

def _order_moves_global(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    wins = immediate_mask(board, me)
    blocks = immediate_mask(board, 3-me)
    return sorted(
        moves,
        key=lambda m: (
            3 if wins and mask_has(wins, m[0], m[1]) else 0,
            2 if blocks and mask_has(blocks, m[0], m[1]) else 0,
            _center_key(m[0], m[1])
        ),
        reverse=True
    )

def alphabeta_global(board: Board, depth: int, alpha: int, beta: int, turn: int, me: int, deadline: float, bud: Budget) -> int:
    if _time_up(deadline) or depth == 0 or bud.tick():
        return evaluate_full(board, me)
    if immediate_mask(board, turn):
        return 200000 - (5 - depth)
    moves = _order_moves_global(board, turn, valid_xy_moves(board))
    if not moves:
        return evaluate_full(board, me)
    best = -10**9
    nxt = 3 - turn
    for (x,y) in moves:
        if _time_up(deadline) or bud.tick(): break
        z = place_inplace(board, x, y, turn)
        if z is None:
            continue
        sc = -alphabeta_global(board, depth-1, -10**9, 10**9, nxt, me, deadline, bud)
        undo_place(board, x, y, z)
        if sc > best: best = sc
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def ab_fallback_move(board: Board, me: int, deadline: float, bud: Budget) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    moves = _order_moves_global(board, me, moves)
    best_mv = moves[0]; best_sc = -10**9
    max_d = AB_MAX_DEPTH
    left = max(0.0, deadline - time.perf_counter())
    if left < 0.4: max_d = 1
    for d in range(1, max_d+1):
        if _time_up(deadline) or bud.tick(): break
        for (x,y) in moves:
            if _time_up(deadline) or bud.tick(): break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -alphabeta_global(board, d, -10**9, 10**9, 3-me, me, deadline, bud)
            undo_place(board, x, y, z)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        moves = [best_mv] + [m for m in moves if m != best_mv]
    return best_mv

# ---- 縦クランプ（直ブロック不要なら z==3直前のみ） ----
def urgent_vertical_clamp(board: Board, me: int, deadline: float, bud: Budget) -> Optional[Coord2]:
    legal = valid_xy_moves(board)
    if not legal: return None
    opp_now = immediate_mask(board, 3-me)
    # 直ブロック不要 → z==3直前のみケア
    if opp_now == 0:
        for (x,y) in legal:
            z = lowest_empty_z(board, x, y)
            if z == 3:
                cnt_you = 0
                for zz in range(3):
                    if board[zz][y][x] == (3 - me):
                        cnt_you += 1
                if cnt_you == 3:
                    mv = (x,y)
                    if mv in safe_filter_moves(board, me, [mv]):
                        return mv
        return None

    # 直ブロック必要局面 → 従来型（控えめ）
    prio: List[Tuple[int,int,Coord2]] = []
    for (x,y) in legal:
        z = lowest_empty_z(board, x, y)
        if z is None: continue
        mc,yc = vertical_profile(board, me, x, y)
        if yc >= 2 and z == 2:
            if (x,y) in safe_filter_moves(board, me, [(x,y)]):
                prio.append((2, _center_key(x,y), (x,y)))
        elif yc == 1 and z == 1:
            if (x,y) in safe_filter_moves(board, me, [(x,y)]):
                prio.append((1, _center_key(x,y), (x,y)))
    if not prio: return None
    prio.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return prio[0][2]

# ---- 手選択 ----
def choose_best(board: Board, me: int, deadline: float, last_move: Coord3, bud: Budget) -> Coord2:
    legal = valid_xy_moves(board)
    if not legal: return (0,0)

    # 1) 自即勝
    my_now = immediate_mask(board, me)
    if my_now:
        for (x,y) in legal:
            if mask_has(my_now, x, y):
                return (x,y)
        # 念のため
        best = max(legal, key=lambda p: _center_key(p[0],p[1]))
        return best

    # 2) 直ブロック
    opp_now = immediate_mask(board, 3-me)
    if opp_now:
        direct: List[Coord2] = [(x,y) for (x,y) in legal if mask_has(opp_now, x, y)]
        if direct:
            best = _choose_by_minimax_opp_after(board, me, direct, deadline, bud, 12)
            if best is not None: return best
            return direct[0]

    # 2.5) Corner Claim Override
    mv_corner = corner_claim_override(board, me, last_move, deadline, bud)
    if mv_corner is not None:
        return mv_corner

    # 3) 縦クランプ（条件付き）
    vb = urgent_vertical_clamp(board, me, deadline, bud)
    if vb is not None:
        return vb

    # 4) 中央クランプ（z∈{1,2}）→ 最悪即勝数最小化
    centers = [(x,y) for (x,y) in CENTERS if (x,y) in legal and lowest_empty_z(board,x,y) in (1,2)]
    centers = safe_filter_moves(board, me, centers)
    if centers:
        best = _choose_by_minimax_opp_after(board, me, centers, deadline, bud, 12)
        if best is not None: return best
        return max(centers, key=lambda p: _center_key(p[0],p[1]))

    # 5) エッジ(2,3層)で側面形（自己列に角がある辺）
    edges: List[Coord2] = []
    for (x,y) in EDGES:
        if (x,y) in legal and lowest_empty_z(board, x, y) in (1,2):
            rc: List[Coord2] = []
            if y == 0: rc += [(0,0),(3,0)]
            if y == 3: rc += [(0,3),(3,3)]
            if x == 0: rc += [(0,0),(0,3)]
            if x == 3: rc += [(3,0),(3,3)]
            hit = False
            for (cx,cy) in rc:
                if board[0][cy][cx] == me:
                    hit = True; break
            if hit:
                edges.append((x,y))
    edges = safe_filter_moves(board, me, edges)
    if edges:
        # 側面テンプレスコア最大
        best_mv = edges[0]; best_sc = -1
        for (x,y) in edges:
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_sc:
                best_sc = sc; best_mv = (x, y)
        return best_mv

    # 6) 脅威空間IDS
    mv_ts = threat_space_best_move_iterative(board, me, deadline, bud)
    if mv_ts is not None:
        return mv_ts

    # 7) 軽量AB
    mv_ab = ab_fallback_move(board, me, deadline, bud)
    if mv_ab is not None:
        return mv_ab

    # 8) 最終フォールバック：最悪即勝数の最小化
    cands = safe_filter_moves(board, me, legal)
    best = _choose_by_minimax_opp_after(board, me, cands, deadline, bud, 12)
    if best is not None:
        return best
    return max(cands, key=lambda p: _center_key(p[0],p[1]))

# ---- Hard Block Gate（角主張中の置換は厳格に） ----
def hard_block_gate(board: Board, me: int, proposed: Coord2, deadline: float, bud: Budget) -> Coord2:
    legal = valid_xy_moves(board)
    if not legal:
        return proposed
    if proposed not in legal:
        return max(legal, key=lambda p: _center_key(p[0],p[1]))

    # 自即勝は通す
    if mask_has(immediate_mask(board, me), proposed[0], proposed[1]):
        return proposed

    # 直ブロック優先
    opp_now = immediate_mask(board, 3-me)
    if opp_now:
        direct = [(x,y) for (x,y) in legal if mask_has(opp_now, x, y)]
        if direct:
            best = _choose_by_minimax_opp_after(board, me, direct, deadline, bud, 12)
            return best if best is not None else direct[0]

    # 角主張中なら、非角への置換は“厳密に改善”のみ
    cur_w = _max_opp_immediate_after_reply(board, me, proposed[0], proposed[1], deadline, bud, 10)
    proposed_is_corner_claim = (proposed in CORNERS and lowest_empty_z(board, proposed[0], proposed[1]) == 0)

    # 別の角（非重ね）で同等以下の危険度なら置換
    alt_corners: List[Coord2] = []
    for (x,y) in legal:
        if (x,y) in CORNERS and lowest_empty_z(board, x, y) == 0 and (x,y) != proposed:
            if not is_cap_on_opponent_corner(board, me, x, y):
                w = _max_opp_immediate_after_reply(board, me, x, y, deadline, bud, 8)
                if w <= cur_w:
                    alt_corners.append((x,y))
    if alt_corners and is_cap_on_opponent_corner(board, me, proposed[0], proposed[1]):
        best = _choose_by_minimax_opp_after(board, me, alt_corners, deadline, bud, 8)
        if best is not None:
            return best

    # 非角への置換：角主張中は w < cur_w のときのみ
    noncorner_better: List[Coord2] = []
    for (x,y) in legal:
        if (x,y) in CORNERS and lowest_empty_z(board, x, y) == 0:
            continue
        if is_cap_on_opponent_corner(board, me, x, y):
            continue
        w = _max_opp_immediate_after_reply(board, me, x, y, deadline, bud, 8)
        if (proposed_is_corner_claim and w < cur_w) or (not proposed_is_corner_claim and w <= cur_w):
            noncorner_better.append((x,y))
    if noncorner_better:
        alt = _choose_by_minimax_opp_after(board, me, noncorner_better, deadline, bud, 8)
        if alt is not None:
            return alt

    return proposed

# ---- エンジン ----
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        deadline = time.perf_counter() + TIME_BUDGET_SEC
        bud = Budget()
        try:
            mv = choose_best(board, player, deadline, last_move, bud)
            mv = force_block_guard(board, player, mv, deadline, bud)
            mv = hard_block_gate(board, player, mv, deadline, bud)
            legal = valid_xy_moves(board)
            return mv if mv in legal else (legal[0] if legal else (0,0))
        except Exception:
            legal = valid_xy_moves(board)
            return max(legal, key=lambda p: _center_key(p[0],p[1])) if legal else (0,0)

AI = MyAI()
