# main.py
# 4x4x4 立体四目並べ AI（角最優先 + 縦列クランプ + トラップ重視 + 脅威限定深読み(TT/キラー)）
# 収録機能：
# - 角z==0徹底（先手は最優先／後手は相手重ね無視→別角）
# - 近接応手（序盤に相手が内側へ置いたら近傍を優先：安全・角・ブロックを守る）
# - 全76ライン精査 / 厳密即勝ち / ダイレクトブロック
# - ≤2手到達の予防ブロック（列優先でクランプ）
# - 縦列クランプ（相手z=1/2 を先取り、自縦列z=1積みで圧）
# - “逆ミッキー完成→サイド直行” と “側面ターゲット形”評価
# - トラップ優先（置いた直後の自DT本数最大化、DT>=2&相手即勝0を最上位）
# - 安全フィルタ（自殺・t支え・相手DT誘発>=2 の抑止。ただし自即勝/直ブロは許容）
# - 脅威限定 Negamax + 反復深化 + TT(Zobrist) + キラームーブ
# - 時間制御（相手が列集中していれば自動で深さ+1）

from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import random, time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)
SIZE = 4

# ===== 時間・探索パラメータ =====
TIME_BUDGET_SEC = 14.8
TS_MAX_DEPTH    = 6     # 基本深さ（局面で+1）
THREAT_BEAM     = 14    # 脅威候補の束幅

# ===== Zobrist / TT / Killer =====
random.seed(2024)
ZKEY = [[[(random.getrandbits(64), random.getrandbits(64))
          for _x in range(SIZE)] for _y in range(SIZE)] for _z in range(SIZE)]

def board_hash(board: Board) -> int:
    h = 0
    for z in range(SIZE):
        for y in range(SIZE):
            for x in range(SIZE):
                v = board[z][y][x]
                if v == 1: h ^= ZKEY[z][y][x][0]
                elif v == 2: h ^= ZKEY[z][y][x][1]
    return h

# TT: (hash, turn) -> (depth, flag, score, best_move)  flag: 0=EXACT, -1=UPPER, +1=LOWER
TT: Dict[Tuple[int,int], Tuple[int,int,int, Optional[Coord2]]] = {}
# Killer: ply -> up to 2 moves
KILLERS: Dict[int, List[Coord2]] = defaultdict(list)

# ===== 基本ユーティリティ =====
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):              # 重力 z=0→上
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

def manhattan(a: Coord2, b: Coord2) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def count_placed(board: Board) -> int:
    c = 0
    for z in range(SIZE):
        for y in range(SIZE):
            for x in range(SIZE):
                if board[z][y][x] != 0:
                    c += 1
    return c

# ===== 勝ち筋（76本） =====
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

# 最後の一手を含むラインのみで判定する索引
LINES_THROUGH: Dict[Coord3, List[int]] = {}
for li, line in enumerate(ALL_LINES):
    for (x,y,z) in line:
        LINES_THROUGH.setdefault((x,y,z), []).append(li)

# ===== 即勝ち・勝ち判定 =====
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

# ===== ラベル→座標 =====
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

# ===== 側面ターゲット形・逆ミッキー =====
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
    if z is None:
        return 0
    if y in (0, 3): req = sideview_required_cells_for_row(y)
    else:           req = sideview_required_cells_for_col(x)
    opp_block = any(board[zz][yy][xx] == you for (xx, yy, zz) in req)
    score = 0 if opp_block else sum(1 for (xx, yy, zz) in req if board[zz][yy][xx] == me)
    undo_place(board, x, y, z)
    return score

def sideview_pattern_score_on_board(board: Board, me: int, x: int, y: int) -> int:
    you = 3 - me
    if y in (0, 3): req = sideview_required_cells_for_row(y)
    else:           req = sideview_required_cells_for_col(x)
    if any(board[zz][yy][xx] == you for (xx, yy, zz) in req):
        return 0
    return sum(1 for (xx, yy, zz) in req if board[zz][yy][xx] == me)

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
                if z in (1, 2): candidates.append((sx, y))
    for x in range(SIZE):
        if is_reverse_mickey_col_complete(board, me, x):
            for sy in (0, 3):
                z = lowest_empty_z(board, x, sy)
                if z in (1, 2): candidates.append((x, sy))
    if not candidates: return None
    best_mv = candidates[0]; best_dt = -1
    for (x, y) in candidates:
        z = place_inplace(board, x, y, me)
        my_dt = len(immediate_winning_squares_try(board, me))
        undo_place(board, x, y, z)
        if my_dt > best_dt:
            best_dt = my_dt; best_mv = (x, y)
    return best_mv

# ===== 安全フィルタ（自殺・t支え・相手DT誘発抑止） =====
def is_t_support_move(board: Board, me: int, x: int, y: int) -> bool:
    you = 3 - me
    z = lowest_empty_z(board, x, y)
    if z != 2:  # t支えは z==2 のみ
        return False
    z2 = place_inplace(board, x, y, me)
    if z2 is None: return True
    opens = is_winning_after(board, you, x, y)
    undo_place(board, x, y, z2)
    return opens

def _opp_dt_count(board: Board, you: int) -> int:
    return len(immediate_winning_squares_try(board, you))

def is_suicide_move(board: Board, me: int, x: int, y: int) -> bool:
    """置いて直後に相手の即勝ち本数が悪化するか？（自即勝/直ブロは許容）"""
    you = 3 - me
    if is_winning_after(board, me, x, y):
        return False
    opp_before = _opp_dt_count(board, you)
    z = place_inplace(board, x, y, me)
    if z is None: return True
    opp_after  = _opp_dt_count(board, you)
    undo_place(board, x, y, z)
    return opp_after > opp_before

def _max_opp_dt_after_reply(board: Board, me: int, x: int, y: int, beam: int = 10) -> int:
    """自手 (x,y) → 相手最善応手後の相手DT最大本数を概算（大きければトラップ失敗）"""
    you = 3 - me
    z0 = place_inplace(board, x, y, me)
    if z0 is None:
        return 99
    moves = valid_xy_moves(board)
    wins_opp = set(immediate_winning_squares_try(board, you))
    cands: Set[Coord2] = set(wins_opp)

    my_wins = set(immediate_winning_squares_try(board, me))
    for mv in moves:
        if mv in my_wins: cands.add(mv)
    for (cx, cy) in CENTERS:
        if (cx, cy) in moves and lowest_empty_z(board, cx, cy) in (1,2):
            cands.add((cx,cy))
    for (ex, ey) in EDGES:
        if (ex, ey) in moves and lowest_empty_z(board, ex, ey) in (1,2):
            cands.add((ex,ey))

    uniq = _center_sorted(list(cands))
    uniq = uniq[:beam] if len(uniq) > beam else uniq

    worst = 0
    for (ox, oy) in uniq if uniq else moves:
        z1 = place_inplace(board, ox, oy, you)
        if z1 is None: continue
        dt = _opp_dt_count(board, you)
        if dt > worst: worst = dt
        undo_place(board, ox, oy, z1)
        if worst >= 2:
            break
    undo_place(board, x, y, z0)
    return worst

def safe_filter_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    """危険手排除。全滅したら元の moves を返す。"""
    you = 3 - me
    opp_now = set(immediate_winning_squares_try(board, you))
    safe: List[Coord2] = []
    for (x, y) in moves:
        # 自即勝 / 直ブロ は無条件許容
        if is_winning_after(board, me, x, y) or (x, y) in opp_now:
            safe.append((x, y)); continue
        # t支え / 自殺 / 相手DT誘発>=2 を拒否
        if is_t_support_move(board, me, x, y): continue
        if is_suicide_move(board, me, x, y):   continue
        if _max_opp_dt_after_reply(board, me, x, y) >= 2: continue
        safe.append((x, y))
    return safe if safe else moves

# ===== 角・近接応手・角関数 =====
def first_layer_corner_moves(board: Board) -> List[Coord2]:
    mv: List[Coord2] = []
    for (x, y) in CORNERS:
        if lowest_empty_z(board, x, y) == 0:
            mv.append((x, y))
    return mv

def proximity_sorted(moves: List[Coord2], target: Coord2) -> List[Coord2]:
    return sorted(moves, key=lambda p: (manhattan(p, target),
                                        abs(1.5 - p[0]) + abs(1.5 - p[1])))

# ===== 相手の≤2手到達脅威：予防ブロック用 =====
def collect_reachable_threats(board: Board, player: int, max_steps: int) -> Set[Coord2]:
    """
    “空/自石のみ”で構成され、空き数<=max_steps（重力可）なラインの着点(x,y)集合を返す。
    2手以内に仕上がる脅威列の着点を先に抑えるために使用。
    """
    you = 3 - player
    res: Set[Coord2] = set()
    for line in ALL_LINES:
        cnt_you = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == you:
                cnt_you = 1; break
            if v == 0:
                empties.append((x,y,z))
        if cnt_you:  # 相手駒が混ざるラインは不可
            continue
        if 0 < len(empties) <= max_steps:
            # 重力が合う着点だけ登録
            for (ex,ey,ez) in empties:
                if lowest_empty_z(board, ex, ey) == ez:
                    res.add((ex,ey))
    return res

# ===== 縦列クランプ =====
def opponent_focus_columns(board: Board, me: int) -> List[Coord2]:
    you = 3 - me
    stats: List[Tuple[int, int, Coord2]] = []
    for y in range(SIZE):
        for x in range(SIZE):
            c = 0
            for z in range(SIZE):
                if board[z][y][x] == you:
                    c += 1
            if c > 0:
                center = int(abs(1.5 - x) + abs(1.5 - y))
                stats.append((c, -center, (x, y)))
    stats.sort(reverse=True)
    return [p for _,__,p in stats]

def vertical_profile(board: Board, me: int, x: int, y: int) -> Tuple[int,int]:
    you = 3 - me
    m = o = 0
    for z in range(SIZE):
        v = board[z][y][x]
        if v == me: m += 1
        elif v == you: o += 1
    return (m, o)

def urgent_vertical_clamp(board: Board, me: int) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    focus = opponent_focus_columns(board, me)
    rank: Dict[Coord2, int] = {p:i for i,p in enumerate(focus)}
    prio: List[Tuple[int,int,int,Coord2]] = []
    for (x, y) in moves:
        z = lowest_empty_z(board, x, y)
        if z is None: continue
        me_c, you_c = vertical_profile(board, me, x, y)
        if you_c >= 2 and z == 2:
            if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                prio.append((2, -rank.get((x,y),99), -int(abs(1.5-x)+abs(1.5-y)), (x,y)))
        elif you_c == 1 and z == 1:
            if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                prio.append((1, -rank.get((x,y),99), -int(abs(1.5-x)+abs(1.5-y)), (x,y)))
    if not prio: return None
    prio.sort(reverse=True)
    return prio[0][3]

def build_own_vertical_pressure(board: Board, me: int) -> Optional[Coord2]:
    moves = valid_xy_moves(board)
    if not moves: return None
    cands: List[Coord2] = []
    for (x, y) in moves:
        z = lowest_empty_z(board, x, y)
        if z == 1:
            me_c, you_c = vertical_profile(board, me, x, y)
            if me_c >= 1 and you_c == 0:
                cands.append((x, y))
    cands = safe_filter_moves(board, me, cands)
    if not cands: return None
    cands.sort(key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
    return cands[0]

# ===== 中央クランプ =====
def choose_center_clamp_move(board: Board, me: int) -> Optional[Coord2]:
    you = 3 - me
    moves = valid_xy_moves(board)
    if not moves: return None
    cands: List[Coord2] = []
    for (x, y) in CENTERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2):
            cands.append((x, y))
    if not cands: return None
    cands = safe_filter_moves(board, me, cands)
    if not cands: return None
    base = len(immediate_winning_squares_try(board, you))
    scored = []
    for (x, y) in cands:
        z = place_inplace(board, x, y, me)
        aft = len(immediate_winning_squares_try(board, you))
        undo_place(board, x, y, z)
        delta = base - aft
        center_bias = -int(abs(1.5 - x) + abs(1.5 - y))
        scored.append(((delta, center_bias), (x, y)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]

# ===== トラップ（Fork）スコア =====
def trap_score_after_move(board: Board, me: int, x: int, y: int) -> Tuple[int,int]:
    """
    置いた直後の “自分の即勝ち手数(my_next)” と “相手の即勝ち手数(opp_next)”。
    主に my_next を最大化、opp_next==0 を強く優遇。
    """
    you = 3 - me
    z = place_inplace(board, x, y, me)
    if z is None: return (0, 99)
    my_next  = len(immediate_winning_squares_try(board, me))
    opp_next = len(immediate_winning_squares_try(board, you))
    undo_place(board, x, y, z)
    return (my_next, opp_next)

# ===== ダイレクトブロック =====
def _score_direct_block(board: Board, me: int, mv: Coord2, opp_wins_now: List[Coord2]) -> Tuple[int,int,int,int]:
    you = 3 - me
    x, y = mv
    z = place_inplace(board, x, y, me)
    if z is None: return (0,0,0,0)
    before = len(opp_wins_now)
    after  = len(immediate_winning_squares_try(board, you))
    blocked = max(0, before - after)
    my_next = len(immediate_winning_squares_try(board, me))
    tie_sv  = sideview_pattern_score_on_board(board, me, x, y) if (x, y) in EDGES else 0
    fork, _opp = trap_score_after_move(board, me, x, y)
    undo_place(board, x, y, z)
    # ブロック数 → 自分DT → フォーク → 側面の順で比較
    return (blocked, my_next, fork, tie_sv)

# ===== “最終ガード” =====
def force_block_guard(board: Board, me: int, chosen: Coord2) -> Coord2:
    you = 3 - me
    if immediate_winning_squares_try(board, me):
        return chosen
    opp_set = set(line_immediate_winning_moves(board, you)) | set(immediate_winning_squares_try(board, you))
    if not opp_set or chosen in opp_set:
        return chosen
    # ダイレクト差し替え
    moves = valid_xy_moves(board)
    direct = [mv for mv in moves if mv in opp_set]
    if direct:
        scored = [(_score_direct_block(board, me, mv, list(opp_set)), mv) for mv in direct]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    # after==0 優先→最小化
    best = None; best_after = 10**9
    for (x, y) in _center_sorted(moves):
        z = place_inplace(board, x, y, me);   after = 10**9
        if z is not None:
            after = len(immediate_winning_squares_try(board, you))
            undo_place(board, x, y, z)
        if after == 0: return (x, y)
        if after < best_after: best_after = after; best = (x, y)
    return best if best is not None else (_center_sorted(moves)[0] if moves else (0,0))

# ===== 脅威限定読み（TT/キラー/反復深化） =====
def _tactical_eval(board: Board, me: int) -> int:
    you = 3 - me
    my_now  = len(immediate_winning_squares_try(board, me))
    opp_now = len(immediate_winning_squares_try(board, you))
    score = 10_000*(my_now - opp_now)
    if my_now >= 2:  score += 1_500
    if opp_now >= 2: score -= 2_000
    # 自縦列の圧（軽く）
    vsum = 0
    for y in range(SIZE):
        for x in range(SIZE):
            m,o = vertical_profile(board, me, x, y)
            if m >= 2 and o == 0: vsum += 3
    score += vsum
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

    # 逆ミッキー完成サイド
    mv_rm = find_reverse_mickey_side_move(board, turn)
    if mv_rm and mv_rm in moves:
        cands.add(mv_rm)

    # 中央2-3 / 辺2-3
    for (x, y) in CENTERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2): cands.add((x,y))
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1,2): cands.add((x,y))

    # 縦列重視
    for (x, y) in moves:
        z = lowest_empty_z(board, x, y)
        m,o = vertical_profile(board, turn, x, y)
        if o >= 2 and z == 2: cands.add((x,y))
        if o == 1 and z == 1: cands.add((x,y))
        if m >= 1 and o == 0 and z == 1: cands.add((x,y))

    # 足りなければ中央寄り補完
    if len(cands) < 6:
        for mv in _center_sorted(moves):
            cands.add(mv)
            if len(cands) >= 8: break

    # トラップスコアで順序付け（my_next降順、opp_next昇順）
    ordered = list(cands)

    def _score(mv: Coord2) -> Tuple[int,int,int,int,int]:
        x,y = mv
        s1 = 3 if mv in wins_turn else 0
        s2 = 2 if mv in wins_opp  else 0
        fork, anti = trap_score_after_move(board, turn, x, y)
        s3 = fork
        s4 = -anti
        s5 = -int(abs(1.5-x)+abs(1.5-y))
        return (s1, s2, s3, s4, s5)

    ordered.sort(key=_score, reverse=True)
    return ordered[:THREAT_BEAM]

def _negamax(board: Board, depth: int, alpha: int, beta: int,
             turn: int, me: int, deadline: float, ply: int = 0) -> int:
    if time.perf_counter() >= deadline or depth == 0:
        return _tactical_eval(board, me)

    key = (board_hash(board), turn)
    if key in TT:
        tt_depth, tt_flag, tt_score, tt_move = TT[key]
        if tt_depth >= depth:
            if tt_flag == 0:   return tt_score
            if tt_flag == -1 and tt_score <= alpha: return tt_score
            if tt_flag == +1 and tt_score >= beta:  return tt_score

    moves = _tactical_candidates(board, turn, me)
    if not moves:
        return _tactical_eval(board, me)

    you = 3 - turn
    if immediate_winning_squares_try(board, turn):
        return 100_000 - (7 - depth)

    # move ordering: TT→Killer→残り（既にcandsはトラップ順）
    ordered = list(moves)
    tt_best = None
    if key in TT and TT[key][3] is not None and TT[key][3] in ordered:
        tt_best = TT[key][3]
        ordered.remove(tt_best); ordered.insert(0, tt_best)
    if KILLERS.get(ply):
        km = [mv for mv in KILLERS[ply] if mv in ordered]
        for mv in reversed(km):
            ordered.remove(mv); ordered.insert(0, mv)

    best = -10**9
    best_move: Optional[Coord2] = tt_best
    a0 = alpha

    for (x, y) in ordered:
        if time.perf_counter() >= deadline: break
        z = place_inplace(board, x, y, turn)
        if z is None: continue
        if is_t_support_move(board, turn, x, y):
            undo_place(board, x, y, z); continue
        score = -_negamax(board, depth-1, -beta, -alpha, you, me, deadline, ply+1)
        undo_place(board, x, y, z)

        if score > best:
            best = score; best_move = (x, y)
        if best > alpha: alpha = best
        if alpha >= beta:
            if (x, y) not in KILLERS[ply]:
                KILLERS[ply].insert(0, (x, y))
                if len(KILLERS[ply]) > 2: KILLERS[ply].pop()
            break

    flag = 0
    if best <= a0: flag = -1
    elif best >= beta: flag = +1
    TT[key] = (depth, flag, best, best_move)
    return best

def threat_space_best_move_iterative(board: Board, me: int, deadline: float) -> Optional[Coord2]:
    root_moves = _tactical_candidates(board, me, me)
    if not root_moves:
        return None
    root_moves = safe_filter_moves(board, me, root_moves)

    focus = opponent_focus_columns(board, me)
    extra = 1 if focus and len(focus) <= 4 else 0
    max_d = min(TS_MAX_DEPTH + extra, 7)

    best_mv: Optional[Coord2] = root_moves[0]
    best_sc = -10**9
    you = 3 - me

    for d in range(2, max_d+1):
        if time.perf_counter() >= deadline: break
        iter_best_mv = best_mv; iter_best_sc = -10**9

        key = (board_hash(board), me)
        if key in TT and TT[key][3] is not None and TT[key][3] in root_moves:
            mv = TT[key][3]
            root_moves = [mv] + [r for r in root_moves if r != mv]

        for (x, y) in root_moves:
            if time.perf_counter() >= deadline: break
            z = place_inplace(board, x, y, me)
            if z is None: continue
            sc = -_negamax(board, d-1, -10**9, 10**9, you, me, deadline, ply=1)
            undo_place(board, x, y, z)
            if sc > iter_best_sc:
                iter_best_sc = sc; iter_best_mv = (x, y)

        if iter_best_mv is not None:
            best_mv, best_sc = iter_best_mv, iter_best_sc
            root_moves = [best_mv] + [mv for mv in root_moves if mv != best_mv]

    return best_mv

# ===== 手選択 =====
def choose_best(board: Board, me: int, deadline: float, last_move: Coord3) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0, 0)
    you = 3 - me
    total = count_placed(board)

    # 1) 自即勝
    my_wins_now = immediate_winning_squares_try(board, me)
    if my_wins_now:
        for mv in moves:
            if mv in my_wins_now: return mv
        return _center_sorted(my_wins_now)[0]

    # 2) 相手即勝ブロック
    opp_wins_now = list(set(immediate_winning_squares_try(board, you)) |
                        set(line_immediate_winning_moves(board, you)))
    if opp_wins_now:
        direct = [mv for mv in opp_wins_now if mv in moves]
        if direct:
            scored = [(_score_direct_block(board, me, mv, opp_wins_now), mv) for mv in direct]
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored[0][1]
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

    # 2.05) ≤2手到達の予防ブロック（列優先）
    opp2 = collect_reachable_threats(board, you, 2)
    if opp2:
        pre_blocks: List[Coord2] = []
        for (x, y) in moves:
            z = lowest_empty_z(board, x, y)
            if z is None: continue
            if (x, y) in opp2 and z in (1,2):
                pre_blocks.append((x, y))
        pre_blocks = safe_filter_moves(board, me, pre_blocks)
        if pre_blocks:
            pre_blocks.sort(key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
            return pre_blocks[0]

    # 2.1) 角z==0の強制最優先（先手/後手細則）
    corner0 = first_layer_corner_moves(board)
    if corner0:
        lx, ly, lz = last_move
        if me == 2 and (lx, ly) in CORNERS and lz == 0:
            others = [(x, y) for (x, y) in corner0 if (x, y) != (lx, ly)]
            if others:
                cands = safe_filter_moves(board, me, others) or others
                cands.sort(key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
                return cands[0]
        cands = safe_filter_moves(board, me, corner0) or corner0
        cands.sort(key=lambda p: (abs(1.5 - p[0]) + abs(1.5 - p[1])))
        return cands[0]

    # 2.12) 近接応手（序盤で相手が内側=非角/非端に置いてきたなら、近傍から）
    if total <= 6:
        lx, ly, lz = last_move
        near_pool = [mv for mv in moves if (mv not in CORNERS)]
        if near_pool:
            # 安全・t支え回避を通し、近い順
            near_pool = safe_filter_moves(board, me, near_pool)
            near_pool = proximity_sorted(near_pool, (lx, ly))
            if near_pool:
                # ただし、中央クランプ/縦クランプ/トラップに劣らないものだけ出す
                best_near = near_pool[0]
                # 簡易トラップ優先：my_next多いなら採用
                mn, on = trap_score_after_move(board, me, best_near[0], best_near[1])
                if mn >= 2 and on == 0:
                    return best_near

    # 2.15) 中央クランプ（センター z∈{1,2}）
    clamp = choose_center_clamp_move(board, me)
    if clamp is not None: return clamp

    # 2.2) フォーカス列クランプ
    vblock = urgent_vertical_clamp(board, me)
    if vblock is not None: return vblock

    # 2.25) 自縦列の圧（z==1）
    vbuild = build_own_vertical_pressure(board, me)
    if vbuild is not None: return vbuild

    # 2.3) 脅威限定 反復深化（TT/キラー）
    mv_ts = threat_space_best_move_iterative(board, me, deadline)
    if mv_ts is not None: return mv_ts

    # 2.35) トラップ優先（DT>=2 & 相手即勝0）
    trap_cands: List[Tuple[Tuple[int,int,int], Coord2]] = []
    for (x, y) in safe_filter_moves(board, me, moves):
        z = place_inplace(board, x, y, me)
        if z is None: continue
        my_next = len(immediate_winning_squares_try(board, me))
        opp_next = len(immediate_winning_squares_try(board, you))
        sv = sideview_pattern_score_on_board(board, me, x, y) if (x, y) in EDGES else 0
        undo_place(board, x, y, z)
        if my_next >= 2 and opp_next == 0:
            trap_cands.append(((my_next, sv, -int(abs(1.5-x)+abs(1.5-y))), (x, y)))
    if trap_cands:
        trap_cands.sort(key=lambda t: t[0], reverse=True)
        return trap_cands[0][1]

    # 2.4) 逆ミッキー完成→サイド直行（安全）
    mv_rm = find_reverse_mickey_side_move(board, me)
    if mv_rm is not None and mv_rm in moves:
        if not is_suicide_move(board, me, mv_rm[0], mv_rm[1]) and \
           not is_t_support_move(board, me, mv_rm[0], mv_rm[1]):
            return mv_rm

    # 2.5) 逆ミッキー“作成”狙い（DT）
    best_rm_move: Optional[Coord2] = None
    best_dt = 0; best_tie = -1
    cand_moves = safe_filter_moves(board, me, moves)
    for (x, y) in cand_moves:
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

    # 3) 中央 2〜3層（安全）
    for (x, y) in CENTERS:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1, 2):
            if (x, y) in safe_filter_moves(board, me, [(x, y)]):
                return (x, y)

    # 4) 辺 2〜3層（行/列角に自石）→ 側面形最大化（安全）
    edge_cands: List[Coord2] = []
    for (x, y) in EDGES:
        if (x, y) in moves and lowest_empty_z(board, x, y) in (1, 2):
            corners = row_col_corners_for_edge(x, y)
            if any(column_has_my_stone(board, me, cx, cy) for (cx, cy) in corners):
                edge_cands.append((x, y))
    edge_cands = safe_filter_moves(board, me, edge_cands)
    if edge_cands:
        best_mv = edge_cands[0]; best_score = -1
        for (x, y) in edge_cands:
            sc = sideview_pattern_score_after_move(board, me, x, y)
            if sc > best_score: best_score = sc; best_mv = (x, y)
        return best_mv

    # 5) 中央（層問わず; 安全）
    for (x, y) in CENTERS:
        if (x, y) in moves and (x, y) in safe_filter_moves(board, me, [(x, y)]):
            return (x, y)

    # 6) 中央寄りフォールバック（安全）
    fallback = _center_sorted(moves)
    fallback = safe_filter_moves(board, me, fallback)
    return fallback[0]

# ===== エンジン =====
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        # 時間開始
        deadline = time.perf_counter() + TIME_BUDGET_SEC
        try:
            mv = choose_best(board, player, deadline, last_move)
            mv = force_block_guard(board, player, mv)
            ms = valid_xy_moves(board)
            return mv if mv in ms else (ms[0] if ms else (0, 0))
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0, 0)

AI = MyAI()
