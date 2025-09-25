# main.py
# 4x4x4 立体四目並べ AI — 10秒制限向け“実戦最強”エンジン
# 技法: 反復深化 / PVS(Principal Variation Search) / Aspiration Window /
#       Transposition Table / Killer Moves / History Heuristic / 静止探索(タクティクス限定)
# 依存: framework.Alg3D, Board  (get_move(board, player, last_move) -> (x,y))

from typing import List, Tuple, Optional, Dict
import time
import math
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# ---- 時間＆ノード予算 ----
TIME_BUDGET_SEC = 9.5
NODE_BUDGET     = 220_000

# ---- 評価重み（調整済み）----
WIN_SCORE      = 1_000_000
PATTERN_0_3    = 120     # 自3/相0 の3連
PATTERN_0_2    = 10      # 自2/相0
PATTERN_0_1    = 1       # 自1/相0
TPOINT_BONUS   = 120     # z==3 で完成する“浮き”リーチの上乗せ
CENTER_COL_BON = 4       # 2D中心寄り（列の上段が空のときのみ軽く）
DOUBLE_THREAT  = 5_000   # 即勝ち手が2つ以上
OPP_DTH_PEN    = 5_000

# ---- 探索制御 ----
INF  = 10**9
PV_BIAS = 16             # PVSで再探索を減らすバイアス
QUIES_DEPTH = 2          # 静止探索の最大深さ
BEAM_ROOT = 14           # ルートのビーム幅（序中盤の分岐抑制）
BEAM_NODE = 10           # ノードのビーム幅

# ---------------- 盤ユーティリティ ----------------
def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    # 重力: z=0(最下層)から上へ
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

# ---------------- ライン列挙（76本） ----------------
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
CENTERS_2D = {(1,1), (2,1), (1,2), (2,2)}

# ---------------- 終局・即勝ち ----------------
def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    lx, ly, lz = last
    for line in ALL_LINES:
        hit = False
        for (x,y,z) in line:
            if x==lx and y==ly and z==lz:
                hit = True; break
        if hit and all(board[z][y][x] == player for (x,y,z) in line):
            return True
    return False

def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None:
        return False
    ok = check_win_at(board, player, (x,y,z))
    undo_place(board, x, y, z)
    return ok

def immediate_winning_squares(board: Board, player: int) -> List[Coord2]:
    return [(x,y) for (x,y) in valid_xy_moves(board) if is_winning_after(board, player, x, y)]

# ---------------- パターン・評価 ----------------
def lowest_ok(board: Board, e: Coord3) -> bool:
    x,y,z = e
    ze = lowest_empty_z(board, x, y)
    return (ze is not None) and (ze == z)

def eval_board(board: Board, me: int, ply: int) -> int:
    you = 3 - me
    # 終局（全走査）
    for line in ALL_LINES:
        if all(board[z][y][x] == me  for (x,y,z) in line):  return WIN_SCORE - ply
        if all(board[z][y][x] == you for (x,y,z) in line):  return -WIN_SCORE + ply

    score = 0
    for line in ALL_LINES:
        m = y = 0
        empties: List[Coord3] = []
        for (x,yy,z) in line:
            v = board[z][yy][x]
            if v == me: m+=1
            elif v == you: y+=1
            else: empties.append((x,yy,z))
        if y == 0:
            # 自分のみのライン
            if m == 3 and len(empties)==1 and lowest_ok(board, empties[0]):
                score += PATTERN_0_3
                if empties[0][2] == 3: score += TPOINT_BONUS
            elif m == 2 and len(empties)==2 and lowest_ok(board, empties[0]) and lowest_ok(board, empties[1]):
                score += PATTERN_0_2
            elif m == 1:
                score += PATTERN_0_1
        if m == 0:
            # 相手のみのライン（対称減点）
            if y == 3 and len(empties)==1 and lowest_ok(board, empties[0]):
                score -= PATTERN_0_3
                if empties[0][2] == 3: score -= TPOINT_BONUS
            elif y == 2 and len(empties)==2 and lowest_ok(board, empties[0]) and lowest_ok(board, empties[1]):
                score -= PATTERN_0_2
            elif y == 1:
                score -= PATTERN_0_1

    # 列中心ボーナス（軽め）
    top = SIZE-1
    for yy in range(SIZE):
        for xx in range(SIZE):
            if board[top][yy][xx] == 0:
                cx = abs(1.5 - xx); cy = abs(1.5 - yy)
                score += int(CENTER_COL_BON * (1.5 - (cx+cy)/2))

    # ダブルリーチ（即勝ち手の数）
    my_kill  = len(immediate_winning_squares(board, me))
    opp_kill = len(immediate_winning_squares(board, you))
    if my_kill >= 2:  score += DOUBLE_THREAT
    if opp_kill >= 2: score -= OPP_DTH_PEN
    return score

# ---------------- 並べ替えヒューリスティクス ----------------
class MoveOrderer:
    def __init__(self):
        self.killers: Dict[int, List[Coord2]] = {}    # depth -> top2 killers
        self.history: Dict[Coord2, int] = {}          # quiet move → score

    def note_killer(self, depth: int, mv: Coord2):
        ks = self.killers.get(depth, [])
        if mv in ks: return
        ks = [mv] + ks
        if len(ks) > 2: ks = ks[:2]
        self.killers[depth] = ks

    def bump_history(self, mv: Coord2, depth: int):
        self.history[mv] = self.history.get(mv, 0) + depth*depth

    def score(self, tt_move: Optional[Coord2], wins: List[Coord2], blocks: List[Coord2], depth: int, moves: List[Coord2]) -> List[Tuple[int, Coord2]]:
        ks = self.killers.get(depth, [])
        table: List[Tuple[int, Coord2]] = []
        win_set = set(wins); blk_set = set(blocks)
        for mv in moves:
            s = 0
            if tt_move is not None and mv == tt_move: s += 1_000_000
            if mv in win_set:  s += 900_000
            if mv in blk_set:  s += 800_000
            if mv in ks:       s += 100_000
            s += self.history.get(mv, 0)
            # 2D中心寄りの微加点（等価手の順序安定）
            cx = abs(1.5 - mv[0]); cy = abs(1.5 - mv[1])
            s += int(1000 * (1.5 - (cx+cy)/2))
            table.append((s, mv))
        table.sort(reverse=True)
        return table

# ---------------- トランスポジションテーブル ----------------
TTEntry = Tuple[int, int, int, Optional[Coord2]]
# (depth_remain, value, flag, best_move)  flag: 0=EXACT, -1=UPPER, +1=LOWER

# ---------------- 探索器 ----------------
class Searcher:
    def __init__(self, me: int, deadline: float):
        self.me = me
        self.deadline = deadline
        self.nodes = 0
        self.tt: Dict[Tuple, TTEntry] = {}
        self.mo = MoveOrderer()
        self.pv: Dict[int, Coord2] = {}  # 深さ→PV第一手

    def time_up(self) -> bool:
        return self.nodes >= NODE_BUDGET or time.perf_counter() >= self.deadline

    def key(self, board: Board, turn: int) -> Tuple:
        return (turn, tuple(tuple(tuple(board[z][y][x] for x in range(SIZE)) for y in range(SIZE)) for z in range(SIZE)))

    # ---- 静止探索（即勝/即負け/リーチのみの簡易延長）----
    def qsearch(self, board: Board, alpha: int, beta: int, turn: int, ply: int, qd: int) -> int:
        if self.time_up(): return eval_board(board, self.me, ply)
        self.nodes += 1

        stand = eval_board(board, self.me, ply)
        if stand >= beta: return stand
        if stand > alpha: alpha = stand
        if qd <= 0: return stand

        moves = valid_xy_moves(board)
        if not moves: return stand

        # タクティカル候補のみ: 自即勝、相手即勝ブロック、相手リーチ減
        you = 3 - turn
        wins  = [mv for mv in moves if is_winning_after(board, turn, mv[0], mv[1])]
        blks  = immediate_winning_squares(board, you)
        blks_set = set(blks)
        tactical = set(wins) | blks_set

        # 追加: 相手の即勝ち数を減らす手
        before = len(blks)
        for (x,y) in moves:
            z = place_inplace(board, x, y, turn)
            after = len(immediate_winning_squares(board, you))
            undo_place(board, x, y, z)
            if after < before:
                tactical.add((x,y))

        if not tactical: return stand

        ordered = list(tactical)
        # 自即勝は先に
        ordered.sort(key=lambda mv: (mv not in wins, mv not in blks_set))

        for (x,y) in ordered:
            z = place_inplace(board, x, y, turn)
            score = -self.qsearch(board, -beta, -alpha, 3-turn, ply+1, qd-1)
            undo_place(board, x, y, z)
            if score >= beta: return score
            if score > alpha: alpha = score
        return alpha

    # ---- PVS αβ ----
    def pvs(self, board: Board, depth: int, alpha: int, beta: int, turn: int, ply: int, root: bool, d_rem: int) -> int:
        if self.time_up(): return eval_board(board, self.me, ply)
        self.nodes += 1

        # 終深
        if depth == 0:
            return self.qsearch(board, alpha, beta, turn, ply, QUIES_DEPTH)

        # 終了＆即勝刈り取り
        moves = valid_xy_moves(board)
        if not moves:
            return eval_board(board, self.me, ply)

        for (x,y) in moves:
            if is_winning_after(board, turn, x, y):
                return WIN_SCORE - ply if turn == self.me else -WIN_SCORE + ply

        # TT 参照
        key = self.key(board, turn)
        tt_hit = self.tt.get(key)
        tt_mv: Optional[Coord2] = tt_hit[3] if tt_hit and tt_hit[0] >= depth else None

        # 自即勝/相手即勝ブロック列挙（順序に反映）
        you = 3 - turn
        wins = [mv for mv in moves if is_winning_after(board, turn, mv[0], mv[1])]
        blks = immediate_winning_squares(board, you)

        # 並べ替え＋ビーム
        ordered = [mv for _,mv in self.mo.score(tt_mv, wins, blks, depth, moves)]
        beam = BEAM_ROOT if root else BEAM_NODE
        ordered = ordered[:beam]

        best = -INF
        best_mv: Optional[Coord2] = None
        first = True

        for (x,y) in ordered:
            z = place_inplace(board, x, y, turn)

            if first:
                score = -self.pvs(board, depth-1, -beta, -alpha, 3-turn, ply+1, False, d_rem-1)
                first = False
            else:
                # PVS: null-window で試し、越えたら本窓で再探索
                score = -self.pvs(board, depth-1, -(alpha+1), -alpha, 3-turn, ply+1, False, d_rem-1)
                if score > alpha and score < beta:
                    score = -self.pvs(board, depth-1, -beta, -alpha, 3-turn, ply+1, False, d_rem-1)

            undo_place(board, x, y, z)

            if score > best:
                best = score
                best_mv = (x,y)
                if root: self.pv[depth] = (x,y)

            if best > alpha:
                alpha = best

            if alpha >= beta:
                # カット：キラー＆ヒストリー更新（安静手のみ）
                if (x,y) not in wins and (x,y) not in blks:
                    self.mo.note_killer(depth, (x,y))
                    self.mo.bump_history((x,y), depth)
                break

        # TT 保存
        flag = 0
        if best <= -INF//2: flag = -1
        elif best <= alpha: flag = -1
        elif best >= beta:  flag = +1
        self.tt[key] = (depth, best, flag, best_mv)
        return best

# ---------------- ルート選択（反復深化＋アスピレーション） ----------------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves: return (0,0)
    # 即勝ち先取り
    for (x,y) in moves:
        if is_winning_after(board, me, x, y):
            return (x,y)

    deadline = time.perf_counter() + TIME_BUDGET_SEC
    search = Searcher(me, deadline)

    # 初期並べ替え（軽いスコアで降順）→ ルートビーム
    base_scores = []
    for (x,y) in moves:
        z = place_inplace(board, x, y, me)
        s = eval_board(board, me, 1)
        undo_place(board, x, y, z)
        base_scores.append((s, (x,y)))
    base_scores.sort(reverse=True)
    root_moves = [mv for _,mv in base_scores[:BEAM_ROOT]]

    best_move = root_moves[0]
    guess = 0
    max_depth = 6 if stones_count(board) >= 20 else 5  # 終盤は少し深く
    for depth in range(1, max_depth+1):
        if search.time_up(): break
        alpha = guess - PV_BIAS
        beta  = guess + PV_BIAS
        local_best = best_move
        local_score = -INF

        # Aspiration loop
        while True:
            if search.time_up(): break
            score = -INF
            # ルートPVS
            first = True
            for (x,y) in root_moves:
                if search.time_up(): break
                z = place_inplace(board, x, y, me)
                val = -search.pvs(board, depth-1, -beta, -alpha, 3-me, 1, True, depth-1)
                undo_place(board, x, y, z)
                if val > score:
                    score = val
                    local_best = (x,y)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
                first = False

            # アスピレーション失敗 → 窓を広げて再試行
            if score <= guess - PV_BIAS:
                alpha = -INF; beta = guess + PV_BIAS
                if alpha <= -INF and beta >= INF: break
                guess = score
                continue
            elif score >= guess + PV_BIAS:
                alpha = guess - PV_BIAS; beta = INF
                guess = score
                continue
            else:
                guess = score
                break

        if not search.time_up():
            best_move = local_best

    return best_move

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        return choose_best(board, player)

AI = MyAI()
