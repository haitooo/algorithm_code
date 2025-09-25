# main.py
# 4x4x4 立体四目並べ AI — 10秒制限向け“実戦最強”エンジン（安定性修正版）

from typing import List, Tuple, Optional, Dict
import time
from framework import Alg3D, Board

Coord2 = Tuple[int, int]
Coord3 = Tuple[int, int, int]
SIZE = 4

# ---- 時間＆ノード予算 ----
TIME_BUDGET_SEC = 9.5
NODE_BUDGET     = 220_000

# ---- 評価重み ----
WIN_SCORE      = 1_000_000
PATTERN_0_3    = 120
PATTERN_0_2    = 10
PATTERN_0_1    = 1
TPOINT_BONUS   = 120
CENTER_COL_BON = 4
DOUBLE_THREAT  = 5_000
OPP_DTH_PEN    = 5_000

# ---- 探索制御 ----
INF  = 10**9
PV_BIAS = 16
QUIES_DEPTH = 2
BEAM_ROOT = 14
BEAM_NODE = 10
ASP_MAX_EXPAND = 3   # Aspirationの窓拡張回数上限

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

# ---------------- 終局・即勝ち ----------------
def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    lx, ly, lz = last
    for line in ALL_LINES:
        if any((x==lx and y==ly and z==lz) for (x,y,z) in line):
            if all(board[z][y][x] == player for (x,y,z) in line):
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
        self.killers: Dict[int, List[Coord2]] = {}
        self.history: Dict[Coord2, int] = {}

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
        self.pv: Dict[int, Coord2] = {}

    def time_up(self) -> bool:
        return self.nodes >= NODE_BUDGET or time.perf_counter() >= self.deadline

    def key(self, board: Board, turn: int) -> Tuple:
        return (turn, tuple(tuple(tuple(board[z][y][x] for x in range(SIZE)) for y in range(SIZE)) for z in range(SIZE)))

    # ---- 静止探索 ----
    def qsearch(self, board: Board, alpha: int, beta: int, turn: int, ply: int, qd: int) -> int:
        if self.time_up(): return eval_board(board, self.me, ply)
        self.nodes += 1

        stand = eval_board(board, self.me, ply)
        if stand >= beta: return stand
        if stand > alpha: alpha = stand
        if qd <= 0: return stand

        moves = valid_xy_moves(board)
        if not moves: return stand

        you = 3 - turn
        wins  = [mv for mv in moves if is_winning_after(board, turn, mv[0], mv[1])]
        blks  = immediate_winning_squares(board, you)
        blks_set = set(blks)
        tactical = set(wins) | blks_set

        before = len(blks)
        for (x,y) in moves:
            z = place_inplace(board, x, y, turn)
            after = len(immediate_winning_squares(board, you))
            undo_place(board, x, y, z)
            if after < before:
                tactical.add((x,y))

        if not tactical: return stand

        ordered = list(tactical)
        ordered.sort(key=lambda mv: (mv not in wins, mv not in blks_set))

        for (x,y) in ordered:
            z = place_inplace(board, x, y, turn)
            score = -self.qsearch(board, -beta, -alpha, 3-turn, ply+1, qd-1)
            undo_place(board, x, y, z)
            if score >= beta: return score
            if score > alpha: alpha = score
        return alpha

    # ---- PVS αβ ----
    def pvs(self, board: Board, depth: int, alpha: int, beta: int, turn: int, ply: int, root: bool) -> int:
        if self.time_up(): return eval_board(board, self.me, ply)
        self.nodes += 1

        if depth == 0:
            return self.qsearch(board, alpha, beta, turn, ply, QUIES_DEPTH)

        moves = valid_xy_moves(board)
        if not moves:
            return eval_board(board, self.me, ply)

        for (x,y) in moves:
            if is_winning_after(board, turn, x, y):
                return WIN_SCORE - ply if turn == self.me else -WIN_SCORE + ply

        key = self.key(board, turn)
        tte = self.tt.get(key)
        tt_mv: Optional[Coord2] = tte[3] if tte and tte[0] >= depth else None

        you = 3 - turn
        wins = [mv for mv in moves if is_winning_after(board, turn, mv[0], mv[1])]
        blks = immediate_winning_squares(board, you)

        ordered = [mv for _,mv in self.mo.score(tt_mv, wins, blks, depth, moves)]
        beam = BEAM_ROOT if root else BEAM_NODE
        if not ordered:
            ordered = moves[:]  # 念のため
        ordered = ordered[:max(1, min(beam, len(ordered)))]

        best = -INF
        best_mv: Optional[Coord2] = None
        first = True
        alpha0, beta0 = alpha, beta  # ← TTフラグ用

        for (x,y) in ordered:
            z = place_inplace(board, x, y, turn)
            if first:
                score = -self.pvs(board, depth-1, -beta, -alpha, 3-turn, ply+1, False)
                first = False
            else:
                score = -self.pvs(board, depth-1, -(alpha+1), -alpha, 3-turn, ply+1, False)
                if score > alpha and score < beta:
                    score = -self.pvs(board, depth-1, -beta, -alpha, 3-turn, ply+1, False)
            undo_place(board, x, y, z)

            if score > best:
                best = score
                best_mv = (x,y)
                if root: self.pv[depth] = (x,y)

            if best > alpha:
                alpha = best
            if alpha >= beta:
                if (x,y) not in wins and (x,y) not in blks:
                    self.mo.note_killer(depth, (x,y))
                    self.mo.bump_history((x,y), depth)
                break

        # TT 保存（正しく alpha0/beta0 で判定）
        flag = 0
        if best <= alpha0: flag = -1     # UPPER
        elif best >= beta0: flag = +1    # LOWER
        else: flag = 0                   # EXACT
        self.tt[key] = (depth, best, flag, best_mv)
        return best

# ---------------- ルート選択（反復深化＋アスピレーション） ----------------
def choose_best(board: Board, me: int) -> Coord2:
    moves = valid_xy_moves(board)
    if not moves:
        return (0, 0)  # 盤面が詰んでいる場合の保険（フレームワーク互換）

    # 即勝ち先取り（ここで返せば安全）
    for (x,y) in moves:
        if is_winning_after(board, me, x, y):
            return (x,y)

    deadline = time.perf_counter() + TIME_BUDGET_SEC
    search = Searcher(me, deadline)

    # 初期並べ替え（軽評価）→ ルートビーム
    scored: List[Tuple[int, Coord2]] = []
    for (x,y) in moves:
        z = place_inplace(board, x, y, me)
        s = eval_board(board, me, 1)
        undo_place(board, x, y, z)
        scored.append((s, (x,y)))
    scored.sort(reverse=True)
    root_moves = [mv for _,mv in scored[:max(1, min(BEAM_ROOT, len(scored)))]]

    best_move = root_moves[0]
    guess = 0
    # 終盤は少し深く（ただし時間制御あり）
    max_depth = 6 if stones_count(board) >= 20 else 5

    for depth in range(1, max_depth+1):
        if search.time_up(): break
        alpha = guess - PV_BIAS
        beta  = guess + PV_BIAS
        local_best = best_move
        expand_cnt = 0

        while True:
            if search.time_up(): break
            score = -INF
            for (x,y) in root_moves:
                if search.time_up(): break
                z = place_inplace(board, x, y, me)
                val = -search.pvs(board, depth-1, -beta, -alpha, 3-me, 1, True)
                undo_place(board, x, y, z)
                if val > score:
                    score = val
                    local_best = (x,y)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break

            # Aspiration調整（上限付き）
            if score <= guess - PV_BIAS and expand_cnt < ASP_MAX_EXPAND:
                alpha = -INF; beta = guess + PV_BIAS; expand_cnt += 1
                guess = score
                continue
            elif score >= guess + PV_BIAS and expand_cnt < ASP_MAX_EXPAND:
                alpha = guess - PV_BIAS; beta = INF; expand_cnt += 1
                guess = score
                continue
            else:
                guess = score
                break

        if not search.time_up():
            best_move = local_best

    # 念のための最終ガード：必ず合法手を返す
    if best_move not in moves:
        return moves[0]
    return best_move

# ---------------- エンジン ----------------
class MyAI(Alg3D):
    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        # いかなる例外でも合法手を返す
        try:
            start = time.perf_counter()
            mv = choose_best(board, player)
            # 万一の時間オーバーに備えた最終ガード
            if time.perf_counter() - start > TIME_BUDGET_SEC:
                ms = valid_xy_moves(board)
                return ms[0] if ms else (0,0)
            return mv
        except Exception:
            ms = valid_xy_moves(board)
            return ms[0] if ms else (0,0)

AI = MyAI()
