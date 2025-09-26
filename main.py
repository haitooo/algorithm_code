# main.py — 4x4x4 立体四目並べ “内蔵定石 + αβ(反復深化) + 脅威評価” 安定版
# 依存: framework.Alg3D, Board  (あなたの環境に合わせて import を維持)
from typing import List, Tuple, Optional, Dict, Iterable
from math import inf
import time
from framework import Alg3D, Board  # 本番用

Coord3 = Tuple[int, int, int]  # (x, y, z)

# ---------------- 基本ユーティリティ ----------------
def clone(board: Board) -> Board:
    # 4x4x4 固定なので手書きコピーが最速&安全
    return [[row[:] for row in plane] for plane in board]

def in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < 4 and 0 <= y < 4 and 0 <= z < 4

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    # 下(0)から上(3)へ見て最初の空きを返す
    for z in range(4):
        if board[z][y][x] == 0:
            return z
    return None

def column_has_space(board: Board, x: int, y: int) -> bool:
    return board[3][y][x] == 0  # 一番上が空ならまだ置ける

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def undo_inplace(board: Board, x: int, y: int, z: int) -> None:
    # 事前条件: そのマスが自分で埋めた直後
    board[z][y][x] = 0

def board_full(board: Board) -> bool:
    for y in range(4):
        for x in range(4):
            if board[3][y][x] == 0:
                return False
    return True

def iter_legal_moves(board: Board) -> Iterable[Tuple[int, int]]:
    # 中央寄り優先の順序（実戦的に強い）
    order = [
        (1,1),(2,2),(1,2),(2,1),
        (0,1),(1,0),(2,3),(3,2),
        (0,2),(2,0),(1,3),(3,1),
        (0,0),(3,3),(0,3),(3,0),
    ]
    for (x, y) in order:
        if column_has_space(board, x, y):
            yield (x, y)

# ---------------- 全76ライン生成と勝敗判定 ----------------
DIRS = [
    (1,0,0),(0,1,0),(0,0,1),            # x,y,z
    (1,1,0),(1,-1,0),                   # xy 斜め
    (1,0,1),(0,1,1),                    # xz, yz
    (1,1,1),(1,-1,1),(-1,1,1),(-1,-1,1),(1,-1,-1)
]

def gen_all_lines() -> List[List[Coord3]]:
    lines = []
    seen = set()
    for z in range(4):
        for y in range(4):
            for x in range(4):
                for dx,dy,dz in DIRS:
                    tmp = []
                    ok = True
                    for i in range(4):
                        xx,yy,zz = x+i*dx, y+i*dy, z+i*dz
                        if not in_bounds(xx,yy,zz):
                            ok = False
                            break
                        tmp.append((xx,yy,zz))
                    if ok:
                        key = tuple(sorted(tmp))
                        if key not in seen:
                            seen.add(key)
                            lines.append(tmp)
    return lines

ALL_LINES = gen_all_lines()  # 76 本

def winner_after(board: Board, last: Optional[Coord3]) -> int:
    # 最後に置いたマスを含むラインを優先的にチェック
    if last is not None:
        lx,ly,lz = last
        for line in ALL_LINES:
            if (lx,ly,lz) in line:
                vals = [board[z][y][x] for (x,y,z) in line]
                if vals[0] != 0 and all(v == vals[0] for v in vals):
                    return vals[0]
    # 念のため全体確認
    for line in ALL_LINES:
        vals = [board[z][y][x] for (x,y,z) in line]
        if vals[0] != 0 and all(v == vals[0] for v in vals):
            return vals[0]
    return 0

# ---------------- 内蔵 定石（外部ファイル禁止のため固定） ----------------
# 列番号は 0..15 (n = x + 4*y)。中心帯→対称展開の短いブック。
INTERNAL_BOOK: List[List[int]] = [
    [5, 10, 6, 9],
    [6, 9, 5, 10],
    [1, 2, 4, 8],
]

def num_to_xy(n: int) -> Tuple[int,int]:
    return (n % 4, n // 4)

def book_suggest(board: Board, moves_played_nums: List[int]) -> Optional[Tuple[int,int]]:
    for seq in INTERNAL_BOOK:
        L = len(moves_played_nums)
        if L < len(seq) and moves_played_nums == seq[:L]:
            x,y = num_to_xy(seq[L])
            if column_has_space(board, x, y):
                return (x, y)
    return None

# ---------------- 評価関数（脅威＋中心加点） ----------------
def count_threats_for(board: Board, player: int) -> int:
    cnt = 0
    opp = 3 - player
    for line in ALL_LINES:
        vals = [board[z][y][x] for (x,y,z) in line]
        if vals.count(player) == 3 and vals.count(0) == 1 and vals.count(opp) == 0:
            cnt += 1
    return cnt

def evaluate(board: Board, player: int) -> float:
    opp = 3 - player
    score = 0.0
    my_threats = 0
    opp_threats = 0

    for line in ALL_LINES:
        vals = [board[z][y][x] for (x,y,z) in line]
        mc = vals.count(player)
        oc = vals.count(opp)
        ec = 4 - mc - oc

        if oc == 0:
            if mc == 4:
                return 1e9
            if mc == 3 and ec == 1:
                score += 300.0; my_threats += 1
            elif mc == 2 and ec == 2:
                score += 20.0
            elif mc == 1 and ec == 3:
                score += 2.0
        elif mc == 0:
            if oc == 4:
                return -1e9
            if oc == 3 and ec == 1:
                score -= 300.0; opp_threats += 1
            elif oc == 2 and ec == 2:
                score -= 20.0
            elif oc == 1 and ec == 3:
                score -= 2.0
        # 混在は0点扱い

    # ダブルリーチ強化
    if my_threats >= 2: score += 5000.0
    if opp_threats >= 2: score -= 5000.0

    # 中央寄りボーナス（序〜中盤の形作り）
    center_w = 3.0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                v = board[z][y][x]
                if v == 0: 
                    continue
                cx = min(x, 3-x)
                cy = min(y, 3-y)
                bonus = (3 - (cx + cy)) * center_w
                if v == player: score += bonus
                else: score -= bonus

    return score

# ---------------- 即勝/ブロック探索 ----------------
def find_immediate_win(board: Board, player: int) -> Optional[Tuple[int,int,int]]:
    for (x,y) in iter_legal_moves(board):
        z = place_inplace(board, x, y, player)
        if z is None:
            continue
        w = winner_after(board, (x,y,z))
        undo_inplace(board, x, y, z)
        if w == player:
            return (x,y,z)
    return None

# ---------------- Zobrist（決定論的に自前生成） ----------------
# 禁止ライブラリなしで固定64bit擬似乱数を生成
def _lcg64(seed: int) -> int:
    # 定数は一般的なLCGパラメータ
    return (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

ZTABLE: List[List[List[List[int]]]] = [[[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
_seed = 0x9e3779b97f4a7c15  # 黄金比ベースの定数
for z in range(4):
    for y in range(4):
        for x in range(4):
            for v in range(3):  # 0,1,2（0は未使用だが揃えておく）
                _seed = _lcg64(_seed ^ (x*0x10001 ^ y*0x1009e ^ z*0x10f1 ^ v*0x9e37))
                ZTABLE[z][y][x][v] = _seed or 0xA5A5A5A5A5A5A5A5  # 0回避

def zobrist(board: Board) -> int:
    h = 0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                v = board[z][y][x]
                if v:
                    h ^= ZTABLE[z][y][x][v]
    return h

class TTEntry:
    __slots__ = ("depth","value","flag","best")
    def __init__(self, depth:int, value:float, flag:str, best:Optional[Tuple[int,int]]):
        self.depth=depth; self.value=value; self.flag=flag; self.best=best

TT: Dict[int, TTEntry] = {}

# ---------------- αβ探索 + 反復深化（安全ガード付き） ----------------
def alphabeta(board: Board, to_move: int, depth: int,
              alpha: float, beta: float,
              last: Optional[Coord3]) -> Tuple[float, Optional[Tuple[int,int]]]:
    me = to_move
    opp = 3 - me

    w = winner_after(board, last)
    if w == me: return (1e9 - (8 - depth), None)
    if w == opp: return (-1e9 + (8 - depth), None)
    if depth == 0 or board_full(board):
        return (evaluate(board, me), None)

    key = zobrist(board)
    e = TT.get(key)
    if e is not None and e.depth >= depth:
        if e.flag == "EXACT": return (e.value, e.best)
        if e.flag == "LOWER": alpha = max(alpha, e.value)
        if e.flag == "UPPER": beta  = min(beta,  e.value)
        if alpha >= beta:     return (e.value, e.best)

    best_move = None
    value = -inf

    # 即勝ちがあればそれを返す（強制手優先）
    iw = find_immediate_win(board, me)
    first_moves = list(iter_legal_moves(board))
    if iw:
        x,y,z = iw
        # 探索深さを1つ進める（相手番）
        place_inplace(board, x, y, me)
        sc,_ = alphabeta(board, opp, depth-1, -beta, -alpha, (x,y,z))
        sc = -sc
        undo_inplace(board, x, y, z)
        return (sc, (x,y))

    # 相手の即勝ブロック候補を優先
    block_set = set()
    for (x,y) in first_moves:
        z = lowest_empty_z(board, x, y)
        if z is None: 
            continue
        board[z][y][x] = opp
        if winner_after(board, (x,y,z)) == opp:
            block_set.add((x,y))
        board[z][y][x] = 0

    # ムーブオーダリング：ブロック > 自脅威増 > 相手脅威減 > 中心寄り
    def move_key(m: Tuple[int,int]) -> Tuple[int,int,int,int]:
        x,y = m
        z = lowest_empty_z(board, x, y)
        if z is None:
            return (0,0,0,0)
        # 自分置く
        board[z][y][x] = me
        myt = count_threats_for(board, me)
        board[z][y][x] = 0
        # 相手置く
        board[z][y][x] = opp
        oppt = count_threats_for(board, opp)
        board[z][y][x] = 0
        center = -(abs(x-1.5) + abs(y-1.5))  # 中心近いほど大
        return (1 if (x,y) in block_set else 0, myt, -oppt, int(center*10))

    moves = sorted(first_moves, key=move_key, reverse=True)

    for (x,y) in moves:
        z = place_inplace(board, x, y, me)
        if z is None:
            continue
        sc,_ = alphabeta(board, opp, depth-1, -beta, -alpha, (x,y,z))
        sc = -sc
        undo_inplace(board, x, y, z)

        if sc > value:
            value = sc
            best_move = (x,y)
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break

    flag = "EXACT"
    if value <= alpha: flag = "UPPER"
    elif value >= beta: flag = "LOWER"
    TT[key] = TTEntry(depth, value, flag, best_move)
    return (value, best_move)

def search_best(board: Board, player: int, time_budget: float = 0.8) -> Tuple[int,int]:
    start = time.time()
    last_best: Optional[Tuple[int,int]] = None
    # 先に合法手があるか最低1つ確認（異常終了防止）
    first_legal = None
    for mv in iter_legal_moves(board):
        first_legal = mv
        break
    if first_legal is None:
        return (0,0)  # 盤が満杯など。呼び出し側仕様に合わせて返す

    # 反復深化：最大深さは7（重い場合は途中で打ち切り、最後に安定手を返す）
    max_depth = 7
    for depth in range(2, max_depth+1):
        if time.time() - start > time_budget and last_best is not None:
            break
        val, bm = alphabeta(board, player, depth, -inf, inf, None)
        if bm is not None:
            last_best = bm

    return last_best if last_best is not None else first_legal

# ---------------- メインAI ----------------
class MyAI(Alg3D):
    def __init__(self):
        self.history_nums: List[int] = []  # 0..15 の列番号で棋譜を保持

    def get_move(self, board: Board, player: int, last_move: Tuple[int,int,int]) -> Tuple[int,int]:
        # 1) 即勝ち（安全：Noneチェック内蔵）
        iw = find_immediate_win(board, player)
        if iw:
            x,y,_ = iw
            return (x,y)

        # 2) 内蔵定石が合えば即着
        bm = book_suggest(board, self.history_nums)
        if bm is not None:
            x,y = bm
            self.history_nums.append(y*4 + x)
            return (x,y)

        # 3) 探索（失敗時は必ず最初の合法手でフォールバック）
        x,y = search_best(board, player, time_budget=0.9)
        # 棋譜を更新（実際に置かれる前提で列番号を足す）
        if 0 <= x < 4 and 0 <= y < 4:
            self.history_nums.append(y*4 + x)
        return (x,y)
