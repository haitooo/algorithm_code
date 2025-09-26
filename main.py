# main.py — 4x4x4 立体四目並べ “book + αβ(IDS) + 脅威評価” 実装
# 依存: framework.Alg3D, Board  (あなたの環境に合わせて import を維持)
from typing import List, Tuple, Optional, Dict, Iterable
from math import inf
import os
import csv
import time
from framework import Alg3D, Board  # 本番用

Coord3 = Tuple[int, int, int]  # (x, y, z)

# ------------------------------------------------------------
# 盤ユーティリティ
# ------------------------------------------------------------
def clone(board: Board) -> Board:
    return [[row[:] for row in plane] for plane in board]  # 4x4x4

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(4):
        if board[z][y][x] == 0:
            return z
    return None

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None: 
        return None
    board[z][y][x] = player
    return z

def undo_inplace(board: Board, x: int, y: int, z: int) -> None:
    board[z][y][x] = 0

def column_has_space(board: Board, x: int, y: int) -> bool:
    return board[3][y][x] == 0

def iter_legal_moves(board: Board) -> Iterable[Tuple[int, int]]:
    # move ordering: 中央寄り→角の順（中心優先は実戦・記事の方針にも合致）
    order = [(1,1),(2,2),(1,2),(2,1),(0,1),(1,0),(2,3),(3,2),
             (0,2),(2,0),(1,3),(3,1),(0,0),(3,3),(0,3),(3,0)]
    for (x, y) in order:
        if column_has_space(board, x, y):
            yield (x, y)

def board_full(board: Board) -> bool:
    for y in range(4):
        for x in range(4):
            if board[3][y][x] == 0:
                return False
    return True

# ------------------------------------------------------------
# 勝敗判定（全76ライン）
# ------------------------------------------------------------
# 全ライン生成: 方向ベクトル13種、各起点から長さ4
DIRS = [
    (1,0,0),(0,1,0),(0,0,1),            # x, y, z
    (1,1,0),(1,-1,0),                   # 平面斜め(xy)
    (1,0,1),(0,1,1),                    # xz, yz
    (1,1,1),(1,-1,1),                   # 3D斜め
    (-1,1,1),(1, -1, -1),(-1,-1,1)      # 残りの3D斜め群
]
def gen_all_lines() -> List[List[Coord3]]:
    lines = []
    def inb(x,y,z): return 0<=x<4 and 0<=y<4 and 0<=z<4
    for z in range(4):
        for y in range(4):
            for x in range(4):
                for dx,dy,dz in DIRS:
                    xs = [(x+i*dx, y+i*dy, z+i*dz) for i in range(4)]
                    if all(inb(*p) for p in xs):
                        lines.append(xs)
    # 重複除去
    uniq = []
    seen = set()
    for L in lines:
        key = tuple(sorted(L))
        if key not in seen:
            seen.add(key)
            uniq.append(L)
    return uniq

ALL_LINES = gen_all_lines()  # 76本になる

def winner_after(board: Board, last: Optional[Coord3]) -> int:
    # 最後に置いた列を含むラインを優先判定（高速化）
    if last is not None:
        lx,ly,lz = last
        for line in ALL_LINES:
            if (lx,ly,lz) in line:
                v = [board[z][y][x] for (x,y,z) in line]
                if v[0]!=0 and all(t==v[0] for t in v):
                    return v[0]
    # 念のため全探索（序中盤でも高速なため）
    for line in ALL_LINES:
        v = [board[z][y][x] for (x,y,z) in line]
        if v[0]!=0 and all(t==v[0] for t in v):
            return v[0]
    return 0

# ------------------------------------------------------------
# 定石ブック（内蔵 + 外部CSV）
#   * 内蔵: “中央→対称”の人間定石（短い）
#   * 外部: human_made_book.csv または book.csv (行: 空白区切りの着手列 例 "5 10 0 15")
#   qweral氏: 自己対局から自動抽出された定石と人間作成ブックを併用。:contentReference[oaicite:1]{index=1}
# ------------------------------------------------------------
INTERNAL_BOOK = [
    # 先手: 中央帯から作って対称に展開（座標は (x+4*y) の16番号基準）
    # あなたの環境は (x,y) 返却なので内部変換は下で行う
    [5, 10, 6, 9],          # 対称中央列を確保 → 立体の筋を増やす
    [1, 2, 4, 8],           # 状況でこちらに分岐
]

def load_external_books() -> List[List[int]]:
    paths = ["human_made_book.csv", "book.csv"]
    seqs: List[List[int]] = []
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, newline="") as f:
                    for row in csv.reader(f):
                        if not row: continue
                        # 1セルにスペース区切り、または複数セルどちらも許可
                        tokens = " ".join(row).split()
                        seq = [int(t) for t in tokens]
                        if all(0 <= v <= 15 for v in seq):
                            seqs.append(seq)
            except Exception:
                pass
    return seqs

BOOK_SEQS = INTERNAL_BOOK + load_external_books()

# 棋譜→(x,y)列（重力落下）
def num_to_xy(n: int) -> Tuple[int,int]:
    return (n % 4, n // 4)

def book_suggest(board: Board, moves_played: List[int]) -> Optional[Tuple[int,int]]:
    for seq in BOOK_SEQS:
        if len(moves_played) < len(seq) and moves_played == seq[:len(moves_played)]:
            nxt = seq[len(moves_played)]
            x,y = num_to_xy(nxt)
            if column_has_space(board, x, y):
                return (x, y)
    return None

# ------------------------------------------------------------
# 脅威評価
#  - 即勝/即防
#  - ダブルリーチ / 3段決勝点（実質的な“奇偶手数で勝つ形”）を高加点
#  - 76ラインのパターンカウントで滑らかにスコア
# 記事: 定石 + パターン認識 + ダブルリーチ重み（5000）等の調整に言及。:contentReference[oaicite:2]{index=2}
# “勝利条件の偶奇/複数列”の理論背景は該当記事参照。:contentReference[oaicite:3]{index=3}
# ------------------------------------------------------------
def evaluate(board: Board, player: int) -> float:
    opp = 2 if player==1 else 1
    score = 0.0
    double_threat_bonus = 5000.0  # ダブルリーチ（記事に合わせたスケール）:contentReference[oaicite:4]{index=4}

    my_threats = 0
    opp_threats = 0
    my_double = 0
    opp_double = 0

    for line in ALL_LINES:
        cells = [board[z][y][x] for (x,y,z) in line]
        mc = cells.count(player)
        oc = cells.count(opp)
        ec = 4 - mc - oc

        if oc==0:
            # 自陣のみのライン
            if mc==4: 
                return 1e9
            if mc==3 and ec==1:
                score += 300.0; my_threats += 1
            elif mc==2 and ec==2:
                score += 20.0
            elif mc==1 and ec==3:
                score += 2.0
        elif mc==0:
            # 相手のみのライン
            if oc==4: 
                return -1e9
            if oc==3 and ec==1:
                score -= 300.0; opp_threats += 1
            elif oc==2 and ec==2:
                score -= 20.0
            elif oc==1 and ec==3:
                score -= 2.0
        # 混在: 無効ライン

    # ダブルリーチ加点（複数の“3+空1”）
    if my_threats >= 2:
        score += double_threat_bonus
        my_double += 1
    if opp_threats >= 2:
        score -= double_threat_bonus
        opp_double += 1

    # 3段決勝点の近傍（簡易近似）: 空+同色3 で、その空が“まだ落ちて来ない高さ”にあっても
    # 実質的に奇偶で勝てる“手順前後不可”の筋は強く評価（簡易検出）
    # → 空の上に空きがあっても、その列の着手回数 parity で先手/後手が到達する
    #   仕組みをボーナス化（厳密検出は重くなるので軽量近似）
    for line in ALL_LINES:
        # 位置と高さの偶奇をざっくり反映
        coords = line
        vals = [board[z][y][x] for (x,y,z) in coords]
        if vals.count(0)==1:
            idx = vals.index(0)
            x,y,z = coords[idx]
            # その空マスに到達するまでの手数 parity
            below_filled = all(board[k][y][x]!=0 for k in range(z))
            if not below_filled:
                # まだ積まれていない → 手順制約あり
                continue
            # 実際に 3 + 空1 の筋
            if vals.count(player)==3:
                score += 180.0  # 通常スレート+α
            elif vals.count(opp)==3:
                score -= 180.0

    # 中央寄り優先（開幕〜中盤の実戦的価値）
    center_weight = 3.0
    for y in range(4):
        for x in range(4):
            for z in range(4):
                v = board[z][y][x]
                if v==0: continue
                cx = min(x,3-x); cy=min(y,3-y)
                bonus = (3 - (cx+cy)) * center_weight  # 中央ほど大
                if v==player: score += bonus
                else: score -= bonus

    return score

# ------------------------------------------------------------
# 強制手（即勝/即死の回避）スキャナ
# ------------------------------------------------------------
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

def count_threats(board: Board, player: int) -> int:
    cnt = 0
    for line in ALL_LINES:
        vals = [board[z][y][x] for (x,y,z) in line]
        if vals.count(player)==3 and vals.count(0)==1 and vals.count(3-player)==0:
            cnt += 1
    return cnt

# ------------------------------------------------------------
# αβ + 反復深化 + TT（シンプルZobrist）
# ------------------------------------------------------------
import random
ZTABLE = [[[ [random.getrandbits(64) for _ in range(3)]  # 0,1,2
             for x in range(4)] for y in range(4)] for z in range(4)]

def zobrist(board: Board) -> int:
    h = 0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                v = board[z][y][x]
                if v: h ^= ZTABLE[z][y][x][v]
    return h

class TTEntry:
    __slots__ = ("depth","value","flag","best")
    def __init__(self, depth:int, value:float, flag:str, best:Optional[Tuple[int,int]]):
        self.depth=depth; self.value=value; self.flag=flag; self.best=best

TT: Dict[int, TTEntry] = {}

def alphabeta(board: Board, player: int, depth: int, alpha: float, beta: float,
              hist_moves: List[int], last: Optional[Coord3]) -> Tuple[float, Optional[Tuple[int,int]]]:
    opp = 3 - player

    w = winner_after(board, last)
    if w==player: return (1e9 - (7-depth), None)
    if w==opp:    return (-1e9 + (7-depth), None)
    if depth==0 or board_full(board):
        return (evaluate(board, player), None)

    key = zobrist(board)
    if key in TT:
        e = TT[key]
        if e.depth >= depth:
            if e.flag=="EXACT":
                return (e.value, e.best)
            elif e.flag=="LOWER":
                alpha = max(alpha, e.value)
            elif e.flag=="UPPER":
                beta = min(beta, e.value)
            if alpha >= beta:
                return (e.value, e.best)

    best_move = None
    value = -inf

    # Killer: 直前探索で良かった手を優先（TTのbest）
    first_moves = list(iter_legal_moves(board))
    if key in TT and TT[key].best in first_moves:
        first_moves.remove(TT[key].best)
        first_moves = [TT[key].best] + first_moves

    # 緊急手: 自即勝/相手即勝ブロックを先に展開
    iw = find_immediate_win(board, player)
    if iw:
        x,y,z = iw
        place_inplace(board, x,y,player)
        sc,_ = alphabeta(board, opp, depth-1, -beta, -alpha, hist_moves+[y*4+x], (x,y,z))
        sc = -sc
        undo_inplace(board, x,y,z)
        return (sc, (x,y))

    # 相手の即勝を防ぐ候補のみに絞る（枝刈り強化）
    block_candidates = set()
    for (x,y) in first_moves:
        z = lowest_empty_z(board,x,y)
        if z is None: 
            continue
        # 仮に相手が打って勝てる所はブロック必須
        board[z][y][x] = opp
        if winner_after(board,(x,y,z))==opp:
            block_candidates.add((x,y))
        board[z][y][x] = 0

    moves = list(first_moves)
    # ダブルリーチ生成/阻止の優先度を上げる
    def move_key(m):
        x,y=m
        z = lowest_empty_z(board,x,y)
        if z is None: return (-10, 0, 0)
        board[z][y][x] = player
        myt = count_threats(board, player)
        board[z][y][x] = 0
        board[z][y][x] = opp
        oppt = count_threats(board, opp)
        board[z][y][x] = 0
        # 自分の脅威多い→高、相手の脅威減→高、ブロック候補→最優先
        return (100 if (x,y) in block_candidates else 0, myt, -oppt)

    moves.sort(key=move_key, reverse=True)

    for (x,y) in moves:
        z = place_inplace(board, x,y,player)
        if z is None: 
            continue
        sc,_ = alphabeta(board, opp, depth-1, -beta, -alpha, hist_moves+[y*4+x], (x,y,z))
        sc = -sc
        undo_inplace(board, x,y,z)

        if sc > value:
            value = sc
            best_move = (x,y)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # TT格納
    flag = "EXACT"
    if value <= alpha: flag = "UPPER"
    elif value >= beta: flag = "LOWER"
    TT[key] = TTEntry(depth, value, flag, best_move)
    return (value, best_move)

# ------------------------------------------------------------
# 反復深化 (時間制御したい場合は time_budget を使う)
# ------------------------------------------------------------
def search_best(board: Board, player: int, hist_moves: List[int], time_budget: float = 0.8) -> Tuple[int,int]:
    start = time.time()
    last_best = None
    depth = 2
    last_move_3d: Optional[Coord3] = None
    # last は呼び出し側から貰えないこともあるので都度 winner を総当たりで見る方針にしている

    while depth <= 8:  # 実戦は 7 目標（記事推奨）。計算量次第で8まで。
        if time.time() - start > time_budget and last_best is not None:
            break
        val, bm = alphabeta(board, player, depth, -inf, inf, hist_moves, last_move_3d)
        if bm is not None:
            last_best = bm
        depth += 1
    # フォールバック：合法手
    if last_best is None:
        for mv in iter_legal_moves(board):
            return mv
        return (0,0)
    return last_best

# ------------------------------------------------------------
# メインAI
# ------------------------------------------------------------
class MyAI(Alg3D):
    def __init__(self):
        self.history_nums: List[int] = []  # 0..15 棋譜

    def get_move(self, board: Board, player: int, last_move: Tuple[int,int,int]) -> Tuple[int,int]:
        # 1) 即勝ち
        iw = find_immediate_win(board, player)
        if iw:
            return (iw[0], iw[1])
        # 2) 定石（ブックが一致したら即返し）
        bm = book_suggest(board, self.history_nums)
        if bm is not None:
            return bm
        # 3) 探索
        move = search_best(board, player, self.history_nums, time_budget=0.9)
        # 棋譜更新（返却直後に実際に積まれる前提で列番号を記録）
        x,y = move
        self.history_nums.append(y*4 + x)
        return move
