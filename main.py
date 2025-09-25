from typing import List, Tuple, Optional
#from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用
import math
import random

Coord = Tuple[int, int, int]

def inb(x,y,z): return 0 <= x < 4 and 0 <= y < 4 and 0 <= z < 4

def all_win_lines() -> List[List[Coord]]:
    L = []
    rng = range(4)
    # 直線（軸方向）
    for y in rng:
        for z in rng:
            L.append([(x,y,z) for x in rng])  # X軸
    for x in rng:
        for z in rng:
            L.append([(x,y,z) for y in rng])  # Y軸
    for x in rng:
        for y in rng:
            L.append([(x,y,z) for z in rng])  # Z軸
    # 平面内斜め（XY 各 Z）
    for z in rng:
        L.append([(i,i,z) for i in rng])
        L.append([(i,3-i,z) for i in rng])
    # 平面内斜め（XZ 各 Y）
    for y in rng:
        L.append([(i,y,i) for i in rng])
        L.append([(i,y,3-i) for i in rng])
    # 平面内斜め（YZ 各 X）
    for x in rng:
        L.append([(x,i,i) for i in rng])
        L.append([(x,i,3-i) for i in rng])
    # 空間対角（4本）
    L.append([(i,i,i) for i in rng])
    L.append([(i,i,3-i) for i in rng])
    L.append([(i,3-i,i) for i in rng])
    L.append([(3-i,i,i) for i in rng])
    return L

WIN_LINES = all_win_lines()
CENTER_COLS = [(1,1),(1,2),(2,1),(2,2)]  # 中央寄り
PREFERRED_COLS = CENTER_COLS + [
    (1,0),(2,0),(0,1),(3,1),(0,2),(3,2),(1,3),(2,3)  # 準中央
] + [(0,0),(0,3),(3,0),(3,3)]  # 端/隅

def count_bits(board: List[List[List[int]]]) -> int:
    c = 0
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if board[x][y][z] != 0:
                    c += 1
    return c

def lowest_empty_z(board, x, y) -> Optional[int]:
    for z in range(4):
        if board[x][y][z] == 0:
            return z
    return None

def legal_cols(board):
    for x in range(4):
        for y in range(4):
            if lowest_empty_z(board, x, y) is not None:
                yield (x, y)

def place(board, x, y, player) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None: return None
    board[x][y][z] = player
    return z

def unplace(board, x, y, z):
    board[x][y][z] = 0

def is_win_for(board, player) -> bool:
    for line in WIN_LINES:
        ok = True
        for (x,y,z) in line:
            if board[x][y][z] != player:
                ok = False; break
        if ok: return True
    return False

def immediate_winning_moves(board, player):
    res = []
    for (x,y) in legal_cols(board):
        z = place(board, x, y, player)
        if z is None: 
            continue
        if is_win_for(board, player):
            res.append((x,y))
        unplace(board, x, y, z)
    return res

def count_next_wins(board, player) -> int:
    # 次の1手で勝てる着手数（重複列もカウント）を数える
    cnt = 0
    for (x,y) in legal_cols(board):
        z = place(board, x, y, player)
        if z is None: 
            continue
        if is_win_for(board, player):
            cnt += 1
        unplace(board, x, y, z)
    return cnt

def creates_double_threat(board, player) -> List[Tuple[int,int]]:
    # 着手後、「次手で勝てる手」が2つ以上になる手を抽出
    res = []
    for (x,y) in legal_cols(board):
        z = place(board, x, y, player)
        if z is None: 
            continue
        wins = immediate_winning_moves(board, player)
        if len(wins) >= 2:
            res.append((x,y))
        unplace(board, x, y, z)
    return res

def eval_position(board, me, opp) -> int:
    # ライン評価：自分の石のみ/相手の石のみのラインを数える
    score = 0
    for line in WIN_LINES:
        myc = sum(1 for (x,y,z) in line if board[x][y][z] == me)
        opc = sum(1 for (x,y,z) in line if board[x][y][z] == opp)
        if myc > 0 and opc > 0:
            continue  # 相殺ラインは無価値
        if opc == 0:
            # 自分専有のポテンシャル
            if myc == 3: score += 1000
            elif myc == 2: score += 50
            elif myc == 1: score += 5
            else: score += 1
        elif myc == 0:
            # 相手専有のポテンシャル（負）
            if opc == 3: score -= 900  # やや軽く（先に即受けで塞ぐ前提）
            elif opc == 2: score -= 45
            elif opc == 1: score -= 4
            else: score -= 1
    # 中央寄りのコラムを軽く加点
    for (x,y) in CENTER_COLS:
        z = 3
        while z >= 0 and board[x][y][z] != 0:
            z -= 1
        if z >= 0:
            score += 1
    return score

def negamax(board, depth, alpha, beta, me, opp) -> Tuple[int, Optional[Tuple[int,int]]]:
    # 勝ち/負け即判定（直前着手側が勝っていないかを親側で見るのが通常だが、
    # ここでは深さ0や探索途中での即勝ち手で枝刈り）
    wins = immediate_winning_moves(board, me)
    if wins:
        # 一手で勝てるなら最大評価
        return (10_000 + depth, wins[0])
    if depth == 0:
        return (eval_position(board, me, opp), None)

    best_move = None
    # 着手順序：中心寄り → その他
    moves = list(legal_cols(board))
    moves.sort(key=lambda xy: (xy not in PREFERRED_COLS, PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999))
    # 相手の即勝ち受けを優先的に含める（順序のみ）
    opp_wins = set(immediate_winning_moves(board, opp))
    moves.sort(key=lambda xy: (xy not in opp_wins))

    for (x,y) in moves:
        z = place(board, x, y, me)
        if z is None:
            continue
        # 相手の番
        val, _ = negamax(board, depth-1, -beta, -alpha, opp, me)
        val = -val
        unplace(board, x, y, z)
        if val > alpha:
            alpha = val
            best_move = (x,y)
            if alpha >= beta:
                break
    return (alpha, best_move)

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 4x4x4: board[x][y][z] in {0,1,2}
        player: int,                  # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        me = player
        opp = 1 if player == 2 else 2

        # --- 1) 先手の初手ブック：中央寄りの列を最優先 ---
        move_count = count_bits(board)
        if me == 1 and move_count == 0:
            # 8つの中央セルを含む4列のいずれかに置く（ランダム/固定どちらでも）
            for (x,y) in CENTER_COLS:
                if lowest_empty_z(board, x, y) is not None:
                    return (x, y)
            # フォールバック
            for (x,y) in PREFERRED_COLS:
                if lowest_empty_z(board, x, y) is not None:
                    return (x, y)

        # --- 2) 即勝ち手 ---
        wins = immediate_winning_moves(board, me)
        if wins:
            return wins[0]

        # --- 3) 相手の即勝ちを即受け ---
        opp_wins = immediate_winning_moves(board, opp)
        if opp_wins:
            # 相手が勝てる列に直接打ち込んでブロック（重力で同列の勝ち点を塞ぐ）
            # 複数ある場合は中央寄り優先
            opp_wins.sort(key=lambda xy: (xy not in PREFERRED_COLS, PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999))
            for (x,y) in opp_wins:
                if lowest_empty_z(board, x, y) is not None:
                    return (x, y)

        # --- 4) ダブルスレット作成 ---
        dts = creates_double_threat(board, me)
        if dts:
            # 作れれば事実上の必勝筋になりやすい
            dts.sort(key=lambda xy: (xy not in PREFERRED_COLS, PREFERRED_COLS.index(xy) if xy in PREFERRED_COLS else 999))
            return dts[0]

        # --- 5) 浅い探索（Negamax + αβ） ---
        # 中盤以降は分岐も少ないため深さ 3〜5 程度で十分強い
        DEPTH = 5 if move_count >= 8 else 4
        val, best = negamax(board, DEPTH, -10**9, 10**9, me, opp)
        if best is not None:
            return best

        # --- 6) それでも決まらない場合のフォールバック（中心寄り優先） ---
        for (x,y) in PREFERRED_COLS:
            if lowest_empty_z(board, x, y) is not None:
                return (x, y)
        # 最後の手段：任意の合法手
        for (x,y) in legal_cols(board):
            return (x,y)
        # 盤が埋まっているはずだが、安全策
        return (0,0)
