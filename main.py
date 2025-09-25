# main.py
# 4x4x4 立体四目並べ AI
# 方針（qweral記事を反映）
# 1) 終局・1手勝ち・相手の1手勝ちブロックを最優先
# 2) ダブルリーチ（次手で勝ち手が2つ以上になる手）を強く評価
# 3) パターン認識ベースの評価関数（相手石混入ラインは無効化、空き含む自分/相手カウント）
# 4) t点（floatingリーチの空きマスが z==3）ボーナス
# 5) 中央寄り優先の軽い位置ボーナス（序盤寄与）
#
# 依存: framework.Alg3D, Board
# 引数: get_move(board, player, last_move) -> (x, y)

from typing import List, Tuple, Optional
from framework import Alg3D, Board

Coord2 = Tuple[int, int]        # (x, y)
Coord3 = Tuple[int, int, int]   # (x, y, z)

SIZE = 4
PLAYERS = (1, 2)

WIN_SCORE      = 1_000_000
DOUBLE_THREAT  = 5_000   # qweral更新記事での調整方針に合わせる
TPOINT_BONUS   = 150      # t点(z==3)の浮きリーチを加点
CENTER_BONUS   = 5        # 位置ボーナス（序盤寄与）
DEPTH_DEFAULT  = 3        # 計算量と安定性のバランス

def clone(board: Board) -> Board:
    return [[row[:] for row in plane] for plane in board]

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(SIZE):
        if board[z][y][x] == 0:
            return z
    return None

def valid_xy_moves(board: Board) -> List[Coord2]:
    ms: List[Coord2] = []
    for y in range(SIZE):
        for x in range(SIZE):
            if board[SIZE-1][y][x] == 0:
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

# --- 勝ち筋（4連）ラインの全列挙（76本）
def generate_lines() -> List[List[Coord3]]:
    lines: List[List[Coord3]] = []

    # 各軸方向
    for z in range(SIZE):
        for y in range(SIZE):
            lines.append([(x, y, z) for x in range(SIZE)])            # x直線
    for z in range(SIZE):
        for x in range(SIZE):
            lines.append([(x, y, z) for y in range(SIZE)])            # y直線
    for y in range(SIZE):
        for x in range(SIZE):
            lines.append([(x, y, z) for z in range(SIZE)])            # z直線（縦）

    # 各面内の2D斜め
    for z in range(SIZE):
        lines.append([(i, i, z) for i in range(SIZE)])
        lines.append([(i, SIZE-1-i, z) for i in range(SIZE)])
    for y in range(SIZE):
        lines.append([(i, y, i) for i in range(SIZE)])
        lines.append([(i, y, SIZE-1-i) for i in range(SIZE)])
    for x in range(SIZE):
        lines.append([(x, i, i) for i in range(SIZE)])
        lines.append([(x, i, SIZE-1-i) for i in range(SIZE)])

    # 立体対角線（空間対角4本）
    lines.append([(i, i, i) for i in range(SIZE)])
    lines.append([(i, i, SIZE-1-i) for i in range(SIZE)])
    lines.append([(i, SIZE-1-i, i) for i in range(SIZE)])
    lines.append([(SIZE-1-i, i, i) for i in range(SIZE)])

    return lines

ALL_LINES = generate_lines()

def is_winning_after(board: Board, player: int, x: int, y: int) -> bool:
    z = place_inplace(board, x, y, player)
    if z is None:
        return False
    win = check_win_at(board, player, (x, y, z))
    undo_place(board, x, y, z)
    return win

def check_win_at(board: Board, player: int, last: Coord3) -> bool:
    # last を含むラインだけ見てもよいが、4x4x4は軽いので全走査で充分
    for line in ALL_LINES:
        cnt = 0
        for (x, y, z) in line:
            if board[z][y][x] == player:
                cnt += 1
            else:
                break if board[z][y][x] != player else None
        # 上のbreakは使わず単純カウントにする
    # もう一度正確に
    for line in ALL_LINES:
        if all(board[z][y][x] == player for (x, y, z) in line):
            return True
    return False

def count_immediate_wins(board: Board, player: int) -> int:
    c = 0
    for (x, y) in valid_xy_moves(board):
        if is_winning_after(board, player, x, y):
            c += 1
    return c

def is_floating_reach(board: Board, player: int, line: List[Coord3]) -> Optional[Coord3]:
    # そのラインに相手石が無く、自分3石+空1なら空位置を返す
    others = 3 if player == 1 else 1
    cnt_p = cnt_o = 0
    empty: Optional[Coord3] = None
    for (x, y, z) in line:
        v = board[z][y][x]
        if v == player:
            cnt_p += 1
        elif v == others:
            cnt_o += 1
        else:
            empty = (x, y, z)
    if cnt_o == 0 and cnt_p == 3 and empty is not None:
        # 空きマスが実際に打てる（重力OK）かも確認
        x, y, z = empty
        ze = lowest_empty_z(board, x, y)
        if ze is not None and ze == z:
            return empty
    return None

def eval_board(board: Board, me: int, ply: int) -> int:
    you = 3 - me

    # 即勝利/敗北
    if any(all(board[z][y][x] == me  for (x,y,z) in line) for line in ALL_LINES):
        return WIN_SCORE - ply
    if any(all(board[z][y][x] == you for (x,y,z) in line) for line in ALL_LINES):
        return -WIN_SCORE + ply

    score = 0

    # 1) パターン認識（相手混入ラインは無効）
    #    自分ライン: 1,10,100,1000 / 相手ラインは対称に減点
    table = [0, 1, 10, 120]   # 3連は大きめ（t点でさらに上乗せ）
    for line in ALL_LINES:
        cnt_me = cnt_you = 0
        empties: List[Coord3] = []
        for (x,y,z) in line:
            v = board[z][y][x]
            if v == me:
                cnt_me += 1
            elif v == you:
                cnt_you += 1
            else:
                empties.append((x,y,z))
        if cnt_you == 0:
            score += table[cnt_me]
            # t点ボーナス
            if cnt_me == 3:
                # 空が合法で z==3 ならボーナス
                e = empties[0]
                x, y, z = e
                ze = lowest_empty_z(board, x, y)
                if ze is not None and ze == z and z == 3:
                    score += TPOINT_BONUS
        if cnt_me == 0:
            score -= table[cnt_you]
            if cnt_you == 3:
                e = [p for p in line if board[p[2]][p[1]][p[0]] == 0][0]
                x, y, z = e
                ze = lowest_empty_z(board, x, y)
                if ze is not None and ze == z and z == 3:
                    score -= TPOINT_BONUS

    # 2) 位置ボーナス（中心優先）
    #    (1,1) 最中央を+、その近傍もやや加点
    for y in range(SIZE):
        for x in range(SIZE):
            ztop = SIZE-1
            # スタックの一番上が空なら列の重心に対して軽く加点
            if board[ztop][y][x] == 0:
                cx = abs(1.5 - x)
                cy = abs(1.5 - y)
                score += int(CENTER_BONUS * (1.5 - (cx+cy)/2))

    return score

def order_moves(board: Board, me: int, moves: List[Coord2]) -> List[Coord2]:
    # 1) 即勝ち
    wins = []
    blocks = []
    double_threats = []
    rest = []
    you = 3 - me
    for (x,y) in moves:
        if is_winning_after(board, me, x, y):
            wins.append((x,y))
            continue
        # 相手の即勝ちをブロック
        z = place_inplace(board, x, y, me)
        can_block = any(is_winning_after(board, you, bx, by) for (bx,by) in valid_xy_moves(board))
        # ↑このままだと「相手が勝てる手があるか」判定。厳密ブロック判定は重いので簡略化。
        # ダブルリーチ生成判定
        my_immediate = count_immediate_wins(board, me)
        if z is not None:
            undo_place(board, x, y, z)
        if my_immediate >= 2:
            double_threats.append((x,y))
        elif can_block:
            blocks.append((x,y))
        else:
            rest.append((x,y))

    # 適当にまとめる（探索前の並べ替え）
    return wins + double_threats + blocks + rest

def minimax(board: Board, depth: int, alpha: int, beta: int, me: int, turn: int, ply: int) -> int:
    # me: ルート側プレイヤ
    # turn: 手番プレイヤ
    moves = valid_xy_moves(board)
    you = 3 - turn

    # 直前で勝敗決する可能性があるため毎ノードで簡易終局判定
    # （より正確には last_move 起点のラインチェックで軽量化できる）
    if depth == 0 or not moves:
        return eval_board(board, me, ply)

    # 即勝ち手があれば刈り取り
    for (x,y) in moves:
        if is_winning_after(board, turn, x, y):
            return WIN_SCORE - ply if turn == me else -WIN_SCORE + ply

    # move ordering
    moves = order_moves(board, turn, moves)

    if turn == me:
        best = -10**9
        for (x,y) in moves:
            z = place_inplace(board, x, y, turn)
            val = minimax(board, depth-1, alpha, beta, me, 3-turn, ply+1)
            undo_place(board, x, y, z)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = 10**9
        for (x,y) in moves:
            z = place_inplace(board, x, y, turn)
            val = minimax(board, depth-1, alpha, beta, me, 3-turn, ply+1)
            undo_place(board, x, y, z)
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best

def choose_best(board: Board, me: int, depth: int) -> Coord2:
    # 1) まず即勝ち
    for (x,y) in valid_xy_moves(board):
        if is_winning_after(board, me, x, y):
            return (x,y)

    # 2) 相手の即勝ちブロック
    you = 3 - me
    for (x,y) in valid_xy_moves(board):
        if is_winning_after(board, you, x, y):
            return (x,y)

    # 3) ダブルリーチ作成手を優先
    cand = []
    for (x,y) in valid_xy_moves(board):
        z = place_inplace(board, x, y, me)
        k = count_immediate_wins(board, me)
        undo_place(board, x, y, z)
        if k >= 2:
            cand.append((x,y))
    if cand:
        return cand[0]

    # 4) 通常探索（ミニマックス+αβ）
    best_val = -10**9
    best_move = None
    moves = order_moves(board, me, valid_xy_moves(board))
    for (x,y) in moves:
        z = place_inplace(board, x, y, me)
        val = minimax(board, depth-1, -10**9, 10**9, me, 3-me, 1)
        undo_place(board, x, y, z)
        if val > best_val:
            best_val = val
            best_move = (x,y)

    # 5) フォールバック：中心寄り
    if best_move is None:
        cx_order = sorted(valid_xy_moves(board), key=lambda p: (abs(1.5-p[0])+abs(1.5-p[1])))
        best_move = cx_order[0]
    return best_move

class MyAI(Alg3D):
    def __init__(self, depth: int = DEPTH_DEFAULT):
        self.depth = depth

    def get_move(self, board: Board, player: int, last_move: Coord3) -> Coord2:
        # 定石: 初手は(1,1)を好む（中央寄り）※qweralの「定石」方針に倣う簡易版
        if all(board[z][y][x] == 0 for z in range(SIZE) for y in range(SIZE) for x in range(SIZE)):
            return (1, 1)

        return choose_best(board, player, self.depth)

# エントリ（本番環境が import する想定）
AI = MyAI()
