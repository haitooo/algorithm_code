# main.py  —  角→対角→即勝ち→ブロック→ダブトリ→形（軽量版）
from typing import List, Tuple, Optional, Iterable
from framework import Alg3D, Board  # 本番用

Coord = Tuple[int, int, int]  # (x, y, z)

# ============== 基本ユーティリティ（安全：盤面コピー） ==============
def clone(board: Board) -> Board:
    return [[row[:] for row in plane] for plane in board]

def lowest_empty_z(board: Board, x: int, y: int) -> Optional[int]:
    for z in range(4):
        if board[z][y][x] == 0:
            return z
    return None

def column_has_space(board: Board, x: int, y: int) -> bool:
    return board[3][y][x] == 0

def place_inplace(board: Board, x: int, y: int, player: int) -> Optional[int]:
    z = lowest_empty_z(board, x, y)
    if z is None:
        return None
    board[z][y][x] = player
    return z

def legal_cols(board: Board) -> Iterable[Tuple[int, int]]:
    for x in range(4):
        for y in range(4):
            if column_has_space(board, x, y):
                yield (x, y)

def count_bits(board: Board) -> int:
    c = 0
    for z in range(4):
        for y in range(4):
            for x in range(4):
                if board[z][y][x] != 0:
                    c += 1
    return c

# ============== 勝ち筋（76本） ==============
def all_win_lines() -> List[List[Coord]]:
    L: List[List[Coord]] = []
    rng = range(4)
    # 軸方向
    for y in rng:
        for z in rng:
            L.append([(x, y, z) for x in rng])  # X軸
    for x in rng:
        for z in rng:
            L.append([(x, y, z) for y in rng])  # Y軸
    for x in rng:
        for y in rng:
            L.append([(x, y, z) for z in rng])  # Z軸
    # 平面斜め
    for z in rng:
        L.append([(i, i, z) for i in rng])
        L.append([(i, 3 - i, z) for i in rng])
    for y in rng:
        L.append([(i, y, i) for i in rng])
        L.append([(i, y, 3 - i) for i in rng])
    for x in rng:
        L.append([(x, i, i) for i in rng])
        L.append([(x, i, 3 - i) for i in rng])
    # 空間対角
    L.append([(i, i, i) for i in rng])
    L.append([(i, i, 3 - i) for i in rng])
    L.append([(i, 3 - i, i) for i in rng])
    L.append([(3 - i, i, i) for i in rng])
    return L

WIN_LINES = all_win_lines()

def is_win_for(board: Board, player: int) -> bool:
    for line in WIN_LINES:
        ok = True
        for (x, y, z) in line:
            if board[z][y][x] != player:
                ok = False
                break
        if ok:
            return True
    return False

# ============== 即勝ち / 多重スレッド（コピー盤で判定） ==============
def immediate_winning_moves(board: Board, player: int) -> List[Tuple[int, int]]:
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        b2 = clone(board)
        if place_inplace(b2, x, y, player) is not None and is_win_for(b2, player):
            res.append((x, y))
    return res

def k_threat_moves(board: Board, player: int, k: int) -> List[Tuple[int, int]]:
    """ その手の後に『次手で勝てる手』がk個以上になる手（k=2:ダブル, k=3:トリプル） """
    res: List[Tuple[int, int]] = []
    for (x, y) in legal_cols(board):
        b2 = clone(board)
        if place_inplace(b2, x, y, player) is None:
            continue
        wins_next = immediate_winning_moves(b2, player)  # コピー盤
        if len(wins_next) >= k:
            res.append((x, y))
    return res

# ============== 形（フ型 / 逆ミッキー）検出：2D占有カラムで軽量判定 ==============
# 盤は 0..3 の 4x4。形は「(x,y)のセット」で定義。回転(90°刻み)・反転を全生成して検出。
Corner = [(0,0),(0,3),(3,0),(3,3)]

# ※以下のベース形は“例”です。手元のノート定義に合わせて自由に編集してください。
#   ここでは 4x4 に収まる代表的な5点形として用意し、回転・反転を自動展開します。
F_SHAPES_BASE = [
    {(0,0),(1,0),(2,0),(0,1),(0,2)},  # 横棒+縦棒の“F”っぽい
]
REV_MICKEY_BASE = [
    {(0,0),(3,0),(1,1),(2,1),(1,2)},  # “耳耳+顔”を意識した逆ミッキー例
]

def rotate90(pt: Tuple[int,int]) -> Tuple[int,int]:
    x,y = pt
    return (3 - y, x)

def reflect_x(pt: Tuple[int,int]) -> Tuple[int,int]:
    x,y = pt
    return (3 - x, y)

def reflect_y(pt: Tuple[int,int]) -> Tuple[int,int]:
    x,y = pt
    return (x, 3 - y)

def all_transforms(shape: set) -> List[set]:
    variants = []
    cur = set(shape)
    for _ in range(4):
        variants.append(cur)
        variants.append(set(reflect_x(p) for p in cur))
        variants.append(set(reflect_y(p) for p in cur))
        # 次の回転
        cur = set(rotate90(p) for p in cur)
    # unique
    uniq = []
    seen = set()
    for s in variants:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

F_SHAPES = [v for base in F_SHAPES_BASE for v in all_transforms(base)]
REV_MICKEY_SHAPES = [v for base in REV_MICKEY_BASE for v in all_transforms(base)]

def occupied_xy_by(board: Board, player: int) -> set:
    """ そのプレイヤーが1つでも石を入れている (x,y) カラム集合 """
    occ = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if board[z][y][x] == player:
                    occ.add((x,y))
                    break
    return occ

def shape_score_after_move(board: Board, x: int, y: int, player: int) -> int:
    """ 形ボーナスの軽量評価（一致した形の個数で加点）"""
    b2 = clone(board)
    if place_inplace(b2, x, y, player) is None:
        return -10**9  # 置けない
    occ = occupied_xy_by(b2, player)

    score = 0
    # 一致完全包含で判定（必要なら“部分的一致で加点”に緩和可能）
    for s in F_SHAPES:
        if s.issubset(occ):
            score += 3   # フ型は中加点
    for s in REV_MICKEY_SHAPES:
        if s.issubset(occ):
            score += 4   # 逆ミッキーはやや高めに
    # 角・中央近傍のちょい足し（タイブレーク用の弱い嗜好）
    if (x,y) in Corner:
        score += 1
    if (x,y) in [(1,1),(1,2),(2,1),(2,2)]:
        score += 1
    return score

# ============== 角・対角ユーティリティ ==============
def my_corners(board: Board, me: int) -> List[Tuple[int,int]]:
    res = []
    for (x,y) in Corner:
        # カラム内のどこかに自石があれば「その角を持っている」とみなす
        for z in range(4):
            if board[z][y][x] == me:
                res.append((x,y))
                break
    return res

def opp_corners(board: Board, opp: int) -> List[Tuple[int,int]]:
    res = []
    for (x,y) in Corner:
        for z in range(4):
            if board[z][y][x] == opp:
                res.append((x,y))
                break
    return res

def opposite_corner(x: int, y: int) -> Tuple[int,int]:
    return (3 - x, 3 - y)

# ============== メインAI（軽量ヒューリスティック優先） ==============
class MyAI(Alg3D):
    def get_move(
        self,
        board: Board,                 # board[z][y][x]
        player: int,                  # 1:先手, 2:後手
        last_move: Tuple[int, int, int]
    ) -> Tuple[int, int]:
        me = player
        opp = 1 if player == 2 else 2
        mv = count_bits(board)

        # ---------------- 1) 初手は必ず“角取り” ----------------
        if mv == 0:
            for (x,y) in Corner:
                if column_has_space(board, x, y):
                    return (x, y)

        # 後手の初手も“角優先”
        if mv == 1 and me == 2:
            # 可能なら相手角の対角を優先、無理なら別角
            taken = opp_corners(board, opp)
            target_order = []
            if taken:
                # 相手が取った角の対角
                ox, oy = opposite_corner(*taken[0])
                target_order.append((ox, oy))
            # 残りの角
            for c in Corner:
                if c not in target_order:
                    target_order.append(c)
            for (x,y) in target_order:
                if column_has_space(board, x, y):
                    return (x, y)

        # ---------------- 2) 二手目（先手の2手目=全体手数2のとき）
        # 相手が“対角線”を取っていなければ“対角角”を取る。取られていれば別の角。
        if me == 1 and mv == 2:
            myc = my_corners(board, me)
            if myc:
                my_first_corner = myc[0]
                ox, oy = opposite_corner(*my_first_corner)
                opp_took_oxoy = (ox,oy) in opp_corners(board, opp)
                if (not opp_took_oxoy) and column_has_space(board, ox, oy):
                    return (ox, oy)
                # 別の角
                for (x,y) in Corner:
                    if (x,y) != my_first_corner and column_has_space(board, x, y):
                        return (x, y)

        # ---------------- 3) 自分の即勝ち ----------------
        wins = immediate_winning_moves(board, me)
        if wins:
            return wins[0]

        # ---------------- 4) 相手の即勝ちブロック ----------------
        opp_wins = immediate_winning_moves(board, opp)
        if opp_wins:
            # 角 > 他 の軽い優先
            opp_wins.sort(key=lambda xy: (xy not in Corner, ))
            for (x,y) in opp_wins:
                if column_has_space(board, x, y):
                    return (x, y)

        # ---------------- 5) ダブル / トリプルスレッド狙い ----------------
        # まずトリプル > ダブルの順で探す（探索はコピー盤で軽い）
        tris = k_threat_moves(board, me, 3)
        if tris:
            # 角・形スコアでタイブレーク
            tris.sort(key=lambda xy: (-shape_score_after_move(board, xy[0], xy[1], me),
                                      xy not in Corner))
            return tris[0]

        dubs = k_threat_moves(board, me, 2)
        if dubs:
            dubs.sort(key=lambda xy: (-shape_score_after_move(board, xy[0], xy[1], me),
                                      xy not in Corner))
            return dubs[0]

        # ---------------- 6) 形（フ型 / 逆ミッキー）を積極的に ----------------
        # すべての合法手を「形スコア」で評価し、最高を選択
        legal = list(legal_cols(board))
        if legal:
            best = max(legal, key=lambda xy: shape_score_after_move(board, xy[0], xy[1], me))
            return best

        # ---------------- フォールバック（万一） ----------------
        for (x,y) in Corner:
            if column_has_space(board, x, y):
                return (x, y)
        for (x,y) in legal_cols(board):
            return (x, y)
        return (0, 0)
