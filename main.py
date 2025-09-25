# --- ルート選択（厳密ブロック優先; 自分の即勝ちは例外的に先行） ---
def choose_best(board: Board, me: int) -> Coord2:
    start = time.perf_counter()
    moves = valid_xy_moves(board)
    if not moves:
        return (0,0)

    you = 3 - me

    # まず相手の「いまの」即勝ちを検出
    opp_now = immediate_winning_squares(board, you)

    # 自分の「いまの」即勝ちがあるなら最優先で勝ち切る
    my_now = [mv for mv in moves if is_winning_after(board, me, mv[0], mv[1])]
    if my_now:
        return my_now[0]

    # 自分即勝ちが無い場合、相手が即勝ち手を持っていたら
    # after==0（相手の即勝ちを完全に消す）手から必ず選ぶ
    if opp_now:
        zero_block = []
        best = None
        best_after = 10**9
        best_sc = -10**9
        for (x,y) in moves:
            z = place_inplace(board, x, y, me)
            after = len(immediate_winning_squares(board, you))
            sc = move_score(board, me, x, y)
            undo_place(board, x, y, z)
            if after == 0:
                zero_block.append((x,y,sc))
            # 全滅時に備えて最小 after も記録
            if after < best_after or (after == best_after and sc > best_sc):
                best_after, best_sc, best = after, sc, (x,y)
        if zero_block:
            # 同率なら自分評価最大
            zero_block.sort(key=lambda t: -t[2])
            return (zero_block[0][0], zero_block[0][1])
        # どうしても0にできない場合だけ最小 after を選ぶ
        return best if best is not None else moves[0]

    # ---- ここからは通常手選択（常に安全フィルタを通す）----

    # 序盤の角優先（安全なものだけ）
    mv = corner_first_opening(board, me)
    if mv is not None:
        # corner_first_opening 内でも安全確認しているが、二重に守る
        if opponent_immediate_wins_after(board, me, mv[0], mv[1]) == 0:
            return mv

    # 安全手だけに絞る（自手後に相手の即勝ちが発生する案は却下）
    safe_moves: List[Coord2] = []
    for (x,y) in moves:
        if opponent_immediate_wins_after(board, me, x, y) == 0:
            safe_moves.append((x,y))

    cand = safe_moves if safe_moves else moves  # 全滅なら最小被害へ

    # 軽量評価で最大化
    best = cand[0]
    best_sc = -10**9
    for (x,y) in cand:
        sc = move_score(board, me, x, y)
        if sc > best_sc:
            best_sc = sc
            best = (x,y)

    if time.perf_counter() - start > TIME_BUDGET_SEC:
        return best
    return best
