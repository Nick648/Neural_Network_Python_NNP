from time import perf_counter
from functools import cache, lru_cache

FIELD_SIZE = 3
field = [['' for _ in range(FIELD_SIZE)] for _ in range(FIELD_SIZE)]
vals_test_1 = [[i for i in range(j, j + FIELD_SIZE)] for j in range(1, FIELD_SIZE ** 2, FIELD_SIZE)]

field_dis = '''
 X | X | X 
-----------
 X | X | X 
-----------
 X | X | X 
'''


def set_field(size: int = 3):
    global field, FIELD_SIZE
    FIELD_SIZE = size
    field = [['' for _ in range(FIELD_SIZE)] for _ in range(FIELD_SIZE)]


def display_field(arr: list[list[str]], mes: str = None) -> None:
    print('*' * 30)
    if mes:
        print(f"{mes}")
    for line in arr:
        for item in line:
            if item:
                print(item, end=' ')
            else:
                print(' ', end=' ')
        print()
    print('*' * 30)


def check_line_win(array_line: list[str]) -> int:
    if len(set(array_line)) == 1:
        return 1
    else:
        return 0


def check_win(field_arr: list[list[str]]) -> str:
    send_array_l_diagonal = []
    send_array_r_diagonal = []
    for i_row in range(len(field_arr)):
        send_array_l_diagonal.append(field_arr[i_row][i_row])
        send_array_r_diagonal.append(field_arr[FIELD_SIZE - i_row - 1][i_row])
        send_array_row, send_array_column = [], []
        for j_col in range(len(field_arr)):
            send_array_row.append(field_arr[i_row][j_col])
            send_array_column.append(field_arr[j_col][i_row])
        if check_line_win(send_array_row):
            return send_array_row[0]
        if check_line_win(send_array_column):
            return send_array_column[0]
    if check_line_win(send_array_l_diagonal):
        return send_array_l_diagonal[0]
    if check_line_win(send_array_r_diagonal):
        return send_array_r_diagonal[0]
    return ''


def check_end_game(field_arr: list[list[str]]) -> bool:
    if check_win(field_arr):
        return True
    for pos_line, line in enumerate(field_arr):
        for pos_val, val in enumerate(line):
            if not val:
                return False
    return True


def score(game_field: list[list[str]], depth: int, player: str) -> int:
    winner = check_win(game_field)
    if winner:
        if winner == player:
            return 10 - depth
        else:
            return depth - 10
    else:
        return 0


# @lru_cache(maxsize=800)
def minimax(game_field: list[list[str]], depth: int = 1, player: str = 'X', opponent: str = 'O',
            step_player: bool = True):
    # print(f"Minimax: {game_field=}; {depth=}; {step_player=}")
    if check_end_game(game_field) or check_win(game_field):
        # print(f"\tReturn {game_field=}; {depth=}; {step_player=}; {score(game_field, depth, player)}")
        return score(game_field, depth, player), ()
    depth += 1
    scores = []  # an array of scores
    moves = []  # an array of moves

    # Populate the scores array, recursing as needed
    for pos_line, line in enumerate(game_field):
        for pos_val, val in enumerate(line):
            if not val:
                copy_field = [line.copy() for line in game_field]
                if step_player:
                    copy_field[pos_line][pos_val] = player
                else:
                    copy_field[pos_line][pos_val] = opponent
                n_score, n_move = minimax(copy_field, depth, player, opponent, not step_player)
                scores.append(n_score)
                moves.append((pos_line, pos_val))
                # print(f"\tAppend {scores=}; {moves=}; {depth=}; {step_player=}")

    # print(f"{scores=}; {moves=}; {game_field=}; {depth=}; {step_player=}")
    # Do the min or the max calculation
    if step_player:
        # This is the max calculation
        max_score_index = scores.index(max(scores))
        choice = moves[max_score_index]
        # print(f"\tReturn {max_score_index=}; {choice=}; {scores[max_score_index]=}")
        return scores[max_score_index], choice
    else:
        # This is the min calculation
        min_score_index = scores.index(min(scores))
        choice = moves[min_score_index]
        # print(f"\tReturn {min_score_index=}; {choice=}; {scores[min_score_index]=}")
        return scores[min_score_index], choice


def test_game():
    set_field(5)
    # field[2][2] = 'X'
    display_field(field, 'Begin')
    while not check_end_game(field):
        while True:
            try:
                x, y = map(int, input('Your step(x y) --> ').strip().split())
                field[x][y] = 'O'
                break
            except ValueError:
                print("Not right!")
        display_field(field, 'Your step:')
        score_step, step = minimax(game_field=field, player='X', opponent='O')
        print(f"\t{step=} -> {score_step=};")
        try:
            field[step[0]][step[1]] = 'X'
        except IndexError:
            print('End game or error')
        display_field(field, 'Opponent step:')


if __name__ == '__main__':
    # test_game()
    for _ in range(3):
        begin_field = [['O', '', ''], ['', '', 'X'], ['', '', '']]
        start = perf_counter()
        res = minimax(begin_field)
        end = perf_counter()
        print(f"1: {res=} -> {end - start}")

        begin_field2 = [['X', '', ''], ['', '', 'O'], ['', '', '']]
        start2 = perf_counter()
        res2 = minimax(begin_field)
        end2 = perf_counter()
        print(f"2: {res2=} -> {end2 - start2}")
