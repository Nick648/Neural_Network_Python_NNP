from tkinter import *
import random
from Algorithm import minimax

root = Tk()
root.title('Criss-Cross')
game_run = True
field = []
FIELD_SIZE = 3
cross_count = 1
total_cross_count = FIELD_SIZE ** 2
player = 'X'
computer = 'O'


def new_game():
    global cross_count, game_run
    for row in range(FIELD_SIZE):
        for col in range(FIELD_SIZE):
            field[row][col]['text'] = ' '
            field[row][col]['background'] = 'lavender'
    game_run = True
    cross_count = 0


def click(row, col):
    if game_run and field[row][col]['text'] == ' ':
        global cross_count
        field[row][col]['text'] = 'X'
        field[row][col]['fg'] = 'chocolate'
        cross_count += 1
        check_win('X')
        if cross_count < total_cross_count:
            cross_count += 1
            computer_move_minimax()
            # computer_move_algorithm()
            check_win('O')


def check_line_win(array_line: list, step_player: str) -> None:
    win = [btn['text'] for btn in array_line]
    if len(set(win)) == 1 and win[0] == step_player:
        global game_run
        for btn in array_line:
            if step_player == player:
                btn['background'] = 'green'
            else:
                btn['background'] = 'red'
            game_run = False


def check_win(step_player: str) -> None:
    send_array_l_diagonal = []
    send_array_r_diagonal = []
    for i_row in range(FIELD_SIZE):
        send_array_l_diagonal.append(field[i_row][i_row])
        send_array_r_diagonal.append(field[FIELD_SIZE - i_row - 1][i_row])
        send_array_row, send_array_column = [], []
        for j_col in range(FIELD_SIZE):
            send_array_row.append(field[i_row][j_col])
            send_array_column.append(field[j_col][i_row])
        check_line_win(send_array_row, step_player)
        check_line_win(send_array_column, step_player)
    check_line_win(send_array_l_diagonal, step_player)
    check_line_win(send_array_r_diagonal, step_player)


def can_win_line(array_line: list, stop_win: bool) -> bool:
    search_player = computer
    if stop_win:
        search_player = player
    win = [btn['text'] for btn in array_line]
    if win.count(search_player) == len(array_line) - 1 and win.count(' ') == 1:
        btn_index = win.index(' ')
        array_line[btn_index]['text'] = computer
        array_line[btn_index]['fg'] = 'magenta'
        return True
    return False


def check_can_win(stop_win: bool) -> bool:
    send_array_l_diagonal = []
    send_array_r_diagonal = []
    for i_row in range(FIELD_SIZE):
        send_array_l_diagonal.append(field[i_row][i_row])
        send_array_r_diagonal.append(field[FIELD_SIZE - i_row - 1][i_row])
        send_array_row, send_array_column = [], []
        for j_col in range(FIELD_SIZE):
            send_array_row.append(field[i_row][j_col])
            send_array_column.append(field[j_col][i_row])
        if can_win_line(send_array_row, stop_win):
            return True
        if can_win_line(send_array_column, stop_win):
            return True
    if can_win_line(send_array_l_diagonal, stop_win):
        return True
    if can_win_line(send_array_r_diagonal, stop_win):
        return True
    return False


def computer_move_algorithm() -> None:
    # Checking the computer's winnings
    if check_can_win(False):
        return
    # Prevent a person from winning
    if check_can_win(True):
        return

    while True:
        row = random.randint(0, FIELD_SIZE - 1)
        col = random.randint(0, FIELD_SIZE - 1)
        if field[row][col]['text'] == ' ':
            field[row][col]['text'] = computer
            field[row][col]['fg'] = 'magenta'
            break


def computer_move_minimax() -> None:
    game_field = [['' for _ in range(FIELD_SIZE)] for _ in range(FIELD_SIZE)]
    for row in range(FIELD_SIZE):
        for col in range(FIELD_SIZE):
            if field[row][col]['text'] == ' ':
                game_field[row][col] = ''
            else:
                game_field[row][col] = field[row][col]['text']
    result = minimax(game_field=game_field, depth=1, player=computer, opponent=player, step_player=True)
    score, step = result[0], result[1]
    step_x, step_y = step[0], step[1]
    field[step_x][step_y]['text'] = computer
    field[step_x][step_y]['fg'] = 'magenta'


if __name__ == '__main__':
    for row_line in range(FIELD_SIZE):
        line = []
        for col_pos in range(FIELD_SIZE):
            button = Button(root, text=' ', width=4, height=2,
                            font=('Verdana', 20, 'bold'),
                            background='lavender',
                            command=lambda row=row_line, col=col_pos: click(row, col))
            button.grid(row=row_line, column=col_pos, sticky='nsew')
            line.append(button)
        field.append(line)
    new_button = Button(root, text='new game', command=new_game)
    new_button.grid(row=FIELD_SIZE, column=0, columnspan=FIELD_SIZE, sticky='nsew')
    root.mainloop()
