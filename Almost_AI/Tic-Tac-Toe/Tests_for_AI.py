from random import randint
from Algorithm import minimax
import numpy as np

FIELD_SIZE = 3


def step_player_x_o(field_arr: list[list[str]], player: str) -> None:
    while True:
        x = randint(0, 2)
        y = randint(0, 2)
        if not field_arr[x][y]:
            field_arr[x][y] = player
            break


def create_field_test(field_arr: list[list[str]], count_step: int) -> list[list[str]]:
    if not 1 <= count_step <= 4:
        return [['']]
    for _ in range(count_step):
        step_player_x_o(field_arr, 'X')
        step_player_x_o(field_arr, 'O')
    return field_arr


def create_new_field(fields: list, count_fields: int, count_steps: int) -> None:
    for _ in range(count_fields):
        copy_field = [['' for _ in range(FIELD_SIZE)] for _ in range(FIELD_SIZE)]
        while True:
            new_field = create_field_test(copy_field, count_steps)
            if new_field not in fields:
                result = minimax(new_field)
                fields.append((new_field, result))
                break


def display_str_tests(str_test: str, str_ans: str) -> None:
    print("\nResults of test files:\n")
    list_test = str_test.strip().split('\n')
    list_ans = str_ans.strip().split('\n')
    for req, ans in zip(list_test, list_ans):
        print(f"{req=}  {ans=}")


def write_test_files(str_test: str, str_ans: str) -> None:
    with open(file="data/Train_AI_quest.txt", mode="w", encoding="utf-8") as file_quest:
        file_quest.write(str_test)
    with open(file="data/Train_AI_ans.txt", mode="w", encoding="utf-8") as file_ans:
        file_ans.write(str_ans)


def get_step_in_binary(step: int) -> str:
    if step == 0:
        return '0 0'
    elif step == 1:
        return '0 1'
    elif step == 2:
        return '1 0'
    else:
        return ''


def transform_data_to_str(fields: list) -> tuple[str, str]:
    str_test = "Player -> 1, Opponent -> -1, Empty -> 0\n"
    str_ans = "Step_x, Step_y\n"
    for test in fields:
        new_test_req = ''
        field = test[0]
        score, step = test[1][0], test[1][1]
        if not step:
            continue
        for pos_line, line in enumerate(field):
            for pos_val, val in enumerate(line):
                if val == 'X':
                    new_test_req += '1 '
                elif val == 'O':
                    new_test_req += '-1 '
                else:
                    new_test_req += '0 '
        str_test += new_test_req + '\n'
        step_x = get_step_in_binary(step[0])
        step_y = get_step_in_binary(step[1])
        str_ans += f"{step_x} {step_y}\n"
    return str_test, str_ans


def create_test_file() -> None:
    fields = []
    create_new_field(fields, count_fields=10, count_steps=4)
    create_new_field(fields, count_fields=15, count_steps=3)
    create_new_field(fields, count_fields=20, count_steps=2)
    create_new_field(fields, count_fields=25, count_steps=1)

    # for f in fields:
    #     print(f"Field = {f[0]}; Score = {f[1][0]}; Step = {f[1][1]}")

    # Create str for test and answer
    str_test, str_ans = transform_data_to_str(fields)
    # display_str_tests(str_test, str_ans)
    write_test_files(str_test, str_ans)


def get_data_from_file(file_path_name: str) -> list[list[int]]:
    temporary_array = []
    with open(file=file_path_name, mode="r", encoding="utf-8") as test_file:
        test_file.readline()
        while True:
            try:
                input_values = list(map(int, test_file.readline().strip().split()))
                if input_values:
                    temporary_array.append(input_values)
                else:
                    break
            except ValueError:
                continue
            except Exception as ex:
                print(f"Developer's mistake -> {type(ex).__name__}: {ex}")
    return temporary_array


def read_data_from_files() -> tuple:
    temporary_array_quest = get_data_from_file("data/Train_AI_quest.txt")
    temporary_array_ans = get_data_from_file("data/Train_AI_ans.txt")
    try:
        data = np.array(temporary_array_quest)
        all_ans = np.array(temporary_array_ans)
        if len(data) == len(all_ans):
            return data, all_ans
        else:
            print(f"Error in file -> {len(data)=} != {len(all_ans)=}")
            exit()
    except Exception as ex:
        print(f"Developer's mistake -> {type(ex).__name__}: {ex}")
        exit()


def main():
    create_test_file()
    data, all_ans = read_data_from_files()
    print(f"{data=}\n\n{all_ans=}")


if __name__ == '__main__':
    main()
