import json

DICT_AWH = {}


def read_data_from_txt() -> tuple:
    array_quest, array_ans = [], []
    with open(file="data/Train_AI_2_quest.txt", mode="r", encoding="utf-8") as file_quest:
        for _ in range(2):
            input_values = file_quest.readline()
        while input_values:
            try:
                input_values = list(map(float, file_quest.readline().strip().split()))
                if input_values:
                    array_quest.append(input_values)
            except ValueError:
                continue
            except Exception as ex:
                print(f"Developer's mistake -> {type(ex).__name__}: {ex}")

    with open(file="data/Train_AI_2_ans.txt", mode="r", encoding="utf-8") as file_ans:
        file_ans.readline()
        while True:
            try:
                input_values = file_ans.readline().strip()
                if input_values:
                    array_ans.append(int(input_values))
                else:
                    break
            except ValueError:
                continue
            except Exception as ex:
                print(f"Developer's mistake -> {type(ex).__name__}: {ex}")

    if len(array_quest) == len(array_ans):
        return array_quest, array_ans
    else:
        print(f"Developer's mistake -> {len(array_quest)=} != {len(array_ans)}")
        print(f"{array_quest=}\n{array_ans}")
        exit()


def write_data_json(dump_dict: dict) -> None:
    """ Writes the dictionary to a json file """
    with open(file="data/genders.json", mode="w", encoding="utf-8") as write_file:
        json.dump(dump_dict, write_file, ensure_ascii=False, indent=5)
    print(f"File: data/genders.json was created!\n")


def data_txt_to_dict() -> None:
    global DICT_AWH
    array_quest, array_ans = read_data_from_txt()
    dict_awh = {}
    for index in range(len(array_quest)):
        age, height, weight = array_quest[index]
        gender = array_ans[index]
        if gender == 1:
            gender = "Female"
        else:
            gender = "Male"
        if age in dict_awh:
            if height in dict_awh[age]:
                if weight in dict_awh[age][height]:
                    dict_awh[age][height][weight] = "F/M"
                else:
                    dict_awh[age][height][weight] = gender
            else:
                new_dict = {weight: gender}
                dict_awh[age][height] = new_dict
        else:
            new_dict = {height: {weight: gender}}
            dict_awh[age] = new_dict

    DICT_AWH = dict_awh
    write_data_json(dict_awh)


def read_json_file() -> None:
    def update_keys(dict_in: dict, type_key: type) -> dict:
        cash_dict = {}
        for key, val in dict_in.items():
            key = type_key(key)
            if isinstance(val, dict):
                cash_dict[key] = update_keys(val, type_key)
            else:
                cash_dict[key] = val
        return cash_dict

    global DICT_AWH
    with open("data/genders.json", "r", encoding="utf-8") as read_file:
        read_dict = json.load(read_file)
    DICT_AWH = update_keys(read_dict, float)


def check_gender(search_age: float, search_height: float, search_weight: float) -> str:
    read_json_file()
    min_dif = 999999
    find_age, find_height, find_weight = 0, 0, 0
    for age in DICT_AWH:
        if abs(age - search_age) < min_dif:
            min_dif = abs(age - search_age)
            find_age = age
        elif abs(age - search_age) > min_dif:
            min_dif = 999999
            for height in DICT_AWH[find_age]:
                if abs(height - search_height) < min_dif:
                    min_dif = abs(height - search_height)
                    find_height = height
                elif abs(height - search_height) > min_dif:
                    min_dif = 999999
                    for weight in DICT_AWH[find_age][find_height]:
                        if abs(weight - search_weight) < min_dif:
                            min_dif = abs(weight - search_weight)
                            find_weight = weight

                    gender = DICT_AWH[find_age][find_height][find_weight]
                    print(f"Find gender: {find_age=}; {find_height=}; {find_weight=}; Continuation Gender -> {gender}")
                    return gender


def check_input() -> None:
    while True:
        try:
            print(f"Enter age(int) height(sm) weight(kg) separating by a space:")
            user_input = list(map(float, input("---> ").strip().split()))
            if not user_input:
                exit()
            if len(user_input) == 3:
                age, height, weight = user_input[0], user_input[1], user_input[2]
                check_gender(age, height, weight)
            else:
                print(f"It was necessary to enter three integer values separated by a space!\n")
        except Exception as ex:
            print(f"-_- Error -> {type(ex).__name__}: {ex}\n")


def main() -> None:
    data_txt_to_dict()
    check_input()


if __name__ == '__main__':
    main()
