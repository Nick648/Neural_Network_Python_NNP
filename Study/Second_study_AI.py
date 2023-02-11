import numpy as np


def sigmoid(x: float) -> float:
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x: float) -> float:
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


def write_to_file_values(text: str) -> None:
    with open(file="data/Values_AI_2.txt", mode="a", encoding="utf_8") as file:
        file.write(text)


def create_tests() -> None:
    age_step, height_step = 2, 5
    total_input = 0

    def input_gender(total_input_gen: int, gender: str) -> int:
        flag, weight_dict = 0, {}
        print(f"{gender}:")
        file_quest.write(f"# {gender}:\n")
        for age in range(20, 70, age_step):
            if age // 10 != flag:
                flag = age // 10
                for height in range(150, 191, height_step):
                    while True:
                        try:
                            val = float(input(f"{age=}; {height=} -> weight: "))
                            break
                        except Exception as ex:
                            print(f"Check input! {type(ex).__name__}: {ex}")
                    total_input_gen += 1
                    weight_dict[height] = val
            for height in range(150, 191, height_step):
                # weight = int(50+0.75*(height-150)+(age-20)/4)
                file_quest.write(f"{age} {height} {weight_dict[height]}\n")
                if gender == "Men":
                    file_ans.write(f"{0}\n")
                elif gender == "Women":
                    file_ans.write(f"{1}\n")
        return total_input_gen

    # file_quest = open(file="data/Train_AI_2_quest.txt", mode="w", encoding="utf-8")
    file_quest = open(file="data/help_1.txt", mode="w", encoding="utf-8")
    # file_ans = open(file="data/Train_AI_2_ans.txt", mode="w", encoding="utf-8")
    file_ans = open(file="data/help_2.txt", mode="w", encoding="utf-8")
    file_quest.write(f"Age Height Weight\n")
    file_ans.write(f"# Female -> 1; Male -> 0;\n")

    total_input += input_gender(total_input, "Men")
    total_input += input_gender(total_input, "Women")

    file_quest.close()
    file_ans.close()
    print(f"\nFiles have been successfully recorded! Total inputs = {total_input}")


def read_data_from_files() -> tuple:
    temporary_array_quest, temporary_array_ans = [], []
    with open(file="data/Train_AI_2_quest.txt", mode="r", encoding="utf-8") as file_quest:
        for _ in range(2):
            input_values = file_quest.readline()
        while input_values:
            try:
                input_values = list(map(float, file_quest.readline().strip().split()))
                if input_values:
                    age, height, weight = input_values
                    # print(f"{age=};{height=};{weight=}")
                    temporary_array_quest.append([age - 45, height - 170, weight - 75])
                    # temporary_array_quest.append(input_values)
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
                    temporary_array_ans.append(int(input_values))
                else:
                    break
            except ValueError:
                continue
            except Exception as ex:
                print(f"Developer's mistake -> {type(ex).__name__}: {ex}")

    try:
        data = np.array(temporary_array_quest)
        all_ans = np.array(temporary_array_ans)
        if len(data) == len(all_ans):
            return data, all_ans
        else:
            print(f"Developer's mistake -> {len(data)=} != {len(all_ans)}")
            exit()
    except Exception as ex:
        print(f"Developer's mistake -> {type(ex).__name__}: {ex}")
        exit()


class Neuron:
    """
    A neuron with:
        - 3 inputs (count_weights_input)
        - 1 bias
    """

    def __init__(self, count_weights_input: int, name_neuron: str = None):
        self.name = name_neuron
        self.weights = [np.random.normal() for _ in range(count_weights_input)]
        self.bias = np.random.normal()
        # print(f"Create Neuron {self.name} -> {self.weights=} -> {self.bias=};")

    def update_weight(self, index: int, value: float) -> None:
        self.weights[index] = self.weights[index] - value
        # print(f"Update weights[{index}] of Neuron {self.name}")

    def update_bias(self, value: float) -> None:
        self.bias = self.bias - value
        # print(f"Update bias of Neuron {self.name}")

    def display_neuron(self) -> str:
        info_neuron = f"Neuron {self.name}: "
        for index in range(len(self.weights)):
            info_neuron += f"w{index + 1} = {self.weights[index]}; "
        info_neuron += f"b = {self.bias}"
        return info_neuron

    def sum_neuron(self, inputs: list[float]) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        # report = f"sum_neuron: {self.display_neuron()}; {inputs=}\n\t"
        # report += f"{np.dot(self.weights, inputs)=} -> {total=}\n"
        # print(report)
        return total

    def feedforward_neuron(self, inputs: list[float]) -> float:
        total = sigmoid(np.dot(self.weights, inputs) + self.bias)
        # report = f"feedforward_neuron: {self.display_neuron()}; {inputs=}\n\t"
        # report += f"{np.dot(self.weights, inputs)=} -> {total=}\n"
        # print(report)
        return total


class NeuralNetwork:
    """
    A neural network with:
      - 3 inputs
      - a hidden layer with 3 neurons (h1, h2, h3)
      - an output layer with 1 neuron (o1)
    """

    def __init__(self, neuron_weights_input: int, count_h: int, count_output: int = 1):
        # Create neurons
        self.h_neurons = [Neuron(neuron_weights_input, f"h{h_i + 1}") for h_i in range(count_h)]
        self.o_neurons = [Neuron(count_h, f"o{o_i + 1}") for o_i in range(count_output)]
        # print(f"Create NeuralNetwork -> \n\t{self.h_neuron=}; \n\t{self.o_neuron=};\n")
        self.display_values("Start Network")

    def display_values(self, com: str = "Values"):
        values_str = f"{com} -> \n"
        for index_h in range(len(self.h_neurons)):
            values_str += f"\t{self.h_neurons[index_h].display_neuron()}\n"
        for index_o in range(len(self.o_neurons)):
            values_str += f"\t{self.o_neurons[index_o].display_neuron()}\n"
        write_to_file_values(values_str + "\n")

    def feedforward(self, x: list[float]) -> list[float]:
        # x is a numpy array with 3 elements. -> index+1=number in pic
        h_layer = [h_neuron.feedforward_neuron(x) for h_neuron in self.h_neurons]
        o_layer = [o_neuron.feedforward_neuron(h_layer) for o_neuron in self.o_neurons]

        return o_layer  # Return len(o_layer) = len(o_neurons)

    def train(self, data, all_y_trues):
        """
        - data is a (n x 3) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        """

        learn_rate = 0.2
        epochs = 3000  # number of times to loop through the entire dataset

        for epoch in range(epochs + 1):
            for input_arr, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                h_sum_layer = [h_neuron.sum_neuron(input_arr) for h_neuron in self.h_neurons]  # = len(h_neurons)
                h_layer = [h_neuron.feedforward_neuron(input_arr) for h_neuron in self.h_neurons]  # = len(h_neurons)
                o_sum_layer = [o_neuron.sum_neuron(h_layer) for o_neuron in self.o_neurons]  # = len(o_neurons)
                o_layer = [o_neuron.feedforward_neuron(h_layer) for o_neuron in self.o_neurons]  # = len(o_neurons)
                y_pred = o_layer

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                # d_L_d_ypred = -2 * (y_true - y_pred)
                dL_d_ypred_list = list(-2 * (np.array(y_true) - np.array(y_pred)))  # = len(o_neurons)

                # --- Neurons output
                d_ypred_o_dw = []  # = len(o_neurons) -> [ len(h_neurons) ]
                for sum_o in o_sum_layer:
                    new_o = []
                    for h in h_layer:
                        new_o.append(h * derivative_sigmoid(sum_o))  # d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                    d_ypred_o_dw.append(new_o)

                d_ypred_o_db = [derivative_sigmoid(sum_o) for sum_o in o_sum_layer]  # = len(o_neurons)

                d_ypred_o_dh = []  # = len(o_neurons) -> [ len(h_neurons) ]
                for neuron_o, sum_o in zip(self.o_neurons, o_sum_layer):
                    new_o = []
                    for weight in neuron_o.weights:
                        new_o.append(weight * derivative_sigmoid(sum_o))
                    d_ypred_o_dh.append(new_o)  # d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)

                # --- Neurons h_layer
                dh_dw = []  # = len(h_neurons) -> [ len(count_input) ]
                dh_db = []  # = len(h_neurons)
                for sum_h in h_sum_layer:
                    new_h = []
                    for data_x in input_arr:
                        new_h.append(
                            data_x * derivative_sigmoid(sum_h))  # d_h1_d_w1 = input_arr[0] * derivative_sigmoid(sum_h1)
                    dh_dw.append(new_h)
                    dh_db.append(derivative_sigmoid(sum_h))

                # --- Update weights and biases for h Neurons
                for d_L_d_ypred_val, num_o_dy_dh in zip(dL_d_ypred_list, d_ypred_o_dh):
                    for h_neuron, num_h_dh_dw, dh_db_val, o_dy_dh_val in zip(self.h_neurons, dh_dw, dh_db, num_o_dy_dh):
                        for pos, dh_dw_val in enumerate(num_h_dh_dw):
                            h_neuron.update_weight(pos, learn_rate * d_L_d_ypred_val * o_dy_dh_val * dh_dw_val)
                        h_neuron.update_bias(learn_rate * d_L_d_ypred_val * o_dy_dh_val * dh_db_val)

                # --- Update weights and biases for o Neurons
                for o_neuron, d_L_d_ypred_val, num_d_ypred_o_dw, d_ypred_o_db_val in zip(self.o_neurons,
                                                                                         dL_d_ypred_list, d_ypred_o_dw,
                                                                                         d_ypred_o_db):
                    for pos, d_ypred_o_dw_val in enumerate(num_d_ypred_o_dw):
                        o_neuron.update_weight(pos, learn_rate * d_L_d_ypred_val * d_ypred_o_dw_val)
                    o_neuron.update_bias(learn_rate * d_L_d_ypred_val * d_ypred_o_db_val)

                # # Neuron h1
                # self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                # self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                # self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                #
                # # Neuron h2
                # self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                # self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                # self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                #
                # # Neuron o1
                # self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                # self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                # self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 100 == 0 or epoch == 0 or epoch == epochs:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                new_arr = []
                for item in y_preds:
                    if len(item) == 1:
                        for answers in item:
                            new_arr.append(answers)
                    else:
                        max_val = max(item)
                        new_arr.append(max_val)
                y_preds = np.array(new_arr)
                loss = mse_loss(all_y_trues, y_preds)
                # print(f"{y_preds=};\n{all_y_trues};\n{loss=}")
                print("Epoch %d loss: %.5f" % (epoch, loss))


def main():
    data, all_ans = read_data_from_files()
    # Train our neural network!
    network = NeuralNetwork(neuron_weights_input=3, count_h=2, count_output=1)
    network.train(data, all_ans)

    # Display results
    print(f"Artificial intelligence has been successfully trained!\n")
    network.display_values("After train")

    while True:
        try:
            print(f"Enter age(int) height(sm) weight(kg) separating by a space:")
            user_input = list(map(float, input("---> ").strip().split()))
            if not user_input:
                exit()
            if len(user_input) == 3:
                age, height, weight = user_input[0], user_input[1], user_input[2]
                # age, height = user_input[0], user_input[1]
                ans_user = network.feedforward([age, height, weight])
                # ans_user = network.feedforward([age, height])
                print(f"Continuation: {ans_user=}; Gender -> {1}\n")
            else:
                print(f"It was necessary to enter three integer values separated by a space!\n")
        except Exception as ex:
            print(f"-_- Error -> {type(ex).__name__}: {ex}\n")


if __name__ == '__main__':
    # create_tests()
    main()
