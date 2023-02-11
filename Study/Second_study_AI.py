import numpy as np


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
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
                    temporary_array_quest.append(input_values)
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


class OurNeuralNetwork:
    """
    A neural network with:
      - 3 inputs
      - a hidden layer with 3 neurons (h1, h2, h3)
      - an output layer with 1 neuron (o1)
    """

    def __init__(self):
        # Weights
        self.weights = [np.random.normal() for _ in range(12)]
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.biases = [np.random.normal() for _ in range(4)]
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.display_values("Start Network")

    def display_values(self, com: str = "Values"):
        values_str = f"{com} -> \n"
        values_str += f"\tWeights:\n"
        for index_w in range(len(self.weights)):
            values_str += f"\tw{index_w + 1} = {self.weights[index_w]}\n"
        values_str += f"\tBiases:\n"
        for index_b in range(len(self.biases)):
            values_str += f"\tb{index_b + 1} = {self.biases[index_b]}\n"
        write_to_file_values(values_str + "\n")

    def feedforward(self, x):
        # x is a numpy array with 3 elements. -> index+1=number in pic
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        # h1 = sigmoid(self.weights[0] * x[0] + self.weights[1] * x[1] + self.weights[2] * x[2] + self.biases[0])
        # h2 = sigmoid(self.weights[3] * x[0] + self.weights[4] * x[1] + self.weights[5] * x[2] + self.biases[1])
        # h3 = sigmoid(self.weights[6] * x[0] + self.weights[7] * x[1] + self.weights[8] * x[2] + self.biases[2])
        # o1 = sigmoid(self.weights[9] * h1 + self.weights[10] * h2 + self.weights[11] * h3 + self.biases[3])
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        """

        learn_rate = 0.1
        epochs = 3000  # number of times to loop through the entire dataset

        for epoch in range(epochs + 1):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)

                # sum_h1_h2_h3 = [
                #     self.weights[0] * x[0] + self.weights[1] * x[1] + self.weights[2] * x[2] + self.biases[0],
                #     self.weights[3] * x[0] + self.weights[4] * x[1] + self.weights[5] * x[2] + self.biases[1],
                #     self.weights[6] * x[0] + self.weights[7] * x[1] + self.weights[8] * x[2] + self.biases[2]
                # ]
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                # sum_h1 = self.weights[0] * x[0] + self.weights[1] * x[1] + self.weights[2] * x[2] + self.biases[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                # sum_h2 = self.weights[3] * x[0] + self.weights[4] * x[1] + self.weights[5] * x[2] + self.biases[1]
                h2 = sigmoid(sum_h2)

                # sum_h3 = self.weights[6] * x[0] + self.weights[7] * x[1] + self.weights[8] * x[2] + self.biases[2]
                # h3 = sigmoid(sum_h3)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                # sum_o1 = self.weights[9] * h1 + self.weights[10] * h2 + self.weights[11] * h3 + self.biases[3]
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = derivative_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 100 == 0 or epoch == 0 or epoch == epochs:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.5f" % (epoch, loss))


def main():
    # Define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_ans = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    # Train our neural network!
    network = OurNeuralNetwork()
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
                user_data = np.array([age, height, weight])
                ans_user = network.feedforward(user_data)
                gender = "Male"
                if ans_user > 0.5:
                    gender = "Female"
                print(f"Continuation: {round(ans_user, 5)}; Gender -> {gender}\n")
            else:
                print(f"It was necessary to enter three integer values separated by a space!\n")
        except Exception as ex:
            print(f"-_- Error -> {type(ex).__name__}: {ex}\n")


if __name__ == '__main__':
    create_tests()
    # read_data_from_files()
    # main()
