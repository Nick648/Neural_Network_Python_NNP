import numpy as np


def sigmoid(x: float) -> float:
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x: float) -> float:
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


class Neuron:
    """
    A neuron with:
        - 3 inputs (count_weights_input)
        - 1 bias
        - 4 func (1 init)
    """

    def __init__(self, count_weights_input: int, name_neuron: str = None):
        self.name = name_neuron
        self.weights = []
        for _ in range(count_weights_input):
            self.weights.append(np.random.normal())
        self.bias = np.random.normal()
        # print(f"Neuron {self.name} -> {self.weights=} -> {self.bias=};")

    def display_neuron(self) -> str:
        info_neuron = f"Neuron {self.name}: "
        for index in range(len(self.weights)):
            info_neuron += f"w{index + 1} = {self.weights[index]}; "
        info_neuron += f"b = {self.bias}"
        return info_neuron

    def sum_neuron(self, inputs) -> float:
        total = np.dot(self.weights, inputs) + self.bias

        # report = f"sum_neuron: {self.display_neuron()}; {inputs=}\n\t"
        # report += f"{np.dot(self.weights, inputs)=} -> {total=}\n"
        # print(report)

        return total

    def feedforward_neuron(self, inputs) -> float:
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
        self.h_neuron = [Neuron(neuron_weights_input, f"h{h_i + 1}") for h_i in range(count_h)]
        self.o_neuron = [Neuron(neuron_weights_input, f"o{o_i + 1}") for o_i in range(count_output)]
        # print(f"Create NeuralNetwork -> \n\t{self.h_neuron=}; \n\t{self.o_neuron=};\n")
        self.display_values("Start Network")

    def display_values(self, com: str = "Values"):
        values_str = f"{com} -> \n"
        for index_h in range(len(self.h_neuron)):
            values_str += f"\t{self.h_neuron[index_h].display_neuron()}\n"
        for index_o in range(len(self.o_neuron)):
            values_str += f"\t{self.o_neuron[index_o].display_neuron()}\n"
        print(values_str + "\n")

    def feedforward(self, x):
        # x is a numpy array with 3 elements. -> index+1=number in pic
        h_layer = [h_neuron.feedforward_neuron(x) for h_neuron in self.h_neuron]
        o_layer = [o_neuron.feedforward_neuron(h_layer) for o_neuron in self.o_neuron]

        # print(f"\n{h_layer=};\n{o_layer=}\n")
        # h_sum_layer = [h_neuron.sum_neuron(x) for h_neuron in self.h_neuron]
        # o_sum_layer = [o_neuron.sum_neuron(h_sum_layer) for o_neuron in self.o_neuron]
        # print(f"\n{h_sum_layer=};\n{o_sum_layer=}\n")

        return o_layer

    def train(self, data, all_y_trues):
        for x, y_true in zip(data, all_y_trues):
            h_sum_layer = [h_neuron.sum_neuron(x) for h_neuron in self.h_neuron]
            h_layer = [h_neuron.feedforward_neuron(x) for h_neuron in self.h_neuron]
            o_sum_layer = [o_neuron.sum_neuron(h_layer) for o_neuron in self.o_neuron]
            o_layer = [o_neuron.feedforward_neuron(h_layer) for o_neuron in self.o_neuron]

            d_ypred_o_dh = []
            for n_o, sum_o in zip(self.o_neuron, o_sum_layer):
                print(f"{n_o=}, {sum_o=}")
                for weight in n_o.weights:
                    d_ypred_o_dh.append(weight * derivative_sigmoid(sum_o))
            print(f"\t{d_ypred_o_dh=}")
            # d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
            # d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)


def main():
    # Define dataset
    data = np.array([
        [-2, -1, 1],  # Alice
        [25, 6, 2],  # Bob
        [17, 4, 3],  # Charlie
        [-15, -6, 4],  # Diana
    ])
    all_ans = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    # Train our neural network!
    network = NeuralNetwork(3, 3, 1)
    network.feedforward([1, 2, 3])
    network.train(data, all_ans)
    # Display results
    network.display_values("After train")


def check_sub_array() -> None:
    data = np.array([[2, 3, 4], [1, 5, 2], [3, 5, 1]])
    ans = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]])
    print(mse_loss(ans, data))

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

if __name__ == '__main__':
    check_sub_array()
    # main()
