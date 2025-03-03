# Python program for linear regression with one variable
import random
import time


def get_current_milliseconds() -> int:
    return round(time.time() * 1000)


# Get random value of b
def get_b(min: int = 0, max:int = 1000) -> float:
    return random.random()*max - min


# Get random value of w
def get_w(min: int = 0, max:int = 1000) -> float:
    return random.random()*max - min


# get random input output
def get_single_variable_input_output(m: int, w_actual: float) -> tuple[float,float]:
    x_input: list = []
    y_output: list = []
    for i in range(m):
        temp_x: float = random.random()
        x_input.append(temp_x)
        temp_y: float = temp_x * w_actual
        y_output.append(temp_y)
    return x_input, y_output


def get_input_output(m: int, w_actual: float, b_actual: float) -> tuple[float, float]:
    x_input: list = []
    y_output: list = []
    for i in range(m):
        temp_x: float = random.random()
        x_input.append(temp_x)
        temp_y: float = temp_x * w_actual + b_actual
        y_output.append(temp_y)
    return x_input, y_output


# Get training set
def get_single_variable_training_set(m: int, w_actual: float) -> tuple[float, float]:
    return get_single_variable_input_output(m, w_actual)


def print_algo_info(m: int, alpha: float, epoch: int):
    print("Running LR with one variable with " + str(m) 
          + " training examples " + str(alpha) + " learning rate, and " + str(epoch) + " epochs.")


# Run linear regression
def run_linear_regression_with_single_variable(m: int, alpha: float, epoch: int):
    print_algo_info()

    w_actual: float = get_w()
    #print("Actual value of w: " + str(w_actual))

    x_training, y_training = get_single_variable_training_set(m, w_actual)
    '''
    print("Training data input: ")
    print(x_training)
    print()

    print("Training data output")
    print(y_training)
    print()
    '''

    w_predicted: float = get_w()
    print("Predicted w before training: " + str(w_predicted))
    print()

    for z in range(epoch):
        print("Training for epoch " + str(z))
        cost: float = 0
        for i in range(m):
            cost = cost + (w_predicted * x_training[i] - y_training[i]) * x_training[i]
        cost = cost/m
        w_predicted = w_predicted - alpha * cost
        print("Predicted w: " + str(w_predicted))
        print()
    
    print("Predicted w after training: " + str(w_predicted))
    print("Actual value of w to be expected: " + str(w_actual))
    print()


def run_linear_regression(m: int, alpha: float, epoch: int):
    print_algo_info(m, alpha, epoch)

    w_actual: float = get_w()
    b_actual: float = get_b()

    x_training, y_training = get_input_output(m, w_actual, b_actual)

    w_predicted: float = get_w()
    b_predicted: float = get_b()
    print("Predicted w before training: " + str(w_predicted))
    print("Predicted b before training: " + str(b_predicted))

    time_start: int = get_current_milliseconds()

    for z in range(epoch):
        print("Training for epoch " + str(z))
        cost_w: float = 0
        cost_b: float = 0
        for i in range(m):
            cost_temp = (w_predicted * x_training[i] + b_predicted - y_training[i])
            cost_w = cost_w + cost_temp * x_training[i]
            cost_b = cost_b + cost_temp
        cost_w = cost_w/m
        cost_b = cost_b/m
        temp_w = w_predicted - alpha * cost_w
        temp_b = b_predicted - alpha * cost_b

        w_predicted = temp_w
        b_predicted = temp_b

        print("Predicted w: " + str(w_predicted))
        print("Predicted b: " + str(b_predicted))
        print()
    
    time_end: int = get_current_milliseconds()
    
    print("Predicted w after training: " + str(w_predicted))
    print("Predicted b after training: " + str(b_predicted))
    print("Actual value of w to be expected: " + str(w_actual))
    print("Actual value of b to be expected: " + str(b_actual))
    print("Time to train: " + str(time_end - time_start) + " milliseconds.")
    print()


def main() -> None:
    random_seed: int = 15
    m: int = 150
    alpha: float = 0.8
    epoch: int = 300
    random.seed(random_seed)
    #run_linear_regression_with_single_variable(m, alpha, epoch)
    run_linear_regression(m, alpha, epoch)


if __name__ == "__main__":
    main()