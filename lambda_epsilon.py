import numpy as np
import argparse
import matplotlib.pyplot as plt


def g(x, lambda_, epsilon):
    return lambda_ * np.log(epsilon + abs(x))


def main():
    parser = argparse.ArgumentParser(description='Plot function g(x) with fixed lambda or epsilon')
    parser.add_argument('--fix', type=str, choices=['lambda', 'epsilon'], required=True,
                        help='Choose to fix lambda or epsilon')
    parser.add_argument('--value', type=float, default = 1, help='Value of the fixed parameter, 1 if no value provided')

    args = parser.parse_args()
    fixed_parameter = args.fix
    fixed_value = args.value
    x = np.linspace(-10, 10, 1000)

    if fixed_parameter == 'epsilon':
        for lambda_ in np.linspace(0, 1, 5):
            plt.plot(x, g(x, lambda_, fixed_value), label=f'lambda = {lambda_:.2f}')
    else:
        for epsilon in np.linspace(0, 1, 5):
            plt.plot(x, g(x, fixed_value, epsilon), label=f'epsilon = {epsilon:.2f}')

    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title(f'Function g(x) with fixed {fixed_parameter} = {fixed_value}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

