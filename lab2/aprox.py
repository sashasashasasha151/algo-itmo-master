import numpy as np
import matplotlib.pyplot as plt
import math
import os
import lab2.methods as dm
from scipy import optimize

EPS = 0.001


def to_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def approx_func_linear(p, a, b):
    return p * a + b


approx_func_linear_grad_a = lambda p, a, b: p
approx_func_linear_grad_b = lambda p, a, b: 1


def approx_func_rational(p, a, b):
    return a / (1 + b * p)


approx_func_rational_grad_a = lambda p, a, b: 1 / (1 + b * p)
approx_func_rational_grad_b = lambda p, a, b: - a * p / ((1 + b * p) ** 2)


def generate_data():
    a, b = np.random.random(2)
    x = [i / 100 for i in range(100)]
    tow = np.random.normal(size=100)
    y = [a * xx + tt + b for (xx, tt) in zip(x, tow)]
    return list(zip(x, y))


def visualize_least_sq_error(func, func_name, data):
    x = [i / 100 for i in range(100)]
    y = [i / 100 for i in range(100)]
    z = []
    for i in x:
        z_ = []
        for j in y:
            z_.append(sum([(func(a, i, j) - b) ** 2 for (a, b) in data]))
        z.append(z_)
    plt.contourf(x, y, z)
    plt.savefig("{}\\images_2\\{}-{}.png".format(os.path.dirname(os.path.abspath(__file__)), "depth", func_name))
    plt.clf()


def exhaustive_search(data, x_0, x_1, y_0, y_1, approximation):
    iter_x = 110
    iter_y = 110
    delta_x = (x_1 - x_0) / iter_x
    delta_y = (y_1 - y_0) / iter_y

    best_pair = (2 ** 100, x_0, y_0)

    for a in range(iter_x):
        for b in range(iter_y):
            least_square_error = 0
            a_coord = x_0 + a * delta_x
            b_coord = y_0 + b * delta_y
            for x_pos, y_pos in data:
                least_square_error += (approximation(x_pos, a_coord, b_coord) - y_pos) ** 2

            if least_square_error < best_pair[0]:
                best_pair = (least_square_error, a_coord, b_coord)
    return best_pair, iter_x


def visualize_data(data, a, b, name_1, name_2):
    x_line = [i / 100 for i in range(100)]
    y_line = [a * i + b for i in x_line]
    plt.scatter([v for (v, _) in data], [v for (_, v) in data], color="red", label="initial data")
    plt.plot(x_line, y_line, color="green", label="approximation line: {} {}".format(name_1, name_2))
    plt.legend()
    plt.savefig("{}\\images_3\\{}-{}.png".format(os.path.dirname(os.path.abspath(__file__)), name_1, name_2))
    plt.clf()


def visualize_exhaustive(data, aprox):
    (least_sq_err, a, b), iter = exhaustive_search(data, 0, 1, 0, 1, aprox)
    visualize_data(data, a, b, "exhaustive", to_name(aprox))
    return (a, b), iter


def build_differencial(data, approx_func, approx_func_grad_a, approx_func_grad_b):
    f_a_b = lambda a, b: sum([(approx_func(x_p, a, b) - y_p) ** 2 for (x_p, y_p) in data])
    f_grad_a_a_b = lambda a, b: sum(
        [(approx_func(x_p, a, b) - y_p) * 2 * approx_func_grad_a(x_p, a, b) for (x_p, y_p) in data])
    f_grad_b_a_b = lambda a, b: sum(
        [(approx_func(x_p, a, b) - y_p) * 2 * approx_func_grad_b(x_p, a, b) for (x_p, y_p) in data])

    return f_a_b, f_grad_a_a_b, f_grad_b_a_b


def minimize_lambda(a, b, f_a_b, f_grad_a, f_grad_b):
    _, _, _, argmim, _ = dm.dichotomy(lambda l: f_a_b(a - l * f_grad_a(a, b), b - l * f_grad_b(a, b)), 0, 2)
    return argmim


def gauss_method(iter_count, a, b, f, f_grad_a, f_grad_b):
    current_iter = 0
    points = [(a, b)]
    ls = [-1]
    while True:
        cur_a, cur_b = points[-1]
        if len(points) % 2 == 0:
            l = optimize.golden(lambda l1: f(l1, cur_b), brack=(-1, 1))
            next_a = l
            next_b = cur_b
        else:
            l = optimize.golden(lambda l1: f(cur_a, l1), brack=(-1, 1))
            next_a = cur_a
            next_b = l
        ls.append(l)
        points.append((next_a, next_b))
        current_iter += 1
        if abs(f(cur_a, cur_b) - f(next_a, next_b)) < EPS or current_iter > iter_count:
            break
    return points, ls, current_iter


def visualize_newton_or_gauss_or_whatever(data, method, apox, aprox_grad_a, aprox_grad_b, L):
    f_a_b, f_grad_a_a_b, f_grad_b_a_b = build_differencial(data, apox, aprox_grad_a, aprox_grad_b)
    points, ls, iters = method(100, 1, 1, f_a_b, f_grad_a_a_b, f_grad_b_a_b)

    visualize_data(data, points[-1][0], points[-1][1], to_name(method), to_name(apox))
    L.append((points[-1], "gauss_linear", apox))
    return points[-1], iters


def nelder_mead(func_a_b, alpha=1, beta=0.5, gamma=2, max_iter=2_000):
    res = optimize.minimize(lambda x: func_a_b(x[0], x[1]), np.array((0, 0)), method='Nelder-Mead')
    print(res)

    return [], res.x, 0

def visualize_nelder_mead(data, apox, L:list):
    f_a_b, _, _ = build_differencial(data, apox, lambda a, b, c: 0, lambda a, b, c: 0)
    _, best, iters = nelder_mead(f_a_b)

    visualize_data(data, best[0], best[1], "nelder_mead", to_name(apox))
    L.append((best, "nelder_mead", apox))
    return best, iters


if __name__ == "__main__":
    data = generate_data()

    lse = lambda a, b, m: sum([(m(x, a, b) - y) ** 2 for (x, y) in data])
    gauss_linear, gauss_linear_iter = visualize_newton_or_gauss_or_whatever(data, gauss_method, approx_func_linear,
                                                                            approx_func_linear_grad_a,
                                                                            approx_func_linear_grad_b)
    gauss_ratio, gauss_ratio_iters = visualize_newton_or_gauss_or_whatever(data, gauss_method, approx_func_rational,
                                                                           approx_func_rational_grad_a,
                                                                           approx_func_rational_grad_b)
    exhaustive_linear, exhanelder_linear_iters = visualize_exhaustive(data, approx_func_linear)
    exhaustive_rational, exhaustive_rational_iters = visualize_exhaustive(data, approx_func_rational)

    nelder_linear, nelder_linear_iters = visualize_nelder_mead(data, approx_func_linear)
    nelder_rational, nelder_rational_iters = visualize_nelder_mead(data, approx_func_rational)

    print("exhaustive_linear", exhaustive_linear, "iter", exhanelder_linear_iters,
          lse(exhaustive_linear[0], exhaustive_linear[1], approx_func_linear))
    print("gauss_linear", gauss_linear, "iter", gauss_linear_iter,
          lse(gauss_linear[0], gauss_linear[1], approx_func_linear))
    print("nelder_mead_linear", nelder_linear, "iter", nelder_linear_iters,
          lse(nelder_linear[0], nelder_linear[1], approx_func_linear))

    print()
    print("exhaustive_rational", exhaustive_rational, "iter", exhaustive_rational_iters,
          lse(exhaustive_rational[0], exhaustive_rational[1], approx_func_rational))
    print("gauss_ratio", gauss_ratio, "iter", gauss_ratio_iters,
          lse(gauss_ratio[0], gauss_ratio[1], approx_func_rational))
    print("nelder_mead_rational", nelder_rational, "iter", nelder_rational_iters,
          lse(nelder_rational[0], nelder_rational[1], approx_func_rational))
