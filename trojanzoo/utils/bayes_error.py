import numpy as np
from scipy.stats import norm

'''
compute Bayes error rate between two Gaussian distributions (assuming equal weights)
'''


def solve(m1, sig1, m2, sig2):  # intersection point(s) between two gaussians
    # copied from https://stackoverflow.com/a/22579904/8821118
    a = 1 / (2 * sig1 ** 2) - 1 / (2 * sig2 ** 2)
    b = m2 / (sig2 ** 2) - m1 / (sig1 ** 2)
    c = m1 ** 2 / (2 * sig1 ** 2) - m2 ** 2 / (2 * sig2 ** 2) - np.log(sig2 / sig1)
    return np.roots([a, b, c])


def bayes_error_one_point(x, m1, sig1, m2, sig2):
    v1 = norm.pdf(x + 1, m1, sig1)
    v2 = norm.pdf(x + 1, m2, sig2)
    cdf1 = lambda x: norm.cdf(x, m1, sig1)
    cdf2 = lambda x: norm.cdf(x, m2, sig2)
    if v1 < v2:  # p2 is greater on the right
        e1 = 1 - cdf1(x)  # integrate p1 from x to inf
        e2 = cdf2(x)  # integrate p2 from -inf to x
    else:
        e1 = 1 - cdf2(x)
        e2 = cdf1(x)
    e = e1 + e2
    return e / 2


def bayes_error_two_point(xs, m1, sig1, m2, sig2):
    x1 = min(xs)
    x2 = max(xs)
    xm = (x1 + x2) / 2
    v1 = norm.pdf(xm, m1, sig1)
    v2 = norm.pdf(xm, m2, sig2)
    cdf1 = lambda x: norm.cdf(x, m1, sig1)
    cdf2 = lambda x: norm.cdf(x, m2, sig2)
    if v1 < v2:  # p2 is greater in the middle
        e1 = cdf1(x2) - cdf1(x1)  # integrate p1 from x1 to x2
        e2 = cdf2(x1)  # integrate p2 from -inf to x1
        e3 = 1 - cdf2(x2)  # integrate p2 from x2 to inf
    else:
        e1 = cdf2(x2) - cdf2(x1)
        e2 = cdf1(x1)
        e3 = 1 - cdf1(x2)
    e = e1 + e2 + e3
    return e / 2


def bayes_error(m1, sig1, m2, sig2):
    x = solve(m1, sig1, m2, sig2)
    if len(x) == 1:
        return bayes_error_one_point(x[0], m1, sig1, m2, sig2)
    else:
        return bayes_error_two_point(x, m1, sig1, m2, sig2)
