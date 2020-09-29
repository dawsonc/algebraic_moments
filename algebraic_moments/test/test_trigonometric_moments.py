"""
Test the function for computing moments of trigonometric functions of Gaussian random variables.
"""
import sympy as sp
import numpy as np

from algebraic_moments.trigonometric_moment import trigonometric_moment


def test_trigonometric_moments_cos_central():
    """Compute the moment of a single cosine of a zero-centered variable"""
    n = [1]
    m = [0]
    mu = [0]
    mu = sp.Matrix(mu)
    sigma = sp.Matrix([1])
    assert trigonometric_moment(n, m, mu, sigma).evalf() == sp.exp(-1/2)


def test_trigonometric_moments_cos_central_symbolic():
    """Compute the moment of a single cosine of a zero-centered variable with symbolic mean
    and variance"""
    n = [1]
    m = [0]
    mu = [sp.symbols('mu')]
    mu = sp.Matrix(mu)
    sigma = sp.Matrix([sp.symbols('sigma2')])
    assert trigonometric_moment(n, m, mu, sigma) == \
        sp.exp(-sp.I * mu[0] - sigma[0]/2)/2 + sp.exp(sp.I * mu[0] - sigma[0]/2)/2


def test_trigonometric_moments_sin_central():
    """Compute the moment of a single sine of a zero-centered variable"""
    n = [0]
    m = [1]
    mu = [0]
    mu = sp.Matrix(mu)
    sigma = sp.Matrix([1])
    assert trigonometric_moment(n, m, mu, sigma).evalf() == 0.0


def test_trigonometric_moments_cos_non_central():
    """Compute the moment of a single sine of a variable centered around pi/2"""
    n = [1]
    m = [0]
    mu = [sp.pi / 2]
    mu = sp.Matrix(mu)
    sigma = sp.Matrix([1])
    assert trigonometric_moment(n, m, mu, sigma).evalf() == 0.0


def test_trigonometric_moments_sin_noncentral():
    """Compute the moment of a single sine of a variable centered around pi/2"""
    n = [0]
    m = [1]
    mu = [sp.pi / 2]
    mu = sp.Matrix(mu)
    sigma = sp.Matrix([1])
    assert trigonometric_moment(n, m, mu, sigma).evalf() == sp.exp(-1/2)


def test_trigonometric_moments_empirically():
    """Compute the moment of a complicated function and validate empirically"""
    np.random.seed(0)
    n = [1, 2, 1]
    m = [2, 1, 1]
    mu = [0, 1, 2]
    mu_sym = sp.Matrix(mu)
    sigma = [[1.0, 0.2, 0.1],
             [0.2, 1.0, 0.3],
             [0.1, 0.3, 1.0]]
    sigma_sym = sp.Matrix(sigma)
    symbolic_moment = trigonometric_moment(n, m, mu_sym, sigma_sym).evalf()

    # Estimate the same moment empirically
    N = 1000
    samples = np.random.multivariate_normal(mu, sigma, N)
    x0 = samples[:, 0]
    x1 = samples[:, 1]
    x2 = samples[:, 2]
    empirical_moment = np.cos(x0)**1 * np.sin(x0)**5 \
        * np.cos(x1)**2 * np.sin(x1)**4 * np.cos(x2)**3 * np.sin(x2)**3
    empirical_moment = np.mean(empirical_moment)

    # These should be close
    print(symbolic_moment)
    print(empirical_moment)
    np.abs(symbolic_moment - empirical_moment) <= 0.0001
