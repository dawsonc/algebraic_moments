"""
Use symbolic manipulation to determine the moments of the sine and cosine of a random variable

Written by C. Dawson on September 29, 2020
"""
import sympy as sp


def trigonometric_moment(n, m, mu, sigma):
    """
    Compute the moment E[ \\product_{j=0}^{N-1} cos^{n_j} X_j * sin^{m_j} X_j ].

    X = [X_1, X_2, ..., X_N] is assumed to be a multivariate Gaussian random variable with mean mu
    and covariance matrix sigma.

    args:
        n (list of integers): a list where the j-th element denotes n_j in the above equation.
        m (list of integers): a list where the j-th element denotes m_j in the above equation.
        mu (sympy Matrix, Nx1): the mean vector of multivariate Gaussian X
        sigma (sympy Matrix, NxN): the covariance matrix of multivariate Gaussian X
    returns:
        The (n,m)-th moment E[ \\product_{j=0}^N cos^{n_j} X_j * sin^{m_j} X_j ]
    """
    # Define some things for convenience
    i = sp.I  # imaginary unit
    N = len(n)
    # Sanity checks on input dimensions
    assert len(m) == N
    assert mu.shape == (N, 1)
    assert sigma.shape == (N, N)

    # Start with the easy bits: compute the constant factor
    beta = sp.prod([1/(i**m[j] * 2**(n[j] + m[j])) for j in range(N)])

    # We define alpha_j as a convenience variable for exp(i*X_j)
    alpha = sp.symbols(f"alpha0:{N}")
    # q_j represents an individual Laurent polynomial factor in the expectation of this moment
    #   (E[.] = beta * E[ \\product_{j=0}^N q_j ], once we've expanded cos and sin using Euler)
    q = [(alpha[j] + alpha[j]**-1)**n[j] * (alpha[j] - alpha[j]**-1)**m[j] for j in range(N)]

    # Expand the product of all q_j and collect the coefficient and indices of the monomials
    q_product_polynomial = sp.prod(q).as_poly()
    q_product_vars = q_product_polynomial.gens
    q_prod_coefficients = q_product_polynomial.coeffs()
    q_prod_monomial_basis = q_product_polynomial.monoms()

    # Sympy treats 1/alpha_j as a separate monomial, but we want to treat it as alpha_j^-1
    # This should half the length of each multi-index in the monomial basis.
    corrected_monomial_basis = []
    for basis_set in q_prod_monomial_basis:
        # Construct a new basis vector, in which we collect the basis terms for each variable
        new_basis_set = [0] * (len(basis_set) // 2)
        for j, basis_component in enumerate(basis_set):
            var = q_product_vars[j]
            # If the variable is reciprocal, collect the negative of its original basis component
            if 1/var in alpha:
                alpha_idx = alpha.index(1/var)
                new_basis_set[alpha_idx] -= basis_component
            # Otherwise, just collect the basis component
            else:
                alpha_idx = alpha.index(var)
                new_basis_set[alpha_idx] += basis_component
        corrected_monomial_basis.append(new_basis_set)


    # Now we can finally compute the desired moment
    moment_terms = []
    for j, basis in enumerate(corrected_monomial_basis):
        # Each term in the moment is weighted by the corresponding coefficient found above
        c_t = q_prod_coefficients[j]
        # Jargon: each term includes the characteristic function (evaluated at 1) of the scalar
        # Gaussian random variable Y_t = t^T X (i.e. a weighted combination of the vector X_t).
        # This scalar will have mean t^T * mu and variance t^T Sigma t
        t = sp.Matrix(basis)
        mean = t.transpose() * mu
        mean = mean[0]
        variance = t.transpose() * sigma * t
        variance = variance[0, 0]
        moment_terms.append(c_t * sp.exp(i * mean - variance / 2))
    moment = beta * sum(moment_terms)

    return moment
