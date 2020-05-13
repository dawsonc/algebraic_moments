import sympy as sp
import numpy as np
import networkx as nx
from enum import Enum

from code_printer import CodePrinter
from objects import Moment

def generate_moment_constraints(expressions, random_vector, deterministic_variables, language):
    """[summary]

    Args:
        expressions ([type]): [description]
        random_vector ([type]): [description]
        deterministic_variables ([type]): [description]
        language ([type]): [description]
    """
    moments = [] # List of generated moments.
    results = []
    for exp in expressions:
        results.append(moment_form(exp, random_vector, moments))

    code_printer = CodePrinter()
    code_printer.print_moment_constraints(results, moments, deterministic_variables, language)

def moment_form(expression, random_vector, moments):
    raw_polynomial = sp.poly(expression, random_vector.variables)

    terms = []

    for multi_index, coeff in raw_polynomial.terms():
        # Go through each term of the raw_polynomial

        # Get the variable power map of this term
        term_vpm = random_vector.vpm(multi_index)

        components = random_vector.dependence_graph.subgraph_components(list(term_vpm.keys()))

        # The idea is to express this term as ceoff * np.prod([term_moments])
        term_moments = []
        for comp in components:
            # Construct the variable power mapping for this component.
            comp_vpm = {var : term_vpm[var] for var in comp}
            
            # Find a moment for this component in moments. If one doesn't exist,
            # create a new one.
            equivalent_moments = [m for m in moments if m.same_vpm(comp_vpm)]

            if len(equivalent_moments)==0:
                moment = Moment.from_vpm(comp_vpm)
                moments.append(moment)
            elif len(equivalent_moments)==1:
                moment = equivalent_moments[0]
            else:
                raise Exception("Found two instances of Moment in the set 'moments' that match comp_vpm. This is an issue because the elements in 'moments' should be unique.")
            
            term_moments.append(moment)
        terms.append(coeff * np.prod(term_moments))
    return sum(terms)