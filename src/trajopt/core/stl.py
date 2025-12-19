'''
Logical Operators parameterized with Generalized Mean-based Smooth Robustness (GMSR)

Author: Samet Uzun
Reference: [https://doi.org/10.48550/arXiv.2405.10996]
'''

import sympy as sp

# =============================
# *———* Penalty Functions *———*
# =============================

def smooth_inequality(x, c=1e-4):
    """
    Smooth penalty function for inequality constraints.
    
    Args:
        x (float): Input value.
        c (float): Smoothing parameter.
    
    Returns:
        float: Smoothed maximum function.
    """
    return sp.sqrt(sp.Max(x, 0)**2 + c**2) - c

def smooth_equality(x, c=1e-4):
    """
    Smooth penalty function for equality constraints.
    
    Args:
        x (float): Input value.
        c (float): Smoothing parameter.
    
    Returns:
        float: Smoothed absolute function.
    """
    return sp.sqrt(x**2 + c**2) - c

# ==============================
# *———* Core STL Functions *———*
# ==============================

def negation(x):
    """
    Negation operator.
    
    Args:
        x (float): Input value.
    
    Returns:
        float: Negated input.
    """
    return -x

def AND(x, c=1e-4):
    """
    Smooth parameterization for the conjunction (AND) operator.
    
    Args:
        x (list of float): List of values.
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for AND composition.
    """
    K = len(x)
    sum_terms = sum(sp.Max(0, xi)**2 for xi in x)
    product_terms = sp.prod(sp.Max(0, -xi)**2 for xi in x)
    
    Mp = (sum_terms / K) + c
    M0 = (product_terms + c**K)**(1/K)
    
    return sp.sqrt(Mp) - sp.sqrt(M0)

def OR(x, c=1e-4):
    """
    Smooth parameterization for the disjunction (OR) operator.
    
    Args:
        x (list of float): List of values.
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for OR composition.
    """
    return -AND([-xi for xi in x], c=c)

# =================================
# *———* Derived STL Functions *———*
# =================================

def IfElse(x, c=1e-4):
    """
    Smooth parameterization of implication (x[0] -> x[1]).
    
    Args:
        x (list of float): List containing two values [trigger, condition].
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for implication.
    """
    return OR([negation(x[0]), x[1]], c=c)

def integer_variable(x, values, c=1e-4):
    """
    Smooth parameterization for integer variables.
    
    Args:
        x (list of float): List of values.
        c (float): Smoothing parameter.
    
    Returns:
        float: Value of integer constraint (zero if x matches an integer).
    """
    return OR([smooth_equality(x - v_i, c=c) for v_i in values])

# =========================
# *———* Lite Versions *———*
# =========================

def AND_lite(x, c=1e-4):
    """
    Lite version of the conjunction (AND) operator.
    Considers only the positive part of the function.
    
    Args:
        x (list of float): List of values.
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for AND.
    """
    K = len(x)
    sum_terms = sum(sp.Max(0, xi)**2 for xi in x)
    Mp = (sum_terms / K) + c
    return sp.sqrt(Mp) - sp.sqrt(c)

def OR_lite(x, c=1e-4):
    """
    Lite version of the disjunction (OR) operator.
    
    Args:
        x (list of float): List of values.
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for OR.
    """
    K = len(x)
    product_terms = sp.prod(sp.Max(0, xi)**2 for xi in x)
    M0 = (product_terms + c**K)**(1/K)
    return sp.sqrt(M0) - sp.sqrt(c)

def IfElse_lite(x, c=1e-4):
    """
    Lite version of the implication operator.
    
    Args:
        x (list of float): List containing two values [trigger, condition].
        c (float): Smoothing parameter.
    
    Returns:
        float: Robustness value for implication.
    """
    return OR_lite([negation(x[0]), x[1]], c=c)
