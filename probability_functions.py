"""
Probability and Pairing Calculations

This module provides functions for common probability problems and
for generating pairings between two sets of people. It includes
methods to calculate the probability of two people in a group
sharing the same birthday, calculate combinations with defectives
in a sample, and generate pairings between two groups.

Functions:
    - birthday_probability(n: int) -> float:
        Computes the probability that at least two people in a
        group of size `n` share the same birthday.

    - sample_with_fixed_defects(n: int, k: int, r: int, x: int) -> int:
        Determines the number of ways to form a sample of size `r`
        from a population of `n` items, where exactly `k` items are
        defective, and the sample contains exactly `x` defective
        items.

    - women_men_pairings(number_of_pairs: int) -> Generator[zip, None, None]:
        Generates all possible pairings between a set of
        `number_of_pairs` women and men.

Additional Utilities:
    - print_aligned_latex_equations(*args: str) -> None:
        Formats and displays a series of LaTeX equations aligned
        for easier readability.

Example Usage:
    >>> birthday_probability(23)
    0.5072972343239857

    >>> sample_with_fixed_defects(10, 3, 5, 2)
    105

    >>> [list(pairing) for pairing in women_men_pairings(2)]
    [[('w1', 'm1'), ('w2', 'm2')], [('w2', 'm1'), ('w1', 'm2')]]

Run as a standalone script to execute doctests and verify all
functionality.
"""

import itertools
from math import comb
from typing import Generator, List, Union

import IPython.display as ipd
from sympy import factorial, init_printing, symbols
from sympy import oo, exp, integrate, Piecewise

init_printing()


def print_aligned_latex_equations(*args: str) -> None:
    """
    Print a series of LaTeX equations aligned at the &= sign.

    This function takes a variable number of LaTeX equation strings as input,
    and displays them in a single output cell, aligned at the &= sign.

    Args:
        *args (str): LaTeX equation strings to be aligned and displayed.

    Returns:
        None

    Examples:
        print_aligned_latex_equations(
            r"x &= 2 + 3",
            r"y &= 4 - 5",
            r"z &= 6 * 7"
        )

    """

    result = r"\\".join(
        [
            r"\begin{equation}",
            r"\begin{split}",
            *args,
            r"\nonumber" r"\end{split}",
            r"\end{equation}",
        ]
    )

    ipd.display(ipd.Math(rf"{result}"))  # type: ignore


def birthday_probability(n: int) -> float:
    """
    Calculates the probability that at least two people in a group
    of n have the same birthday.

    Args:
      n (int): The number of people in the group.

    Returns:
      float: The probability that at least two people have the same birthday.

    Raises:
      ValueError: If n is less than or equal to 0.

    Examples:
        >>> birthday_probability(1)
        0.0
        >>> birthday_probability(2)
        0.002739726027397249
        >>> birthday_probability(23)
        0.5072972343239857
        >>> birthday_probability(50)
        0.9703735795779884
        >>> birthday_probability(365)
        1.0
        >>> birthday_probability(0)
        Traceback (most recent call last):
        ...
        ValueError: The number of people must be positive.
    """

    if n <= 0:
        raise ValueError("The number of people must be positive.")

    # Calculate the probability that all n people have different birthdays
    probability_of_different_birthdays = 1.0
    for i in range(1, n):
        probability_of_different_birthdays *= (365 - i) / 365

    # The probability of at least two people having the same birthday
    probability_of_same_birthday = 1 - probability_of_different_birthdays

    return probability_of_same_birthday


def sample_with_fixed_defects(n: int, k: int, r: int, x: int) -> int:
    """
    Calculates the number of ways to choose a sample of size r from a population
    of n items, with exactly k defective items, where the sample contains
    exactly x defective items.

    Args:
        n (int): The total population size.
        k (int): The number of defective items in the population.
        r (int): The sample size.
        x (int): The number of defective items to be chosen in the sample.

    Returns:
        int: The number of ways to choose such a sample.

    Raises:
        ValueError: If inputs are invalid (e.g., sample size larger than population, x > k, etc.).

    Examples:


        >>> sample_with_fixed_defects(n=1, k=0, r=1, x=0)
        1

        >>> sample_with_fixed_defects(n=1, k=1, r=1, x=1)
        1

        >>> sample_with_fixed_defects(n=1, k=1, r=1, x=0)
        0

        >>> sample_with_fixed_defects(n=2, k=2, r=2, x=0)
        0

        >>> sample_with_fixed_defects(n=2, k=1, r=2, x=0)
        0

        >>> sample_with_fixed_defects(n=2, k=2, r=2, x=0)
        0

        >>> sample_with_fixed_defects(n=2, k=2, r=2, x=2)
        1

        >>> sample_with_fixed_defects(n=10, k=4, r=5, x=3)
        60

        >>> sample_with_fixed_defects(n=10, k=3, r=5, x=2)
        105

        # >>> sample_with_fixed_defects(6, 2, 4, 1)
        # 8

        >>> sample_with_fixed_defects(20, 5, 10, 3)
        64350

        >>> sample_with_fixed_defects(5, 2, 3, 1)
        6


    Example of invalid input:
        >>> sample_with_fixed_defects(10, 3, 5, 4)
        Traceback (most recent call last):
        ...
        ValueError: Invalid input: Sample size or defective count out of range.

        >>> sample_with_fixed_defects(10, 2, 3, 3)
        Traceback (most recent call last):
        ...
        ValueError: Invalid input: Sample size or defective count out of range.
    """
    # sample_with_fixed_defects(n=10, k=3, r=5, x=2)
    # Validate inputs
    if r > n or x > k or r < x:
        raise ValueError("Invalid input: Sample size or defective count out of range.")

    # The number of ways to choose x defective items from k
    choose_defective = comb(k, x)

    # The number of ways to choose r - x non-defective items from n - k
    choose_non_defective = comb(n - k, r - x)

    # Total number of ways to form the sample
    return choose_defective * choose_non_defective


def women_men_pairings(number_of_pairs: int) -> Generator[zip, None, None]:
    """
    Generates pairings of all permutations of women with a set of men,
    where both sets have `number_of_pairs` individuals.

    Args:
        number_of_pairs (int): The number of women and men.

    Returns:
        Generator[zip, None, None]: A generator yielding zips of each
        permutation of women with men.

    Examples:
        >>> [list(pairing) for pairing in women_men_pairings(2)]
        [[('w1', 'm1'), ('w2', 'm2')], [('w2', 'm1'), ('w1', 'm2')]]
    """

    # Define generators for women and a factory function for men
    def women(n: int) -> Generator[str, None, None]:
        """
        Creates a generator for women labels in the format "w1", "w2", ..., "wn".

        Args:
            n (int): The number of women.

        Returns:
            Generator[str, None, None]: A generator of women labels.

        Examples:
            >>> list(women(2))
            ['w1', 'w2']
        """
        return (f"w{_}" for _ in range(1, n + 1))

    def men(n: int) -> Generator[str, None, None]:
        """
        Creates a generator for men labels in the format "m1", "m2", ..., "mn".

        Args:
            n (int): The number of men.

        Returns:
            Generator[str, None, None]: A generator of men labels.

        Examples:
            >>> list(men(2))
            ['m1', 'm2']
        """
        return (f"m{_}" for _ in range(1, n + 1))

    return (
        zip(perm_women, men(number_of_pairs))
        for perm_women in itertools.permutations(women(number_of_pairs))
    )


def multinomial_coefficient(
    n: Union[int, symbols], k: List[Union[int, symbols]]
) -> Union[int, symbols]:
    """
    Calculate the multinomial coefficient for dividing n items
    into groups of sizes specified in the list k. This version
    supports both numerical values and symbolic expressions.

    The formula for the multinomial coefficient is:
        n! / (k1! * k2! * ... * kr!)
    where n = sum(k), and k is a list of integers or symbols
    representing the sizes of the groups.

    Args:
        n (Union[int, symbols]): The total number of items or a symbolic variable.
        k (List[Union[int, symbols]]): A list specifying the sizes of the groups.

    Returns:
        Union[int, symbols]: The multinomial coefficient as an integer
        or a symbolic expression.

    Raises:
        ValueError: If the sum of k does not equal n when given as integers.

    Examples:
        >>> from sympy import symbols
        >>> n = 5
        >>> k = [2, 3]
        >>> multinomial_coefficient(n, k)
        10

        >>> n = 10
        >>> k = [4, 3, 3]
        >>> multinomial_coefficient(n, k)
        4200

        >>> x = symbols('x')
        >>> y = symbols('y')
        >>> z = symbols('z')
        >>> multinomial_coefficient(x + y + z, [x, y, z])
        factorial(x + y + z)/(factorial(x)*factorial(y)*factorial(z))

    Example of invalid input:
        >>> multinomial_coefficient(5, [1, 2, 1])  # sum(k) != n
        Traceback (most recent call last):
        ...
        ValueError: The sum of the group sizes must equal n when specified as integers.
    """
    if all(isinstance(ki, int) for ki in k) and isinstance(n, int) and sum(k) != n:
        raise ValueError(
            "The sum of the group sizes must equal n when specified as integers."
        )

    # Calculate the multinomial coefficient using factorials
    coefficient = factorial(n)
    for ki in k:
        coefficient /= factorial(ki)

    return coefficient


def exponential_distribution_pdf(time_interval, lamda):
    """
    Calculates the probability density function (PDF) of the exponential distribution.

    Parameters
    ----------
    time_interval : sympy.Symbol
        The variable of interest.
    lambda_ : sympy.Symbol
        The rate parameter of the exponential distribution.

    Returns
    -------
    sympy.Expr
        The probability density function of the exponential distribution.

    Examples
    --------
    >>> t = symbols('t', positive=True)
    >>> exponential_distribution_pdf(t, 2)
    2*exp(-2*t)
    
    >>> exponential_distribution_pdf(t, 1)
    exp(-t)

    """
    
    P = Piecewise(
    (lamda * exp(-lamda * time_interval), time_interval >= 0),
    (0, time_interval < 0))

    return P

def exponential_distribution_cdf(time_interval, lamda):
    """
    Calculates the cumulative distribution function (CDF) of the exponential distribution.

    Parameters
    ----------
    time_interval : sympy.Symbol
        The upper limit of integration.
    lamda : sympy.Symbol
        The rate parameter of the exponential distribution.

    Returns
    -------
    sympy.Expr
        The cumulative distribution function of the exponential distribution.

    Examples
    --------
    >>> t = symbols('t', positive=True)
    >>> lamda = symbols('lamda', positive=True)
    >>> exponential_distribution_cdf(t, lamda)
    1 - exp(-lamda*t)
    
    >>> exponential_distribution_cdf(oo, lamda)
    1
    
    >>> exponential_distribution_cdf(-oo, lamda)
    0
    
    """
    x = symbols('x')
    return integrate(
        exponential_distribution_pdf(x, lamda), 
        (x, -oo, time_interval)
    )

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
    
    
