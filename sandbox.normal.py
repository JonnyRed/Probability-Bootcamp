import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_standard_normal_distribution(ax):
    # Create a range of x values from -3 to 3
    """
    Plots the standard normal distribution (Gaussian distribution or bell curve)
    from -3 to 3.

    The plot shows the probability density function (PDF) of the standard
    normal distribution, which is a continuous probability distribution with
    a mean of 0 and a standard deviation of 1. The x-axis represents the
    values of the random variable, and the y-axis represents the
    corresponding probability densities.

    The plot also annotates the mean (μ) and standard deviation (σ) of the
    distribution with dashed red and green lines, respectively.

    The plot is displayed using matplotlib's pyplot interface.
    
    param ax: The axis object on which to plot the distribution.
    
    return: None
    """
    x = np.linspace(-3, 3, 100)

    # Calculate the corresponding y values using the standard normal
    # distribution formula
    y = norm.pdf(x)

    # Create the plot
    ax.plot(x, y)

    # Add title and labels
    ax.set_title("Standard Normal Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")

    # Fill the area from -1 to 1
    ax.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.3)

    # Annotate the mean (μ) and standard deviation (σ)
    mu = 0
    sigma = 1
    ax.axvline(x=mu, color="red", linestyle="--", label=f"mean (μ) = {mu}")
    ax.axvline(
        x=mu + sigma, color="green", linestyle="--", label=f"std dev (σ) = {sigma}"
    )
    ax.axvline(x=mu - sigma, color="green", linestyle="--")

    # Add legend
    ax.legend()

def plot_standard_cdf_normal_distribution(ax):
    """
    Plots the cumulative distribution function (CDF) of the standard
    normal distribution from -3 to 3.

    The plot shows the cumulative probability that a standard normal
    random variable is less than or equal to a given value on the x-axis.
    The y-axis represents the cumulative probability values.

    The plot is displayed using matplotlib's pyplot interface.
    
    param ax: The axis object on which to plot the CDF.
    
    return: None
    
    """

    x = np.linspace(-3, 3, 100)

    # Calculate the corresponding y values using the standard normal
    # distribution formula
    y = norm.cdf(x)

    # Create the plot
    ax.plot(x, y)  

    # Add title and labels
    ax.set_title("Standard Normal CDF")    
    ax.set_xlabel("x")
    ax.set_ylabel("Probability")   

def plot_standard_normal_cdf():
    # Create a range of x values from -3 to 3
    """
    Plots the standard normal distribution and its cumulative distribution
    function (CDF) from -3 to 3.

    The plot on the left shows the probability density function (PDF) of the
    standard normal distribution, which is a continuous probability
    distribution with a mean of 0 and a standard deviation of 1. The x-axis
    represents the values of the random variable, and the y-axis represents the
    corresponding probability densities.

    The plot on the right shows the cumulative distribution function (CDF) of
    the standard normal distribution, which gives the cumulative probability
    that a standard normal random variable is less than or equal to a given
    value on the x-axis. The y-axis represents the cumulative probability
    values.

    The plot is displayed using matplotlib's pyplot interface.

    return: None
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_standard_normal_distribution(ax1)
    plot_standard_cdf_normal_distribution(ax2)
    
    # Display the plot
    plt.show()

# Call the function to plot the standard normal CDF
plot_standard_normal_cdf()


# Flip a coin 400 times (p = 1/2:** This indicates that 
# fair coin is flipped 400 times, with the probability of getting 
#heads (p) being  0.5. Using the normal distribvution we can calculate  
# the probability of getting between 190 and 230 heads in the 400 flips.
# The equation P(X = 190) + P(X = 191) + ... + P(X = 229) + P(X = 230)
# shows the exact way to calculate the probability. It involves summing 
# the probabilities of getting exactly 190 heads, 191 heads, and so on, 
# up to 230 heads. This calculation can be quite tedious.
# Instead, we can use the normal distribution to calculate the probability
# of getting between 190 and 230 heads in the 400 flips.

# Provide a pytynon function to illustrate the use of the normal distribution

def calculate_probability(n, p, lower, upper):
    # Define the parameters
    """
    Calculate the probability of getting a certain number of successes
    in `n` independent trials, each with probability of success `p`.

    Parameters
    ----------
    n : int
        The number of independent trials.
    p : float
        The probability of success in each trial.
    lower : int
        The lower bound of the range of interest.
    upper : int
        The upper bound of the range of interest.

    Returns
    -------
    float
        The probability of getting between `lower_bound` and `upper_bound`
        successes in the `n` trials.

    Notes
    -----
    This function uses the normal distribution to approximate the
    probability. The approximation is close for large `n` and `p` not
    close to 0 or 1.
    """
    mean = n * p  # mean of the normal distribution
    std_dev = np.sqrt(n * p * (1 - p))  # standard deviation of the normal distribution

    # Define the range of interest
    # Standardize the bounds
    lower_bound_std = (lower - mean) / std_dev
    upper_bound_std = (upper - mean) / std_dev

    # Calculate the probability using the normal distribution
    return norm.cdf(upper_bound_std) - norm.cdf(lower_bound_std)


trials = 400
success = 0.5
lower_bound = 190
upper_bound = 230
probability = calculate_probability(trials, success, lower_bound, upper_bound)

print(
    f"The probability of getting between {lower_bound} and {upper_bound} "
    f"heads is approximately {probability:.4f}"
)
