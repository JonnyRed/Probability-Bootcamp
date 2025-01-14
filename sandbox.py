import numpy as np
import pandas as pd
from scipy.stats import geom
import matplotlib.pyplot as plt
import seaborn as sns

# Define the probability of success (p)
p = 0.3

# Generate random samples from the geometric distribution

# In the Python code you provided, `samples` represents a collection of
# randomly generated numbers that follow a geometric distribution.
#
# * Geometric Distribution: This distribution models the number of trials
# needed to achieve the first success in a series of independent trials,
# where each trial has the same probability of success (`p`).
#
# * `samples`: This variable stores the results of those simulated trials.
# Each value within `samples` represents the number of trials it took
# to achieve the first success in one of those simulated experiments.
#
# **In simpler terms:**
#
# Imagine flipping a biased coin repeatedly. The geometric distribution describes how many flips you might need to get the first "heads." `samples` would be a list of the number of flips it took to get the first "heads" in many simulated coin-flipping experiments.
#
# This collection of samples allows you to:
#
# * **Analyze the behavior of the geometric distribution:** You can use
# these samples to calculate statistics like the mean and variance,
# which can help you understand the typical number of trials required for
# the first success.
# * **Conduct simulations and experiments:** You can use these samples to
# simulate real-world scenarios that follow a geometric distribution,
# such as the number of attempts needed to win a game of chance.
#
# By generating these samples, you can gain insights into the properties
# and behavior of the geometric distribution.
samples = geom.rvs(p, size=1000)

# Calculate and print the probability of the first success occurring
# on the 5th trial
x = 5
prob_x = geom.pmf(x, p)
print(f"{'Probability of first success on the 5th trial:':>50}" f"{prob_x:>15.4f}")

# Calculate and print the cumulative probability of the first success
# occurring within 3 trials
prob_within_3 = geom.cdf(3, p)
print(
    f"{'Probability of first success within 3rd trial:':>50}" f"{prob_within_3:>15.4f}"
)

# Plot the probability mass function (PMF)
x_values = np.arange(1, 20)  # Range of trials
pmf_values = geom.pmf(x_values, p)
sns.barplot(x=x_values, y=pmf_values)
plt.xlabel("Number of Trials")
plt.ylabel("Probability")
plt.title(f"Geometric Distribution PMF (p={p:.2f})")
plt.show()

p_series = pd.Series(samples)
print(p_series.describe())
# plt.show()
