# TRADABILITY METRICS 
## the tradability metrics function defined in statistics.py act as a quality-filter to determine ifv a statistically linked pair is actually profitable to trade in a real world scenario

# While cointegration informs about the existence of a relationship, these metrics determine if a relationship moves fast enough and consistently enough to be traded

**Hurst Exponent H** 
# H measures the memeory of the spread to confirm it is truly mean-reverting

# it calculates the variance of the spread scales over different time lags

# H < 0.5 (mean reverting): it is the target, it means a move away from the mean is likely to be followed by a move back toward the mean

# H > 0.5 (trending): with a high H the spread is a trend

**half line**


# by combining these, move to a theoritical relationship to a practical strategy
# look for pairs that has low p-value(statistically significant cointegration), low Hurst (strong memory to return to the mean), short half line (high turnover of capital) and high zero crossing (frequent trading signals)



