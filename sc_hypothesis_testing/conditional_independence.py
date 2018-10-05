import pandas as pd
from .differential_correlation import differential_correlation

def conditional_independence(x, y, z, num_bins=10, p_cut=0.05, which='spearman'):
    """ 
        Tests for conditional independence such that P(x, y|z) == P(x|z)P(y|z).

        Perform differential correlation analysis for different bins of values of z, 
        and test must pass for all bins. 
    """

    bins = pd.cut(z, bins=num_bins, labels=False)

    for bin_ in range(num_bins):

        x_s = x[bins == bin_]
        y_s = y[bins == bin_]

        statistic, pval = differential_correlation(x, y, x_s, y_s, test='>', which=which)

        print(pval)
        if pval > p_cut:
            return False
    return True
