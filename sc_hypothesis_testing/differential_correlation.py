"""
	differential_correlation.py

	Code for performing differential correlation with single cell RNA data.
"""	

from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, spearmanr
import numpy as np


def _differential_correlation_statistic(x1, y1, x2, y2, which='spearman'):
    """ Computes a differential correlation statistic for 4 1d samples. """

    corr_1, pval_1 = correlation(x1, y1, which=which)
    corr_2, pval_2 = correlation(x2, y2, which=which)
    return (np.arctanh(corr_1) - np.arctanh(corr_2))/(np.sqrt(np.absolute((1/len(x1)) - (1/len(x2)))))


def _null_distribution(x1, y1, x2, y2, which='spearman', num_null=100):
    """ Generates null p values by shuffling the labels. """

    n_1, n_2 = len(x1), len(x2)
    x_all, y_all = np.concatenate([x1, x2]), np.concatenate([y1, y2])

    null_stats = []
    for i in range(num_null):
        idx1 = np.random.permutation(n_1+n_2)[:n_1]
        idx2 = np.random.permutation(n_1+n_2)[:n_2]
        null_stats.append(
            _differential_correlation_statistic(
                x_all[idx1],
                y_all[idx1],
                x_all[idx2],
                y_all[idx2], which=which))
    return np.array(null_stats)


def correlation(x, y, which='spearman'):
	""" Measures a metric of correlation between two 1d samples. """
    
	if which == 'spearman':
		return spearmanr(x, y)
	elif which == 'pearson':
		return pearsonr(x, y)
	else:
		raise 'Not yet implemented'


def differential_correlation(x1, y1, x2, y2, which='spearman', num_null=200, test='!='):
	""" 
		Performs differential correlation between 4 1d samples, 2 in each condition. 
		Returns a p-value based on random shuffling of data.
	"""

	statistic = _differential_correlation_statistic(x1, y1, x2, y2, which=which)

	null_stats = _null_distribution(x1, y1, x2, y2, which=which, num_null=num_null)

	if test == '!=':
		return statistic, (np.absolute(null_stats) > np.absolute(statistic)).sum()/num_null
	elif test == '>=':
		return statistic, (null_stats >= statistic).sum()/num_null
	else: #test == '<='
		return statistic, (null_stats >= statistic).sum()/num_null