import torch
import numpy as np
import pandas as pd
import itertools
from scipy.stats import norm
from scvi_extensions.dataset.label_data_loader import LabelDataLoaders
from scvi_extensions.hypothesis_testing.utils import sample_scale, expression_stats, compute_p_value, breakup_batch, get_bayes_factors


def gene_variance_test(model, dataset, desired_labels, M_sampling, M_permutation):
    """
    Performs differential expression given a dataset between all desired labels.
    """
    data_loader = LabelDataLoaders(
        gene_dataset=dataset, 
        desired_labels=desired_labels,
        num_samples=10000)

    print('Sampling for variance testing...')
    px_scale, all_labels = expression_stats(
        model, 
        data_loader['all'], 
        M_sampling=M_sampling)
    print('Done sampling for variance testing...')

    variances = (px_scale.reshape(-1, M_sampling, px_scale.shape[-1])*1e6).var(axis=1)
    labels = all_labels.reshape(-1, M_sampling).mean(axis=1)

    # Generate the null distribution
    wl_rates, wl_bfs = get_bayes_factors(1, 1, variances, labels, null=True)
    null_loc, null_scale = norm.fit(wl_rates)

    # Combination of labels and perform hypothesis testing
    results = {}
    for label_1, label_2 in itertools.combinations(desired_labels, 2):
        if label_1 not in results:
            results[label_1] = {}
        il_rates, il_bfs = get_bayes_factors(label_1, label_2, variances, labels)
        pvalues = np.array([compute_p_value(null_loc, null_scale, rate) for rate in il_rates])
        direction = np.array([np.sign(rate - 0.5) for rate in il_rates])
        results[label_1][label_2] = pd.DataFrame(
            data=list(zip(
                dataset.gene_names,
                np.arange(dataset.gene_names.shape[0]),
                il_rates, 
                il_bfs,
                np.absolute(il_bfs),
                pvalues, 
                direction)),
            columns=['gene', 'gene_index', 'P(H1)', 'bayes_factor', 'bayes_factor_mag' ,'pval', 'direction'])\
        .sort_values('bayes_factor_mag', ascending=False)
    return results