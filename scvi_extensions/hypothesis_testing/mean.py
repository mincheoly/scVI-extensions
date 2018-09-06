"""
    mean.py

    This file contains code for the hypothesis testing of means, aka differential expression. 
    Most of the code is from scVI, adapted to the supervised case.
"""

import torch
import numpy as np
import pandas as pd
import itertools
from scipy.stats import norm
from scvi_extensions.dataset.label_data_loader import LabelDataLoaders
from scvi_extensions.hypothesis_testing.utils import sample_scale, expression_stats, compute_p_value, breakup_batch, get_bayes_factors


def batch_differential_expression(vae, data_loader, M_sampling=100, desired_labels=[0, 1], num_batch=200):
    """
    Visualizes the DE process. Each batch is considered a data point, giving a distribution of Bayes factor
    estimates for a total of num_batch points. 
    """
    alt_bayes_factors = []
    null_bayes_factors = []
    count = 0

    for tensors in data_loader:

        # Draw some sample data points and sample the px scales
        sample_batch, _, _, batch_index, labels = tensors
        sample_batch_list, batch_index_list, labels_list = breakup_batch(sample_batch, batch_index, labels, desired_labels)
        px_list = [sample_scale(vae, a, b, c, M_sampling).data.numpy() for a, b, c in zip(sample_batch_list, batch_index_list, labels_list)]

        # Generate Bayes factors for alternate hypothesis
        batch_alt_rates = np.mean((px_list[0] >= px_list[1]), 0)
        alt_bayes_factors.append(np.log(batch_alt_rates + 1e-8) - np.log(1 - batch_alt_rates + 1e-8))

        # Generate Bayes factors for null hypothesis
        batch_null_bfs = []
        for px in px_list:
            shuffled_px = px.reshape(M_sampling, -1, px.shape[-1])[np.random.permutation(M_sampling), :, :].reshape(-1, px.shape[-1])
            batch_null_bfs.append(np.mean((px >= shuffled_px), 0))
        batch_null_rates = np.concatenate(batch_null_bfs)
        null_bayes_factors.append(np.log(batch_null_rates + 1e-8) - np.log(1 - batch_null_rates + 1e-8))

        count += 1
        if count >= num_batch:
            break

    return np.vstack(alt_bayes_factors), np.vstack(null_bayes_factors)


def differential_expression(model, dataset, desired_labels, M_sampling, M_permutation):
    """
    Performs differential expression given a dataset between all desired labels.
    """
    data_loader = LabelDataLoaders(
        gene_dataset=dataset, 
        desired_labels=desired_labels,
        num_samples=10000)

    print('Sampling for differential expression...')
    px_scale, all_labels = expression_stats(
        model, 
        data_loader['all'], 
        M_sampling=M_sampling)
    print('Done sampling for differential expression...')

    # Generate the null distribution
    wl_rates, wl_bfs = get_bayes_factors(1, 1, px_scale, all_labels, null=True)
    null_loc, null_scale = norm.fit(wl_rates)

    # Combination of labels
    results = {}
    for label_1, label_2 in itertools.combinations(desired_labels, 2):
        if label_1 not in results:
            results[label_1] = {}
        il_rates, il_bfs = get_bayes_factors(label_1, label_2, px_scale, all_labels)
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

