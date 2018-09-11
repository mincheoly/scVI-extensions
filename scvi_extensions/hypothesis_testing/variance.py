import torch
import numpy as np
import pandas as pd
import itertools
from scipy.stats import norm
from scvi_extensions.dataset.label_data_loader import LabelDataLoaders
from scvi_extensions.hypothesis_testing.utils import sample_scale, expression_stats, compute_p_value, breakup_batch, get_bayes_factors


def batch_variance_test(vae, dataset, desired_labels, M_sampling=100):
    """
    Visualizes the variance testing process. Each batch is considered a data point, giving a sampling distribution of Bayes factor
    estimate. 
    """
    alt_bayes_factors = []
    null_bayes_factors = []

    data_loader = LabelDataLoaders(
        gene_dataset=dataset, 
        desired_labels=desired_labels,
        num_samples=10000)

    for tensors in data_loader['all']:

        # Draw some sample data points and sample the px scales
        sample_batch, _, _, batch_index, labels, testing_labels = tensors
        sample_batch_list, batch_index_list, labels_list = breakup_batch(sample_batch, batch_index, testing_labels, desired_labels)
        px_list = [sample_scale(vae, a, b, c, M_sampling).data.numpy() for a, b, c in zip(sample_batch_list, batch_index_list, labels_list)]

        # Generate Bayes factors for alternate hypothesis
        px_list[0] = np.vstack([px_list[0][np.random.permutation(px_list[0].shape[0]), :] for i in range(10)])
        px_list[1] = np.vstack([px_list[1][np.random.permutation(px_list[1].shape[0]), :] for i in range(10)])
        variances_0 = (px_list[0].reshape(-1, M_sampling, px_list[0].shape[-1])*1e6).var(axis=1)
        variances_1 = (px_list[1].reshape(-1, M_sampling, px_list[1].shape[-1])*1e6).var(axis=1)
        batch_alt_rates = np.mean((variances_0 >= variances_1), 0)
        alt_bayes_factors.append(np.log(batch_alt_rates + 1e-8) - np.log(1 - batch_alt_rates + 1e-8))

        # Generate Bayes factors for null hypothesis
        combined_px = np.vstack(px_list)
        null_variances_0 = (combined_px[np.random.permutation(combined_px.shape[0]), :].reshape(-1, M_sampling, combined_px.shape[-1])*1e6).var(axis=1)
        null_variances_1 = (combined_px[np.random.permutation(combined_px.shape[0]), :].reshape(-1, M_sampling, combined_px.shape[-1])*1e6).var(axis=1)
        batch_null_rates = np.mean((null_variances_0 >= null_variances_1), 0)
        null_bayes_factors.append(np.log(batch_null_rates + 1e-8) - np.log(1 - batch_null_rates + 1e-8))

    return np.vstack(alt_bayes_factors), np.vstack(null_bayes_factors)


def gene_variance_test(model, dataset, desired_labels, M_sampling):
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

    # Create pseudobulk within a label
    for label in np.unique(all_labels):
        px_scale_subset = px_scale[all_labels==label, :]
        shuffled_subset = px_scale_subset[np.random.permutation(px_scale_subset.shape[0]), :]
        px_scale[all_labels == label, :] = shuffled_subset

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
    return wl_rates, results