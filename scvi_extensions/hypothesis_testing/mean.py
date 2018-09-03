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


def breakup_batch(sample_batch, batch_index, labels, desired_labels):

    sample_batch_list, batch_index_list, labels_list = [], [], []
    rarest_label_count = 1e10

    for label in desired_labels:
        sample_batch_list.append(sample_batch[labels.view(-1) == label, :])
        batch_index_list.append(batch_index[labels.view(-1) == label, :])
        labels_list.append(labels[labels.view(-1) == label, :])

        if sample_batch_list[-1].size(0) < rarest_label_count:
            rarest_label_count = sample_batch_list[-1].size(0)

    for idx in range(len(desired_labels)):
        sample_batch_list[idx] = sample_batch_list[idx][:rarest_label_count, :]
        batch_index_list[idx] = batch_index_list[idx][:rarest_label_count, :]
        labels_list[idx] = labels_list[idx][:rarest_label_count, :]

    return sample_batch_list, batch_index_list, labels_list


def sample_scale(vae, sample_batch, batch_index, labels, M_sampling):

    return (
        vae.get_sample_scale(
            sample_batch.repeat(1, M_sampling).view(-1, sample_batch.size(1)), 
            batch_index=batch_index.repeat(1, M_sampling).view(-1, 1), 
            y=labels.repeat(1, M_sampling).view(-1, 1)
        ).squeeze()).cpu()


def batch_differential_expression(vae, data_loader, M_sampling=100, desired_labels=[0, 1], num_batch=10000):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    :param vae: The generative vae and encoder network
    :param data_loader: a data loader for a particular dataset
    :param M_sampling: number of samples
    :return: A 1-d vector of statistics of size n_genes
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


def expression_stats(vae, data_loader, M_sampling=100):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    :param vae: The generative vae and encoder network
    :param data_loader: a data loader for a particular dataset
    :param M_sampling: number of samples
    :return: A 1-d vector of statistics of size n_genes

    (From scVI)
    """

    px_scales = []
    all_labels = []
    cell_count = 0
    for sample_batch, _, _, batch_index, labels in data_loader:
        px_scales += [sample_scale(vae, sample_batch, batch_index, labels, M_sampling)]
        all_labels += [labels.repeat(1, M_sampling).view(-1, 1).cpu()]
    px_scale = torch.cat(px_scales)
    all_labels = torch.cat(all_labels)
    
    return px_scale, all_labels


def get_bayes_factors(label_1, label_2, px_scale, labels, M_permutation=10000, null=False):
    """
    Returns the rate of hypothesis one being true, as well as the Bayes Factors corresponding to these rates.
    """
    np_labels = labels.view(-1).data.numpy()
    if null:
        first_label_idx = second_label_idx = np.arange(px_scale.size(0))
    else:
        first_label_idx = np.where(np_labels == label_1)[0]
        second_label_idx = np.where(np_labels == label_2)[0]
    first_set = px_scale[np.random.choice(first_label_idx, size=M_permutation), :].data.numpy()
    second_set = px_scale[np.random.choice(second_label_idx, size=M_permutation), :].data.numpy()
    res1 = np.mean(first_set > second_set, 0)
    return res1, np.log(res1 + 1e-8) - np.log(1 - res1 + 1e-8)


def compute_p_value(mean, std, val):
    """
    Two tailed p-value.
    """
    pval = norm.cdf(x=val, loc=mean, scale=std)
    return pval if pval < 0.5 else 1 - pval


def differential_expression(model, dataset, desired_labels, M_sampling, M_permutation):
    """
    Performs differential expression given a dataset between all desired labels.
    """
    data_loader = LabelDataLoaders(
        gene_dataset=dataset, 
        desired_labels=desired_labels,
        num_samples=5000)

    px_scale, all_labels = expression_stats(
        model, 
        data_loader['all'], 
        M_sampling=M_sampling)

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
                il_rates, 
                il_bfs,
                np.absolute(il_bfs),
                pvalues, 
                direction)),
            columns=['gene', 'P(H1)', 'bayes_factor', 'bayes_factor_mag' ,'pval', 'direction'])\
        .sort_values('bayes_factor_mag', ascending=False)
    return results

