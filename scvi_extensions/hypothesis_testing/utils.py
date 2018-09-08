"""
    utils.py

    This file contains code for utility functions for hypothesis testing.
"""

import torch
import numpy as np
from scipy.stats import norm


def get_bayes_factors(label_1, label_2, measurement, labels, M_permutation=10000, null=False):
    """
    Returns the rate of hypothesis one being true, as well as the Bayes Factors corresponding to these rates.
    measurement and labels should be numpy arrays. labels should be a 1d array.
    """

    if null:
        first_label_idx = second_label_idx = np.arange(measurement.shape[0])
    else:
        first_label_idx = np.where(labels == label_1)[0]
        second_label_idx = np.where(labels == label_2)[0]
    first_set = measurement[np.random.choice(first_label_idx, size=M_permutation), :]
    second_set = measurement[np.random.choice(second_label_idx, size=M_permutation), :]
    res1 = np.mean(first_set > second_set, 0)
    return res1, np.log(res1 + 1e-8) - np.log(1 - res1 + 1e-8)


def sample_scale(vae, sample_batch, batch_index, labels, M_sampling):

    return (
        vae.get_sample_scale(
            sample_batch.repeat(1, M_sampling).view(-1, sample_batch.size(1)), 
            batch_index=batch_index.repeat(1, M_sampling).view(-1, 1), 
            y=labels.repeat(1, M_sampling).view(-1, 1)
        ).squeeze()).cpu()


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
    for sample_batch, _, _, batch_index, labels , testing_labels in data_loader:
        px_scales += [sample_scale(vae, sample_batch, batch_index, labels, M_sampling)]
        all_labels += [testing_labels.repeat(1, M_sampling).view(-1, 1).cpu()]
    px_scale = torch.cat(px_scales).data.numpy()
    all_labels = torch.cat(all_labels).data.numpy().reshape(-1)
    
    return px_scale, all_labels


def compute_p_value(mean, std, val):
    """
    Two tailed p-value.
    """
    pval = norm.cdf(x=val, loc=mean, scale=std)
    return pval if pval < 0.5 else 1 - pval


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
        if sample_batch_list[idx].size(0) == 0:
            return [], [], []
        sample_batch_list[idx] = sample_batch_list[idx][:rarest_label_count, :]
        batch_index_list[idx] = batch_index_list[idx][:rarest_label_count, :]
        labels_list[idx] = labels_list[idx][:rarest_label_count, :]

    return sample_batch_list, batch_index_list, labels_list