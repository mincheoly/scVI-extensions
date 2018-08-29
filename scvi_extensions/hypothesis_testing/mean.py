"""
    mean.py

    This file contains code for the hypothesis testing of means, aka differential expression. 
    Most of the code is from scVI, adapted to the supervised case.
"""

import torch
import numpy as np


def breakup_batch(sample_batch, batch_index, labels, desired_labels):

    sample_batches, batch_index, labels, desired_labels = [], [], [], []
    rarest_label_count = 1e8

    for label in desirend_labels:
        sample_batch_list = sample_batch[labels.view(-1) == label, :]
        batch_index_list = batch_index[labels.view(-1) == label, :]
        labels_list = labels[labels.view(-1) == label, :]

        if sample_batch_list.size(0) < rarest_label_count:
            rarest_label_count = sample_batch_list.size(0)

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
        ).squeeze()).cpu().data.numpy()


def differential_expression(vae, data_loader, M_sampling=100, desired_labels=['NO_GUIDE'], desired_cell_count=500, num_pairs=10000):
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
    pair_count = 0

    for tensors in data_loader:

        # Draw some sample data points and sample the px scales
        sample_batch, _, _, batch_index, labels = tensors
        sample_batch_list, batch_index_list, labels_list = breakup_batch(sample_batch, batch_index, labels, desired_labels)
        px_list = map(lambda a, b, c: sample_scale(a, b, c, M_sampling), zip(sample_batch_list, batch_index_list, labels_list))

        # Generate Bayes factors for alternate hypothesis
        batch_alt_rates = np.mean((px_list[0] >= px_list[0]).reshape(-1, M_sampling, px_samples.shape[-1]), 0)
        alt_bayes_factors.append(np.log(batch_alt_rates + 1e-8) - np.log(1 - batch_alt_rates + 1e-8))

        # Geberate Bayes factors for null hypothesis
        batch_null_bfs = []
        for px in px_list:
            shuffled_px = px.reshape(M_sampling, -1, px.shape[-1])[np.random.permutation(M), :, :].reshape(-1, px.shape[-1])
            batch_null_bfs.append(np.mean((px >= shuffled_px).reshape(-1, M_sampling, px_samples.shape[-1]), 0))
        batch_null_rates = np.concatenate(batch_null_bfs)
        null_bayes_factors.append(np.log(batch_null_rates + 1e-8) - np.log(1 - batch_null_rates + 1e-8))

        pair_count += sample_batch_list[0].size(0)
        if pair_count >= num_pairs:
            break

    return np.concatenate(alt_bayes_factors), np.concatenate(null_bayes_factors)


# def test_means(label_1, label_2, px_scale, labels, M_permutation=10000):

#     np_labels = labels.view(-1).data.numpy()

#     first_label_idx = np.where(np_labels == label_1)[0]
#     second_label_idx = np.where(np_labels == label_2)[0]

#     first_set = px_scale[np.random.choice(first_label_idx, size=M_permutation), :].data.numpy()
#     second_set = px_scale[np.random.choice(second_label_idx, size=M_permutation), :].data.numpy()
#     res1 = np.mean(first_set >= second_set, 0)
#     res1 = np.log(res1 + 1e-8) - np.log(1 - res1 + 1e-8)

#     return res1