"""
    mean.py

    This file contains code for the hypothesis testing of means, aka differential expression. 
    Most of the code is from scVI, adapted to the supervised case.
"""

import torch
import numpy as np

def expression_stats(vae, data_loader, M_sampling=100, desired_labels=['NO_GUIDE'], desired_cell_count=500):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    :param vae: The generative vae and encoder network
    :param data_loader: a data loader for a particular dataset
    :param M_sampling: number of samples
    :return: A 1-d vector of statistics of size n_genes
    """
    px_scales = []
    all_labels = []
    cell_count = 0
    for tensors in data_loader:
        sample_batch, _, _, batch_index, labels = tensors
        
        # Only retain data about the labels of interest
        indices = torch.zeros([labels.shape[0]], dtype=torch.uint8)
        for dlabel in desired_labels:
            indices |= (labels.view(-1) == dlabel)
        
        if indices.sum() > 0:
            sample_batch = sample_batch[indices, :]
            batch_index = batch_index[indices, 0]
            labels = labels[indices, 0]
            cell_count += indices.sum()

            sample_batch = sample_batch.repeat(1, M_sampling).view(-1, sample_batch.size(1))
            batch_index = batch_index.repeat(1, M_sampling).view(-1, 1)
            labels = labels.repeat(1, M_sampling).view(-1, 1)
            px_scales += [(vae.get_sample_scale(sample_batch, batch_index=batch_index, y=labels).squeeze()).cpu()]
            all_labels += [labels.cpu()]
            
        if cell_count >= desired_cell_count:
            print(cell_count)
            break

    px_scale = torch.cat(px_scales)
    all_labels = torch.cat(all_labels)
    
    return px_scale, all_labels


def test_means(label_1, label_2, px_scale, labels, M_permutation=10000):

    np_labels = labels.view(-1).data.numpy()

    first_label_idx = np.where(np_labels == label_1)[0]
    second_label_idx = np.where(np_labels == label_2)[0]

    first_set = px_scale[np.random.choice(first_label_idx, size=M_permutation), :].data.numpy()
    second_set = px_scale[np.random.choice(second_label_idx, size=M_permutation), :].data.numpy()
    res1 = np.mean(first_set >= second_set, 0)
    res1 = np.log(res1 + 1e-8) - np.log(1 - res1 + 1e-8)

    return res1