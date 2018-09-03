from scvi.dataset import DataLoaders
from scvi.dataset.data_loaders import DataLoaderWrapper

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler, WeightedRandomSampler
from sklearn.model_selection._split import _validate_shuffle_split
import numpy as np

class LabelDataLoaders(DataLoaders):
    to_monitor = []
    loop = ['all']

    def __init__(self, gene_dataset, desired_labels, num_samples=None, **kwargs):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        """
        super(LabelDataLoaders, self).__init__(gene_dataset, **kwargs)

        n = len(self.gene_dataset)

        # Get indices for desired labels
        label_indices = {}
        label_weights = {}
        total_cell_count = 0
        weights = np.zeros(len(gene_dataset))

        for label in desired_labels:
            label_indices = np.where(gene_dataset.labels.ravel() == label)[0]
            label_weight = 1.0/len(label_indices) * 1.0/len(desired_labels)
            total_cell_count += len(label_indices)
            weights[label_indices] = label_weight

        data_loader = self(weights=weights, num_samples=total_cell_count if not num_samples else num_samples)

        self.dict.update({
            'all': data_loader
        })

    def __call__(self, shuffle=False, indices=None, weights=None, num_samples=None):
        if indices is not None and shuffle:
            raise ValueError('indices is mutually exclusive with shuffle')
        if indices is None:
            if shuffle:
                sampler = RandomSampler(self.gene_dataset)
            elif weights is not None:
                if num_samples is None:
                    raise ValueError('num_samples should be set when using WeightedRandomSampler')
                sampler = WeightedRandomSampler(weights, num_samples=num_samples)
            else:
                sampler = SequentialSampler(self.gene_dataset)
        else:
            sampler = SubsetRandomSampler(indices)
        return DataLoaderWrapper(self.gene_dataset, use_cuda=self.use_cuda, sampler=sampler,
                                 **self.kwargs)