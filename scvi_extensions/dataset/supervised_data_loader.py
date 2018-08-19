from scvi.dataset import DataLoaders
from scvi.dataset.data_loaders import DataLoaderWrapper

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler, WeightedRandomSampler
from sklearn.model_selection._split import _validate_shuffle_split
import numpy as np

class SupervisedTrainTestDataLoaders(DataLoaders):
    to_monitor = ['train', 'test']
    loop = ['train']

    def __init__(self, gene_dataset, train_size=0.1, test_size=None, seed=0, **kwargs):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        """
        super(SupervisedTrainTestDataLoaders, self).__init__(gene_dataset, **kwargs)

        n = len(self.gene_dataset)
        n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        np.random.seed(seed=seed)
        permutation = np.random.permutation(n)

        # Get indices
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test:(n_test + n_train)]

        # Get weights for each label
        unique_labels, label_counts = np.unique(gene_dataset.labels[:, 0], return_counts=True)
        self.weight_lookup = 1.0/label_counts * 1.0/len(unique_labels)

        # Create weights
        weights_all = np.zeros(len(gene_dataset))
        weights_train = np.zeros(len(gene_dataset))
        weights_test = np.zeros(len(gene_dataset))
        for idx in indices_train:
            weights_train[idx] = self.weight_lookup[gene_dataset.labels[idx, 0]]
        for idx in indices_test:
            weights_test[idx] = self.weight_lookup[gene_dataset.labels[idx, 0]]
        for idx in range(len(gene_dataset)):
            weights_all[idx] = self.weight_lookup[gene_dataset.labels[idx, 0]]

        data_loader_train = self(weights=weights_train)
        data_loader_test = self(weights=weights_test)
        data_loader_all = self(weights=weights_all)

        self.dict.update({
            'train': data_loader_train,
            'test': data_loader_test,
            'all': data_loader_all
        })

    def __call__(self, shuffle=False, indices=None, weights=None):
        if indices is not None and shuffle:
            raise ValueError('indices is mutually exclusive with shuffle')
        if indices is None:
            if shuffle:
                sampler = RandomSampler(self.gene_dataset)
            elif weights is not None:
                sampler = WeightedRandomSampler(weights, num_samples=len(self.gene_dataset))
            else:
                sampler = SequentialSampler(self.gene_dataset)
        else:
            sampler = SubsetRandomSampler(indices)
        return DataLoaderWrapper(self.gene_dataset, use_cuda=self.use_cuda, sampler=sampler,
                                 **self.kwargs)