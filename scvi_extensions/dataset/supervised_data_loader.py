from scvi.dataset import DataLoaders

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler, WeightedRandomSampler


class SupervisedTrainTestDataLoaders(DataLoaders):
    to_monitor = ['train', 'test']
    loop = ['train']

    def __init__(self, gene_dataset, train_size=0.1, test_size=None, seed=0, **kwargs):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        """
        super(TrainTestDataLoaders, self).__init__(gene_dataset, **kwargs)

        n = len(self.gene_dataset)
        n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        np.random.seed(seed=seed)
        permutation = np.random.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test:(n_test + n_train)]

        data_loader_train = self(indices=indices_train)
        data_loader_test = self(indices=indices_test)

        self.dict.update({
            'train': data_loader_train,
            'test': data_loader_test
        })

    def __call__(self, shuffle=False, indices=None, weights=None):
        if indices is not None and shuffle:
            raise ValueError('indices is mutually exclusive with shuffle')
        if indices is None:
            if shuffle:
                sampler = RandomSampler(self.gene_dataset)
            else:
                sampler = SequentialSampler(self.gene_dataset)
        elif weighted and shuffle:
            sampler = WeightedRandomSampler(weights, )
        else:
            sampler = SubsetRandomSampler(indices)
        return DataLoaderWrapper(self.gene_dataset, use_cuda=self.use_cuda, sampler=sampler,
                                 **self.kwargs)