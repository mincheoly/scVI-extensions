from scvi.dataset import DataLoaders

class SupervisedDataLoaders(DataLoaders):
    to_monitor = ['labelled', 'unlabelled']

    def __init__(self, gene_dataset, n_labelled_samples_per_class=50, seed=0, use_cuda=True, **kwargs):
        """
        :param n_labelled_samples_per_class: number of labelled samples per class
        """
        super(SemiSupervisedDataLoaders, self).__init__(gene_dataset, use_cuda=use_cuda, **kwargs)

        n_labelled_samples_per_class_array = [n_labelled_samples_per_class] * gene_dataset.n_labels
        labels = np.array(gene_dataset.labels).ravel()
        np.random.seed(seed=seed)
        permutation_idx = np.random.permutation(len(labels))
        labels = labels[permutation_idx]
        indices = []
        current_nbrs = np.zeros(len(n_labelled_samples_per_class_array))
        for idx, (label) in enumerate(labels):
            label = int(label)
            if current_nbrs[label] < n_labelled_samples_per_class_array[label]:
                indices.insert(0, idx)
                current_nbrs[label] += 1
            else:
                indices.append(idx)
        indices = np.array(indices)
        total_labelled = sum(n_labelled_samples_per_class_array)
        indices_labelled = permutation_idx[indices[:total_labelled]]
        indices_unlabelled = permutation_idx[indices[total_labelled:]]

        data_loader_all = self(shuffle=True)
        data_loader_labelled = self(indices=indices_labelled)
        data_loader_unlabelled = self(indices=indices_unlabelled)

        self.dict.update({
            'all': data_loader_all,
            'labelled': data_loader_labelled,
            'unlabelled': data_loader_unlabelled,
        })