import torch
from torch.nn import functional as F

from scvi.inference.variational_inference import SemiSupervisedVariationalInference, VariationalInference
from scvi_extensions.dataset.supervised_data_loader import SupervisedTrainTestDataLoaders

class SupervisedVariationalInference(SemiSupervisedVariationalInference):
    r"""The SupervisedVariationalInference class for the completely supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = JointSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    default_metrics_to_monitor = VariationalInference.default_metrics_to_monitor

    def __init__(self, model, gene_dataset, classification_ratio=100, train_size=0.9, **kwargs):
        super(SupervisedVariationalInference, self).__init__(model, gene_dataset, train_size=train_size, **kwargs)
        self.data_loaders = SupervisedTrainTestDataLoaders(gene_dataset, use_cuda=self.use_cuda, train_size=train_size)
        self.classification_ratio = classification_ratio

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, y = tensors
        reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index, y)
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        return loss