import torch
from torch.distributions import Normal, Categorical, kl_divergence as kl

from scvi.models.classifier import Classifier
from scvi.models.modules import Encoder, DecoderSCVI
from scvi.models.utils import broadcast_labels
from scvi.models.vae import VAE


class CVAE(VAE):
    r"""A conditional variational autoencoder model,

    Args:
        :n_input: Number of input genes.
        :n_batch: Default: ``0``.
        :n_labels: Default: ``0``.
        :n_hidden: Number of hidden. Default: ``128``.
        :n_latent: Default: ``1``.
        :n_layers: Number of layers. Default: ``1``.
        :dropout_rate: Default: ``0.1``.
        :dispersion: Default: ``"gene"``.
        :log_variational: Default: ``True``.
        :reconstruction_loss: Default: ``"zinb"``.
        :y_prior: Default: None, but will be initialized to uniform probability over the cell types if not specified

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vaec = VAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> gene_dataset = SyntheticDataset(n_labels=3)
        >>> vaec = VAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=3, y_prior=torch.tensor([[0.1,0.5,0.4]]))
    """

    def __init__(self, n_input, n_batch, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1,
                 y_prior=None, dispersion="gene", log_variational=True, reconstruction_loss="zinb"):
        super(CVAE, self).__init__(n_input, n_batch, n_labels, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                   dropout_rate=dropout_rate, dispersion=dispersion, log_variational=log_variational,
                                   reconstruction_loss=reconstruction_loss)

        self.z_encoder = Encoder(n_input, n_latent, n_cat_list=[n_batch, n_labels], n_hidden=n_hidden, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.decoder = DecoderSCVI(n_latent, n_input, n_cat_list=[n_batch, n_labels], n_layers=n_layers,
                                   n_hidden=n_hidden, dropout_rate=dropout_rate)

        self.y_prior = torch.nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels), requires_grad=False
        )


    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        qz_m, qz_v, z = self.z_encoder(torch.log(1 + x), batch_index, y)
        qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
        qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
        z = Normal(qz_m, qz_v).sample()
        px = self.decoder.px_decoder(z, batch_index, y)  # y only used in VAEC - won't work for batch index not None
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):

        # Prepare for sampling
        x_ = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x_)

        # Enumerate choices of label
        ys, xs, library_s, batch_index_s = (
            broadcast_labels(
                y, x, library, batch_index, n_broadcast=self.n_labels
            )
        )

        if self.log_variational:
            xs_ = torch.log(1 + xs)

        # Sampling
        qz_m, qz_v, zs = self.z_encoder(xs_, batch_index_s, ys)

        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, zs, library_s, batch_index_s, ys)

        reconst_loss = self._reconstruction_loss(xs, px_rate, px_r, px_dropout, batch_index_s, ys)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        return reconst_loss, kl_divergence_z + kl_divergence_l