#!/usr/bin/env python

import os

import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, RetinaDataset, CropseqDataset
from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.metrics.differential_expression import de_stats
from scvi.metrics.imputation import imputation
from scvi.models import VAE, SVAEC
from scvi.inference import VariationalInference

import torch

if __name__ == '__main__':

	h5_filename = '/netapp/home/mincheol/raw_gene_bc_matrices_h5.h5'
	metadata_filename = '/ye/yelabstore2/dosageTF/tfko_140/combined/nsnp20.raw.sng.km_vb1.norm.meta.txt'

	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		use_donors=True,
		use_labels=True,
		save_path='')

	print('loaded dataset!')

	n_epochs=300
	lr=1e-3
	use_batches=True
	use_cuda=True

	vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
	infer = VariationalInference(
		vae, 
		gene_dataset, 
		train_size=0.9, 
		use_cuda=use_cuda,
		verbose=True,
		frequency=5)
	infer.train(n_epochs=n_epochs, lr=lr)

	# Save the model states
	torch.save(vae.state_dict(), '/netapp/home/mincheol/vae_states_test.model_states')

	# Save the model itself
	torch.save(vae, '/netapp/home/mincheol/vae_model_test.model')

	# Print history
	ll_train = infer.history["ll_train"]
	ll_test = infer.history["ll_test"]

	print(list(ll_train))
	print('---')
	print(list(ll_test))

	print('finished training!')