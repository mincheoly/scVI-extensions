"""
	visualize_cropseq.py
	Visualize the latent space of the CROP-seq experiment.
"""


import os

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

from scvi.dataset import CropseqDataset

from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.models import VAE, SVAEC
from scvi.inference import VariationalInference

import torch

if __name__ == '__main__':

	# Load the model
	model_path = '/netapp/home/mincheol/vae_model_test.model'
	vae = torch.load(model_path, map_location=lambda storage, loc: storage)

	# Load the dataset
	h5_filename = '/netapp/home/mincheol/raw_gene_bc_matrices_h5.h5'
	metadata_filename = '/ye/yelabstore2/dosageTF/tfko_140/combined/nsnp20.raw.sng.km_vb1.norm.meta.txt'
	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		use_donors=True,
		use_labels=True,
		save_path='')

	# Create an inference class
	use_batches=True
	use_cuda=True
	infer = VariationalInference(
		vae, 
		gene_dataset, 
		train_size=0.9, 
		use_cuda=use_cuda,
		verbose=True,
		frequency=5)

	# Get the latent space
	n_samples = 5000
	latent, batch_indices, labels = get_latent(vae, infer.data_loaders['sequential'])
	print(type(labels))
	print(type(latent))
	print(len(labels))
	latent, idx_t_sne = infer.apply_t_sne(latent, n_samples)
	batch_indices = batch_indices[idx_t_sne].ravel()
	labels = labels[idx_t_sne].ravel()

	# Plot
	plt.figure(figsize=(10, 10))
	for i, guide in enumerate(np.unique(gene_dataset.labels)):
		plt.scatter(latent[labels == guide, 0], latent[labels == guide, 1], label=guide)
	plt.title('tSNE plot of the scVI latent space')
	plt.xlabel('tSNE1')
	plt.ylabel('tSNE2')

	# Save the plot
	plt.savefig('/netapp/home/mincheol/plots/scvi_tsne.png')





