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
import argparse
import torch

from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.models import VAE, SVAEC
from scvi.inference import VariationalInference

from scvi_extensions.dataset.cropseq import CropseqDataset
from scvi_extensions.dataset.supervised_data_loader import SupervisedTrainTestDataLoaders

if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Visulaizing the latent space of scVI')
	parser.add_argument('--model', type=str, metavar='M', help='path to the model to be used')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--label', type=str, help='labels to use for coloring')
	parser.add_argument('--metadata', type=str,metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--output', type=str, metavar='O', help='output file name')
	args = parser.parse_args()

	# Load the model
	model_path = args.model
	vae = torch.load(model_path, map_location=lambda storage, loc: storage)

	# Load the dataset
	h5_filename = args.data
	metadata_filename = args.metadata
	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		use_donors=True,
		use_labels=True,
		save_path='')

	# Data Loader
	n_samples = 10000
	data_loader = SupervisedTrainTestDataLoaders(gene_dataset, num_samples=n_samples*10)

	# Infer
	n_epochs=500
	lr=1e-3
	use_batches=False
	use_cuda=False

	infer = VariationalInference(
		vae, 
		gene_dataset, 
		train_size=0.9, 
		use_cuda=use_cuda,
		frequency=5)

	# Get the latent space
	latent, batch_indices, labels = get_latent(vae, data_loader['all'])
	latent, idx_t_sne = infer.apply_t_sne(latent, n_samples)
	batch_indices = batch_indices[idx_t_sne].ravel()
	labels = labels[idx_t_sne].ravel()

	# Plot
	plt.figure(figsize=(10, 10))
	ko_genes = np.array([gene_dataset[label] for label in labels], dtype=str)
	for gene in np.unique(ko_genes):
		plt.scatter(latent[ko_genes == gene, 0], latent[ko_genes == gene, 1], label=gene)
	plt.title('tSNE plot of the scVI latent space')
	plt.xlabel('tSNE1')
	plt.ylabel('tSNE2')

	# Save the plot
	plt.savefig(args.output)





