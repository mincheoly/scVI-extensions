#!/usr/bin/env python

import os

import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.metrics.differential_expression import de_stats
from scvi.metrics.imputation import imputation
from scvi.models import VAE, SVAEC, VAEC
from scvi.inference import VariationalInference

from scvi_extensions.dataset.cropseq import CropseqDataset
from scvi_extensions.inference.supervised_variational_inference import SupervisedVariationalInference

import torch

if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Training a VAE(C)')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--metadata', type=str,metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--output', type=str, metavar='O', help='output model name')
	args = parser.parse_args()

	h5_filename = args.data
	metadata_filename = args.metadata

	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		use_donors=True,
		use_labels='louvain',
		new_n_genes=1000,
		save_path='')

	print('loaded dataset!')

	n_epochs=200
	lr=5e-4
	use_batches=True
	use_cuda=True

	print('Using learning rate', lr)

	vae = VAEC(gene_dataset.nb_genes, n_labels=gene_dataset.n_labels, n_batch=gene_dataset.n_batches * use_batches)
	infer = SupervisedVariationalInference(
		vae, 
		gene_dataset, 
		train_size=0.9, 
		use_cuda=use_cuda,
		verbose=True,
		frequency=1)
	infer.train(n_epochs=n_epochs, lr=lr)

	# Save the model states
	torch.save(vae.state_dict(), args.output + '.model_states')

	# Save the model itself
	torch.save(vae, args.output + '.model')

	# Print history
	ll_train = infer.history["ll_train"]
	ll_test = infer.history["ll_test"]

	print(list(ll_train))
	print('---')
	print(list(ll_test))

	print('finished training!')