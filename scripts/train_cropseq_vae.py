#!/usr/bin/env python

import sys
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


class ForceIOStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if not self.stream.isatty():
            os.fsync(self.stream.fileno())

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = ForceIOStream(sys.stdout)
sys.stderr = ForceIOStream(sys.stderr)

if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Training a VAE(C)')
	parser.add_argument('--model', type=str, metavar='M', help='vaec for classifier, vae for vanilla')
	parser.add_argument('--label', type=str, metavar='L', help='what to use as label, one of: gene, louvain, guide')
	parser.add_argument('--n_genes', type=int, metavar='N', help='how many genes to keep, based on variance.')
	parser.add_argument('--n_latent', type=int, metavar='A', help='dimensionality of the latent space')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--metadata', type=str,metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--output', type=str, metavar='O', help='output model name')
	args = parser.parse_args()

	h5_filename = args.data
	metadata_filename = args.metadata

	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		batch='wells',
		use_labels=args.label,
		new_n_genes=args.n_genes,
		save_path='')

	print('loaded dataset!')

	n_epochs=20
	lr=1e-4
	use_batches=True
	use_cuda=True

	print('Using learning rate', lr)

	if args.model == 'vae':
		vae = VAE(
			gene_dataset.nb_genes,
			n_batch=gene_dataset.n_batches * use_batches,
			n_latent=args.n_latent)
	else:
		vae = CVAE(
			gene_dataset.nb_genes, 
			n_labels=gene_dataset.n_labels, 
			n_batch=gene_dataset.n_batches * use_batches,
			n_latent=args.n_latent)

	infer = SupervisedVariationalInference(
		vae, 
		gene_dataset, 
		train_size=0.9, 
		use_cuda=use_cuda,
		verbose=True,
		frequency=1)
	infer.train(n_epochs=n_epochs, lr=lr)

	# Save the model itself
	torch.save(vae, args.output + '.model')

	# Print history
	ll_train = infer.history["ll_train"]
	ll_test = infer.history["ll_test"]

	print(list(ll_train))
	print('---')
	print(list(ll_test))

	print('finished training!')