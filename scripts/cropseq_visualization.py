"""
	cropseq_visualization.py
	Visualize the latent space of the CROP-seq experiment using scanpy.
"""

import sys
import os

import torch
import numpy as np
import itertools
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import scanpy.api as sc
from anndata import AnnData

from scvi_extensions.dataset.supervised_data_loader import SupervisedTrainTestDataLoaders
from scvi_extensions.dataset.cropseq import CropseqDataset


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


def get_cropseq_latent(vae, data_loader):

    latent = []
    batch_indices = []
    labels = []
    for tensors in data_loader:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        latent += [vae.sample_from_posterior_z(sample_batch, y=label)]
        batch_indices += [batch_index]
        labels += [label]
    return np.array(torch.cat(latent)), np.array(torch.cat(batch_indices)), np.array(torch.cat(labels)).ravel()


if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Clustering and visualizing latent space of scVI.')
	parser.add_argument('--n_neighbors', type=int, default=14, help='n_neighbors for SCANPY neighborhood calcuation')
	parser.add_argument('--model_path', type=str, metavar='M', help='path to a trained torch model')
	parser.add_argument('--model_label', type=str, help='the labels with which the model was trained on, one of: gene, louvain, or guide')
	parser.add_argument('--n_genes', type=int, metavar='N', help='how many genes to keep, based on variance.')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--metadata', type=str, metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--output', type=str, metavar='O', help='where the output files should go')
	args = parser.parse_args()

	# Create a gene dataset
	gene_dataset = CropseqDataset(
		filename=args.data,
		metadata_filename=args.metadata,
		batch='wells',
		use_labels=args.model_label,
		new_n_genes=args.n_genes,
		save_path='')

	# Read the model
	model = torch.load(args.model_path, map_location=lambda storage, loc: storage)

	# Create a data_loader
	data_loader = SupervisedTrainTestDataLoaders(gene_dataset)

	# Sample the latent space
	latent, batch_indices, labels = get_cropseq_latent(model, data_loader['sequential'])

	# Load the numpy array file into AnnData
	adata = AnnData(latent)

	print('Using n_neighbors:', args.n_neighbors)

	# Compute neighborhood
	sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=0, use_rep=None)

	# Compute louvain
	sc.tl.louvain(adata)
	print('Number of louvain clusters:', len(adata.obs['louvain'].value_counts()))

	# Compute, show, and save UMAP
	sc.tl.umap(adata)
	sc.pl.umap(adata, color='louvain');
	plt.savefig(args.output + '/umap_louvain_{}.png'.format(args.n_neighbors))
	plt.close()

	# Compute, show, and save tSNE
	sc.tl.tsne(adata, n_pcs=0, use_rep=None)
	sc.pl.tsne(adata, color='louvain')
	plt.savefig(args.output + '/tsne_louvain_{}.png'.format(args.n_neighbors))
	plt.close()

	adata.write(args.output + '/scvi_vis_{}.h5ad'.format(args.n_neighbors))