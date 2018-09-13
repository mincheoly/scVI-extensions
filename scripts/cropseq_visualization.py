"""
	tsne_comparison.py
	Visualize the latent space of the CROP-seq experiment.
	Quantify the visualizations.
"""


import os

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


if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Clustering and visualizing latent space of scVI.')
	parser.add_argument('--latent_space', type=str, help='path to the csv latent space file')
	parser.add_argument('--n_neighbors', type=int, help='n_neighbors for SCANPY neighborhood calcuation')
	parser.add_argument('--output', type=str, metavar='O', help='where the output files should go')

	args = parser.parse_args()

	# Load the CSV file into AnnData
	adata = sc.read_csv(args.latent_space)

	print('Using n_neighbors:', args.n_neighbors)

	# Compute neighborhood
	sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=0, use_rep=None)

	# Compute louvain
	sc.tl.louvain(adata)
	print('Number of louvain clusters:', len(adata.obs['louvain'].value_counts()))
	adata.obs.to_csv(args.output + '/louvain_cluster_labels_{}.csv'.format(args.n_neighbors))

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
