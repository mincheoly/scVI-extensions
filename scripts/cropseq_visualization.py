"""
	cropseq_visualization.py
	Visualize the latent space of the CROP-seq experiment.
	Quantify the visualizations.
"""

import sys
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
	parser = argparse.ArgumentParser(description='Clustering and visualizing latent space of scVI.')
	parser.add_argument('--latent_space', type=str, help='path to the csv latent space file')
	parser.add_argument('--n_neighbors', type=int, default=14, help='n_neighbors for SCANPY neighborhood calcuation')
	parser.add_argument('--output', type=str, metavar='O', help='where the output files should go')
	parser.add_argument('--louvain', type=str, default=None, help='louvain cluster assignments')

	args = parser.parse_args()

	# Load the CSV file into AnnData
	adata = sc.read_csv(args.latent_space)

	print('Using n_neighbors:', args.n_neighbors)

	# Compute neighborhood
	sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=0, use_rep=None)

	if args.louvain is None:

		# Compute louvain
		sc.tl.louvain(adata)
		print('Number of louvain clusters:', len(adata.obs['louvain'].value_counts()))

	else:
		print('Given previously calculated louvain clusters...')
		adata.obs['louvain'] = pd.read_csv(args.louvain)['louvain'].astype('category').values

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

	adata.write(args.output + '/anndata.h5ad')