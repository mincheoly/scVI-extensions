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
import torch
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy

from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.models import VAE, SVAEC
from scvi.inference import VariationalInference

from scvi_extensions.dataset.cropseq import CropseqDataset
from scvi_extensions.dataset.supervised_data_loader import SupervisedTrainTestDataLoaders


def get_cropseq_latent(vae, data_loader):
    latent = []
    batch_indices = []
    labels = []
    for tensors in data_loader:
        sample_batch, local_l_mean, local_l_var, batch_index, label, testing_label = tensors
        latent += [vae.sample_from_posterior_z(sample_batch, y=label)]
        batch_indices += [batch_index]
        labels += [testing_label]
    return np.array(torch.cat(latent)), np.array(torch.cat(batch_indices)).ravel(), np.array(torch.cat(labels)).ravel()


if __name__ == '__main__':

	# Argparse
	parser = argparse.ArgumentParser(description='Visulaizing the latent space of scVI')
	parser.add_argument('--model_path', type=str, metavar='M', help='path to a trained torch model')
	parser.add_argument('--model_label', type=str, help='the labels with which the model was trained on, one of: gene, louvain, or guide')
	parser.add_argument('--label', type=str, metavar='L', help='what to use as label, one of: gene, louvain, guide')
	parser.add_argument('--n_genes', type=int, metavar='N', help='how many genes to keep, based on variance.')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--metadata', type=str, metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--n_samples', type=int, help='how many samples to plot')
	parser.add_argument('--output', type=str, metavar='O', help='where the output files should go')

	args = parser.parse_args()

	# Load the model
	model = torch.load(args.model_path, map_location=lambda storage, loc: storage)

	# Load the dataset
	gene_dataset = CropseqDataset(
		filename=args.data,
		metadata_filename=args.metadata,
		use_donors=True,
		use_labels=args.model_label,
		new_n_genes=args.n_genes,
		testing_labels=args.label,
		save_path='')

	# Data Loader
	data_loader = SupervisedTrainTestDataLoaders(gene_dataset, num_samples=args.n_samples)

	# Sample the latent space
	latent, batch_indices, labels = get_cropseq_latent(model, data_loader['all'])

	# Perform t-SNE
	embedded = TSNE(n_components=2).fit_transform(latent)

	# Save an overall t-SNE picture. Attach legend only if the labels are louvain.
	plt.figure(figsize=(10, 10))
	for label in np.unique(labels):
	    plt.scatter(embedded[labels == label, 0], embedded[labels == label, 1], s=12)
	if args.label == 'louvain':
		plt.legend(np.unique(labels))
	plt.xlabel('tSNE1')
	plt.ylabel('tSNE2')
	plt.title('tSNE of the latent space')
	plt.savefig(args.output + '/overall_tsne.png')
	plt.close()

	# Save an overall t-SNE with donors as labels.
	plt.figure(figsize=(10, 10))
	for donor in np.unique(batch_indices):
	    plt.scatter(embedded[batch_indices == donor, 0], embedded[batch_indices == donor, 1], s=12)
	plt.legend(np.unique(batch_indices))
	plt.xlabel('tSNE1')
	plt.ylabel('tSNE2')
	plt.title('tSNE of the latent space')
	plt.savefig(args.output + '/overall_tsne_by_donor.png')
	plt.close()

	heatmaps = {}
	# Draw heatmaps of guides
	if args.label == 'gene':
		lookup = gene_dataset.ko_gene_lookup
	elif args.label == 'guide':
		lookup = gene_dataset.guide_lookup
	else: #louvain
		lookup = np.unique(labels)
	for label in np.unique(labels):
		heatmap, _, _ = np.histogram2d(
			embedded[labels == label, 0], 
			embedded[labels == label, 1],
			bins=40,
			normed=False)
		heatmap = gaussian_filter(heatmap, sigma=2)
		heatmaps[label] = heatmap
		plt.figure(figsize=(10, 10))
		plt.imshow(heatmap.T, extent=None, origin='lower', cmap=matplotlib.cm.jet, aspect=1)
		plt.title('Distribution of {} in tSNE space'.format(lookup[label]))
		plt.savefig(args.output + '/{}_heatmap.png'.format(lookup[label]))
		plt.close()

	distances = []
	# Compute shift in distribution for each label
	for label_1, label_2 in itertools.combinations(np.unique(labels), 2):
		l2_dist = np.sqrt(np.sum((heatmaps[label_1] - heatmaps[label_2])**2))
		distances.append((
			lookup[label_1],
			lookup[label_2],
			l2_dist,
			kl_dist))
	distance_df = pd.DataFrame(distances, columns=['label_1', 'label_2', 'L2', 'KL'])\
		.sort_values('L2', ascending=False)

	distance_df.to_csv(args.output + '/distribution_shifts.csv', index=False)





