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
from scvi_extensions.hypothesis_testing.mean import differential_expression
from scvi_extensions.hypothesis_testing.variance import gene_variance_test
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
	parser = argparse.ArgumentParser(description='Performing differential expression')
	parser.add_argument('--model_path', type=str, metavar='M', help='path to a trained torch model')
	parser.add_argument('--label', type=str, metavar='L', help='what to use as label, one of: gene, louvain, guide')
	parser.add_argument('--n_genes', type=int, metavar='N', help='how many genes to keep, based on variance.')
	parser.add_argument('--data', type=str, metavar='D', help='path to the h5 data file')
	parser.add_argument('--metadata', type=str,metavar='E', help='path to the tab separated metadata file')
	parser.add_argument('--desired_labels', type=str, help='List of desired labels')
	parser.add_argument('--output', type=str, metavar='O', help='where the output files should go')
	parser.add_argument('--gpu', help='using a GPU?', action='store_true')
	args = parser.parse_args()

	# # Create a dataset
	h5_filename = args.data
	metadata_filename = args.metadata
	gene_dataset = CropseqDataset(
		filename=h5_filename,
		metadata_filename=metadata_filename,
		use_donors=True,
		use_labels=args.label,
		new_n_genes=args.n_genes,
		save_path='')

	# Parse the desired labels
	named_desired_labels = [x.strip("'") for x in args.desired_labels[1:-1].split(', ')]
	if args.label == 'louvain':
		desired_labels = [int(x) for x in named_desired_labels]
	elif args.label == 'guide':
		desired_labels = [np.where(gene_dataset.guide_lookup == guide)[0][0] for guide in named_desired_labels]
	elif args.label == 'gene':
		desired_labels = [np.where(gene_dataset.ko_gene_lookup == gene)[0][0] for gene in named_desired_labels]
	else:
		raise AssertionError('labels must be one of: gene, guide, or louvain.')
	label_lookup = dict(zip(desired_labels, named_desired_labels))

	# Read the model
	if args.gpu:
		model = torch.load(args.model_path)
	else:
		model = torch.load(args.model_path, map_location=lambda storage, loc: storage)

	# Perform hypothesis testing
	de_results = differential_expression(model, gene_dataset, desired_labels, 100, 10000)
	variance_results = gene_variance_test(model, gene_dataset, desired_labels, 100, 10000)

	# Save the results
	for label_1, label_2 in itertools.combinations(desired_labels, 2):
		de_results[label_1][label_2].to_csv(args.output + '/de_result_{}_{}.csv'.format(label_lookup[label_1], label_lookup[label_2]), index=False)
		variance_results[label_1][label_2].to_csv(args.output + '/variance_result_{}_{}.csv'.format(label_lookup[label_1], label_lookup[label_2]), index=False)
