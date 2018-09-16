from scvi.dataset import GeneExpressionDataset
from scvi.dataset.dataset import arrange_categories
import numpy as np
import pandas as pd
import h5py
import torch
import scipy.sparse as sp_sparse


class CropseqDataset(GeneExpressionDataset):
    r"""Loads a `.h5` file from a CROP-seq experiment.

    Args:
        :filename: Name of the `.h5` file.
        :metadata_filename: Name of the tab separated metadata file.
        :save_path: Save path of the dataset. Default: ``'data/'``.
        :url: Url of the remote dataset. Default: ``None``.
        :new_n_genes: Number of subsampled genes. Default: ``False``.
        :subset_genes: List of genes for subsampling. Default: ``None``.
        :use_donors: Whether to use donors as batches. Default: ``False``.
        :use_labels: Whether to use guides as labels. Default: ``False``.


    Examples:
        >>> # Loading a local dataset
        >>> local_cropseq_dataset = CropseqDataset("TM_droplet_mat.h5", save_path = 'data/')

    """

    def __init__(
            self, 
            filename, 
            metadata_filename, 
            save_path='data/', 
            url=None, 
            new_n_genes=False, 
            subset_genes=None, 
            batch='wells', 
            use_labels='guide',
            testing_labels=None,
            remove_guides=True
            ):

        self.download_name = filename
        self.metadata_filename = metadata_filename
        self.save_path = save_path
        self.url = url
        self.use_donors = use_donors
        self.use_labels = use_labels
        self.remove_guides = remove_guides

        data, gene_names, guides, donor_batches, louvain, ko_gene, wells = self.preprocess()

        self.guide_lookup = np.unique(guides)
        self.ko_gene_lookup = np.unique(ko_gene)
        self.guides = guides
        self.louvain = louvain
        self.ko_gene = ko_gene
        self.donor_batches = donor_batches
        self.wells = wells

        if not self.use_labels:
            labels = None
        elif self.use_labels == 'guide':
            labels = guides
        elif self.use_labels == 'louvain':
            labels = louvain
        elif self.use_labels == 'gene':
            labels = ko_gene
        else:
            labels = None

        super(CropseqDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                data, 
                batch_indices=self.donor_batches if self.batch == 'donor' else self.wells, 
                labels=labels),
                gene_names=gene_names,
                cell_types=np.unique(labels))

        if not testing_labels:
            self.testing_labels = None
        elif testing_labels == 'guide':
            self.testing_labels, _ = arrange_categories(self.guides)
        elif testing_labels == 'louvain':
            self.testing_labels, _ = arrange_categories(self.louvain)
        elif testing_labels == 'gene':
            self.testing_labels, _ = arrange_categories(self.ko_gene)
        else:
            self.testing_labels = None # Perhaps raise an exception.
        
        self.subsample_genes(new_n_genes=new_n_genes, subset_genes=subset_genes)


    def preprocess(self):
        print("Preprocessing CROP-seq dataset")

        barcodes, gene_names, matrix = self.read_h5_file()

        if self.remove_guides:

            is_gene = ~pd.Series(gene_names, dtype=str).str.contains('guide').values

            # Remove guides from the gene list
            gene_names = gene_names[is_gene]
            data = matrix[:, is_gene]
        else:

            data = matrix

        # Get labels and wells from metadata
        metadata = pd.read_csv(self.metadata_filename, sep='\t')
        keep_cell_indices, guides, donor_batches, louvain, ko_gene, wells = self.process_metadata(metadata, data, barcodes)

        print('Number of cells kept after filtering with metadata:', len(keep_cell_indices))

        # Filter the data matrix
        data = data[keep_cell_indices, :]

        # Remove all 0 cells
        has_umis = (data.sum(axis=1) > 0).A1
        data = data[has_umis, :]
        guides = guides[has_umis]
        donor_batches = donor_batches[has_umis]
        louvain = louvain[has_umis]
        ko_gene = ko_gene[has_umis]
        wells = wells[has_umis]
        print('Number of cells kept after removing all zero cells:', has_umis.sum())

        print("Finished preprocessing CROP-seq dataset")
        return data, gene_names, guides, donor_batches, louvain, ko_gene, wells


    def process_metadata(self, metadata, data, barcodes):

        # Attach original row number to the metadata
        matrix_barcodes = pd.DataFrame()
        matrix_barcodes['index'] = barcodes
        matrix_barcodes['row_number'] = np.arange(barcodes.shape[0])
        full_metadata = metadata.merge(matrix_barcodes, on='index', how='left')

        # Apply some filtering criteria
        keep_cells_metadata = full_metadata\
            .query('guide_cov != "Undetermined"')\
            .copy()

        # Clean up data
        keep_cells_metadata['guide_cov'] = keep_cells_metadata['guide_cov'].replace('0', 'NO_GUIDE')
        guides = keep_cells_metadata['guide_cov'].values.reshape(-1, 1)

        # Extract louvain cluster
        louvain = keep_cells_metadata['louvain'].values.reshape(-1, 1)

        # Extract the gene being knocked out
        ko_gene = keep_cells_metadata['guide_cov'].str.extract(r'^([^.]*).*').values.reshape(-1, 1)

        # Extract the well
        wells = keep_cells_metadata['well_cov'].values.reshape(-1, 1)

        # Assign codes to each donor
        donor_labels = keep_cells_metadata['donor_cov'].values
        unique_donors = np.unique(donor_labels)
        self.donor_lookup = dict(zip(list(unique_donors), list(range(len(unique_donors)))))
        donor_batches = np.zeros(len(donor_labels))
        for i in range(len(donor_labels)):
            donor_batches[i] = self.donor_lookup[donor_labels[i]]
        donor_batches = donor_batches.reshape(-1, 1)

        return keep_cells_metadata['row_number'].values, guides, donor_batches, louvain, ko_gene, wells


    def read_h5_file(self, key=None):
        
        with h5py.File(self.save_path + self.download_name, 'r') as f:
            
            keys = [k for k in f.keys()]
            
            if not key:
                key = keys[0]
                
            group = f[key]
            attributes = {key:val[()] for key, val in group.items()}
            matrix = sp_sparse.csc_matrix(
                (
                    attributes['data'], 
                    attributes['indices'], 
                    attributes['indptr']), 
                shape=attributes['shape'])
            
        return attributes['barcodes'].astype(str), attributes['gene_names'].astype(str), matrix.transpose()

    def collate_fn(self, batch):

        indexes = np.array(batch)
        if self.testing_labels is None:
            if self.dense:
                X = torch.from_numpy(self.X[indexes])
            else:
                X = torch.FloatTensor(self.X[indexes].toarray())
            if self.x_coord is None or self.y_coord is None:
                return X, torch.FloatTensor(self.local_means[indexes]), \
                       torch.FloatTensor(self.local_vars[indexes]), \
                       torch.LongTensor(self.batch_indices[indexes]), \
                       torch.LongTensor(self.labels[indexes])
            else:
                return X, torch.FloatTensor(self.local_means[indexes]), \
                       torch.FloatTensor(self.local_vars[indexes]), \
                       torch.LongTensor(self.batch_indices[indexes]), \
                       torch.LongTensor(self.labels[indexes]), \
                       torch.FloatTensor(self.x_coord[indexes]), \
                       torch.FloatTensor(self.y_coord[indexes])
        else:
            if self.dense:
                X = torch.from_numpy(self.X[indexes])
            else:
                X = torch.FloatTensor(self.X[indexes].toarray())
            if self.x_coord is None or self.y_coord is None:
                return X, torch.FloatTensor(self.local_means[indexes]), \
                       torch.FloatTensor(self.local_vars[indexes]), \
                       torch.LongTensor(self.batch_indices[indexes]), \
                       torch.LongTensor(self.labels[indexes]), \
                       torch.LongTensor(self.testing_labels[indexes])
            else:
                return X, torch.FloatTensor(self.local_means[indexes]), \
                       torch.FloatTensor(self.local_vars[indexes]), \
                       torch.LongTensor(self.batch_indices[indexes]), \
                       torch.LongTensor(self.labels[indexes]), \
                       torch.FloatTensor(self.x_coord[indexes]), \
                       torch.FloatTensor(self.y_coord[indexes])