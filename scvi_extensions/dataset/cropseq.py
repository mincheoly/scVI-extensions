from scvi.dataset import GeneExpressionDataset
import numpy as np
import pandas as pd
import h5py
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
            use_donors=False, 
            use_labels='guide'):

        self.download_name = filename
        self.metadata_filename = metadata_filename
        self.save_path = save_path
        self.url = url
        self.use_donors = use_donors
        self.use_labels = use_labels

        data, gene_names, guides, donor_batches, louvain, ko_gene = self.preprocess()

        self.guide_lookup = np.unique(guides)
        self.ko_gene_lookup = np.array([x.split('.')[0] for x in self.guide_lookup], dtype=str)
        self.guides = guides
        self.louvain = louvain
        self.ko_gene = ko_gene
        self.donor_batches = donor_batches

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
                batch_indices=donor_batches if self.use_donors else 0, 
                labels=labels),
                gene_names=gene_names,
                cell_types=np.unique(labels))
        
        self.subsample_genes(new_n_genes=new_n_genes, subset_genes=subset_genes)


    def preprocess(self):
        print("Preprocessing CROP-seq dataset")

        barcodes, gene_names, matrix = self.read_h5_file()

        is_gene = ~pd.Series(gene_names, dtype=str).str.contains('guide').values

        # Remove guides from the gene list
        gene_names = gene_names[is_gene]
        data = matrix[:, is_gene]

        # Get labels and wells from metadata
        metadata = pd.read_csv(self.metadata_filename, sep='\t')
        keep_cell_indices, guides, donor_batches, louvain, ko_gene = self.process_metadata(metadata, data, barcodes)

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
        print('Number of cells kept after removing all zero cells:', has_umis.sum())

        print("Finished preprocessing CROP-seq dataset")
        return data, gene_names, guides, donor_batches, louvain, ko_gene


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
        ko_gene = keep_cells_metadata['guide_cov'].str.extract(r'(.*)\.')

        # Assign codes to each donor
        donor_labels = keep_cells_metadata['donor_cov'].values
        unique_donors = np.unique(donor_labels)
        self.donor_lookup = dict(zip(list(unique_donors), list(range(len(unique_donors)))))
        donor_batches = np.zeros(len(donor_labels))
        for i in range(len(donor_labels)):
            donor_batches[i] = self.donor_lookup[donor_labels[i]]
        donor_batches = donor_batches.reshape(-1, 1)

        return keep_cells_metadata['row_number'].values, guides, donor_batches, louvain, ko_gene


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
