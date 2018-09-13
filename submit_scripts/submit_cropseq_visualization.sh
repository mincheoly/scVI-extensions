#!/bin/bash                        
#                           
#$ -S /bin/bash                  
#$ -o /netapp/home/mincheol/outputs
#$ -e /netapp/home/mincheol/outputs
#$ -cwd                          
#$ -r y                          
#$ -j y                          
#$ -l mem_free=30G
#$ -l arch=linux-x64             
#$ -l netapp=5G,scratch=5G      
#$ -l h_rt=20:00:00
#$ -t 1-12               

# If you used the -t option above, this same script will be run for each task,
# but with $SGE_TASK_ID set to a different value each time (1-10 in this case).
# The commands below are one way to select a different input (PDB codes in
# this example) for each task.  Note that the bash arrays are indexed from 0,
# while task IDs start at 1, so the first entry in the tasks array variable
# is simply a placeholder

n_neighbor_array=(4 5 6 7 8 9 10 11 13 14 16 18)
n_neighbor="${n_neighbor_array[$SGE_TASK_ID]}"

export PYTHONPATH=/netapp/home/mincheol/scVI:/netapp/home/mincheol/scVI-extensions
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/netapp/home/mincheol/anaconda3/lib

export CUDA_PATH=/ye/yelabstore2/mincheol/cuda-8.0
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/ye/yelabstore2/mincheol/cuda-8.0/bin

source activate scvi
python /netapp/home/mincheol/scVI-extensions/scripts/cropseq_visualization.py \
	--latent_space /netapp/home/mincheol/scvi_latent_space.csv \
	--n_neighbors $n_neighbor \
	--output /ye/yelabstore2/mincheol/outputs/scvi/scvi_louvain
source deactivate

qstat -j $JOB_ID                                  # This is useful for debugging and usage purposes,
                                                  # e.g. "did my job exceed its memory request?"