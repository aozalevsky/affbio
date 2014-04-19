#!/bin/bash

#Load structures
mpirun -n ${1} /home/domain/silwer/work/aff_cluster/app/load_structures.py *.pdb
#Calculate matrix
mpirun -n ${1} /home/domain/silwer/work/aff_cluster/app/calculate_rmsd_matrix.py
#Prepare matrix
mpirun -n ${1} /home/domain/silwer/work/aff_cluster/app/prepare_cluster_matrix.py
#Calculate and set preference
/home/domain/silwer/work/aff_cluster/app/calculate_median.py
#Cluster matrix
mpirun -n ${1} /home/domain/silwer/work/aff_cluster/app/affinity_propagation_mpi.py
