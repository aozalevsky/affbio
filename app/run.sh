#!/bin/bash

#Exit on first error
set -e

#Get path to app dir
AFFDIR=$(dirname ${0})


#Load structures
mpirun -n ${1} ${AFFDIR}/load_structures.py *.pdb
#Calculate matrix
mpirun -n ${1} ${AFFDIR}/calculate_rmsd_matrix.py
#Prepare matrix
mpirun -n ${1} ${AFFDIR}/prepare_cluster_matrix.py
#Calculate and set preference
${AFFDIR}/calculate_median.py
#Cluster matrix
mpirun -n ${1} ${AFFDIR}/affinity_propagation_mpi.py

#print centers
${AFFDIR}/get_stat.py
