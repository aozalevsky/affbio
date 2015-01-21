#!/bin/sh

[ x"$TMPDIR" == x"" ] && TMPDIR=/tmp
HOSTFILE=${TMPDIR}/hostfile.${SLURM_JOB_ID}
srun hostname -s|sort|uniq -c|awk '{print $2" slots="$1}' > $HOSTFILE || { rm -f $HOSTFILE; exit 255; }

#mpirun --hostfile $HOSTFILE "$@"

#Load structures
#mpirun -n 2 /home/silwer/work/aff_cluster/app/load_structures.py *.pdb

#Calculate matrix
mpirun --hostfile $HOSTFILE /home/users/golovin/progs/python2.7/bin/python2.7 /home/users/golovin/progs/aff_cluster/app/calculate_rmsd_matrix.py
#Prepare matrix
mpirun --hostfile $HOSTFILE /home/users/golovin/progs/python2.7/bin/python2.7 /home/users/golovin/progs/aff_cluster/app/prepare_cluster_matrix.py
#Calculate and set preference
srun -n 1 /home/users/golovin/progs/python2.7/bin/python2.7 /home/users/golovin/progs/aff_cluster/app/calculate_median.py
#Cluster matrix
mpirun --hostfile $HOSTFILE -npernode 1 /home/users/golovin/progs/python2.7/bin/python2.7 /home/users/golovin/progs/aff_cluster/app/affinity_propagation_mpi.py

rm -f $HOSTFILE
