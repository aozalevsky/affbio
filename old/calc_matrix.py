#!/usr/bin/python

import logging
import datetime as dt
import numpy as np
import prody

from mpi4py import MPI
import h5py


#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size


#Init logging
if comm.rank == 0:
    #Get current time
    t0 = dt.datetime.now()
    t = t0
logging.basicConfig(filename='aff.log', level=logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#Add logging to console
console = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(message)s')
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#Change logging level for prody module
pr_log = logging.getLogger("prody")
pr_log.setLevel(logging.ERROR)


def calc(i, j):
    """calculate RMSD"""
    mob, trans = prody.superpose(j, i)
    return prody.calcRMSD(i, mob)


#Now RMSD calculation
stf = 'aff.struct'
matf = 'aff.raw'
#Reread structures by every process
pdb_struct = np.load(stf, mmap_mode='r')
ln = np.count_nonzero(pdb_struct)
#Init storage for matrices
#HDF5 file
f = h5py.File(matf, 'w', driver='mpio', comm=comm)
#Table for RMSD
rmsd_matrix = f.create_dataset('rmsd', (ln, ln), dtype='f')
#Table for clusterization
cl_matrix = f.create_dataset('cluster', (ln, ln), dtype='f')

i = comm.rank
while i < ln:
    if comm.rank == 0:
        logging.info('Processing row %d of %d' % (i, ln))
    for j in range(i):
        rmsd = calc(pdb_struct[i], pdb_struct[j])
        n_rmsd = -1.0 * rmsd ** 2.0
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd
        cl_matrix[i, j] = n_rmsd
        cl_matrix[j, i] = n_rmsd
    i += NPROCS

#Wait for all processes
comm.Barrier()

if comm.rank == 0:
    logging.info("RMSD matrix have been calculated")
    logging.info("RMSD matrix have been successfully written to %s" % matf)
    t = dt.datetime.now() - t
    logging.info("RMSD calculation time is %s" % t)

#Cleanup
#Close matrix file
f.close()
comm.Barrier()
