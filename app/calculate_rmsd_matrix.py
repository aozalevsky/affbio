#!/usr/bin/python

import logging
import datetime as dt
import numpy as np
import prody

from mpi4py import MPI

import h5py
from h5py import h5s


#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

#Init logging
if rank == 0:
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
#Reread structures by every process
pdb_struct = np.load(stf, mmap_mode='r')
#Count number of structures
N = np.count_nonzero(pdb_struct)

#Init storage for matrices
Mfn = 'aff_rmsd_matrix.hdf5'
#HDF5 file
Mf = h5py.File(Mfn, 'w', driver='mpio', comm=comm)
#Table for RMSD
M = Mf.create_dataset(
    'rmsd',
    (N, N),
    dtype='float32',
    chunks=(1, N))
    #compression='lzf', Not for parallel version
    #fletcher32=True)

Ms = M.id.get_space()

j = rank
while j < N:
    if rank == 0:
        logging.info('Processing column %d of %d' % (j, N))
    jj = j + 1
    tM = np.fromiter(
        (calc(pdb_struct[i], pdb_struct[j]) for i in xrange(jj, N)),
        dtype='float32')

    ms = h5s.create_simple((N - jj,))
    Ms.select_hyperslab((jj, j), (N - jj, 1))
    M.id.write(ms, Ms, tM)

    j += NPROCS

#Wait for all processes
comm.Barrier()

if rank == 0:
    logging.info("RMSD matrix have been calculated")
    logging.info("RMSD matrix have been successfully written to %s" % Mfn)
    t = dt.datetime.now() - t
    logging.info("RMSD calculation time is %s" % t)

#Cleanup
#Close matrix file
Mf.close()
