#!/usr/bin/python

import logging
import datetime as dt
import numpy as np
import prody

import mpi4py
from mpi4py import MPI



#Get MPI info
comm = MPI.COMM_WORLD
info = MPI.INFO_ENV
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

import h5py
from h5py import h5s

from h5py import h5p, h5fd

dxpl = h5p.create(h5p.DATASET_XFER)
dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

fapl = h5p.create(h5p.FILE_ACCESS) 
fapl.set_fapl_mpio(comm, info)
fapl.set_alignment(0, 1048576) 

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
fid = h5f.create(MFn, h5f.ACC_TRUNC, fapl=fapl)

Mf = h5py.File(fid) 
#Mf = h5py.File(Mfn, 'w', driver='mpio', comm=comm)
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
    tN = N - jj
    if tN > 0:
        tM = np.fromiter(
            (calc(pdb_struct[i], pdb_struct[j]) for i in xrange(jj, N)),
            dtype='float32')

        ms = h5s.create_simple((N - jj,))
        Ms.select_hyperslab((jj, j), (N - jj, 1))
        M.id.write(ms, Ms, tM, dxpl=dxpl)

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
