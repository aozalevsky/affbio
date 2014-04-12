#!/usr/bin/python

import logging
import time
import numpy as np
import prody

from mpi4py import MPI

#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

import h5py
from h5py import h5s


debug = False
#debug = True

if rank == 0:
    #Get current time
    t0 = time.time()

    if debug is True:
        import cProfile
        import pstats
        import StringIO
        pr = cProfile.Profile()
        pr.enable()

#Init logging
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
Sfn = 'aff_struct.hdf5'
#Reread structures by every process
Sf = h5py.File(Sfn, 'r', driver='mpio', comm=comm)
S = Sf['struct']
#Count number of structures
N = S.len()

#Init storage for matrices
Mfn = 'aff_rmsd_matrix.hdf5'
#HDF5 file
#fid = h5f.create(Mfn, h5f.ACC_TRUNC, fapl=fapl)
#Mf = h5py.File(fid)
Mf = h5py.File(Mfn, 'w', driver='mpio', comm=comm)
Mf.atomic = True
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
    tM = np.fromiter(
        (calc(S[i], S[j]) for i in xrange(jj, N)),
        dtype='float32')

    ms = h5s.create_simple((N - jj,))
    Ms.select_hyperslab((jj, j), (N - jj, 1))
    M.id.write(ms, Ms, tM)

    j = j + NPROCS

#Wait for all processes
comm.Barrier()

if rank == 0:
    logging.info("RMSD matrix have been calculated")
    logging.info("RMSD matrix have been successfully written to %s" % Mfn)
    logging.info("RMSD calculation time is %s" % (time.time() - t0))
    print S[100, :5]

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

#Cleanup
#Close matrix file
Mf.close()
Sf.close()
