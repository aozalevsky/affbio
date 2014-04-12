#!/usr/bin/python

import time
import numpy as np
import pyRMSD.RMSDCalculator
from pyRMSD import condensedMatrix

from mpi4py import MPI

#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
size = NPROCS
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


def task(rk, ln):
    b = rk * ln
    return (b, b + ln)


def calc_diag_chunk(ic, tS):
    calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
        "QCP_OMP_CALCULATOR",
        ic)
    rmsd = calculator.pairwiseRMSDMatrix()
    rmsd_matrix = condensedMatrix.CondensedMatrix(rmsd)
    ln = len(tS)
    for i in xrange(ln):
        for j in xrange(i):
            tS[i, j] = rmsd_matrix[i, j]


def calc_chunk(ic, jc, tS):
    ln, n, d = ic.shape
    ttS = np.empty((ln + 1, n, d))
    ttS[1:] = jc
    for i in xrange(ln):
        ttS[0] = ic[i]
        calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
            "QCP_OMP_CALCULATOR",
            ttS)
        tS[i] = calculator.oneVsFollowing(0)


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
Ms = M.id.get_space()


#Partiotioning
l = N // size
lr = N % size

if lr > 0:
    print 'Truncating matrix to %dx%d to fit' % (l * size, l * size)

lN = (size + 1) * size / 2

m = lN // size
mr = lN % size

if mr > 0:
    m = m + 1 if rank % 2 == 0 else m


#Init calculations
tS = np.zeros((l, l), dtype=np.float)
ms = h5s.create_simple((l, l))

i, j = rank, rank
ic = S[i * l: (i + 1) * l]
jc = ic

for c in xrange(0, m):
    try:
        assert i == j
        calc_diag_chunk(ic, tS)
    except AssertionError:
        calc_chunk(ic, jc, tS)

    Ms.select_hyperslab((i * l, j * l), (l, l))
    M.id.write(ms, Ms, tS)

    if rank == 0:
        print "Step %d of %d" % (c, m)

    if 0 < (rank - c):
        j = j - 1
        jc = S[j * l: (j + 1) * l]
    elif rank - c == 0:
        i = size - rank - 1
        ic = S[i * l: (i + 1) * l]
    else:
        j = j + 1
        jc = S[j * l: (j + 1) * l]


#Wait for all processes
comm.Barrier()

if rank == 0:
    print "RMSD matrix have been calculated"
    print "RMSD matrix have been successfully written to %s" % Mfn
    print "RMSD calculation time is %s" % (time.time() - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

#Cleanup
#Close matrix file
Sf.close()
Mf.close()
