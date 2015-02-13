#!/usr/bin/python

#General modules
import time

#NumPy for arrays
import numpy as np

#RMPI parallelism
from mpi4py import MPI
#Get RMPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

#H5PY for storage
import h5py
from h5py import h5s

#pyRMSD for calculations
import pyRMSD.RMSDCalculator
from pyRMSD import condensedMatrix


if rank == 0:
    print "Calculating RMSD matrix"
    #Get current time
    t0 = time.time()

    debug = False
    #debug = True

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
        "KABSCH_SERIAL_CALCULATOR",
        ic)
    rmsd = calculator.pairwiseRMSDMatrix()
    rmsd_matrix = condensedMatrix.CondensedMatrix(rmsd)
    ln = len(tS)
    for i in xrange(ln):
        for j in xrange(i):
            tS[i, j] = rmsd_matrix[i, j]


def calc_chunk(ic, jc, tS):
    ln, n, d = ic.shape
    ttS = np.zeros((ln + 1, n, d))
    ttS[1:] = jc
    for i in xrange(ln):
        ttS[0] = ic[i]
        calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
            "KABSCH_SERIAL_CALCULATOR",
            ttS)
        tS[i] = calculator.oneVsFollowing(0)


#Now RMSD calculation
Sfn = 'aff_struct.hdf5'
#Reread structures by every process
Sf = h5py.File(Sfn, 'r', driver='mpio', comm=comm)
S = Sf['struct']
#Count number of structures
N = S.len()


#Partiotioning
l = N // NPROCS
lr = N % NPROCS

if lr > 0:
    print 'Truncating matrix to %dx%d to fit' % (l * NPROCS, l * NPROCS)

lN = (NPROCS + 1) * NPROCS / 2

m = lN // NPROCS
mr = lN % NPROCS

if mr > 0:
    m = m + 1 if rank % 2 == 0 else m

#Init storage for matrices
RMfn = 'aff_rmsd_matrix.hdf5'
#HDF5 file
#fid = h5f.create(RMfn, h5f.ACC_TRUNC, fapl=fapl)
#RMf = h5py.File(fid)
RMf = h5py.File(RMfn, 'w', driver='mpio', comm=comm)
RMf.atomic = True
#Table for RMSD
RM = RMf.create_dataset(
    'rmsd',
    (N, N),
    dtype=np.float,
    chunks=(l, l))
RM.attrs['chunk'] = l
RMs = RM.id.get_space()


#Init calculations
tS = np.zeros((l, l), dtype=np.float)
ms = h5s.create_simple((l, l))

i, j = rank, rank
ic = S[i * l: (i + 1) * l]
jc = ic

for c in xrange(0, m):
    if rank == 0:
        tit = time.time()

    try:
        assert i == j
        calc_diag_chunk(ic, tS)
    except AssertionError:
        calc_chunk(ic, jc, tS)

    RMs.select_hyperslab((i * l, j * l), (l, l))
    RM.id.write(ms, RMs, tS)

    if rank == 0:
        teit = time.time()
        print "Step %d of %d T %s" % (c, m, teit - tit)

    if 0 < (rank - c):
        j = j - 1
        jc = S[j * l: (j + 1) * l]
    elif rank - c == 0:
        i = NPROCS - rank - 1
        ic = S[i * l: (i + 1) * l]
    else:
        j = j + 1
        jc = S[j * l: (j + 1) * l]


#Wait for all processes
comm.Barrier()

if rank == 0:
    print "RMSD matrix have been calculated"
    print "RMSD matrix have been successfully written to %s" % RMfn
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
RMf.close()
