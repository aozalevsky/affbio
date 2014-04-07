#!/usr/bin/python

import sys
import os
from os.path import join as pj

import datetime as dt

import numpy as np

from mpi4py import MPI

import h5py
from h5py import h5s

#debug = False
debug = True


def task(rk, ln):
    b = rk * ln
    return (b, b + l)

#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
rank = comm.rank

#Init RMSD matrix
#Get file name
RMfn = 'aff_rmsd_matrix.hdf5'
#Open matrix file in parallel mode
RMf = h5py.File(RMfn, 'r', driver='mpio', comm=comm)
#Open table with data for clusterization
RM = RMf['rmsd']
RMs = RM.id.get_space()

N = 0
l = 0

if rank == 0:
    t0 = dt.datetime.now()

    if debug is True:
        import cProfile
        import pstats
        import StringIO
        pr = cProfile.Profile()
        pr.enable()

    N, N1 = RM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(RM.shape))

    l = N / NPROCS
    r = N - l * NPROCS
    if r != 0:
        l = l
        N = N - r
        print 'Truncating matrix to %dx%d to fit on %d procs' % (N, N, NPROCS)

N = comm.bcast(N, root=0)
l = comm.bcast(l, root=0)

tb, te = task(rank, l)

# Fix for last row, because it diagonal and so empty
if rank == NPROCS - 1:
    te -= 1

#Init cluster matrix
#Get file name
CMfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
CMf = h5py.File(CMfn, 'w', driver='mpio', comm=comm)
#Open table with data for clusterization
CM = CMf.create_dataset(
    'cluster',
    (N, N),
    dtype=np.float)

CMs = CM.id.get_space()

random_state = np.random.RandomState(0)
x = np.finfo(np.float).eps
y = np.finfo(np.float).tiny * 100

# Remove degeneracies
for j in xrange(tb, te):

    if rank == 0:
        print 'Processing row %d of %d' % ((j - tb) * NPROCS, N)

    #Ignore diagonals
    jj = j + 1
    tN = N - jj

    ms = h5s.create_simple((tN,))

    tRM = np.empty((tN,), dtype=np.float)
    RMs.select_hyperslab((jj, j), (tN, 1))
    RM.id.read(ms, RMs, tRM)

    tRM = -1 * tRM ** 2
    tCM = tRM * x + y

    ttCM = tCM * random_state.randn(tN)
    ttCM = ttCM + tRM

    CMs.select_hyperslab((jj, j), (tN, 1))
    CM.id.write(ms, CMs, ttCM)

    ttCM = tCM * random_state.randn(tN)
    ttCM = ttCM + tRM

    CMs.select_hyperslab((j, jj), (1, tN))
    CM.id.write(ms, CMs, ttCM)

RMf.close()
CMf.close()

if rank == 0:

    t1 = dt.datetime.now()
    print "Time is %s" % (t1 - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
