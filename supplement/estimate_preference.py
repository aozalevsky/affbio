#!/usr/bin/python

import sys
import os
from os.path import join as pj

import datetime as dt

import numpy as np

from mpi4py import MPI

import h5py
from h5py import h5s

from livestats import livestats


def task(rk, ln):
    b = rk * ln
    return (b, b + l)

#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
rank = comm.rank

#Init cluster matrix
#Get file name
CMfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
CMf = h5py.File(CMfn, 'r', driver='mpio', comm=comm)
#Open table with data for clusterization
CM = CMf['cluster']

N = 0
l = 0

if rank == 0:
    t0 = dt.datetime.now()
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    N, N1 = CM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(CM.shape))

    l = N / NPROCS
    r = N - l * NPROCS
    if r != 0:
        l = l
        N = N - r
        print 'Truncating matrix to NxN to fit on %d procs' % NPROCS

    med = livestats.LiveStats()
    madd = np.vectorize(med.add)

N = comm.bcast(N, root=0)
l = comm.bcast(l, root=0)

CMs = CM.id.get_space()
tCM = np.empty((N,), dtype=np.float)

ms = h5s.create_simple((N,))

tb, te = task(NPROCS - 1 - rank, l)

if rank == 0:
    te -= 1

# Remove degeneracies
for i in xrange(tb, te):

    CMs.select_hyperslab((i, 0), (1, N))
    CM.id.read(ms, CMs, tCM)

    if rank != 0:
        comm.Send([tCM, MPI.FLOAT], dest=0)

    if rank == 0:
        madd(tCM)
        for n in range(1, NPROCS):
            comm.Recv([tCM, MPI.FLOAT], source=n)
            madd(tCM)

if rank == 0:
    level, median = med.quantiles()[0]
    t1 = dt.datetime.now()

    print 'Med', median
    pr.disable()
    print 'NP', np.median(CM)
    print "Time is %s" % (t1 - t0)
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

CMf.close()
