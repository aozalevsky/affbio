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
    import cProfile, pstats, StringIO
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
        print 'Truncating matrix to NxN to fit on %d procs' % NPROCS

    med = livestats.LiveStats()
    madd = np.vectorize(med.add)

N = comm.bcast(N, root=0)
l = comm.bcast(l, root=0)

tb, te = task(NPROCS - 1 - rank, l)

if rank == 0:
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
CM.attrs['preference'] = 0.0


random_state = np.random.RandomState(0)
x = np.finfo(np.float).eps
y = np.finfo(np.float).tiny * 100

# Remove degeneracies
for j in xrange(tb, te):

    jj = j + 1
    tN = N - jj

    tRM = np.empty((tN,), dtype=np.float)
    ms = h5s.create_simple((tN,))
    RMs.select_hyperslab((jj, j), (tN, 1))
    RM.id.read(ms, RMs, tRM)

    tRM = -1 * tRM ** 2
    tCM = tRM * x + y

    ttCM = tCM * random_state.randn(tN)
    ttCM = ttCM + tRM

    CMs.select_hyperslab((jj, j), (tN, 1))
    CM.id.write(ms, CMs, ttCM)

    if rank != 0:
        comm.send(ttCM, dest=0)
#        print rank, 'Sent'

    if rank == 0:
        madd(ttCM)
        for n in range(1, NPROCS):
            madd(comm.recv(source=n))

    ttCM = tCM * random_state.randn(tN)
    ttCM = ttCM + tRM

    CMs.select_hyperslab((j, jj), (1, tN))
    CM.id.write(ms, CMs, ttCM)


preference = np.empty((1,), dtype=np.float)

if rank == 0:
    level, median = med.quantiles()[0]
    preference = np.array([median], dtype=np.float)
    CM.attrs['preference'] = median
    te += 1

comm.Bcast([preference, MPI.FLOAT])

ms_e = h5s.create_simple((1,))

for i in xrange(tb, te):

    tCM = (preference * x + y) * random_state.randn(1) + preference

    CMs.select_elements([(i, i)])
    CM.id.write(ms_e, CMs, tCM)


if rank == 0:

    t1 = dt.datetime.now()

    print 'Med', median

    pr.disable()
    print 'NP', np.median(CM)
    print "Time is %s" % (t1 - t0)
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #print s.getvalue()

RMf.close()
CMf.close()
