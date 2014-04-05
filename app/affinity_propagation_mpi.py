#!/usr/bin/python


import sys

import os
from os.path import join as pj

import datetime as dt

import numpy as np
import bottleneck as bn

from mpi4py import MPI

import h5py
from h5py import h5s


def task(rk, l):
    b = rk * l
    return (b, b + l)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
rank = comm.rank

#Init storage for matrices
#Get file name
Sfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
Sf = h5py.File(Sfn, 'r', driver='mpio', comm=comm)
#Open table with data for clusterization
S = Sf['cluster']

params = {
    'N': 0,
    'l': 0,
    'TMfn': '',
    'preference': 0.0,
    'conv_iter': 0,
    'max_iter': 0,
    'damping': 0.0,
    'verbose': False}

P = Bunch(params)

if rank == 0:
    t0 = dt.datetime.now()
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    N, N1 = S.shape

    if N != N1:
        raise ValueError("S must be a square array (shape=%s)" % repr((N, N1)))
    else:
        P.N = N

    conv_iter = 15

    if conv_iter < 1:
        raise ValueError('conv_iter must be > 1')
    else:
        P.conv_iter = conv_iter

    max_iter = 2000

    if max_iter < 0:
        raise ValueError('max_iter must be > 0')
    else:
        P.max_iter = max_iter

    damping = 0.9

    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')
    else:
        P.damping = damping

    verbose = True
    P.verbose = verbose

    #tmpd = osp.join(osp.abspath(osp.dirname(s.file.filename)), 'tmp.hdf5')
    P.TMfn = pj('/tmp', 'tmp.hdf5')

    l = N / NPROCS
    r = N - l * NPROCS

    if r != 0:
        print 'Truncating matrix to NxN to fit on %d procs' % NPROCS
        l = l
        N = N - r
    P.N = N
    P.l = l

P = comm.bcast(P)

Ss = S.id.get_space()
tS = np.ndarray((P.N,), dtype=np.float)
tdS = np.ndarray((1,), dtype=np.float)

TMf = h5py.File(P.TMfn, 'w', driver='mpio', comm=comm)

R = TMf.create_dataset('R', (P.N, P.N), dtype=np.float)
Rs = R.id.get_space()
tRold = np.ndarray((P.N,), dtype=np.float)
tdR = np.ndarray((P.l,), dtype=np.float)

Rp = TMf.create_dataset('Rp', (P.N, P.N), dtype=np.float)
Rps = Rp.id.get_space()
tRp = np.ndarray((P.N,), dtype=np.float)

A = TMf.create_dataset('A', (P.N, P.N), dtype=np.float)
As = A.id.get_space()

tAold = np.ndarray((P.N,), dtype=np.float)
tAS = np.ndarray((P.N,), dtype=np.float)
tdA = np.ndarray((P.l,), dtype=np.float)

e = np.ndarray((P.N, P.conv_iter), dtype=np.int)
tE = np.ndarray((P.N,), dtype=np.int)
ttE = np.ndarray((P.l,), dtype=np.int)

ms = h5s.create_simple((P.N,))
ms_e = h5s.create_simple((1,))

z = - np.finfo(np.double).max

tb, te = task(rank, P.l)

for it in xrange(P.max_iter):

    # Compute responsibilities
    for i in xrange(tb, te):

        Ss.select_hyperslab((i, 0), (1, P.N))
        S.id.read(ms, Ss, tS)
        #tS = S[i, :]

        As.select_hyperslab((i, 0), (1, P.N))
        A.id.read(ms, As, tAS)
        #tAS = A[i, :]
        tAS += tS

        Rs.select_hyperslab((i, 0), (1, P.N))
        R.id.read(ms, Rs, tRold)
        #tRold = R[i, :]

        tI = bn.nanargmax(tAS)
        tY = tAS[tI]
        tAS[tI] = z
        tY2 = np.max(tAS)

        tR = tS - tY
        tR[tI] = tS[tI] - tY2
        tR = (1 - P.damping) * tR + P.damping * tRold

        tdR[i - tb] = tR[i]
        tRp = np.maximum(tR, 0)
        tRp[i] = tdR[i - tb]

        R.id.write(ms, Rs, tR)
        #R[i, :] = tR

        Rps.select_hyperslab((i, 0), (1, P.N))
        Rp.id.write(ms, Rps, tRp)
        #Rp[i, :] = tRp

    comm.Barrier()

    # Compute availabilities
    for j in xrange(tb, te):

        As.select_hyperslab((0, j), (P.N, 1))
        A.id.read(ms, As, tAold)
        #tAold = A[:, j]

        Rps.select_hyperslab((0, j), (P.N, 1))
        Rp.id.read(ms, Rps, tRp)
        #tRp = Rp[:, j]

        tA = bn.nansum(tRp) - tRp
        tdA[j - tb] = tA[j]
        tA = np.minimum(tA, 0)
        tA[j] = tdA[j - tb]

        tA = (1 - P.damping) * tA + P.damping * tAold

        A.id.write(ms, As, tA)
        #A[:, j] = (1 - P.damping) * tA + P.damping * tAold

    K = 0

    for i in xrange(tb, te):
        n = 1 if ((tdA[i - tb] + tdR[i - tb]) > 0) else 0

        ttE[i - tb] = n
        K += n

    comm.Gather([ttE, MPI.INT], [tE, MPI.INT])
    comm.Bcast([tE, MPI.INT])
    e[:, it % P.conv_iter] = tE

    converged = False

    if it >= P.conv_iter:

        se = 0

        for i in xrange(tb, te):
            n = bn.nansum(e[i, :])
            if n == P.conv_iter or n == 0:
                se += 1

        K = comm.allreduce(K)
        se = comm.reduce(se)

        if rank == 0:

            print 'It %d K %d' % (it, K)

            converged = (se == P.N)

            if (converged is True) and (K > 0):
                if P.verbose is True:
                    print("Converged after %d iterations." % it)
                converged = True
            else:
                converged = False

        converged = comm.bcast(converged, root=0)

    if converged is True:
        break

if K > 0:

    I = np.nonzero(e[:, 0])[0]
    C = np.zeros((P.N,), dtype=np.int)
    tC = np.zeros((P.l,), dtype=np.int)

    for i in xrange(tb, te):

        Ss.select_hyperslab((i, 0), (1, P.N))
        S.id.read(ms, Ss, tS)

        tC[i - tb] = bn.nanargmax(tS[I])

    comm.Gather([tC, MPI.INT], [C, MPI.INT])

    if rank == 0:
        C[I] = np.arange(K)

    comm.Bcast([C, MPI.INT])

    for k in xrange(K):
        ii = np.where(C == k)[0]
        tN = ii.shape[0]

        tI = np.zeros((tN, ), dtype=np.float)
        ttI = np.zeros((tN, ), dtype=np.float)
        tttI = np.zeros((tN, ), dtype=np.float)

        j = rank
        while j < tN:
            for i in xrange(tN):

                jj = ii[j]
                ind = [(ii[i], jj)]

                Ss.select_elements(ind)
                S.id.read(ms_e, Ss, tdS)
                tttI[i] = tdS[0]

            ttI[j] = bn.nansum(tttI)
            j += NPROCS

        comm.Reduce([ttI, MPI.FLOAT], [tI, MPI.FLOAT])

        if rank == 0:
            I[k] = ii[bn.nanargmax(tI)]

    comm.Bcast([I, MPI.INT])

    for i in xrange(tb, te):
        Ss.select_hyperslab((i, 0), (1, P.N))
        S.id.read(ms, Ss, tS)

        tC[i - tb] = bn.nanargmax(tS[I])

    comm.Gather([tC, MPI.INT], [C, MPI.INT])

    if rank == 0:
        C[I] = np.arange(K)

else:
    if rank == 0:
        I = np.empty(())
        C = np.empty((P.N, ))
        C.fill(np.nan)

if rank == 0:
    os.remove(P.TMfn)
    t1 = dt.datetime.now()
    print I[:], C[:]
    print "APM time is %s" % (t1 - t0)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

Sf.close()
TMf.close()
