#!/usr/bin/python

#General modules
import time
import os

#NumPy for arrays
import numpy as np
import bottleneck as bn

#MPI parallelism
from mpi4py import MPI
#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

#H5PY for storage
import h5py
from h5py import h5s

import multiprocessing

NCORES = multiprocessing.cpu_count()
NCORES = NCORES / NPROCS


def task(rk, l):
    b = rk * l
    return (b, b + l)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


#Init storage for matrices
#Get file name
Sfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
Sf = h5py.File(Sfn, 'r', driver='mpio', comm=comm)
Sf.atomic = True
#Open table with data for clusterization
S = Sf['cluster']

params = {
    'N': 0,
    'l': 0,
    'll': 0,
    'TMfn': '',
    'preference': 0.0,
    'conv_iter': 0,
    'max_iter': 0,
    'damping': 0.0,
    'verbose': False}

P = Bunch(params)

debug = False
#debug = True

if rank == 0:
    t0 = time.time()
#    import cProfile, pstats, StringIO
#    pr = cProfile.Profile()
#    pr.enable()

    if debug is True:
        import cProfile
        import pstats
        import StringIO
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
    #P.TMfn = pj('/tmp', 'tmp.hdf5')
    P.TMfn = 'aff_tmp.hdf5'

    r = N % (NPROCS * NCORES)
    N -= r
    l = N // NPROCS
    if r > 0:
        print 'Truncating matrix to NxN to fit on %d procs' % (NPROCS * l)
    P.N = N
    P.l = l
    P.ll = NCORES

P = comm.bcast(P)

Ss = S.id.get_space()
tS = np.ndarray((P.ll, P.N), dtype=np.float)
tSl = np.ndarray((P.N,), dtype=np.float)
tdS = np.ndarray((1,), dtype=np.float)

TMf = h5py.File(P.TMfn, 'w', driver='mpio', comm=comm)
TMf.atomic = True

R = TMf.create_dataset('R', (P.N, P.N), dtype=np.float)
Rs = R.id.get_space()
tRold = np.ndarray((P.ll, P.N), dtype=np.float)
tR = np.ndarray((P.ll, P.N), dtype=np.float)
tdR = np.ndarray((P.l,), dtype=np.float)

Rp = TMf.create_dataset('Rp', (P.N, P.N), dtype=np.float)
#Rp = TMf.create_dataset('Rp', (P.N, P.N), dtype=np.float, chunks=(P.N, 1))
#Rp = TMf.create_dataset('Rp', (P.N, P.N), dtype=np.float, chunks=(1, P.N))
#Rp = TMf.create_dataset('Rp', (P.N, P.N), dtype=np.float, chunks=(100, 100))

Rps = Rp.id.get_space()
tRp = np.ndarray((P.ll, P.N), dtype=np.float)
tRpa = np.ndarray((P.N, P.ll), dtype=np.float)

#A = TMf.create_dataset('A', (P.N, P.N), dtype=np.float, chunks=(100, 100))
#A = TMf.create_dataset('A', (P.N, P.N), dtype=np.float, chunks=(1, P.N))
A = TMf.create_dataset('A', (P.N, P.N), dtype=np.float)
As = A.id.get_space()

tAold = np.ndarray((P.ll, P.N), dtype=np.float)
tAS = np.ndarray((P.ll, P.N), dtype=np.float)
tdA = np.ndarray((P.l,), dtype=np.float)

tA = np.ndarray((P.N, P.ll), dtype=np.float)
tAolda = np.ndarray((P.N, P.ll), dtype=np.float)

e = np.ndarray((P.N, P.conv_iter), dtype=np.int)
tE = np.ndarray((P.N,), dtype=np.int)
ttE = np.ndarray((P.l,), dtype=np.int)

ms = h5s.create_simple((P.ll, P.N))

ms_l = h5s.create_simple((P.N,))
ms_e = h5s.create_simple((1,))

z = - np.finfo(np.double).max

tb, te = task(rank, P.l)
#P.ll = P.l // NCORES

for it in xrange(P.max_iter):
    if rank == 0:
        tit = time.time()

    # Compute responsibilities
    for i in xrange(tb, te, P.ll):
        Ss.select_hyperslab((i, 0), (P.ll, P.N))
        S.id.read(ms, Ss, tS)
        #tS = S[i, :]

        As.select_hyperslab((i, 0), (P.ll, P.N))
        A.id.read(ms, As, tAS)
        #tAS = A[i, :]
        tAS += tS

        Rs.select_hyperslab((i, 0), (P.ll, P.N))
        R.id.read(ms, Rs, tRold)
        #tRold = R[i, :]

        for ii in xrange(P.ll):

            tI = np.argmax(tAS[ii])
            tY = tAS[ii][tI]
            tAS[ii][tI] = z
            tY2 = np.max(tAS[ii])

            tR[ii] = tS[ii] - tY
            tR[ii][tI] = tS[ii][tI] - tY2
            tR[ii] = (1 - P.damping) * tR[ii] + P.damping * tRold[ii]

            tdR[i - tb + ii] = tR[ii][i + ii]
            tRp[ii] = np.maximum(tR[ii], 0)
            tRp[ii][i + ii] = tdR[i - tb + ii]

        R.id.write(ms, Rs, tR)
        #R[i, :] = tR

        Rps.select_hyperslab((i, 0), (P.ll, P.N))
        Rp.id.write(ms, Rps, tRp)
        #Rp[i, :] = tRp

    comm.Barrier()

    # Compute availabilities
    for j in xrange(tb, te, P.ll):

        As.select_hyperslab((0, j), (P.N, P.ll))
        A.id.read(ms, As, tAolda)
        #tAold = A[:, j]

        Rps.select_hyperslab((0, j), (P.N, P.ll))
        Rp.id.read(ms, Rps, tRpa)
        #tRp = Rp[:, j]

        for jj in xrange(P.ll):
            tA[:, jj] = np.sum(tRpa[:, jj]) - tRpa[:, jj]

            tdA[j - tb + jj] = tA[j + jj, jj]
            tA[:, jj] = np.minimum(tA[:, jj], 0)
            tA[j + jj, jj] = tdA[j - tb + jj]

            tA[:, jj] = (1 - P.damping) * tA[:, jj] + P.damping * tAolda[:, jj]

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
            n = np.sum(e[i, :])
            if n == P.conv_iter or n == 0:
                se += 1

        K = comm.allreduce(K)
        se = comm.reduce(se)

        if rank == 0:

            converged = (se == P.N)

            if (converged is True) and (K > 0):
                if P.verbose is True:
                    print("Converged after %d iterations." % it)
                converged = True
            else:
                converged = False

        converged = comm.bcast(converged, root=0)

    if rank == 0:
        teit = time.time()

        print 'It %d K %d T %s' % (it, K, teit - tit)

    if converged is True:
        break

if K > 0:

    I = np.nonzero(e[:, 0])[0]
    C = np.zeros((P.N,), dtype=np.int)
    tC = np.zeros((P.l,), dtype=np.int)


    for i in xrange(tb, te):

        Ss.select_hyperslab((i, 0), (1, P.N))
        S.id.read(ms_l, Ss, tSl)

        tC[i - tb] = np.argmax(tSl[I])

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
        ms_k = h5s.create_simple((tN,))

        j = rank
        while j < tN:
            ind = [(ii[i], ii[j]) for i in xrange(tN)]

            Ss.select_elements(ind)
            S.id.read(ms_k, Ss, tttI)

            ttI[j] = np.sum(tttI)
            j += NPROCS

        comm.Reduce([ttI, MPI.FLOAT], [tI, MPI.FLOAT])

        if rank == 0:
            I[k] = ii[np.argmax(tI)]

    I.sort()
    comm.Bcast([I, MPI.INT])

    for i in xrange(tb, te):
        Ss.select_hyperslab((i, 0), (1, P.N))
        S.id.read(ms_l, Ss, tSl)

        tC[i - tb] = np.argmax(tSl[I])

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
    t1 = time.time()
    print I[:], C[:]
    I.dump('aff.centers')
    C.dump('aff.labels')
    print "APM time is %s" % (t1 - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

#Cleanup
Sf.close()
TMf.close()
