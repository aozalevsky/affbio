#!/usr/bin/python

#General modules
import time
import os
import sys
import psutil
import gc


gc.disable()

#NumPy for arrays
import numpy as np
import bottleneck as bn

#MPI parallelism
from mpi4py import MPI
#Get MPI info
comm = MPI.COMM_WORLD
NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
#Get number of processes
NPROCS = comm.size
#Get rank
rank = comm.rank

#H5PY for storage
import h5py
from h5py import h5s

from h5py import h5p, h5fd

dxpl = h5p.create(h5p.DATASET_XFER)
dxpl.set_dxpl_mpio(h5fd.MPIO_INDEPENDENT)


def task(rk, l):
    b = rk * l
    return (b, b + l)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


#Init storage for matrices
#Get file name
SSfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
SSf = h5py.File(SSfn, 'r', driver='mpio', comm=comm)
#Open table with data for clusterization
SS = SSf['cluster']
SSs = SS.id.get_space()

params = {
    'N': 0,
    'l': 0,
    'll': 0,
    'TMfn': '',
    'disk': False,
    'preference': 0.0,
    'conv_iter': 0,
    'max_iter': 0,
    'damping': 0.0,
    'verbose': False}

P = Bunch(params)

debug = False
#debug = True

if rank == 0:
    print 'Clusterizing matrix'
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

    N, N1 = SS.shape

    if N != N1:
        raise ValueError("S must be a square array (shape=%s)" % repr((N, N1)))
    else:
        P.N = N

    try:
        preference = SS.attrs['median']
        P.preference = preference
    except:
        raise ValueError('Unable to get median from cluster matrix')

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
    P.TMfn = 'aff_tmp'

    r = N % NPROCS
    N -= r
    l = N // NPROCS
    if r > 0:
        print 'Truncating matrix to NxN to fit on %d procs' % (NPROCS * l)
    P.N = N

    #Fit to memory
    MEM = psutil.phymem_usage().available / NPROCS_LOCAL
#    MEM = 500 * 10 ** 3
    tt = np.arange(1, dtype=np.float)
    ts = (sys.getsizeof(tt) + sys.getsizeof(tt[0]) * N) / 8  # Python give bits
    ts *= 8  # Allocate memory for e, tE, and ...
    MEM -= ts  # ----
    tl = MEM // ts  # Allocate memory for tS, tA, tR....
    cache = l
    if tl < l:
        P.disk = True
        try:
            cache = int(sys.argv[1])
            print sys.argv[1]
            assert cache < l
        except:
            cache = 1
            print 'Wrong cache settings, set cache to 1'

        print 'Cache size is %d of %d' % (cache, l)

    P.l = l
    P.ll = cache

P = comm.bcast(P)

N = P.N
l = P.l
ll = P.ll

ms = h5s.create_simple((ll, N))
ms_l = h5s.create_simple((N,))
ms_e = h5s.create_simple((1,))


tb, te = task(rank, l)

tS = np.ndarray((ll, N), dtype=np.float32)
tSl = np.ndarray((N,), dtype=np.float32)
tdS = np.ndarray((1,), dtype=np.float32)

disk = P.disk

if disk is True:

    TMLf = h5py.File(P.TMfn + '_' + str(rank) + '.hdf5', 'w')

    S = TMLf.create_dataset('S', (l, N), dtype='f4')
    Ss = S.id.get_space()

#Copy input data and
#place preference on diagonal
preference = P.preference
random_state = np.random.RandomState(0)
x = np.finfo(np.double).eps
y = np.finfo(np.double).tiny * 100
z = - np.finfo(np.double).max

for i in xrange(tb, te, ll):
    SSs.select_hyperslab((i, 0), (ll, N))
    SS.id.read(ms, SSs, tS, dxpl=dxpl)
    for il in xrange(ll):
        tS[il, i + il] = (preference * x + y) * random_state.randn() \
            + preference
    if disk is True:
        Ss.select_hyperslab((i - tb, 0), (ll, N))
        S.id.write(ms, Ss, tS)

if disk is True:
    R = TMLf.create_dataset('R', (l, N), dtype='f4')
    Rs = R.id.get_space()

tRold = np.zeros((ll, N), dtype=np.float32)
tR = np.zeros((ll, N), dtype=np.float32)
tdR = np.zeros((l,), dtype=np.float32)

#Shared storage
TMf = h5py.File(P.TMfn + '.hdf5', 'w', driver='mpio', comm=comm)
TMf.atomic = True

Rp = TMf.create_dataset('Rp', (N, N), dtype='f4')
Rps = Rp.id.get_space()

tRp = np.ndarray((ll, N), dtype=np.float32)
tRpa = np.ndarray((N, ll), dtype=np.float32)

A = TMf.create_dataset('A', (N, N), dtype='f4')
As = A.id.get_space()

tAS = np.ndarray((ll, N), dtype=np.float32)
tAold = np.ndarray((N, ll), dtype=np.float32)
tA = np.ndarray((N, ll), dtype=np.float32)
tdA = np.ndarray((l,), dtype=np.float32)

conv_iter = P.conv_iter
e = np.ndarray((N, conv_iter), dtype=np.int8)
tE = np.ndarray((N,), dtype=np.int8)
ttE = np.ndarray((l,), dtype=np.int8)

damping = P.damping
max_iter = P.max_iter

converged = False
ind = np.arange(ll)

for it in xrange(max_iter):
    if rank == 0:
        print '=' * 10 + 'It %d' % (it) + '=' * 10
        tit = time.time()
    # Compute responsibilities
    for i in xrange(tb, te, ll):
        if disk is True:
            il = i - tb
            Ss.select_hyperslab((il, 0), (ll, N))
            S.id.read(ms, Ss, tS)
        #tS = S[i, :]
            Rs.select_hyperslab((il, 0), (ll, N))
            R.id.read(ms, Rs, tRold)
        else:
            tRold = tR.copy()

        As.select_hyperslab((i, 0), (ll, N))
        A.id.read(ms, As, tAS)
        #tAS = A[i, :]
        tAS += tS
        #tRold = R[i, :]

        tI = bn.nanargmax(tAS, axis=1)
        tY = tAS[ind, tI]
        tAS[ind, tI[ind]] = z
        tY2 = bn.nanmax(tAS, axis=1)

        tR = tS - tY[:, np.newaxis]
        tR[ind, tI[ind]] = tS[ind, tI[ind]] - tY2[ind]
        tR = (1 - damping) * tR + damping * tRold

        tRp = np.maximum(tR, 0)

        for il in xrange(ll):
            tRp[il, i + il] = tR[il, i + il]
            tdR[i - tb + il] = tR[il, i + il]

        if disk is True:
            R.id.write(ms, Rs, tR)
            #R[i, :] = tR

        Rps.select_hyperslab((i, 0), (ll, N))
        Rp.id.write(ms, Rps, tRp, dxpl=dxpl)

            #Rp[i, :] = tRp
    if rank == 0:
        teit1 = time.time()
        print 'R T %s' % (teit1 - tit)

    comm.Barrier()

    # Compute availabilities
    for j in xrange(tb, te, ll):

        As.select_hyperslab((0, j), (N, ll))
        A.id.read(ms, As, tAold, dxpl=dxpl)
        #tAold = A[:, j]

        Rps.select_hyperslab((0, j), (N, ll))
        Rp.id.read(ms, Rps, tRpa, dxpl=dxpl)
        #tRp = Rp[:, j]

        tA = bn.nansum(tRpa, axis=0)[np.newaxis, :] - tRpa
        for jl in xrange(ll):
            tdA[j - tb + jl] = tA[j + jl, jl]

        tA = np.minimum(tA, 0)

        for jl in xrange(ll):
            tA[j + jl, jl] = tdA[j - tb + jl]

        tA *= (1 - damping)
        tA += damping * tAold

        for jl in xrange(ll):
            tdA[j - tb + jl] = tA[j + jl, jl]

        A.id.write(ms, As, tA, dxpl=dxpl)

    if rank == 0:
        teit2 = time.time()
        print 'A T %s' % (teit2 - teit1)

    ttE = np.array(((tdA + tdR) > 0), dtype=np.int8)

    comm.Gather([ttE, MPI.INT], [tE, MPI.INT])
    comm.Bcast([tE, MPI.INT])
    e[:, it % conv_iter] = tE
    K = bn.nansum(tE)

    if rank == 0:
        teit = time.time()
        print 'Total K %d T %s' % (K, teit - tit)

    if it >= conv_iter:

        if rank == 0:
            se = bn.nansum(e, axis=1)
            converged = (bn.nansum((se == conv_iter) + (se == 0)) == N)

            if (converged == np.bool_(True)) and (K > 0):
                if P.verbose is True:
                    print("Converged after %d iterations." % (it))
                converged = True
            else:
                converged = False

        converged = comm.bcast(converged, root=0)

    if converged is True:
        break

if K > 0:

    I = np.nonzero(e[:, 0])[0]
    C = np.zeros((N,), dtype=np.int8)
    tC = np.zeros((l,), dtype=np.int8)

    for i in xrange(ll):
        if disk is True:
            Ss.select_hyperslab((i, 0), (1, N))
            S.id.read(ms_l, Ss, tSl)
        else:
            tSl = tS[i]

        tC[i] = bn.nanargmax(tSl[I])

    comm.Gather([tC, MPI.INT], [C, MPI.INT])

    if rank == 0:
        C[I] = np.arange(K)

    comm.Bcast([C, MPI.INT])

    for k in xrange(K):
        ii = np.where(C == k)[0]
        tN = ii.shape[0]
        if tN > 0:
            tI = np.zeros((tN, ), dtype=np.float32)
            ttI = np.zeros((tN, ), dtype=np.float32)
            tttI = np.zeros((tN, ), dtype=np.float32)
            ms_k = h5s.create_simple((tN,))

            j = rank
            while j < tN:
                ind = [(ii[i], ii[j]) for i in xrange(tN)]
                SSs.select_elements(ind)
                SS.id.read(ms_k, SSs, tttI)

                ttI[j] = bn.nansum(tttI)
                j += NPROCS

            comm.Reduce([ttI, MPI.FLOAT], [tI, MPI.FLOAT])

            if rank == 0:
                I[k] = ii[bn.nanargmax(tI)]
        else:
            I[k] = ii[0]

    I.sort()
    comm.Bcast([I, MPI.INT])

    for i in xrange(ll):
        if disk is True:
            Ss.select_hyperslab((i, 0), (1, N))
            S.id.read(ms_l, Ss, tSl)
        else:
            tSl = tS[i]

        tC[i] = bn.nanargmax(tSl[I])

    comm.Gather([tC, MPI.INT], [C, MPI.INT])

    if rank == 0:
        C[I] = np.arange(K)

else:
    if rank == 0:
        I = np.zeros(())
        C = np.zeros((N, ))
        C.fill(np.nan)

#Cleanup
SSf.close()
TMf.close()
if disk is True:
    TMLf.close()
    os.remove(P.TMfn + '_' + str(rank) + '.hdf5')

if rank == 0:
    os.remove(P.TMfn + '.hdf5')
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
