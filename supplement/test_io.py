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
dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)


def task(rk, l):
    b = rk * l
    return (b, b + l)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def factors(n):
    return sorted(set(reduce(list.__add__, ([i, n // i] for i in range(
        1, int(n ** 0.5) + 1) if n % i == 0))))

#Init storage for matrices
#Get file name
SSfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
SSf = h5py.File(SSfn, 'r', driver='mpio', comm=comm)
#Open table with data for clusterization
SS = SSf['cluster']
SSs = SS.id.get_space()

params = {
    'c': [],
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
    print 'Testing cache size'
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
    MEM = 500 * 10 ** 3
    tt = np.arange(1, dtype=np.float)
    ts = (sys.getsizeof(tt) + sys.getsizeof(tt[0]) * N) / 8  # Python give bits
    ts *= 8  # Allocate memory for e, tE, and ...
    MEM -= ts  # ----
    tl = MEM // ts  # Allocate memory for tS, tA, tR....
    if tl < l:
        P.disk = True
        while l % tl > 0:
            tl -= 1

        for i in factors(l):
            if i < l:
                P.c.append(i)

        P.l = l
        P.ll = P.c[len(P.c) // 2]
    else:
        print 'Dataset fits memory'
        comm.Abort()

P = comm.bcast(P)


N = P.N
l = P.l
ll = P.ll

tb, te = task(rank, l)
disk = P.disk

damping = P.damping

ms_l = h5s.create_simple((N,))
tSl = np.ndarray((N,), dtype=np.float)

ms = h5s.create_simple((ll, N))
tS = np.ndarray((ll, N), dtype=np.float)
tdS = np.ndarray((1,), dtype=np.float)


TMLf = h5py.File(P.TMfn + '_' + str(rank) + '.hdf5', 'w')

S = TMLf.create_dataset(
    'S', (l, N), dtype=np.float)
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
        Ss.select_hyperslab((i - tb, 0), (ll, N))
        S.id.write(ms, Ss, tS)

R = TMLf.create_dataset(
    'R', (l, N), dtype=np.float)
Rs = R.id.get_space()

tdR = np.zeros((l,), dtype=np.float)
tdA = np.ndarray((l,), dtype=np.float)

conv_iter = P.conv_iter
e = np.ndarray((N, conv_iter), dtype=np.int)
tE = np.ndarray((N,), dtype=np.int)
ttE = np.ndarray((l,), dtype=np.int)

times = []


it = 0
for ll in P.c:
    ms = h5s.create_simple((ll, N))
    tS = np.zeros((ll, N), dtype=np.float)

    tRold = np.zeros((ll, N), dtype=np.float)
    tR = np.zeros((ll, N), dtype=np.float)

    #Shared storage
    TMf = h5py.File(P.TMfn + '.hdf5', 'w', driver='mpio', comm=comm)
    TMf.atomic = True

    Rp = TMf.create_dataset('Rp', (N, N), dtype=np.float, fillvalue=0)
    Rps = Rp.id.get_space()

    tRp = np.zeros((ll, N), dtype=np.float)
    tRpa = np.zeros((N, ll), dtype=np.float)

    A = TMf.create_dataset('A', (N, N), dtype=np.float, fillvalue=0)
    As = A.id.get_space()

    tAS = np.zeros((ll, N), dtype=np.float)
    tAold = np.zeros((N, ll), dtype=np.float)
    tA = np.zeros((N, ll), dtype=np.float)

    ind = np.arange(ll)

    if rank == 0:
        print '=' * 10 + 'It %d' % (it) + '=' * 10
        print 'Cache size is %d of %d' % (ll, P.l)
        tit = time.time()
    # Compute responsibilities
    for i in xrange(tb, te, ll):
        il = i - tb
        Ss.select_hyperslab((il, 0), (ll, N))
        S.id.read(ms, Ss, tS)
        Rs.select_hyperslab((il, 0), (ll, N))
        R.id.read(ms, Rs, tRold)

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
        tAold *= damping
        tA += tAold

    for jl in xrange(ll):
            tdA[j - tb + jl] = tA[j + jl, jl]

    A.id.write(ms, As, tA, dxpl=dxpl)

    if rank == 0:
        teit2 = time.time()
        print 'A T %s' % (teit2 - teit1)

    ttE = np.array(((tdA + tdR) > 0), dtype=np.int)

    comm.Gather([ttE, MPI.INT], [tE, MPI.INT])
    comm.Bcast([tE, MPI.INT])
    e[:, it % conv_iter] = tE
    K = bn.nansum(tE)

    if rank == 0:
        teit = time.time()
        tot = teit - tit
        times.append(tot)
        print 'Total K %d T %s' % (K, tot)

    TMf.close()
    it += 1
    comm.Barrier()


#Cleanup
SSf.close()
TMLf.close()
os.remove(P.TMfn + '_' + str(rank) + '.hdf5')


if rank == 0:
    #os.remove(P.TMfn + '.hdf5')
    t1 = time.time()
    tm = np.argmin(times)
    print 'Fastest: cache: %d of %d T: %s' % (P.c[tm], P.l, times[tm])
    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
