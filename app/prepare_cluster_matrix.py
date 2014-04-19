#!/usr/bin/python

#General modules
import time

#NumPy for arrays
import numpy as np

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


if rank == 0:
    print 'Preparing cluster matrix'
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


def calc_chunk(l, tRM, tCM):
    ttCM = tCM * random_state.randn(l, l)
    ttCM += tRM
    return ttCM


def calc_chunk_diag(l, tRM, tCM):
    tCM += tCM.transpose()
    tRM += tRM.transpose()
    ttCM = calc_chunk(l, tRM, tCM)
    return ttCM


#Init RMSD matrix
#Get file name
RMfn = 'aff_rmsd_matrix.hdf5'
#Open matrix file in parallel mode
RMf = h5py.File(RMfn, 'r', driver='mpio', comm=comm)
RMf.atomic = True
#Open table with data for clusterization
RM = RMf['rmsd']
RMs = RM.id.get_space()

N = RM.len()
l = N // NPROCS

if rank == 0:
    N, N1 = RM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(RM.shape))

    print RM.attrs['chunk']
    if RM.attrs['chunk'] % l > 0:
        raise ValueError(
            "Wrong chunk size in RMSD matrix")

l = comm.bcast(l, root=0)

#Init cluster matrix
#Get file name
CMfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
CMf = h5py.File(CMfn, 'w', driver='mpio', comm=comm)
CMf.atomic = True
#Open table with data for clusterization
CM = CMf.create_dataset(
    'cluster',
    (N, N),
    dtype=np.float,
    chunks=(l, l))
CM.attrs['chunk'] = l
CMs = CM.id.get_space()

random_state = np.random.RandomState(0)
x = np.finfo(np.float).eps
y = np.finfo(np.float).tiny * 100

#Partiotioning
lN = (NPROCS + 1) * NPROCS / 2

m = lN // NPROCS
mr = lN % NPROCS

if mr > 0:
    m = m + 1 if rank % 2 == 0 else m


#Init calculations
tRM = np.zeros((l, l), dtype=np.float)
tCM = np.zeros((l, l), dtype=np.float)
ttCM = np.zeros((l, l), dtype=np.float)
ms = h5s.create_simple((l, l))

i, j = rank, rank

for c in xrange(m):
    if rank == 0:
        tit = time.time()
    RMs.select_hyperslab((i * l, j * l), (l, l))
    RM.id.read(ms, RMs, tRM)

    #tRM = -1 * tRM ** 2
    tRM **= 2
    tRM *= -1
    tCM = tRM * x + y

    try:
        assert i != j

        ttCM = calc_chunk(l, tRM, tCM)
        ttCM.transpose()
        CMs.select_hyperslab((j * l, i * l), (l, l))
        CM.id.write(ms, CMs, ttCM)

        ttCM = calc_chunk(l, tRM, tCM)

    except AssertionError:
        ttCM = calc_chunk_diag(l, tRM, tCM)

    CMs.select_hyperslab((i * l, j * l), (l, l))
    CM.id.write(ms, CMs, ttCM)

    if rank == 0:
        teit = time.time()
        print "Step %d of %d T %s" % (c, m, teit - tit)

    if (rank - c) > 0:
        j = j - 1
    elif (rank - c) == 0:
        i = NPROCS - rank - 1
    else:
        j = j + 1


#Wait for all processes
comm.Barrier()

if rank == 0:
    print "Time is %s" % (time.time() - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

RMf.close()
CMf.close()
