#!/usr/bin/python

#General modules
import sys
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

#Multiprocessing parallellism
import multiprocessing
import joblib
NCORES = multiprocessing.cpu_count()

#H5PY for storage
import h5py
from h5py import h5s

#pyRMSD for calculations
import prody


def task(rk, ln):
    b = rk * ln
    return (b, b + ln)


def parse(i):
    """Parse PDB files"""
    ps = prody.parsePDB(i)
    pc = ps.getCoords()
    return pc

debug = False
#debug = True

#Init logging
if rank == 0:
    #Get current time
    t0 = time.time()

    if debug is True:
        import cProfile
        import pstats
        import StringIO
        pr = cProfile.Profile()
        pr.enable()


pdb_list = sys.argv[1:]
N = len(pdb_list)

r = N % (NPROCS * NCORES)
if r != 0:
    r = (NPROCS * NCORES) - r
    N = N + r
l = N / NPROCS

tb, te = task(NPROCS - 1 - rank, l)

if rank == 0:
    te = te - r

na = 0  # Number of atoms
nc = 3  # Number of atom coordinates
if rank == 0:
    t = parse(pdb_list[0])
    na = t.shape[0]
na = comm.bcast(na)


#Init storage for matrices
Sfn = 'aff_struct.hdf5'
#HDF5 file
Sf = h5py.File(Sfn, 'w', driver='mpio', comm=comm)
Sf.atomic = True
#Table for RMSD
S = Sf.create_dataset(
    'struct',
    (N, na, nc),
    dtype=np.float,
    chunks=(1, na, nc))
Ss = S.id.get_space()
tS = np.ndarray((l, na, nc), dtype=np.float32)
ms = h5s.create_simple((l, na, nc))
for i in xrange(tb, te):
    try:
        print 'Parsing %s' % pdb_list[i]
        tS[i - tb] = parse(pdb_list[i])
        print 'Parsed %s' % pdb_list[i]
    except:
        raise ValueError('Broken structure %s' % pdb_list[i])

Ss.select_hyperslab((tb, 0, 0), (l, na, nc))
S.id.write(ms, Ss, tS)

#Wait for all processes
comm.Barrier()

Sf.close()

if rank == 0:

    Sf = h5py.File(Sfn, 'a', driver='sec2')
    S = Sf['struct']

    vls = h5py.special_dtype(vlen=str)
    L = Sf.create_dataset(
        'labels',
        (N,),
        dtype=vls)

    for i in xrange(r):
        rd = np.random.randint(N - r)
        S[te + i] = S[rd]
        pdb_list.append('dummy%d_%s' % (i, pdb_list[rd]))
    L[:] = pdb_list[:]

    print "Structure reading time is %f" % (time.time() - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    Sf.close()
