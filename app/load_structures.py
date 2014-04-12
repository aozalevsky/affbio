#!/usr/bin/python

import sys
import time
import prody
import numpy as np

from mpi4py import MPI

#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size
rank = comm.rank

import h5py
from h5py import h5s


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
    t0 = time.clock()

    if debug is True:
        import cProfile
        import pstats
        import StringIO
        pr = cProfile.Profile()
        pr.enable()


pdb_list = sys.argv[1:]
N = len(pdb_list)

r = N % NPROCS
if r != 0:
    r = NPROCS - r
    N = N + r
l = N / NPROCS

tb, te = task(NPROCS - 1 - rank, l)

if rank == 0:
    te = te - r

s = np.ndarray((3,), dtype=np.int)
if rank == 0:
    t = parse(pdb_list[0])
    s[0] = N
    s[1], s[2] = t.shape
comm.Bcast([s, MPI.INT])


#Init storage for matrices
Sfn = 'aff_struct.hdf5'
#HDF5 file
Sf = h5py.File(Sfn, 'w', driver='mpio', comm=comm)
Sf.atomic = True
#Table for RMSD
S = Sf.create_dataset(
    'struct',
    (s[0], s[1], s[2]),
    dtype=np.float,
    chunks=(1, s[1], s[2]))
Ss = S.id.get_space()

ms = h5s.create_simple((te - tb, s[1], s[2]))

tS = np.array([parse(pdb_list[i]) for i in xrange(tb, te)])
Ss.select_hyperslab((tb, 0, 0), (te - tb, s[1], s[2]))
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
        rd = np.random.randint(N)
        S[te + i] = S[rd]
        pdb_list.append('dummy%d_%s' % (i, pdb_list[rd]))
    L[:] = pdb_list[:]

    print "Structure reading time is %f" % (time.clock() - t0)

    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    Sf.close()
