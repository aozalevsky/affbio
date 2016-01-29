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

#Livestats for median
#from livestats import livestats
import pyximport
pyximport.install()
import lvc_double

print 'Calculating median'

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

#Init cluster matrix
#Get file name
CMfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
CMf = h5py.File(CMfn, 'r+', driver='sec2')
#CMf.atomic = True
#Open table with data for clusterization
CM = CMf['cluster']
CMs = CM.id.get_space()

N = CM.len()
l = CM.attrs['chunk']
m = N // l
m2 = m ** 2

N, N1 = CM.shape

if N != N1:
    raise ValueError(
        "S must be a square array (shape=%s)" % repr(CM.shape))

if l <= 0:
    raise ValueError(
        "Wrong chunk size in RMSD matrix")

#Init calculations
#med = livestats.LiveStats()
med = lvc_double.Quantile(0.5)
madd = np.vectorize(med.add)

tCM = np.zeros((N,), dtype=np.float)
ms = h5s.create_simple((N,))

c = 0
for i in xrange(N):
    #for j in xrange(m):
        #print 'Processing chunk %d of %d' % (c, m2)
        #CMs.select_hyperslab((i * l, j * l), (l, l))
        CMs.select_hyperslab((i, 0), (1, N))
        CM.id.read(ms, CMs, tCM)
        med.add(tCM)
        #madd(tCM)
#       for x in np.nditer(tCM):
#           med.add(x)
        c += 1


#level, median = med.quantiles()[0]
median = med.quantile()
t1 = time.time()
tdCM = np.array([median], dtype=np.float)
ms = h5s.create_simple((1, 1))

print 'Med', median
#print 'NP', np.median(CM)

CM.attrs['median'] = median
#CM.attrs['preference'] = np.median(CM)
print "Time is %s" % (time.time() - t0)

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

CMf.close()
