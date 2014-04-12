#!/usr/bin/python


import time

import h5py
from h5py import h5s

from livestats import livestats
import numpy as np

debug = False
#debug = True

#Init cluster matrix
#Get file name
CMfn = 'aff_cluster_matrix.hdf5'
#Open matrix file in parallel mode
CMf = h5py.File(CMfn, 'r+', driver='sec2')
#Open table with data for clusterization
CM = CMf['cluster']
CMs = CM.id.get_space()

t0 = time.time()

if debug is True:
    import cProfile
    import pstats
    import StringIO
    pr = cProfile.Profile()
    pr.enable()

N, N1 = CM.shape

if N != N1:
    raise ValueError(
        "S must be a square array (shape=%s)" % repr(CM.shape))

med = livestats.LiveStats()
madd = np.vectorize(med.add)


ms = h5s.create_simple((N,))
tCM = np.empty((N,), dtype=np.float)

# Remove degeneracies
for i in xrange(N):
    print 'Processing row %d of %d' % (i, N)

    CMs.select_hyperslab((i, 0), (1, N))
    CM.id.read(ms, CMs, tCM)

    madd(tCM)

level, median = med.quantiles()[0]
t1 = time.time()

print 'Med', median
#print 'NP', np.median(CM)
print "Time is %s" % (t1 - t0)

ms = h5s.create_simple((1,))
tCM = np.array([median], dtype=np.float)

for i in xrange(N):
    print 'Processing row %d of %d' % (i, N)
    CMs.select_elements([(i, i)])
    CM.id.write(ms, CMs, tCM)

CM.attrs['preference'] = median

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

CMf.close()
