#!/usr/bin/python

import logging
import os
import os.path as osp
import datetime as dt
import numpy as np

import h5py

#Init logging
logging.basicConfig(filename='aff.log', level=logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#Add logging to console
console = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(message)s')
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#Init storage for matrices
pdb = 'aff.list'
matf = 'aff.raw'
#HDF5 file
f = h5py.File(matf, 'r')
#Table for RMSD
#rmsd_matrix = f['rmsd']
#Table for clusterization
cl_matrix = f['cluster']
#Reread structures by every process
pdb_list = np.load(pdb)
ln = len(pdb_list)

t = dt.datetime.now()
logging.info("Starting clusterization")
#Now clusterization routine
from sklearn.cluster import AffinityPropagation as AP
##Convert all values to float
af = AP(
    affinity="precomputed",
    max_iter=2000,
    convergence_iter=50).fit(cl_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

with open('aff.res', 'w') as out:
    for i in range(ln):
        out.write("%s\t%d\n" % (pdb_list[i], labels[i]))
with open('aff.ref', 'w') as out:
    for i in cluster_centers_indices:
        out.write('%s\n' % pdb_list[i])

logging.info("Info about run:")
#logging.info('%s: %s' % ("Average distance", d_avg))
logging.info('%s: %d' % ("Number of clusters", n_clusters_))
logging.info('%s\t%s' % ("Center of cluster", "Number of cluster"))
for i in cluster_centers_indices:
    logging.info('%s\t%d' % (pdb_list[i], labels[i]))
t = dt.datetime.now() - t
logging.info("Clusterization time is %s" % t)


#Cleanup
#Close matrix file
#f.close()
#Some logging info
logging.info("Run finished successfully")

#def create_dirs(n, cl):
#    import shutil as sh
#    test = np.array([osp.exists(str(i)) for i in range(n)], dtype=np.bool)
#    if np.any(test):
#        logging.warning("Removing old directories")
#        try:
#            [sh.rmtree(str(i)) for i in range(999)]
#        except OSError:
#            pass
#    [os.mkdir(str(i)) for i in range(n)]
#    for k in cl.keys():
#        sh.copy(k, osp.join(str(cl[k]), k))
#
#create_dirs(n_clusters_, cl)
#
