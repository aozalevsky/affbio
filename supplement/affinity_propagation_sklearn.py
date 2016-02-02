from sklearn.cluster import AffinityPropagation as apn
#from affinity_propagation_dummy import affinity_propagation as apd

import datetime as dt

import h5py
import sys


#Init storage for matrices
#pdb = 'aff.list'
matf = sys.argv[1]
#HDF5 file
f = h5py.File(matf, 'r')
#Table for RMSD
#rmsd_matrix = f['rmsd']
#Table for clusterization
cl_matrix = f['tier1']['cluster']
pref = f['tier1']['cluster'].attrs['median']
#Reread structures by every process

t0 = dt.datetime.now()
af = apn(
    affinity="precomputed",
    verbose=True,
    convergence_iter=15,
    max_iter=2000,
    preference=pref,
    damping=0.95).fit(cl_matrix)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
print "Preference: %f" % af.get_params()['preference']
print 'APN: %d' % n_clusters_
t1 = dt.datetime.now()
print "APN time is %s" % (t1 - t0)
print cluster_centers_indices
#print labels
