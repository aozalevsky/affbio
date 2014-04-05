from sklearn.cluster import AffinityPropagation as apn
#from affinity_propagation_dummy import affinity_propagation as apd

import datetime as dt

import h5py
#Init storage for matrices
pdb = 'aff.list'
matf = 'aff_cluster_matrix.hdf5'
#HDF5 file
f = h5py.File(matf, 'r')
#Table for RMSD
#rmsd_matrix = f['rmsd']
#Table for clusterization
cl_matrix = f['cluster']
#Reread structures by every process

t0 = dt.datetime.now()
af = apn(
    affinity="precomputed",
    verbose=True,
#    preference=-0.0122545,
    convergence_iter=15,
    max_iter=2000,
    damping=0.9).fit(cl_matrix)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
print 'APN: %d' % n_clusters_
t1 = dt.datetime.now()
print cluster_centers_indices, labels
print "APN time is %s" % (t1 - t0)
