#!/usr/bin/python

import os
import os.path as osp
import sys
import numpy as np
import logging
import prody
import pp

NPROCS = 16
NJOBS = NPROCS * 4
matf = 'rmsd_matrix.raw'


#Init logging
with open('aff.log', 'w'):
    pass
logging.basicConfig(filename='aff.log', level=logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

#Add logging to console
console = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(message)s')
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

#Change logging level for pp module
p_log = logging.getLogger("pp")
p_log.setLevel(logging.ERROR)

pr_log = logging.getLogger("prody")
pr_log.setLevel(logging.ERROR)


def call_res(i):
    """We have to call results after pp"""
    if i:
        return i()
    else:
        return 0


def symmetrize(a):
    """Return symmetric matrice"""
    return a + a.T - np.diag(a.diagonal())


def parse(i):
#    from prody import *
    return prody.parsePDB(i)


def calc(i, j):
    r = prody.calcRMSD(i, j)
    return (r, -1 * r ** 2)


#Get CWD
cwd = os.getcwd()

#
pool = pp.Server(NPROCS)

#Get pdb files
pdb_list = sys.argv[1:]
ln = len(pdb_list)
#pdb_struct = np.memmap(matf, dtype=object, shape=(ln,), mode='write')
#pdb_struct = np.memmap(matf, dtype=object, shape=(ln,), mode='write')
pdb_struct = np.ndarray((ln,), dtype='object')

c = 0
nj = 0
while c < ln:

    if NJOBS < (ln - c):
        nj = NJOBS
    else:
        nj = ln - c

    for m in range(nj):
        pdb_struct[c + m] = pool.submit(
            parse, (pdb_list[c + m], ), (), ('prody',))
    pool.wait()

    c += nj
    sys.stdout.write('%d structures of %d loaded\r' % (c, ln))

res = np.frompyfunc(call_res, 1, 1)
pdb_struct = res(pdb_struct)

#Matrix for storing pool of processes
#matrix_f = h5py.File(matf, 'w')
#rmsd_matrix = matrix_f.create_dataset("rmsd", (ln, ln), dtype='float')
o_matrix = np.memmap('aff.tmp.o', dtype='object', shape=(ln, ln), mode='write')
r_matrix = np.memmap('aff.tmp.r', dtype='float', shape=(ln, ln), mode='write')
rmsd_matrix = np.memmap(matf, dtype='float', shape=(ln, ln), mode='write')
#dmatrix = matrix_f.create_dataset("dmatrix", (ln, ln), dtype='float')
#o_matrix = np.ndarray((NJOBS,), dtype='object')
#matrix = np.memmap(matf, dtype=np.dtype(object), shape=(ln, ln), mode='write')
#r_matrix = np.ndarray((NJOBS,), dtype='float32')

c = 0
b = (ln - 1) / 2.0 * ln
#for i in range(ln):
##    print 'Row %d of %d' % (i, ln)
#    j = 0
#    while j < i:
#        if NJOBS < (i - j):
#            nj = NJOBS
#        else:
#            nj = i - j
#        for m in range(nj):
#            o_matrix[m] = pool.submit(
#                calc, (pdb_struct[i], pdb_struct[j]), (), ('prody', ))
#        pool.wait()
#        r_matrix = res(o_matrix)
#        for m in range(nj):
#            matrix[i][j + m] = r_matrix[m]
#            matrix[j + m][i] = r_matrix[m]
#            dmatrix[i][j + m] = -1 * r_matrix[m] ** 2
#            dmatrix[j + m][i] = -1 * r_matrix[m] ** 2
#
#        j += nj
#        c += nj
#        sys.stdout.write('%d00 distances of %d calculated\r' % (c / 100 , b))

for i in range(ln):
    print 'Row %d of %d' % (i, ln)
    for j in range(i):
        o_matrix[i][j] = pool.submit(
            calc, (pdb_struct[i], pdb_struct[j]), (), ('prody', ))


pool.wait()
pool.destroy()

r_matrix = symmetrize(res(o_matrix))
rmsd_matrix = r_matrix
r_matrix = -1 * r_matrix ** 2

#S = np.memmap('aff.tmp', dtype='object', shape=(ln, ln), mode='write')
#matrix = res_matrix.astype(np.float32)


logging.info("RMSD matrix have been calculated")
#logging.info("Writing RMSD matrix to file")

logging.info("RMSD matrix have been successfully written to %s" % matf)

logging.info("Starting clusterization")

#Now clusterization routine
from sklearn.cluster import AffinityPropagation as AP

#Prepare values
S = -1 * r_matrix ** 2
#Calc average for logfile
d_avg = np.average(S)


#Convert all values to float
af = AP(
    affinity="precomputed", max_iter=2000, convergence_iter=50).fit(S)
af.labels = pdb_list
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
labels_true = pdb_list
n_clusters_ = len(cluster_centers_indices)

cl = {}
for i, v in enumerate(pdb_list):
    cl[v] = labels[i]

with open('aff.res', 'w') as f:
    for k in sorted(cl.keys()):
        f.write("%s\t%d\n" % (k, cl[k]))


logging.info("Info about run:")
logging.info('%s: %s' % ("Average distance", d_avg))
logging.info('%s: %d' % ("Number of clusters", n_clusters_))
logging.info('%s\t%s' % ("Center of cluster", "Number of cluster"))
for i in cluster_centers_indices:
    logging.info('%s\t%d' % (pdb_list[i], cl[pdb_list[i]]))

with open('aff.ref', 'w') as f:
    for i in cluster_centers_indices:
        f.write('%s\n' % pdb_list[i])


def create_dirs(n, cl):
    import shutil as sh
    test = np.array([osp.exists(str(i)) for i in range(n)], dtype=np.bool)
    if np.any(test):
        logging.warning("Removing old directories")
        try:
            [sh.rmtree(str(i)) for i in range(999)]
        except OSError:
            pass
    [os.mkdir(str(i)) for i in range(n)]
    for k in cl.keys():
        sh.copy(k, osp.join(str(cl[k]), k))

create_dirs(n_clusters_, cl)

logging.info("Run finished successfully")
os.remove('aff.raw')
