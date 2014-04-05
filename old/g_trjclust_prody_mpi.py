#!/usr/bin/python

import logging
import sys
import os
import os.path as osp
import datetime as dt
import numpy as np
import prody

from mpi4py import MPI
import h5py



#Get MPI info
comm = MPI.COMM_WORLD
#Get number of processes
NPROCS = comm.size


#Init logging
if comm.rank == 0:
    with open('aff.log', 'w'):
        pass
    #Get current time
    t0 = dt.datetime.now()
    t = t0
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

#Change logging level for prody module
pr_log = logging.getLogger("prody")
pr_log.setLevel(logging.ERROR)


def call_res(i):
    """We have to call results after pp"""
    if i:
        return i()
    else:
        return 0


def parse(i):
    """Parse PDB files"""
    return prody.parsePDB(i)


def calc(i, j):
    """calculate RMSD"""
    return prody.calcRMSD(i, j)


#Get pdb files
pdb_list = np.array(sys.argv[1:])
ln = len(pdb_list)
n = ln / NPROCS
vln = ln
vn = n
if n == 0:
    vn = ln
elif n > 0 and (n * NPROCS) != ln:
    vn = n + 1
    vln = vn * NPROCS


#File with imported structure objects
stf = 'aff.struct'

stf_t = stf + '.' + str(comm.rank)

if (n == 0 and comm.rank == 0) or n > 0:
    pdb_struct = np.ndarray((vn,), dtype='object')
    for i in range(vn):
        ind = vn * comm.rank + i
        if ind < ln:
            pdb_struct[i] = parse(pdb_list[ind])
    #Dump all structures to file
    pdb_struct.dump(stf_t)

#Wait for all processes
comm.Barrier()

if comm.rank == 0:

    if n == 0:
        pdb_struct[:] = np.load(stf + '.' + str(0))
        os.remove(stf + '.' + str(0))

    elif n > 0:
        pdb_struct = np.ndarray((vln, ), dtype='object')
        for i in range(NPROCS):
            pdb_struct[vn * i: vn * (i + 1)] = np.load(stf + '.' + str(i))
            os.remove(stf + '.' + str(i))
        pdb_struct.dump(stf)

    t = dt.datetime.now() - t0
    logging.info("Structure reading time is %s" % t)
    t = dt.datetime.now()


comm.Barrier()
#Now RMSD calculation
#Init storage for matrices
matf = 'matrix.raw'
#HDF5 file
f = h5py.File(matf, 'w', driver='mpio', comm=comm)
#Table for RMSD
rmsd_matrix = f.create_dataset('rmsd', (ln, ln))
#Table for clusterization
cl_matrix = f.create_dataset('cluster', (ln, ln))
#Reread structures by every process
pdb_struct = np.load(stf, mmap_mode='r')


i = comm.rank
while i < ln:
    if comm.rank == 0:
        logging.info('Processing row %d of %d' % (i, ln))
    for j in range(i):
        rmsd = calc(pdb_struct[i], pdb_struct[j])
        n_rmsd = -1 * rmsd ** 2
        rmsd_matrix[i][j] = rmsd
        rmsd_matrix[j][i] = rmsd
        cl_matrix[i][j] = n_rmsd
        cl_matrix[j][i] = n_rmsd
    i += NPROCS

del pdb_struct
#Wait for all processes
comm.Barrier()

if comm.rank == 0:
    logging.info("RMSD matrix have been calculated")
    logging.info("RMSD matrix have been successfully written to %s" % matf)
    t = dt.datetime.now() - t
    logging.info("RMSD calculation time is %s" % t)
    t = dt.datetime.now()

    logging.info("Starting clusterization")
    #Now clusterization routine
    from sklearn.cluster import AffinityPropagation as AP
    #
    #S = np.memmap('aff.raw', dtype='float32', shape=(ln, ln), mode='write')
    ##Prepare values
    #S = -1 * matrix ** 2
    ##Calc average for logfile
    #d_avg = np.mean(cl_matrix)
    #
    #
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
f.close()
#Some logging info
if comm.rank == 0:
    logging.info("Run finished successfully")
    t = dt.datetime.now() - t0
    logging.info("Total run time is %s" % t)
comm.Barrier()

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
