#!/usr/bin/python

import logging
import sys
import os
import datetime as dt

import prody
import numpy as np

from mpi4py import MPI


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


def parse(i):
    """Parse PDB files"""
    return prody.parsePDB(i)

pdb_f = 'aff.list'

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
        pdb_list.dump(pdb_f)

    t = dt.datetime.now() - t0
    logging.info("Structure reading time is %s" % t)
    t = dt.datetime.now()


comm.Barrier()
