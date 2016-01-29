#!/usr/bin/python

#General modules
import os
import re
import uuid
import subprocess

#NumPy for arrays
import numpy as np

#H5PY for storage
import h5py

from .AffRender import AffRender


def cluster_to_trj(
        Sfn,
        index=None,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    def copy_connects(src, dst):
        with open(src, 'r') as fin, open(dst, 'r') as fout:
            inpdb = np.array(fin.readlines())
            ind = np.array(
                map(lambda x: re.match('CONECT', x), inpdb),
                dtype=np.bool)
            con = inpdb[ind]

            outpdb = fout.readlines()
            endmdl = 'ENDMDL\n'
            endmdl_ind = outpdb.index(endmdl)
            outpdb.pop(endmdl_ind)
            outpdb.extend(con)
            outpdb.append(endmdl)

        with open(dst, 'w') as fout:
            fout.write(''.join(outpdb))

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2+')

    I = Sf['aff_labels'][:]

    L = Sf['labels'][:]

    TMbfn = str(uuid.uuid1())
    TMtrj = TMbfn + '.pdb'
    fout = open(TMtrj, 'w')

    ind = np.where(I == index)
    for j in L[ind]:
        with open(j, 'r') as fin:
            fout.write(fin.read())
    fout.close()

    return TMtrj


def render_b_factor(
        Sfn,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    def copy_connects(src, dst):
        with open(src, 'r') as fin, open(dst, 'r') as fout:
            inpdb = np.array(fin.readlines())
            ind = np.array(
                map(lambda x: re.match('CONECT', x), inpdb),
                dtype=np.bool)
            con = inpdb[ind]

            outpdb = fout.readlines()
            endmdl = 'ENDMDL\n'
            endmdl_ind = outpdb.index(endmdl)
            outpdb.pop(endmdl_ind)
            outpdb.extend(con)
            outpdb.append(endmdl)

        with open(dst, 'w') as fout:
            fout.write(''.join(outpdb))

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2')

    C = Sf['aff_centers'][:]
    NC = len(C)

    I = Sf['aff_labels'][:]
    NI = len(I)

    L = Sf['labels'][:]

    cs = np.bincount(I)
    pcs = cs * 100.0 / NI

    centers = []

    for i in range(NC):

        TMtrj = cluster_to_trj(Sfn, i)
        TMbfn = TMtrj[:-4]  # strip .pdb from end
        TMbfac = TMbfn + '_b.pdb'
        TMxvg = TMbfn + '.xvg'

        top = kwargs['topology']

        call = [
            'g_rmsf',
            '-s', L[C[i]],
            '-f', TMtrj,
            '-ox', TMbfac,
            '-o', TMxvg,
            '-fit']

        g_rmsf = subprocess.Popen(call, stdin=subprocess.PIPE)
        # Pass index group 0 to gromacs
        g_rmsf.communicate(input='0')
        g_rmsf.wait()
        os.remove(TMxvg)
        os.remove(TMtrj)

        copy_connects(top, TMbfac)

        centers.append(TMbfac)

        kwargs['pdb_list'] = centers
        kwargs['nums'] = pcs

    AffRender(**kwargs)

    comm.Barrier()


def render_b_factor(
        Sfn,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    def copy_connects(src, dst):
        with open(src, 'r') as fin, open(dst, 'r') as fout:
            inpdb = np.array(fin.readlines())
            ind = np.array(
                map(lambda x: re.match('CONECT', x), inpdb),
                dtype=np.bool)
            con = inpdb[ind]

            outpdb = fout.readlines()
            endmdl = 'ENDMDL\n'
            endmdl_ind = outpdb.index(endmdl)
            outpdb.pop(endmdl_ind)
            outpdb.extend(con)
            outpdb.append(endmdl)

        with open(dst, 'w') as fout:
            fout.write(''.join(outpdb))

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2')

    C = Sf['aff_centers'][:]
    NC = len(C)

    I = Sf['aff_labels'][:]
    NI = len(I)

    L = Sf['labels'][:]

    cs = np.bincount(I)
    pcs = cs * 100.0 / NI

    centers = []

    for i in range(NC):

        TMtrj = cluster_to_trj(Sfn, i)
        TMbfn = TMtrj[:-4]  # strip .pdb from end
        TMbfac = TMbfn + '_b.pdb'
        TMxvg = TMbfn + '.xvg'

        top = kwargs['topology']

        call = [
            'g_rmsf',
            '-s', L[C[i]],
            '-f', TMtrj,
            '-ox', TMbfac,
            '-o', TMxvg,
            '-fit']

        g_rmsf = subprocess.Popen(call, stdin=subprocess.PIPE)
        # Pass index group 0 to gromacs
        g_rmsf.communicate(input='0')
        g_rmsf.wait()
        os.remove(TMxvg)
        os.remove(TMtrj)

        copy_connects(top, TMbfac)

        centers.append(TMbfac)

        kwargs['pdb_list'] = centers
        kwargs['nums'] = pcs

    AffRender(**kwargs)

    comm.Barrier()
