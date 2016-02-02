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
        tier=1,
        index=None,
        merged=False,
        out=None,
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
            outpdb.reverse()
            endmdl_ind = -1 - outpdb.index(endmdl)
            outpdb.reverse()
            outpdb.pop(endmdl_ind)
            outpdb.extend(con)
            outpdb.append(endmdl)

        with open(dst, 'w') as fout:
            fout.write(''.join(outpdb))

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)

    if tier > 1 and merged:
        I = G['aff_labels_merged'][:]
        L = Sf['tier1']['labels'][:]
    else:
        L = G['labels'][:]
        I = G['aff_labels'][:]

    ind = np.where(I == index)
    frames = L[ind]

    j = 0

    with open(frames[j], 'r') as fin, open(out, 'w') as fout:
        fout.write(fin.read())

    top = Sf['tier1']['labels'].attrs['topology']
    copy_connects(top, out)

    with open(out, 'r+') as fout:
        for j in range(1, len(frames)):
            with open(frames[0], 'r') as fin:
                fout.write(fin.read())


def render_b_factor(
        Sfn,
        tier=1,
        merged=False,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)

    C = G['aff_centers'][:]
    NC = len(C)

    if tier > 1 and merged:
        I = G['aff_labels_merged'][:]
    else:
        I = G['aff_labels'][:]

    NI = len(I)

    L = G['labels'][:]

    cs = np.bincount(I)
    pcs = cs * 100.0 / NI

    centers = []

    for i in range(NC):

        TMbfn = str(uuid.uuid1())
        TMtrj = TMbfn + '.pdb'
        cluster_to_trj(Sfn,
                       index=i,
                       merged=merged,
                       out=TMtrj,
                       mpi=mpi)
        TMbfac = TMbfn + '_b.pdb'
        TMxvg = TMbfn + '.xvg'

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

        centers.append(TMbfac)

        kwargs['pdb_list'] = centers
        kwargs['nums'] = pcs

    AffRender(**kwargs)

    map(os.remove, centers)


def render_b_factor_old(
        Sfn,
        mpi=None,
        tier=1,
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
        os.remove(TMbfac)

        copy_connects(top, TMbfac)

        centers.append(TMbfac)

        kwargs['pdb_list'] = centers
        kwargs['nums'] = pcs

    AffRender(**kwargs)

    comm.Barrier()
