#
# This file is part of the AffBio package for clustering of
# biomolecular structures.
#
# Copyright (c) 2015-2016, by Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

# General modules
import os

# NumPy for arrays
import numpy as np

# H5PY for storage
import h5py

from .AffRender import AffRender
from .checks import AffBioError
from .rmsf import cluster_bfactors

# Records kept when a frame is wrapped into a MODEL block
COORD_RECORDS = ('ATOM', 'HETATM', 'ANISOU', 'TER', 'CONECT')


def cluster_members(Sf, tier=1, merged=False):
    """Cluster label of every structure and the structure file names."""
    G = Sf['tier%d' % tier]
    if tier > 1 and merged:
        return G['aff_labels_merged'][:], Sf['tier1']['labels'].asstr()[:]
    return G['aff_labels'][:], G['labels'].asstr()[:]


def model_lines(fname, number):
    """Records of one frame as a MODEL ... ENDMDL block, without END."""
    with open(fname, 'r') as f:
        lines = [l for l in f if l[:6].strip() != 'END']
    if any(l.startswith('MODEL') for l in lines):
        return lines
    body = [l for l in lines if l[:6].strip() in COORD_RECORDS]
    return ['MODEL     %4d\n' % number] + body + ['ENDMDL\n']


def cluster_to_trj(
        Sfn,
        tier=1,
        index=None,
        merged=False,
        output=None,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    if index is None or output is None:
        raise AffBioError(
            'cluster_to_trj needs a cluster --index and an -o output file.')

    with h5py.File(Sfn, 'r', driver='sec2') as Sf:
        I, L = cluster_members(Sf, tier, merged)
        top = Sf['tier1']['labels'].attrs['topology']

    frames = L[I == index]
    if len(frames) == 0:
        raise AffBioError('There is no cluster %d; clusters are numbered '
                          '0 to %d.' % (index, I.max()))

    with open(output, 'w') as fout:
        fout.writelines(model_lines(frames[0], 1))

    copy_connects(top, output)

    with open(output, 'a') as fout:
        for k, frame in enumerate(frames[1:], 2):
            fout.writelines(model_lines(frame, k))
        fout.write('END\n')


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

    with h5py.File(Sfn, 'r', driver='sec2') as Sf:
        top = Sf['tier1']['labels'].attrs['topology']
        G = Sf['tier%d' % tier]
        C = G['aff_centers'][:]
        LC = G['labels'].asstr()[:]
        I, L = cluster_members(Sf, tier, merged)

    NI = len(I)

    cs = np.bincount(I)
    pcs = cs * 100.0 / NI

    centers = []

    for i in range(len(C)):
        TMbfac = 'cluster_%d_bfac.pdb' % i
        cluster_bfactors(LC[C[i]], L[I == i], TMbfac)
        copy_connects(top, TMbfac)
        centers.append(TMbfac)

    kwargs['pdb_list'] = centers
    kwargs['nums'] = pcs

    AffRender(**kwargs)

    for c in centers:
        os.remove(c)


def copy_connects(src, dst):
    """Copy the CONECT records of src into dst before ENDMDL or END."""
    with open(src, 'r') as fin:
        con = [l for l in fin if l.startswith('CONECT')]
    if not con:
        return

    with open(dst, 'r') as fout:
        lines = fout.readlines()

    records = [l[:6].strip() for l in lines]
    for marker in ('ENDMDL', 'END'):
        if marker in records:
            pos = len(records) - 1 - records[::-1].index(marker)
            break
    else:
        pos = len(lines)

    lines[pos:pos] = con

    with open(dst, 'w') as fout:
        fout.write(''.join(lines))
