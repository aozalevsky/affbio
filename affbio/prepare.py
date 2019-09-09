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

#General modules
import time

#NumPy for arrays
import numpy as np

#H5PY for storage
import h5py
from h5py import h5s


def prepare_cluster_matrix(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        *args, **kwargs):

    def calc_chunk(l, tRM, tCM):
        ttCM = tRM + tCM * random_state.randn(l, l)
        return ttCM

    def calc_chunk_diag(l, tRM, tCM):
        ttCM = tCM + tCM.transpose()
        ttRM = tRM + tRM.transpose()
        ttCM = calc_chunk(l, ttRM, ttCM)
        return ttCM

    comm, NPROCS, rank = mpi

    #Init RMSD matrix
    #Open matrix file in parallel mode
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    RM = G['rmsd']
    RMs = RM.id.get_space()

    N = RM.len()
    l = N // NPROCS

    if rank == 0:
        N, N1 = RM.shape

        if N != N1:
            raise ValueError(
                "S must be a square array (shape=%s)" % repr(RM.shape))

        if RM.attrs['chunk'] % l > 0:
            raise ValueError(
                "Wrong chunk size in RMSD matrix")

    CM = G.require_dataset(
        'cluster',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    CM.attrs['chunk'] = l
    CMs = CM.id.get_space()

    random_state = np.random.RandomState(0)
    x = np.finfo(np.float32).eps
    y = np.finfo(np.float32).tiny * 100

    #Partiotioning
    lN = (NPROCS + 1) * NPROCS / 2

    m = lN // NPROCS
    mr = lN % NPROCS

    if mr > 0:
        m = m + 1 if rank % 2 == 0 else m

    #Init calculations
    tRM = np.zeros((l, l), dtype=np.float32)
    tCM = np.zeros((l, l), dtype=np.float32)
    ttCM = np.zeros((l, l), dtype=np.float32)
    ms = h5s.create_simple((l, l))

    i, j = rank, rank

    for c in range(m):
        if rank == 0:
            tit = time.time()
        RMs.select_hyperslab((i * l, j * l), (l, l))
        RM.id.read(ms, RMs, tRM)

        #tRM = -1 * tRM ** 2
        tRM **= 2
        tRM *= -1
        tCM = tRM * x + y

        if i == j:
            ttCM = calc_chunk_diag(l, tRM[:], tCM[:])
            CMs.select_hyperslab((i * l, j * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

        else:
            ttCM = calc_chunk(l, tRM[:], tCM[:])
            CMs.select_hyperslab((i * l, j * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

            ttCM = calc_chunk(l, tRM.transpose(), tCM.transpose())
            CMs.select_hyperslab((j * l, i * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

        if rank == 0:
            teit = time.time()
            if verbose:
                print "Step %d of %d T %s" % (c, m, teit - tit)

        if (rank - c) > 0:
            j = j - 1
        elif (rank - c) == 0:
            i = NPROCS - rank - 1
        else:
            j = j + 1

    #Wait for all processes
    comm.Barrier()

    Sf.close()


def calc_median(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    #Livestats for median
    #from livestats import livestats
    import pyximport
    pyximport.install()
    import lvc

    #Init cluster matrix
    #Open matrix file in single mode
    Sf = h5py.File(Sfn, 'r+', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    CM = G['cluster']

    N = CM.len()
    l = CM.attrs['chunk']

    N, N1 = CM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(CM.shape))

    if l <= 0:
        raise ValueError(
            "Wrong chunk size in RMSD matrix")


    if N * N1 > 10000:
    #Init calculations
        #med = livestats.LiveStats()
        med = lvc.Quantile(0.5)

        for i in range(N):
            #CMs.select_hyperslab((i, 0), (1, i - 1))
            #CM.id.read(ms, CMs, tCM)
            med.add(CM[i, :i])

        #level, median = med.quantiles()[0]
        median = med.quantile()
    else:
        median = np.median(CM[:])

    if verbose:
        print 'Median: %f' % median

    CM.attrs['median'] = median

    Sf.close()


def set_preference(
        Sfn,
        tier=1,
        preference=None,
        factor=1.0,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    #Init storage for matrices
    #Get file name
    #Open matrix file in parallel mode
    Sf = h5py.File(Sfn, 'r+', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    SS = G['cluster']
    SSs = SS.id.get_space()
    ms = h5s.create_simple((1, 1))
    tS = np.zeros((1,), dtype=np.float32)

    ft = np.float32

    N, N1 = SS.shape

    if N != N1:
        raise ValueError("S must be a square array \
            (shape=%s)" % repr((N, N1)))

    if not preference:
        try:
            preference = SS.attrs['median']
        except:
            raise ValueError(
                'Unable to get preference from cluster matrix')

    preference = ft(preference * factor)

    #Copy input data and
    #place preference on diagonal
    random_state = np.random.RandomState(0)
    x = np.finfo(ft).eps
    y = np.finfo(ft).tiny * 100

    for i in range(N):
        tS[0] = preference + (preference * x + y) * random_state.randn()
        SSs.select_hyperslab((i, i), (1, 1))
        SS.id.write(ms, SSs, tS)

    SS.attrs['preference'] = preference

    if verbose:
        print 'Preference: %f' % preference

    Sf.close()
