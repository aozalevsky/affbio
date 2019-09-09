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

# This code is heavily relies on the Affiniti Propagation
# code from Scikit-learn package
# see http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation


# General modules
import os
import time
import uuid
import shutil
import psutil
import tempfile
from os.path import join as osp

# NumPy for arrays
import numpy as np
import bottleneck as bn

# MPI parallelism
from mpi4py import MPI

# H5PY for storage
import h5py
from h5py import h5s

from .utils import Bunch, task


def aff_cluster(
        Sfn,
        tier=1,
        conv_iter=15,
        max_iter=2000,
        damping=0.95,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    else:
        NPROCS_LOCAL = 1

    # Init storage for matrices
    # Get file name
    # Open matrix file in parallel mode
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)
        Sf.atomic = True

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    # Open table with data for clusterization
    SS = G['cluster']
    SSs = SS.id.get_space()

    params = {
        'N': 0,
        'l': 0,
        'll': 0,
        'TMfn': '',
        'disk': False,
        'preference': 0.0}

    P = Bunch(params)

    ft = np.float32

    if rank == 0:

        N, N1 = SS.shape

        if N != N1:
            raise ValueError("S must be a square array \
                (shape=%s)" % repr((N, N1)))
        else:
            P.N = N

        try:
            preference = SS.attrs['preference']
        except:
            raise ValueError(
                'Unable to get preference from cluster matrix')

        if max_iter < 0:
            raise ValueError('max_iter must be > 0')

        if not 0 < conv_iter < max_iter:
            raise ValueError('conv_iter must lie in \
                interval between 0 and max_iter')

        if damping < 0.5 or damping >= 1.0:
            raise ValueError('damping must lie in interval between 0.5 and 1')

        print '#' * 10, 'Main params', '#' * 10
        print 'preference: %.3f' % preference
        print 'damping: %.3f' % damping
        print 'conv_iter: %d' % conv_iter
        print 'max_iter: %d' % max_iter
        print '#' * 31

        P.TMbfn = str(uuid.uuid1())
        P.TMfn = P.TMbfn + '.hdf5'

        if NPROCS > 1:

            # Magic 4 to fit MPI.Gather
            r = N % (NPROCS * 4)
        else:
            r = 0

        N -= r
        l = N // NPROCS
        if r > 0:
            print 'Truncating matrix to %sx%s to fit on %d procs' \
                % (N, N, NPROCS)
        P.N = N

        # Fit to memory
        MEM = psutil.virtual_memory().available / NPROCS_LOCAL
        # MEM = 500 * 10 ** 6
        ts = np.dtype(ft).itemsize * N  # Python give bits
        ts *= 8 * 1.1  # Allocate memory for e, tE, and ...
        # MEM -= ts  # ----
        tl = int(MEM // ts)  # Allocate memory for tS, tA, tR....

        def adjust_cache(tl, l):
            while float(l) % float(tl) > 0:
                tl -= 1
            return tl

        if tl < l:
            P.disk = True
            try:
                cache = 0
#                cache = int(sys.argv[1])
#                print sys.argv[1]
                assert cache < l
            except:
                cache = tl
                # print 'Wrong cache settings, set cache to %d' % tl
            tl = adjust_cache(tl, l)
            P.l = l
            P.ll = tl
        else:
            P.l = l
            P.ll = l

        if verbose:
            print "Available memory per process: %.2fG" % (MEM / 10.0 ** 9)
            print "Memory per row: %.2fM" % (ts / 10.0 ** 6)
            print "Estimated memory per process: %.2fG" \
                % (ts * P.ll / 10.0 ** 9)
            print 'Cache size is %d of %d' % (P.ll, P.l)

    P = comm.bcast(P)

    N = P.N
    l = P.l
    ll = P.ll

    ms = h5s.create_simple((ll, N))
    ms_l = h5s.create_simple((N,))

    tb, te = task(N, NPROCS, rank)

    tS = np.zeros((ll, N), dtype=ft)
    tSl = np.zeros((N,), dtype=ft)

    disk = P.disk

    if disk is True:
        TMLfd = tempfile.mkdtemp()
        TMLfn = osp(TMLfd, P.TMbfn + '_' + str(rank) + '.hdf5')
        TMLf = h5py.File(TMLfn, 'w')
        TMLf.atomic = True

        S = TMLf.create_dataset('S', (l, N), dtype=ft)
        Ss = S.id.get_space()

    # Copy input data and
    # place preference on diagonal
    z = - np.finfo(ft).max

    for i in range(tb, te, ll):
        SSs.select_hyperslab((i, 0), (ll, N))
        SS.id.read(ms, SSs, tS)

        if disk is True:
            Ss.select_hyperslab((i - tb, 0), (ll, N))
            S.id.write(ms, Ss, tS)

    if disk is True:
        R = TMLf.create_dataset('R', (l, N), dtype=ft)
        Rs = R.id.get_space()

    tRold = np.zeros((ll, N), dtype=ft)
    tR = np.zeros((ll, N), dtype=ft)
    tdR = np.zeros((l,), dtype=ft)

    # Shared storage
    if NPROCS == 1:
        TMf = h5py.File(P.TMfn, 'w', driver='sec2')
    else:
        TMf = h5py.File(P.TMfn, 'w', driver='mpio', comm=comm)
        TMf.atomic = True

    Rp = TMf.create_dataset('Rp', (N, N), dtype=ft)
    Rps = Rp.id.get_space()

    tRp = np.zeros((ll, N), dtype=ft)
    tRpa = np.zeros((N, ll), dtype=ft)

    A = TMf.create_dataset('A', (N, N), dtype=ft)
    As = A.id.get_space()

    tAS = np.zeros((ll, N), dtype=ft)
    tAold = np.zeros((N, ll), dtype=ft)
    tA = np.zeros((N, ll), dtype=ft)
    tdA = np.zeros((l,), dtype=ft)

    e = np.zeros((N, conv_iter), dtype=np.int8)
    tE = np.zeros((N,), dtype=np.int8)
    ttE = np.zeros((l,), dtype=np.int8)

    converged = False
    cK = 0
    K = 0
    ind = np.arange(ll)

    comm.Barrier()


    # Starting clustering
    for it in range(max_iter):
        if rank == 0:
            if verbose is True:
                print '=' * 10 + 'It %d' % (it) + '=' * 10
                tit = time.time()

        # Compute responsibilities
        for i in range(tb, te, ll):
            if disk is True:
                il = i - tb
                Ss.select_hyperslab((il, 0), (ll, N))
                S.id.read(ms, Ss, tS)
            # tS = S[i, :]
                Rs.select_hyperslab((il, 0), (ll, N))
                R.id.read(ms, Rs, tRold)
            else:
                tRold = tR.copy()

            if NPROCS > 1:
                As.select_hyperslab((i, 0), (ll, N))
                A.id.read(ms, As, tAS)
            else:
                tAS = tA.copy()

            # Tas = a[I, :]
            tAS += tS
            # tRold = R[i, :]

            tI = bn.nanargmax(tAS, axis=1)
            tY = tAS[ind, tI]
            tAS[ind, tI[ind]] = z
            tY2 = bn.nanmax(tAS, axis=1)

            tR = tS - tY[:, np.newaxis]
            tR[ind, tI[ind]] = tS[ind, tI[ind]] - tY2[ind]
            tR = (1 - damping) * tR + damping * tRold

            tRp = np.maximum(tR, 0)

            tRp[:ll, i:i + ll].flat[::ll + 1] = tR[:ll, i:i + ll].flat[::ll + 1]
            tdR[i - tb: i - tb + ll] = tR[:ll, i:i + ll].flat[::ll + 1]

            if disk is True:
                R.id.write(ms, Rs, tR)
                # R[i, :] = tR

            if NPROCS > 1:
                Rps.select_hyperslab((i, 0), (ll, N))
                Rp.id.write(ms, Rps, tRp)

            # Rp[i, :] = tRp
        if rank == 0:
            if verbose is True:
                teit1 = time.time()
                print 'R T %s' % (teit1 - tit)

        comm.Barrier()

        # Compute availabilities
        for j in range(tb, te, ll):

            if NPROCS > 1 or disk is True:
                As.select_hyperslab((0, j), (N, ll))

            if disk is True:
                A.id.read(ms, As, tAold)
            else:
                tAold = tA.copy()

            if NPROCS > 1:
                Rps.select_hyperslab((0, j), (N, ll))
                Rp.id.read(ms, Rps, tRpa)
            else:
                tRpa = tRp.copy()
            # tRp = Rp[:, j]

            tA = bn.nansum(tRpa, axis=0)[np.newaxis, :] - tRpa
            tdA[j - tb: j - tb + ll] = tA[j: j + ll, :ll].flat[::ll + 1]

            tA = np.minimum(tA, 0)

            tA[j:j + ll, :ll].flat[::ll + 1] = tdA[j - tb: j - tb + ll]

            tA *= (1.0 - damping)

            tA += damping * tAold

            for jl in range(ll):
                tdA[j - tb + jl] = tA[j + jl, jl]

            if NPROCS > 1:
                A.id.write(ms, As, tA)

        if rank == 0:
            if verbose is True:
                teit2 = time.time()
                print 'A T %s' % (teit2 - teit1)

        ttE = np.array(((tdA + tdR) > 0), dtype=np.int8)

        if NPROCS > 1:
            comm.Gather([ttE, MPI.INT], [tE, MPI.INT])
            comm.Bcast([tE, MPI.INT])
        else:
            tE = ttE

        e[:, it % conv_iter] = tE
        pK = K
        K = bn.nansum(tE)

        if rank == 0:
            if verbose is True:
                teit = time.time()
                cc = ''
                if K == pK:
                    if cK == 0:
                        cK += 1
                    elif cK > 1:
                        cc = ' Conv %d of %d' % (cK, conv_iter)
                else:
                    cK = 0

                print 'Total K %d T %s%s' % (K, teit - tit, cc)

        if it >= conv_iter:

            if rank == 0:
                se = bn.nansum(e, axis=1)
                converged = (bn.nansum((se == conv_iter) + (se == 0)) == N)

                if (converged == np.bool_(True)) and (K > 0):
                    if verbose is True:
                        print("Converged after %d iterations." % (it))
                    converged = True
                else:
                    converged = False

            converged = comm.bcast(converged, root=0)

        if converged is True:
            break

    if not converged and verbose and rank == 0:
        print("Failed to converge after %d iterations." % (max_iter))

    if K > 0:

        I = np.nonzero(e[:, 0])[0]
        C = np.zeros((N,), dtype=np.int)
        tC = np.zeros((l,), dtype=np.int)

        for i in range(l):
            if disk is True:
                Ss.select_hyperslab((i, 0), (1, N))
                S.id.read(ms_l, Ss, tSl)
            else:
                tSl = tS[i]

            tC[i] = bn.nanargmax(tSl[I])

        comm.Gather([tC, MPI.INT], [C, MPI.INT])

        if rank == 0:
            C[I] = np.arange(K)

        comm.Bcast([C, MPI.INT])

        for k in range(K):
            if NPROCS > 1:
                ii = np.where(C == k)[0]
                tN = ii.shape[0]

                tI = np.zeros((tN, ), dtype=np.float32)
                ttI = np.zeros((tN, ), dtype=np.float32)
                tttI = np.zeros((tN, ), dtype=np.float32)
                ms_k = h5s.create_simple((tN,))

                j = rank
                while j < tN:
                    ind = [(ii[i], ii[j]) for i in range(tN)]
                    SSs.select_elements(ind)
                    SS.id.read(ms_k, SSs, tttI)

                    ttI[j] = bn.nansum(tttI)
                    j += NPROCS

                comm.Reduce([ttI, MPI.FLOAT], [tI, MPI.FLOAT])

            else:
                ii = np.where(C == k)[0]
                tI = bn.nansum(tS[ii[:, np.newaxis], ii], axis=0)

            if rank == 0:
                I[k] = ii[bn.nanargmax(tI)]

        I.sort()
        comm.Bcast([I, MPI.INT])

        for i in range(l):
            if disk is True:
                Ss.select_hyperslab((i, 0), (1, N))
                S.id.read(ms_l, Ss, tSl)
            else:
                tSl = tS[i]

            tC[i] = bn.nanargmax(tSl[I])

        comm.Gather([tC, MPI.INT], [C, MPI.INT])

        if rank == 0:
            C[I] = np.arange(K)

    else:
        if rank == 0:
            I = np.zeros(())
            C = np.zeros(())

    # Cleanup
    Sf.close()
    TMf.close()

    if disk is True:
        TMLf.close()
        shutil.rmtree(TMLfd)

    comm.Barrier()

    if rank == 0:

        os.remove(P.TMfn)

        if verbose:
            print 'APN: %d' % K

        if I.size and C.size:

            Sf = h5py.File(Sfn, 'r+', driver='sec2')
            Gn = 'tier%d' % tier
            G = Sf.require_group(Gn)

            if 'aff_labels' in G.keys():
                del G['aff_labels']

            L = G.require_dataset(
                'aff_labels',
                shape=C.shape,
                dtype=np.int)

            L[:] = C[:]

            if tier > 1:
                PGn = 'tier%d' % (tier - 1)
                PG = Sf.require_group(PGn)
                PL = PG['aff_labels'][:]
                NL = np.copy(PL)

                for i in range(len(C)):
                    ind = np.where(PL == i)
                    NL[ind] = C[i]

                LM = G.require_dataset(
                    'aff_labels_merged',
                    shape=PL.shape,
                    dtype=np.int)

                LM[:] = NL[:]

            if 'aff_centers' in G.keys():
                del G['aff_centers']

            CM = G.require_dataset(
                'aff_centers',
                shape=I.shape,
                dtype=np.int)
            CM[:] = I[:]
            Sf.close()


def print_stat(
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

    C = G['aff_centers']
    NC = C.len()

    I = G['aff_labels']

    LC = G['labels']
    L = G['labels']
    if tier > 1 and merged:
        I = G['aff_labels_merged']
        L = Sf['tier1']['labels']
    NI = I.len()

    with open('aff_centers.out', 'w') as f:
        for i in range(NC):
            f.write("%s\t%d\n" % (LC[C[i]], i))

    with open('aff_labels.out', 'w') as f:
        for i in range(NI):
            f.write("%s\t%d\n" % (L[i], I[i]))

    with open('aff_stat.out', 'w') as f:
        f.write("AFF tier: %d\n" % tier)
        f.write("NUMBER OF CLUSTERS: %d\n" % NC)

        cs = np.bincount(I)
        pcs = cs * 100.0 / NI

        f.write('INDEX CENTER SIZE PERCENTAGE\n')

        for i in range(NC):
            f.write("%d\t%s\t%d\t%.3f\n" % (i, LC[C[i]], cs[i], pcs[i]))
