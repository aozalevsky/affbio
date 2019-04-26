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
import glob
import time

# H5PY for storage
import h5py
from h5py import h5s

# NumPy
import numpy as np

# pyRMSD for calculations
import prody

# pyRMSD for calculations
import pyRMSD.RMSDCalculator
from pyRMSD import condensedMatrix

from natsort import natsorted

from .utils import task


def load_pdb_coords(
        Sfn,
        pdb_list,
        tier=1,
        topology=None,
        pbc=True,
        threshold=10.0,
        mpi=None,
        verbose=False,
        selection='all',
        *args, **kwargs):

    def check_pbc(coords, threshold=10.0, selection='all'):
        for i in range(len(coords) - 1):
            assert np.linalg.norm(coords[i] - coords[i + 1]) < threshold

    def parse_pdb(i, pbc=True, threshold=10.0, selection='all'):
        """Parse PDB files"""
        ps = prody.parsePDB(i)
        pc_ = ps.select(selection)
        if pc_ is None:
            raise ValueError('Empty selection "%s"' % selection)

        pc = pc_.getCoords()
        if pbc:
            check_pbc(pc, threshold)
        return pc

    def estimate_pdb_numatoms(
            topology, pbc=True, threshold=10.0, selection='all'):

        pdb_t = parse_pdb(
            topology, pbc=pbc, threshold=threshold, selection=selection)

        return pdb_t.shape

    def estimate_coord_shape(
            ftype='pdb',
            pdb_list=None,
            topology=None,
            pbc=True,
            threshold=10.0,
            selection='all',
            NPROCS=1,
            ):

        N = len(pdb_list)
        r = N % NPROCS

        if r > 0:
            N = N - r
            print('Truncating number to %d to fit %s procs' % (N, NPROCS))

        if ftype == 'pdb':
            if not topology:
                topology = pdb_list[0]
            na, nc = estimate_pdb_numatoms(
                topology,
                pbc=pbc,
                threshold=threshold,
                selection=selection)

        shape = (N, na, nc)

        return shape

    def load_pdb_names(Sfn, pdb_list, topology=None, tier=1):
        N = len(pdb_list)

        Sf = h5py.File(Sfn, 'w', driver='sec2')

        vls = h5py.special_dtype(vlen=str)
        Gn = 'tier%d' % tier
        G = Sf.require_group(Gn)
        L = G.create_dataset(
            'labels',
            (N,),
            dtype=vls)

        L[:] = pdb_list[:]

        if not topology:
            topology = pdb_list[0]

        L.attrs['topology'] = topology

        Sf.close()

    def load_from_previous_tier(Sfn, tier):
        Sf = h5py.File(Sfn, 'r+', driver='sec2')

        PGn = 'tier%d' % (tier - 1)
        PG = Sf.require_group(PGn)

        PS = PG['struct']
        nstruct, natoms, ncoords = PS.shape
        PNL = PG['labels']

        PC = PG['aff_centers'][:]
        nstruct = PC.shape[0]

        shape = (nstruct, natoms, ncoords)
        chunk = (1, natoms, ncoords)

        Gn = 'tier%d' % tier
        G = Sf.require_group(Gn)
        S = G.require_dataset(
            'struct',
            shape,
            dtype=np.float,
            chunks=chunk)

        vls = h5py.special_dtype(vlen=str)
        L = G.require_dataset(
            'labels',
            (nstruct,),
            dtype=vls)

        for i in range(nstruct):
            S[i] = PS[PC[i]][:]
            L[i] = PNL[PC[i]][:]

        Sf.close()

    comm, NPROCS, rank = mpi

    if tier > 1:
        if rank == 0:
            load_from_previous_tier(Sfn, tier)
            return
        else:
            return

    if len(pdb_list) == 1:
        ptrn = pdb_list[0]
        if '*' in ptrn or '?' in ptrn:
            pdb_list = glob.glob(ptrn)
            pdb_list = natsorted(pdb_list)

    shape = None

    if rank == 0:
        shape = estimate_coord_shape(
            pdb_list=pdb_list,
            topology=topology,
            pbc=pbc,
            threshold=threshold,
            selection=selection,
            NPROCS=NPROCS)

        N = shape[0]
        load_pdb_names(Sfn, pdb_list[:N], topology=topology)

    shape = comm.bcast(shape)
    N = shape[0]
    chunk = (1,) + shape[1:]

    # Init storage for matrices
    # HDF5 file
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    # Table for RMSD
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    S = G.require_dataset(
        'struct',
        shape,
        dtype=np.float,
        chunks=chunk)

    # A little bit of dark magic for faster io
    Ss = S.id.get_space()
    tS = np.ndarray(chunk, dtype=np.float)
    ms = h5s.create_simple(chunk)

    tb, te = task(N, NPROCS, rank)

    for i in range(tb, te):
        try:
            tS = parse_pdb(
                pdb_list[i],
                pbc=pbc, threshold=threshold, selection=selection)

            if verbose:
                print('Parsed %s' % pdb_list[i])
        except:
            raise ValueError('Broken structure %s' % pdb_list[i])

        Ss.select_hyperslab((i, 0, 0), chunk)
        S.id.write(ms, Ss, tS)

    # Wait for all processes
    comm.Barrier()

    Sf.close()


def calc_rmsd_matrix(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        noalign=False,
        *args, **kwargs):

    if noalign:
        cl = 'NOSUP_SERIAL_CALCULATOR'
    else:
        cl = "KABSCH_SERIAL_CALCULATOR"

    def calc_diag_chunk(ic, tS, cl):
        calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
            cl,
            ic)
        rmsd = calculator.pairwiseRMSDMatrix()
        rmsd_matrix = condensedMatrix.CondensedMatrix(rmsd)
        ln = len(tS)
        for i in range(ln):
            for j in range(i):
                tS[i, j] = rmsd_matrix[i, j]

    def calc_chunk(ic, jc, tS, cl):
        ln, n, d = ic.shape
        ttS = np.zeros((ln + 1, n, d))
        ttS[1:] = jc
        for i in range(ln):
            ttS[0] = ic[i]
            calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
                cl,
                ttS)
            tS[i] = calculator.oneVsFollowing(0)

    def partition(N, NPROCS, rank):
        # Partiotioning
        l = N // NPROCS
        lr = N % NPROCS

        if lr > 0 and rank == 0:
            print('Truncating matrix to %dx%d to fit %d procs' % (
                    l * NPROCS, l * NPROCS, NPROCS))

        lN = (NPROCS + 1) * NPROCS / 2

        m = lN // NPROCS
        mr = lN % NPROCS

        if mr > 0:
            m = m + 1 if rank % 2 == 0 else m

        return (l, m)

    comm, NPROCS, rank = mpi

    # Reread structures by every process
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    S = G['struct']
    # Count number of structures
    N = S.len()

    l, m = partition(N, NPROCS, rank)

    # HDF5 file
    # Table for RMSD
    RM = G.require_dataset(
        'rmsd',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    RM.attrs['chunk'] = l
    RMs = RM.id.get_space()

    # Init calculations
    tS = np.zeros((l, l), dtype=np.float32)
    ms = h5s.create_simple((l, l))

    i, j = rank, rank
    ic = S[i * l: (i + 1) * l]
    jc = ic

    for c in range(0, m):
        if rank == 0:
            tit = time.time()

        if i == j:
            calc_diag_chunk(ic, tS, cl)
        else:
            calc_chunk(ic, jc, tS, cl)

        RMs.select_hyperslab((i * l, j * l), (l, l))
        RM.id.write(ms, RMs, tS)

        if rank == 0:
            teit = time.time()
            if verbose:
                print("Step %d of %d T %s" % (c, m, teit - tit))

        # Dark magic of task assingment

        if 0 < (rank - c):
            j = j - 1
            jc = S[j * l: (j + 1) * l]
        elif rank - c == 0:
            i = NPROCS - rank - 1
            ic = S[i * l: (i + 1) * l]
        else:
            j = j + 1
            jc = S[j * l: (j + 1) * l]

    # Wait for all processes
    comm.Barrier()

    # Cleanup
    # Close matrix file
    Sf.close()
