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
import warnings

# H5PY for storage
import h5py
from h5py import h5s

# NumPy
import numpy as np

# MDAnalysis for reading structures
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBReader
from MDAnalysis.exceptions import SelectionError

from natsort import natsorted

from .checks import AffBioError, check_decomposition, check_stage, \
    effective_n
from .rmsd import rmsd_block
from .utils import task

# Rows of a diagonal block per RMSD call, so only its lower half is computed
DIAG_ROWS = 256


def expand_pdb_list(pdb_list):
    """Expand a single quoted glob pattern into a naturally sorted list."""
    if len(pdb_list) == 1:
        ptrn = pdb_list[0]
        if '*' in ptrn or '?' in ptrn:
            return natsorted(glob.glob(ptrn))
    return list(pdb_list)


def selection_indices(topology, selection='all'):
    """Atom count of the topology and indices of the selected atoms."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = mda.Universe(topology)
    try:
        sel = u.select_atoms(selection)
    except (SelectionError, ValueError) as e:
        raise AffBioError(
            'Invalid --selection "%s": %s. AffBio uses MDAnalysis selection '
            'syntax, e.g. "chainID A" instead of ProDy\'s "chain A".'
            % (selection, e))
    if sel.n_atoms == 0:
        raise AffBioError('Empty selection "%s"' % selection)
    return u.atoms.n_atoms, sel.indices


def read_coords(fname, n_atoms, idx):
    """Coordinates of the selected atoms in the first model of a PDB file."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with PDBReader(fname) as reader:
            ts = reader.ts
            if ts.n_atoms != n_atoms:
                raise ValueError('has %d atoms, the topology has %d'
                                 % (ts.n_atoms, n_atoms))
            return ts.positions[idx].astype(np.float64)


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

    def check_pbc(coords, threshold=10.0):
        dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        bad = np.nonzero(dist >= threshold)[0]
        if bad.size:
            i = bad[0]
            raise ValueError('atoms %d and %d are %.1f A apart '
                             '(PBC artifact?)' % (i, i + 1, dist[i]))

    def parse_pdb(fname, n_atoms, idx):
        """Parse PDB files"""
        coords = read_coords(fname, n_atoms, idx)
        if pbc:
            check_pbc(coords, threshold)
        return coords

    def load_pdb_names(Sfn, pdb_list, topology):
        N = len(pdb_list)

        Sf = h5py.File(Sfn, 'w', driver='sec2')

        Gn = 'tier%d' % tier
        G = Sf.require_group(Gn)
        L = G.create_dataset(
            'labels',
            (N,),
            dtype=h5py.string_dtype())

        L[:] = pdb_list
        L.attrs['topology'] = topology

        Sf.close()

    def load_from_previous_tier(Sfn, tier, NPROCS):
        Sf = h5py.File(Sfn, 'r+', driver='sec2')

        PG = Sf['tier%d' % (tier - 1)]
        PS = PG['struct']
        nstruct, natoms, ncoords = PS.shape
        PNL = PG['labels'].asstr()

        PC = PG['aff_centers'][:]
        check_decomposition(len(PC), NPROCS)
        nstruct = effective_n(len(PC), NPROCS)
        if nstruct < len(PC):
            print('Using %d of %d centers of tier %d to split evenly across '
                  '%d processes; dropped: %s'
                  % (nstruct, len(PC), tier - 1, NPROCS,
                     ', '.join(PNL[c] for c in PC[nstruct:])))

        shape = (nstruct, natoms, ncoords)
        chunk = (1, natoms, ncoords)

        G = Sf.require_group('tier%d' % tier)
        S = G.require_dataset(
            'struct',
            shape,
            dtype=np.float64,
            chunks=chunk)
        S.attrs['nprocs'] = NPROCS

        L = G.require_dataset(
            'labels',
            (nstruct,),
            dtype=h5py.string_dtype())

        for i in range(nstruct):
            S[i] = PS[PC[i]][:]
            L[i] = PNL[PC[i]]

        Sf.close()

    comm, NPROCS, rank = mpi

    if tier > 1:
        if rank == 0:
            load_from_previous_tier(Sfn, tier, NPROCS)
        return

    pdb_list = expand_pdb_list(pdb_list)
    check_decomposition(len(pdb_list), NPROCS)
    N = effective_n(len(pdb_list), NPROCS)

    if not topology:
        topology = pdb_list[0]
    n_atoms, idx = selection_indices(topology, selection)

    shape = (N, len(idx), 3)
    chunk = (1, len(idx), 3)

    if rank == 0:
        if N < len(pdb_list):
            print('Using %d of %d structures to split evenly across %d '
                  'processes; dropped: %s'
                  % (N, len(pdb_list), NPROCS, ', '.join(pdb_list[N:])))
        load_pdb_names(Sfn, pdb_list[:N], topology)

    # Wait until the file exists
    comm.Barrier()

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
        dtype=np.float64,
        chunks=chunk)
    S.attrs['nprocs'] = NPROCS

    # A little bit of dark magic for faster io
    Ss = S.id.get_space()
    ms = h5s.create_simple(chunk)

    tb, te = task(N, NPROCS, rank)

    for i in range(tb, te):
        try:
            tS = parse_pdb(pdb_list[i], n_atoms, idx)
        except Exception as e:
            raise AffBioError('Broken structure %s: %s' % (pdb_list[i], e))

        if verbose:
            print('Parsed %s' % pdb_list[i])

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

    # --noalign compares raw coordinates: no centering, no rotation
    superpose = not noalign

    def calc_diag_chunk(ic, tS):
        # Strict lower triangle only, DIAG_ROWS rows at a time
        ln = len(ic)
        for r0 in range(0, ln, DIAG_ROWS):
            r1 = min(r0 + DIAG_ROWS, ln)
            rmsd_block(ic[r0:r1], ic[:r1], superpose, out=tS[r0:r1, :r1])
        for i in range(ln):
            tS[i, i:] = 0

    def calc_chunk(ic, jc, tS):
        rmsd_block(ic, jc, superpose, out=tS)

    def partition(N, NPROCS, rank):
        # Partiotioning
        l = N // NPROCS

        lN = (NPROCS + 1) * NPROCS // 2

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

    try:
        check_decomposition(N, NPROCS)
        check_stage('calc_rmsd', N, NPROCS,
                    stored_nprocs=S.attrs.get('nprocs'))
    except AffBioError:
        Sf.close()
        raise

    l, m = partition(N, NPROCS, rank)

    # HDF5 file
    # Table for RMSD
    RM = G.require_dataset(
        'rmsd',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    RM.attrs['chunk'] = l
    RM.attrs['nprocs'] = NPROCS
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
            calc_diag_chunk(ic, tS)
        else:
            calc_chunk(ic, jc, tS)

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
