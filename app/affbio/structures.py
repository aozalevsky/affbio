#General modules
import glob
import time

#H5PY for storage
import h5py
from h5py import h5s

#NumPy
import numpy as np

#pyRMSD for calculations
import prody

#pyRMSD for calculations
import pyRMSD.RMSDCalculator
from pyRMSD import condensedMatrix

from .utils import task


def load_pdb_coords(
        Sfn,
        pdb_list,
        tier=1,
        topology=None,
        mpi=None,
        verbose=False,
        *args, **kwargs):

    def check_pbc(coords, threshold=50):
        for i in range(len(coords) - 1):
            assert np.linalg.norm(coords[i] - coords[i + 1]) < threshold

    def parse_pdb(i):
        """Parse PDB files"""
        ps = prody.parsePDB(i)
        pc = ps.getCoords()
        check_pbc(pc)
        return pc

    def estimate_pdb_numatoms(topology):

        pdb_t = parse_pdb(topology)

        return pdb_t.shape

    def estimate_coord_shape(
            ftype='pdb',
            pdb_list=None,
            topology=None,
            NPROCS=1):

        N = len(pdb_list)
        r = N % NPROCS

        if r > 0:
            N = N - r
            print 'Truncating number to %d to fit %s procs' % (N, NPROCS)

        if ftype == 'pdb':
            if not topology:
                topology = pdb_list[0]
            na, nc = estimate_pdb_numatoms(topology)

        shape = (N, na, nc)

        return shape

    def load_pdb_names(Sfn, pdb_list, topology=None):
        N = len(pdb_list)

        Sf = h5py.File(Sfn, 'r+', driver='sec2')

        vls = h5py.special_dtype(vlen=str)
        L = Sf.create_dataset(
            'labels',
            (N,),
            dtype=vls)

        L[:] = pdb_list[:]

        if not topology:
            topology = pdb_list[0]

        L.attrs['topology'] = topology

        Sf.close()

    comm, NPROCS, rank = mpi

    if len(pdb_list) == 1:
        ptrn = pdb_list[0]
        if '*' in ptrn or '?' in ptrn:
            pdb_list = glob.glob(ptrn)

    shape = None

    if rank == 0:
        shape = estimate_coord_shape(
            pdb_list=pdb_list,
            topology=topology,
            tier=tier)

    shape = comm.bcast(shape)
    N = shape[0]
    chunk = (1,) + shape[1:]

    #Init storage for matrices
    #HDF5 file
    Sf = h5py.File(Sfn, 'w', driver='mpio', comm=comm)
    #Table for RMSD
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
            tS = parse_pdb(pdb_list[i])
            if verbose:
                print 'Parsed %s' % pdb_list[i]
        except:
            raise ValueError('Broken structure %s' % pdb_list[i])

        Ss.select_hyperslab((i, 0, 0), chunk)
        S.id.write(ms, Ss, tS)

    #Wait for all processes
    comm.Barrier()

    Sf.close()

    load_pdb_names(Sfn, pdb_list[:N])


def calc_rmsd_matrix(
        Sfn,
        mpi=None,
        verbose=False,
        *args, **kwargs):

    def calc_diag_chunk(ic, tS):
        calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
            "KABSCH_SERIAL_CALCULATOR",
            ic)
        rmsd = calculator.pairwiseRMSDMatrix()
        rmsd_matrix = condensedMatrix.CondensedMatrix(rmsd)
        ln = len(tS)
        for i in range(ln):
            for j in range(i):
                tS[i, j] = rmsd_matrix[i, j]

    def calc_chunk(ic, jc, tS):
        ln, n, d = ic.shape
        ttS = np.zeros((ln + 1, n, d))
        ttS[1:] = jc
        for i in range(ln):
            ttS[0] = ic[i]
            calculator = pyRMSD.RMSDCalculator.RMSDCalculator(
                "KABSCH_SERIAL_CALCULATOR",
                ttS)
            tS[i] = calculator.oneVsFollowing(0)

    def partition(N, NPROCS, rank):
        #Partiotioning
        l = N // NPROCS
        lr = N % NPROCS

        if lr > 0 and rank == 0:
            print 'Truncating matrix to %dx%d to fit %d procs' \
                % (l * NPROCS, l * NPROCS, NPROCS)

        lN = (NPROCS + 1) * NPROCS / 2

        m = lN // NPROCS
        mr = lN % NPROCS

        if mr > 0:
            m = m + 1 if rank % 2 == 0 else m

        return (l, m)

    comm, NPROCS, rank = mpi

    #Reread structures by every process
    Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)
    Sf.atomic = True
    S = Sf['struct']
    #Count number of structures
    N = S.len()

    l, m = partition(N, NPROCS, rank)

    #HDF5 file
    #Table for RMSD
    RM = Sf.require_dataset(
        'rmsd',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    RM.attrs['chunk'] = l
    RMs = RM.id.get_space()

    #Init calculations
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
                print "Step %d of %d T %s" % (c, m, teit - tit)

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

    #Wait for all processes
    comm.Barrier()

    #Cleanup
    #Close matrix file
    Sf.close()

    #Wait for all processes
    comm.Barrier()
