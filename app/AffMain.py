#!/usr/bin/python

#General modules
import os
import re
import time
import uuid
import shutil
import psutil
import tempfile
import subprocess
import glob
from os.path import join as osp
from collections import OrderedDict as OD

#NumPy for arrays
import numpy as np
import bottleneck as bn

#MPI parallelism
from mpi4py import MPI

#H5PY for storage
import h5py
from h5py import h5s

#pyRMSD for calculations
import prody

#pyRMSD for calculations
import pyRMSD.RMSDCalculator
from pyRMSD import condensedMatrix


from AffRender import AffRender


import argparse as ag


def dummy(*args, **kwargs):
    pass


def init_mpi():
    #Get MPI info
    comm = MPI.COMM_WORLD
    #Get number of processes
    NPROCS = comm.size
    #Get rank
    rank = comm.rank

    return (comm, NPROCS, rank)


def master(fn):
    comm, NPROCS, rank = init_mpi()

    if rank == 0:
        return fn
    else:
        return dummy


@master
def get_args(choices):
    """Parse cli arguments"""

    print choices
    parser = ag.ArgumentParser(
        description='Parallel ffitiny propagation for biomolecules')

    parser.add_argument('-m',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('-t', '--task',
                        nargs='+',
                        required=True,
                        choices=choices,
                        metavar='TASK',
                        help='Task to do. Available options \
                        are: %s' % ", ".join(choices))

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    load_pdb = parser.add_argument_group('load_pdb')

    load_pdb.add_argument('-f',
                          nargs='*',
                          type=str,
                          dest='pdb_list',
                          metavar='FILE',
                          help='PDB files')

    load_pdb.add_argument('-s',
                          type=str,
                          dest='topology',
                          help='Topology PDB file')

    preference = parser.add_argument_group('calculate_preference')

    preference.add_argument('--factor',
                            type=float,
                            dest='factor',
                            metavar='FACTOR',
                            default=1.0,
                            help='Multiplier for median')

    aff_cluster = parser.add_argument_group('aff_cluster')

    aff_cluster.add_argument('--preference',
                             type=float,
                             dest='preference',
                             metavar='PREFERENCE',
                             help='Override computed preference')

    aff_cluster.add_argument('--conv_iter',
                             type=int,
                             dest='conv_iter',
                             metavar='ITERATIONS',
                             default=15,
                             help='Iterations to converge')

    aff_cluster.add_argument('--max_iter',
                             type=int,
                             dest='max_iter',
                             metavar='ITERATIONS',
                             default=2000,
                             help='Maximum iterations')

    aff_cluster.add_argument('--damping',
                             type=float,
                             dest='damping',
                             metavar='DAMPING',
                             default=0.95,
                             help='Damping factor')

    render = parser.add_argument_group('render')

    render.add_argument('-o', '--output',
                        metavar='output.png',
                        help='Output PNG image')

    render.add_argument('--nums',
                        nargs='*', type=int,
                        help='Values for numerical labels')

    render.add_argument('--draw_nums',
                        action='store_true',
                        help='Draw numerical labels')

    render.add_argument('--guess_nums',
                        action='store_true',
                        help='Guess nums from filenames',
                        default=False)

    render.add_argument('--clear',
                        action='store_true',
                        help='Clear intermidiate files')

    render.add_argument('--width',
                        nargs='?', type=int, default=640,
                        help='Width of individual image')

    render.add_argument('--height',
                        nargs='?', type=int, default=480,
                        help='Height of individual image')

    render.add_argument('--moltype',
                        nargs='?', type=str, default="general",
                        choices=["general", "origami"],
                        help='Height of individual image')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def task(N, NPROCS, rank):
    l = N / NPROCS
    b = rank * l
    return (b, b + l)


def init_logging(task, verbose=False):
    if verbose:

        print 'Starting task: %s' % task

    #Get current time
    t0 = time.time()

    return t0


def finish_logging(task, t0, verbose=False):
    if verbose:
        print "Task: %s execution time is %f" % (task, time.time() - t0)


def load_pdb_coords(
        Sfn,
        pdb_list,
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

    @master
    def estimate_pdb_numatoms(topology):

        pdb_t = parse_pdb(topology)

        return pdb_t.shape

    @master
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

    @master
    def load_pdb_names(Sfn, pdb_list, topology=None):
        N = len(pdb_list)

        Sf = h5py.File(Sfn, 'r+', driver='sec2')
        Sf.atomic = True

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

    shape = estimate_coord_shape(pdb_list=pdb_list, topology=topology)
    shape = comm.bcast(shape)
    N = shape[0]
    chunk = (1,) + shape[1:]

    #Init storage for matrices
    #HDF5 file
    Sf = h5py.File(Sfn, 'w', driver='mpio', comm=comm)
    Sf.atomic = True
    #Table for RMSD
    S = Sf.create_dataset(
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

    if rank == 0:
        print np.sum(RM)
    #Cleanup
    #Close matrix file
    Sf.close()


def prepare_cluster_matrix(
        Sfn,
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
    RMf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)
    RMf.atomic = True
    #Open table with data for clusterization
    RM = RMf['rmsd']
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

    CM = RMf.require_dataset(
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

        print i, j
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

    if rank == 0:
#        print CM[10, 3], CM[3, 10]
#        print sum(CM[-1])
        print CM[:]
        print np.sum(CM)

    RMf.close()


@master
def calc_median(
        Sfn,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    #Livestats for median
    #from livestats import livestats
    import pyximport
    pyximport.install()
    import lvc

    #Init cluster matrix
    #Open matrix file in single mode
    CMf = h5py.File(Sfn, 'r+', driver='sec2')
    CMf.atomic = True
    #Open table with data for clusterization
    CM = CMf['cluster']

    N = CM.len()
    l = CM.attrs['chunk']

    N, N1 = CM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(CM.shape))

    if l <= 0:
        raise ValueError(
            "Wrong chunk size in RMSD matrix")

    #Init calculations
    #med = livestats.LiveStats()
    med = lvc.Quantile(0.5)

    for i in range(N):
        #CMs.select_hyperslab((i, 0), (1, i - 1))
        #CM.id.read(ms, CMs, tCM)
        med.add(CM[i, :i])

    #level, median = med.quantiles()[0]
    median = med.quantile()

    if verbose:
        print 'Median: %f' % median

    CM.attrs['median'] = median

    CMf.close()


@master
def set_preference(
        Sfn,
        preference=None,
        factor=1.0,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    #Init storage for matrices
    #Get file name
    #Open matrix file in parallel mode
    SSf = h5py.File(Sfn, 'r+', driver='sec2')
    SSf.atomic = True
    #Open table with data for clusterization
    SS = SSf['cluster']
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

    SSf.close()


def aff_cluster(
        Sfn,
        conv_iter=15,
        max_iter=2000,
        damping=0.95,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])

    #Init storage for matrices
    #Get file name
    #Open matrix file in parallel mode
    SSf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)
    SSf.atomic = True
    #Open table with data for clusterization
    SS = SSf['cluster']
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

        if damping < 0.5 or damping >= 1:
            raise ValueError('damping must lie in interval between 0.5 and 1')

        print '#' * 10, 'Main params', '#' * 10
        print 'preference: %.3f' % preference
        print 'damping: %.3f' % damping
        print 'conv_iter: %d' % conv_iter
        print 'max_iter: %d' % max_iter
        print '#' * 31

        P.TMbfn = str(uuid.uuid1())
        P.TMfn = P.TMbfn + '.hdf5'

        # Magic 4 to fit MPI.Gather
        r = N % (NPROCS * 4)
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
                #print 'Wrong cache settings, set cache to %d' % tl
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

    tS = np.ndarray((ll, N), dtype=ft)
    tSl = np.ndarray((N,), dtype=ft)

    disk = P.disk

    if disk is True:
        TMLfd = tempfile.mkdtemp()
        TMLfn = osp(TMLfd, P.TMbfn + '_' + str(rank) + '.hdf5')
        TMLf = h5py.File(TMLfn, 'w')
        TMLf.atomic = True

        S = TMLf.create_dataset('S', (l, N), dtype=ft)
        Ss = S.id.get_space()

    #Copy input data and
    #place preference on diagonal
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

    #Shared storage
    TMf = h5py.File(P.TMfn, 'w', driver='mpio', comm=comm)
    TMf.atomic = True

    Rp = TMf.create_dataset('Rp', (N, N), dtype=ft)
    Rps = Rp.id.get_space()

    tRp = np.ndarray((ll, N), dtype=ft)
    tRpa = np.ndarray((N, ll), dtype=ft)

    A = TMf.create_dataset('A', (N, N), dtype=ft)
    As = A.id.get_space()

    tAS = np.ndarray((ll, N), dtype=ft)
    tAold = np.ndarray((N, ll), dtype=ft)
    tA = np.ndarray((N, ll), dtype=ft)
    tdA = np.ndarray((l,), dtype=ft)

    e = np.ndarray((N, conv_iter), dtype=np.int8)
    tE = np.ndarray((N,), dtype=np.int8)
    ttE = np.ndarray((l,), dtype=np.int8)

    converged = False
    cK = 0
    K = 0
    ind = np.arange(ll)

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
            #tS = S[i, :]
                Rs.select_hyperslab((il, 0), (ll, N))
                R.id.read(ms, Rs, tRold)
            else:
                tRold = tR.copy()

            As.select_hyperslab((i, 0), (ll, N))
            A.id.read(ms, As, tAS)
            #Tas = a[I, :]
            tAS += tS
            #tRold = R[i, :]

            tI = bn.nanargmax(tAS, axis=1)
            tY = tAS[ind, tI]
            tAS[ind, tI[ind]] = z
            tY2 = bn.nanmax(tAS, axis=1)

            tR = tS - tY[:, np.newaxis]
            tR[ind, tI[ind]] = tS[ind, tI[ind]] - tY2[ind]
            tR = (1 - damping) * tR + damping * tRold

            tRp = np.maximum(tR, 0)

            for il in range(ll):
                tRp[il, i + il] = tR[il, i + il]
                tdR[i - tb + il] = tR[il, i + il]

            if disk is True:
                R.id.write(ms, Rs, tR)
                #R[i, :] = tR

            Rps.select_hyperslab((i, 0), (ll, N))
            Rp.id.write(ms, Rps, tRp)

                #Rp[i, :] = tRp
        if rank == 0:
            if verbose is True:
                teit1 = time.time()
                print 'R T %s' % (teit1 - tit)

        comm.Barrier()

        # Compute availabilities
        for j in range(tb, te, ll):

            As.select_hyperslab((0, j), (N, ll))

            if disk is True:
                A.id.read(ms, As, tAold)
            else:
                tAold = tA.copy()

            Rps.select_hyperslab((0, j), (N, ll))
            Rp.id.read(ms, Rps, tRpa)
            #tRp = Rp[:, j]

            tA = bn.nansum(tRpa, axis=0)[np.newaxis, :] - tRpa
            for jl in range(ll):
                tdA[j - tb + jl] = tA[j + jl, jl]

            tA = np.minimum(tA, 0)

            for jl in range(ll):
                tA[j + jl, jl] = tdA[j - tb + jl]

            tA *= (1 - damping)
            tA += damping * tAold

            for jl in range(ll):
                tdA[j - tb + jl] = tA[j + jl, jl]

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

    #Cleanup
    SSf.close()
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

            if 'aff_labels' in Sf.keys():
                del Sf['aff_labels']

            LM = Sf.require_dataset(
                'aff_labels',
                shape=C.shape,
                dtype=np.int)
            LM[:] = C[:]

            if 'aff_centers' in Sf.keys():
                del Sf['aff_centers']

            CM = Sf.require_dataset(
                'aff_centers',
                shape=I.shape,
                dtype=np.int)
            CM[:] = I[:]
            Sf.close()


@master
def print_stat(
        Sfn,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    Sf = h5py.File(Sfn, 'r', driver='sec2')

    C = Sf['aff_centers']
    NC = C.len()

    I = Sf['aff_labels']
    NI = I.len()

    L = Sf['labels']

    with open('aff_centers.out', 'w') as f:
        for i in range(NC):
            f.write("%s\t%d\n" % (L[C[i]], i))

    with open('aff_labels.out', 'w') as f:
        for i in range(NI):
            f.write("%s\t%d\n" % (L[i], I[i]))

    with open('aff_stat.out', 'w') as f:
        f.write("NUMBER OF CLUSTERS: %d\n" % NC)

        cs = np.bincount(I)
        pcs = cs * 100.0 / NI

        f.write('INDEX CENTER SIZE PERCENTAGE\n')

        for i in range(NC):
            f.write("%d\t%s\t%d\t%.3f\n" % (i, L[C[i]], cs[i], pcs[i]))


@master
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

        TMbfn = str(uuid.uuid1())
        TMtrj = TMbfn + '.pdb'
        fout = open(TMtrj, 'w')
        TMbfac = TMbfn + '_b.pdb'
        TMxvg = TMbfn + '.xvg'

        top = kwargs['topology']

        ind = np.where(I == i)
        for j in L[ind]:
            with open(j, 'r') as fin:
                fout.write(fin.read())
        fout.close()

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
#    map(os.remove, centers)

#        for i in range(NC):
#            f.write("%d\t%s\t%d\t%.3f\n" % (i, L[C[i]], cs[i], pcs[i]))


@master
def render_aff(*args, **kwargs):

    AffRender(*args, **kwargs)


def get_tasks():

    tasks = OD([
        ('load_pdb', load_pdb_coords),
        ('calc_rmsd', calc_rmsd_matrix),
        ('prepare_matrix', prepare_cluster_matrix),
        ('calc_median', calc_median),
        ('set_preference', set_preference),
        ('aff_cluster', aff_cluster),
        ('print_stat', print_stat),
        ('render_results', render_b_factor)])

    return tasks


def get_tasks_wrapper():
    tasks = get_tasks().keys()
    tasks.extend(['cluster', 'all'])
    return tasks


def run_tasks(tasks, args):

    comm, NPROCS, rank = args['mpi']

    if len(tasks) == 1:
        tsk = tasks[0]

        if tsk == 'all':
            tasks = get_tasks().keys()

        elif tsk == 'cluster':
            ntasks = get_tasks()
            ntasks.pop('render_results')
            tasks = ntasks.keys()

    for t in tasks:
        run_task(t, args)
        comm.Barrier()


def run_task(task, args):

    comm, NPROCS, rank = args['mpi']

    #Init logging
    if rank == 0:
        t0 = init_logging(task, args['verbose'])

        if args['debug'] is True:

            import cProfile
            import pstats
            import StringIO

            pr = cProfile.Profile()
            pr.enable()

    tasks = get_tasks()
    fn = tasks[task]
    fn(**args)

    if rank == 0:
        finish_logging(task, t0, args['verbose'])

        if args['debug'] is True:
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'tottime'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()

    comm.Barrier()


if __name__ == '__main__':

    mpi = init_mpi()

    comm, NPROCS, rank = mpi

    args = None
    exit = False

    if rank == 0:
        try:
            args = get_args(get_tasks_wrapper())
        except SystemExit:
            comm.Abort()

    args = comm.bcast(args)

    args['mpi'] = mpi

    run_tasks(args['task'], args)
