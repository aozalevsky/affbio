import os
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from helpers import run_affbio

pytest.importorskip('mpi4py')
if not h5py.get_config().mpi:
    pytest.skip('h5py is built without MPI', allow_module_level=True)
MPIRUN = shutil.which('mpirun') or shutil.which('mpiexec')
if MPIRUN is None:
    pytest.skip('mpirun not found', allow_module_level=True)

ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def launcher(nprocs):
    cmd = [MPIRUN, '-n', str(nprocs)]
    version = subprocess.run([MPIRUN, '--version'], capture_output=True,
                             text=True).stdout
    if 'Open MPI' in version or 'OpenRTE' in version:
        cmd.append('--oversubscribe')
    return cmd


def affbio(args, cwd, nprocs=1):
    return run_affbio(args, cwd=cwd, env=ENV,
                      launcher=launcher(nprocs) if nprocs > 1 else ())


def tier1(path, name):
    with h5py.File(path / 'm.hdf5', 'r') as f:
        return f['tier1'][name][:]


def rmsd_args(frames, noalign):
    return (['-m', 'm.hdf5', '-t', 'load_pdb', 'calc_rmsd', '-f'] + frames
            + (['--noalign'] if noalign else []))


@pytest.fixture(scope='module')
def serial_rmsd(tmp_path_factory, adk_frames):
    out = {}
    for noalign in (False, True):
        d = tmp_path_factory.mktemp('serial_rmsd')
        r = affbio(rmsd_args(adk_frames[:240], noalign), d)
        assert r.returncode == 0, r.stderr
        out[noalign] = tier1(d, 'rmsd')
    return out


@pytest.mark.parametrize('nprocs', [2, 3, 4])
@pytest.mark.parametrize('noalign', [False, True])
def test_rmsd_matrix_matches_serial(serial_rmsd, adk_frames, tmp_path,
                                    nprocs, noalign):
    # 240 is a multiple of 4 * nprocs for 2, 3 and 4: nothing is dropped
    r = affbio(rmsd_args(adk_frames[:240], noalign), tmp_path, nprocs)
    assert r.returncode == 0, r.stdout + r.stderr
    R = tier1(tmp_path, 'rmsd')
    il = np.tril_indices(240, -1)
    np.testing.assert_allclose(R[il], serial_rmsd[noalign][il],
                               rtol=1e-6, atol=1e-5)
    assert np.all(np.triu(R) == 0)


def test_cluster_matches_serial(adk_frames, tmp_path):
    frames = adk_frames[:1040]  # multiple of 4 * 4
    serial, parallel = tmp_path / 's', tmp_path / 'p'
    serial.mkdir()
    parallel.mkdir()
    args = ['-m', 'm.hdf5', '-t', 'cluster', '-f'] + frames
    r = affbio(args, serial)
    assert r.returncode == 0, r.stderr
    r = affbio(args, parallel, 4)
    assert r.returncode == 0, r.stdout + r.stderr
    assert np.array_equal(tier1(parallel, 'aff_centers'),
                          tier1(serial, 'aff_centers'))
    assert np.array_equal(tier1(parallel, 'aff_labels'),
                          tier1(serial, 'aff_labels'))


def test_trimming_is_logged(adk_frames, tmp_path):
    r = affbio(['-m', 'm.hdf5', '-t', 'load_pdb', '-f'] + adk_frames,
               tmp_path, 3)
    assert r.returncode == 0, r.stderr
    assert tier1(tmp_path, 'struct').shape[0] == 1044
    assert 'Using 1044 of 1047 structures' in r.stdout
    for f in adk_frames[1044:]:
        assert f in r.stdout


def test_other_process_count_is_rejected(adk_frames, tmp_path):
    r = affbio(rmsd_args(adk_frames[:240], False), tmp_path, 4)
    assert r.returncode == 0, r.stderr
    r = affbio(['-m', 'm.hdf5', '-t', 'prepare_matrix'], tmp_path, 3)
    assert r.returncode != 0
    assert 'prepared with 4 processes' in r.stderr
