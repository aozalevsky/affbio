import shutil
import types

import pytest

from affbio import checks
from affbio.checks import (AffBioError, check_decomposition, check_disk,
                           check_parallel_io, check_stage, disk_needs,
                           effective_n, estimate_memory, local_procs,
                           memory_warning)

GIB = 2 ** 30


def test_error_is_value_error():
    assert issubclass(AffBioError, ValueError)


def test_effective_n():
    assert effective_n(1047, 1) == 1047
    assert effective_n(1047, 3) == 1044
    assert effective_n(1040, 4) == 1040
    assert effective_n(200000, 7) == 199976


@pytest.mark.parametrize('n, nprocs', [(2, 1), (16, 4), (200000, 7)])
def test_decomposition_ok(n, nprocs):
    check_decomposition(n, nprocs)


@pytest.mark.parametrize('n, nprocs, message', [
    (1, 1, 'Need at least 2 structures'),
    (15, 4, 'Need at least 16 structures for 4 processes'),
    (200000, 6, 'Use at least 7 processes'),
    (40000, 1, 'Use at least 2 processes'),
])
def test_decomposition_fails(n, nprocs, message):
    with pytest.raises(AffBioError, match=message):
        check_decomposition(n, nprocs)


def test_stage_ok():
    check_stage('calc_rmsd', 240, 4)
    check_stage('prepare_matrix', 240, 4, stored_chunk=60)
    check_stage('prepare_matrix', 240, 8, stored_chunk=60)
    check_stage('aff_cluster', 240, 4)
    check_stage('aff_cluster', 1047, 1)


@pytest.mark.parametrize('stage, n, nprocs, chunk', [
    ('calc_rmsd', 240, 7, None),
    ('prepare_matrix', 240, 3, 60),
    ('prepare_matrix', 240, 1, 60),
    ('aff_cluster', 240, 7, None),
])
def test_stage_mismatch_names_original_nprocs(stage, n, nprocs, chunk):
    with pytest.raises(AffBioError, match='prepared with 4 processes'):
        check_stage(stage, n, nprocs, stored_chunk=chunk, stored_nprocs=4)


def test_disk_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 1000
    needs = disk_needs(str(tmp_path / 'm.hdf5'), n,
                       ['load_pdb', 'calc_rmsd', 'prepare_matrix',
                        'aff_cluster'])
    # rmsd + cluster next to the file, Rp + A in the working directory
    assert needs == {str(tmp_path): 4 * 4 * n * n}


def test_disk_needs_skips_existing_and_reclaims(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sfn = tmp_path / 'm.hdf5'
    sfn.write_bytes(b'x' * 1000)
    needs = disk_needs(str(sfn), 100, ['calc_rmsd', 'prepare_matrix'],
                       existing=('rmsd',), overwrite=True)
    assert needs == {str(tmp_path): 4 * 100 * 100 - 1000}


def test_check_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(checks.shutil, 'disk_usage',
                        lambda path: types.SimpleNamespace(free=10 * GIB))
    check_disk({str(tmp_path): 9 * GIB})
    with pytest.raises(AffBioError, match='need 11.0 GiB, 10.0 GiB free'):
        check_disk({str(tmp_path): 11 * GIB})
    sub = tmp_path / 'sub'
    sub.mkdir()
    # two directories on one filesystem add up
    with pytest.raises(AffBioError, match='need 12.0 GiB'):
        check_disk({str(tmp_path): 6 * GIB, str(sub): 6 * GIB})
    with pytest.raises(AffBioError, match='does not exist'):
        check_disk({str(tmp_path / 'missing'): 1})


def test_local_procs(monkeypatch):
    monkeypatch.delenv('OMPI_COMM_WORLD_LOCAL_SIZE', raising=False)
    monkeypatch.delenv('MPI_LOCALNRANKS', raising=False)
    assert local_procs() == 1
    monkeypatch.setenv('MPI_LOCALNRANKS', '6')
    assert local_procs() == 6


def test_memory(monkeypatch):
    monkeypatch.delenv('OMPI_COMM_WORLD_LOCAL_SIZE', raising=False)
    monkeypatch.delenv('MPI_LOCALNRANKS', raising=False)
    monkeypatch.setattr(checks.psutil, 'virtual_memory',
                        lambda: types.SimpleNamespace(available=4 * GIB))
    assert estimate_memory('prepare_matrix', 10000, 1) == 40 * 10000 ** 2
    assert estimate_memory('calc_rmsd', 1000, 1, natoms=214) == \
        4 * 1000 ** 2 + 96 * 1000 * 214 + 128 * 2 ** 20
    assert memory_warning('prepare_matrix', 1000, 1) is None
    w = memory_warning('prepare_matrix', 20000, 1)
    assert 'prepare_matrix needs about 14.9 GiB' in w


def test_parallel_io():
    check_parallel_io(1, False)
    check_parallel_io(4, True)
    with pytest.raises(AffBioError, match='h5py built with MPI'):
        check_parallel_io(4, False)
