import numpy as np
import pytest

from affbio import mpi
from affbio.checks import AffBioError
from affbio.mpi import SerialComm, get_comm
from affbio.utils import finish_debug, init_debug, init_mpi, task


def test_serial_comm():
    comm = SerialComm()
    assert (comm.size, comm.rank) == (1, 0)
    assert comm.bcast({'a': 1}) == {'a': 1}
    src = np.arange(6, dtype=np.int64)
    dst = np.zeros(6, dtype=np.int64)
    comm.Gather([src, None], [dst, None])
    assert np.array_equal(dst, src)
    dst2 = np.zeros(6, dtype=np.int64)
    comm.Reduce(src, dst2)
    assert np.array_equal(dst2, src)
    comm.Barrier()
    comm.Bcast([dst, None])


def test_without_mpi4py(monkeypatch):
    monkeypatch.setattr(mpi, 'MPI', None)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    monkeypatch.delenv('PMI_SIZE', raising=False)
    assert isinstance(get_comm(), SerialComm)


@pytest.mark.parametrize('var', ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'])
def test_launcher_without_mpi4py(monkeypatch, var):
    monkeypatch.setattr(mpi, 'MPI', None)
    monkeypatch.setenv(var, '4')
    with pytest.raises(AffBioError, match=r"affbio\[mpi\]"):
        get_comm()


def test_utils():
    comm, nprocs, rank = init_mpi()
    assert (nprocs, rank) == (1, 0)
    assert task(10, 2, 1) == (5, 10)
    assert isinstance(task(10, 3, 0)[1], int)
    finish_debug(init_debug(True), True)


@pytest.mark.parametrize('var, value', [('SLURM_STEP_NUM_TASKS', '4'),
                                        ('PMIX_RANK', '2')])
def test_slurm_or_pmix_launch_without_mpi4py(monkeypatch, var, value):
    monkeypatch.setattr(mpi, 'MPI', None)
    for v in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_STEP_NUM_TASKS',
              'PMIX_RANK'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, value)
    with pytest.raises(AffBioError, match=r"affbio\[mpi\]"):
        get_comm()


@pytest.mark.parametrize('var, value', [('SLURM_STEP_NUM_TASKS', '1'),
                                        ('PMIX_RANK', '0')])
def test_single_task_launch_runs_serially(monkeypatch, var, value):
    monkeypatch.setattr(mpi, 'MPI', None)
    for v in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_STEP_NUM_TASKS',
              'PMIX_RANK'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, value)
    assert isinstance(get_comm(), SerialComm)
