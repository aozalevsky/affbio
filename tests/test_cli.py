import os
import shutil
import subprocess
import sys

import h5py
import pytest

from affbio.cli import expand_tasks
from helpers import run_affbio

HIDE_MPI = ("import sys; sys.modules['mpi4py'] = None; "
            "from affbio.cli import run; run()")


def run_without_mpi4py(args, cwd, env=None):
    return subprocess.run([sys.executable, '-c', HIDE_MPI] + list(args),
                          cwd=cwd, env=env, capture_output=True, text=True)


def test_help_lists_tasks(tmp_path):
    r = run_affbio(['--help'], cwd=tmp_path)
    assert r.returncode == 0
    for name in ('load_pdb', 'calc_rmsd', 'prepare_matrix', 'calc_median',
                 'set_preference', 'aff_cluster', 'print_stat',
                 'cluster_to_trj', 'render', 'cluster', 'all'):
        assert name in r.stdout


def test_expand_tasks():
    assert expand_tasks(['cluster'])[0] == 'load_pdb'
    assert expand_tasks(['cluster'])[-1] == 'print_stat'
    assert expand_tasks(['cluster', 'render'])[-1] == 'render'
    assert expand_tasks(['calc_median']) == ['calc_median']


def test_single_structure_is_rejected(tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f', adk_frames[0]],
                   cwd=tmp_path)
    assert r.returncode == 1
    assert 'Need at least 2 structures' in r.stderr
    assert 'Traceback' not in r.stderr


def test_bad_selection_is_explained(tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '--selection', 'chain A',
                    '-f'] + adk_frames[:5], cwd=tmp_path)
    assert r.returncode == 1
    assert 'MDAnalysis selection syntax' in r.stderr
    assert 'Traceback' not in r.stderr


def test_missing_matrix_file_is_explained(tmp_path):
    r = run_affbio(['-m', 'nothing.hdf5', '-t', 'calc_rmsd'], cwd=tmp_path)
    assert r.returncode == 1
    assert 'run load_pdb first' in r.stderr
    assert 'Traceback' not in r.stderr


def test_runs_without_mpi4py(tmp_path, adk_frames):
    r = run_without_mpi4py(['-m', 'm.hdf5', '-t', 'load_pdb', 'calc_rmsd',
                            '-f'] + adk_frames[:20], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / 'm.hdf5', 'r') as f:
        assert f['tier1/rmsd'].shape == (20, 20)


def test_mpi_launcher_without_mpi4py(tmp_path):
    env = dict(os.environ, OMPI_COMM_WORLD_SIZE='2')
    r = run_without_mpi4py(['--help'], cwd=tmp_path, env=env)
    assert r.returncode == 1
    assert 'affbio[mpi]' in r.stderr


def test_unreadable_topology_is_explained(tmp_path, adk_frames):
    garbage = tmp_path / 'garbage.pdb'
    garbage.write_text('this is not a PDB file\n')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-s', str(garbage),
                    '-f'] + adk_frames[:5], cwd=tmp_path)
    assert r.returncode == 1
    assert 'Cannot read topology' in r.stderr
    assert 'Traceback' not in r.stderr


HIDE_PYMOL = ("import sys; sys.modules['pymol2'] = None; "
              "from affbio.cli import run; run()")


def run_without_pymol(args, cwd):
    return subprocess.run([sys.executable, '-c', HIDE_PYMOL] + list(args),
                          cwd=cwd, capture_output=True, text=True)


def test_render_without_pymol_fails_before_any_work(small_run, tmp_path):
    shutil.copy(small_run / 'm.hdf5', tmp_path / 'm.hdf5')
    r = run_without_pymol(['-m', 'm.hdf5', '-t', 'render'], tmp_path)
    assert r.returncode == 1
    assert 'affbio[render]' in r.stderr
    assert 'Traceback' not in r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ['m.hdf5']


def test_cluster_and_render_without_pymol_stops_up_front(adk_frames,
                                                         tmp_path):
    r = run_without_pymol(['-m', 'm.hdf5', '-t', 'cluster', 'render', '-f']
                          + adk_frames[:20], tmp_path)
    assert r.returncode == 1
    assert 'affbio[render]' in r.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def prepared(tmp_path, adk_frames):
    """m.hdf5 with RMSD and cluster matrices, but no median yet."""
    r = run_affbio(['-m', 'm.hdf5', '-t', 'load_pdb', 'calc_rmsd',
                    'prepare_matrix', '-f'] + adk_frames[:20], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    return tmp_path


@pytest.mark.parametrize('task, hint', [
    ('set_preference', 'run calc_median first'),
    ('aff_cluster', 'run set_preference first'),
    ('print_stat', 'run aff_cluster first'),
    ('render', 'run aff_cluster first'),
])
def test_missing_inputs_are_explained(prepared, task, hint):
    if task == 'render':
        # without PyMOL, render stops at the PyMOL check first
        pytest.importorskip('pymol2')
    r = run_affbio(['-m', 'm.hdf5', '-t', task], cwd=prepared)
    assert r.returncode == 1
    assert hint in r.stderr
    assert 'Traceback' not in r.stderr


def test_task_on_missing_file_is_explained(tmp_path):
    r = run_affbio(['-m', 'nothing.hdf5', '-t', 'calc_median'], cwd=tmp_path)
    assert r.returncode == 1
    assert 'run prepare_matrix first' in r.stderr
    assert 'Traceback' not in r.stderr


def test_explicit_preference_needs_no_median(prepared):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'set_preference',
                    '--preference', '-5'], cwd=prepared)
    assert r.returncode == 0, r.stderr
