import shutil

import h5py
import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from helpers import run_affbio


@pytest.fixture(scope='module')
def full_run(tmp_path_factory, adk_frames):
    d = tmp_path_factory.mktemp('full_run')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames,
                   cwd=d)
    assert r.returncode == 0, r.stderr
    return d


def read(run_dir, name, tier=1):
    with h5py.File(run_dir / 'm.hdf5', 'r') as f:
        return f['tier%d' % tier][name][:]


def test_rmsd_matrix(full_run):
    R = read(full_run, 'rmsd')
    X = read(full_run, 'struct')
    assert R.shape == (1047, 1047)
    assert np.all(np.diag(R) == 0)
    a = np.ascontiguousarray(X[700] - X[700].mean(0))
    b = np.ascontiguousarray(X[3] - X[3].mean(0))
    expected = qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)
    assert R[700, 3] == pytest.approx(expected, abs=1e-5)


def test_clusters(full_run):
    centers = read(full_run, 'aff_centers')
    labels = read(full_run, 'aff_labels')
    with h5py.File(full_run / 'm.hdf5', 'r') as f:
        attrs = dict(f['tier1/cluster'].attrs)
    assert 'median' in attrs and 'preference' in attrs
    assert 1 < len(centers) < 1047
    assert len(labels) == 1047
    assert np.array_equal(labels[centers], np.arange(len(centers)))


def test_out_files(full_run, adk_frames):
    labels = read(full_run, 'aff_labels')
    lines = (full_run / 'aff_labels.out').read_text().splitlines()
    assert lines == ['%s\t%d' % (f, k) for f, k in zip(adk_frames, labels)]
    assert "b'" not in (full_run / 'aff_centers.out').read_text()
    stat = (full_run / 'aff_stat.out').read_text()
    assert 'NUMBER OF CLUSTERS: %d' % len(read(full_run, 'aff_centers')) \
        in stat


def test_rerun_gives_identical_clusters(full_run, tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames,
                   cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert np.array_equal(read(tmp_path, 'aff_labels'),
                          read(full_run, 'aff_labels'))
    assert np.array_equal(read(tmp_path, 'aff_centers'),
                          read(full_run, 'aff_centers'))


def test_quoted_glob_in_natural_order(tmp_path, adk_frames):
    for k in range(12):
        shutil.copy(adk_frames[k], tmp_path / ('f%d.pdb' % (k + 1)))
    r = run_affbio(['-m', 'm.hdf5', '-t', 'load_pdb', '-f',
                    str(tmp_path / 'f*.pdb')], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / 'm.hdf5', 'r') as f:
        names = list(f['tier1/labels'].asstr()[:])
    assert names == [str(tmp_path / ('f%d.pdb' % (k + 1)))
                     for k in range(12)]


def test_tier2_merged_labels(full_run, tmp_path):
    shutil.copy(full_run / 'm.hdf5', tmp_path / 'm.hdf5')
    r = run_affbio(['-m', 'm.hdf5', '--tier', '2', '-t', 'cluster',
                    '--merged_labels'], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    k1 = len(read(tmp_path, 'aff_centers', tier=1))
    k2 = len(read(tmp_path, 'aff_centers', tier=2))
    merged = read(tmp_path, 'aff_labels_merged', tier=2)
    assert 0 < k2 <= k1
    assert len(merged) == 1047
    assert merged.min() >= 0 and merged.max() < k2
    assert len((tmp_path / 'aff_labels.out').read_text().splitlines()) == 1047
    assert 'AFF tier: 2' in (tmp_path / 'aff_stat.out').read_text()
