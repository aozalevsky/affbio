import h5py
import numpy as np
import pytest

from affbio.aff_cluster import aff_cluster, print_stat
from affbio.checks import AffBioError
from affbio.prepare import calc_median, prepare_cluster_matrix, \
    set_preference
from affbio.structures import calc_rmsd_matrix, load_pdb_coords
from affbio.utils import init_mpi


@pytest.fixture
def matrix(tmp_path, adk_frames):
    """Cluster matrix with preference for 150 AdK frames."""
    mpi = init_mpi()
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:150], mpi=mpi)
    calc_rmsd_matrix(sfn, mpi=mpi)
    prepare_cluster_matrix(sfn, mpi=mpi)
    calc_median(sfn, mpi=mpi)
    set_preference(sfn, mpi=mpi)
    return sfn, mpi


def test_cluster_matrix_and_preference(matrix):
    sfn, mpi = matrix
    il = np.tril_indices(150, -1)
    with h5py.File(sfn, 'r') as f:
        R = f['tier1/rmsd'][:].astype(np.float64)
        C = f['tier1/cluster'][:]
        attrs = dict(f['tier1/cluster'].attrs)
    assert attrs['nprocs'] == 1
    # similarity = -RMSD^2 plus tiny noise, symmetric
    np.testing.assert_allclose(C[il], -(R[il] ** 2), rtol=1e-5)
    np.testing.assert_allclose(C[il], C.T[il], rtol=1e-5)
    median = np.median(C[il].astype(np.float64))
    assert attrs['median'] == pytest.approx(median)
    assert attrs['preference'] == pytest.approx(median, rel=1e-6)
    np.testing.assert_allclose(np.diag(C), attrs['preference'], rtol=1e-5)


def test_aff_cluster_and_stat(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    aff_cluster(sfn, mpi=mpi)
    print_stat(sfn, mpi=mpi)
    with h5py.File(sfn, 'r') as f:
        centers = f['tier1/aff_centers'][:]
        labels = f['tier1/aff_labels'][:]
        names = f['tier1/labels'].asstr()[:]
    assert labels.dtype == np.int64
    assert 1 < len(centers) < 150
    assert len(labels) == 150
    assert np.array_equal(labels[centers], np.arange(len(centers)))
    stat = (work / 'aff_stat.out').read_text()
    assert 'NUMBER OF CLUSTERS: %d' % len(centers) in stat
    lines = (work / 'aff_labels.out').read_text().splitlines()
    assert lines[0] == '%s\t%d' % (names[0], labels[0])
    assert "b'" not in (work / 'aff_centers.out').read_text()
    # only the three .out files remain; the temporary HDF5 file is gone
    assert sorted(p.name for p in work.iterdir()) == \
        ['aff_centers.out', 'aff_labels.out', 'aff_stat.out']


def test_no_exemplars_is_reported(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(AffBioError, match='no exemplars'):
        aff_cluster(sfn, mpi=mpi, conv_iter=1, max_iter=2, damping=0.99)
    assert list(work.iterdir()) == []


def test_bad_damping(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AffBioError, match='damping'):
        aff_cluster(sfn, mpi=mpi, damping=1.5)
