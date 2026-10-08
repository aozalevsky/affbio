import h5py
import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from affbio.checks import AffBioError
from affbio.structures import (calc_rmsd_matrix, expand_pdb_list,
                               load_pdb_coords, read_coords,
                               selection_indices)
from affbio.utils import init_mpi


@pytest.fixture
def mpi():
    return init_mpi()


def qcp(a, b):
    a = np.ascontiguousarray(a - a.mean(0))
    b = np.ascontiguousarray(b - b.mean(0))
    return qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)


def test_load_with_selection(tmp_path, adk_frames, mpi):
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:40], mpi=mpi, selection='resid 1:100')
    with h5py.File(sfn, 'r') as f:
        g = f['tier1']
        assert g['struct'].shape == (40, 100, 3)
        assert g['struct'].attrs['nprocs'] == 1
        assert list(g['labels'].asstr()[:]) == adk_frames[:40]
        assert g['labels'].attrs['topology'] == adk_frames[0]
        np.testing.assert_allclose(
            g['struct'][0], read_coords(adk_frames[0], 214, np.arange(100)))


def test_broken_structure(tmp_path, adk_frames, mpi):
    lines = open(adk_frames[1]).readlines()
    del lines[next(i for i, l in enumerate(lines) if l.startswith('ATOM'))]
    broken = tmp_path / 'broken.pdb'
    broken.write_text(''.join(lines))
    with pytest.raises(AffBioError, match='Broken structure .*broken.pdb'):
        load_pdb_coords(str(tmp_path / 'm.hdf5'),
                        [adk_frames[0], str(broken)], mpi=mpi)


def test_old_prody_selection_is_explained(adk_frames):
    with pytest.raises(AffBioError, match='MDAnalysis selection syntax'):
        selection_indices(adk_frames[0], 'chain A')


def test_empty_selection(adk_frames):
    with pytest.raises(AffBioError, match='Empty selection'):
        selection_indices(adk_frames[0], 'resname XYZ')


def test_trjconv_style_frames(tmp_path, adk_frames):
    """gmx trjconv -sep frames have MODEL/TER/ENDMDL and may have CONECT."""
    atoms = [l for l in open(adk_frames[0]) if l.startswith('ATOM')]
    frame = tmp_path / 'frame0.pdb'
    frame.write_text('TITLE     Protein\nMODEL        1\n' + ''.join(atoms)
                     + 'TER\nCONECT    1    2\nENDMDL\n')
    idx = np.arange(214)
    np.testing.assert_allclose(read_coords(str(frame), 214, idx),
                               read_coords(adk_frames[0], 214, idx))


def test_expand_glob_natural_order(tmp_path):
    for k in (1, 2, 10, 9):
        (tmp_path / ('f%d.pdb' % k)).write_text('')
    assert expand_pdb_list([str(tmp_path / 'f*.pdb')]) == \
        [str(tmp_path / ('f%d.pdb' % k)) for k in (1, 2, 9, 10)]
    assert expand_pdb_list(['b.pdb', 'a.pdb']) == ['b.pdb', 'a.pdb']


@pytest.mark.parametrize('noalign', [False, True])
def test_rmsd_matrix(tmp_path, adk_frames, mpi, noalign):
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:300], mpi=mpi)
    calc_rmsd_matrix(sfn, mpi=mpi, noalign=noalign)
    with h5py.File(sfn, 'r') as f:
        X = f['tier1/struct'][:]
        R = f['tier1/rmsd'][:]
        assert f['tier1/rmsd'].attrs['chunk'] == 300
        assert f['tier1/rmsd'].attrs['nprocs'] == 1
    assert np.all(np.triu(R) == 0)
    # pairs inside and across the DIAG_ROWS = 256 row tiles
    for i, j in [(1, 0), (299, 0), (257, 256), (299, 298), (150, 3)]:
        if noalign:
            expected = np.sqrt(((X[i] - X[j]) ** 2).sum(-1).mean())
        else:
            expected = qcp(X[i], X[j])
        assert R[i, j] == pytest.approx(expected, abs=1e-5)
