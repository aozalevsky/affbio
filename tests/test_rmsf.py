import h5py
import MDAnalysis as mda
import numpy as np
import pytest

from affbio.checks import AffBioError
from affbio.misc import cluster_to_trj, copy_connects
from affbio.rmsf import BFACTOR, cluster_bfactors
from affbio.utils import init_mpi

ELEMENTS = ['N', 'C', 'C', 'O', 'S']


def write_pdb(path, coords, elements, conect=(), end='END'):
    lines = ['ATOM  %5d %-4s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00'
             '          %2s\n' % (k, el, k, x, y, z, el)
             for k, ((x, y, z), el) in enumerate(zip(coords, elements), 1)]
    lines += ['CONECT%5d%5d\n' % pair for pair in conect]
    if end:
        lines.append(end + '\n')
    path.write_text(''.join(lines))
    return str(path)


def reference_bfactors(center, members, masses):
    """Mass-weighted Kabsch fit onto the center, then RMSF (gmx rmsf)."""
    w = masses / masses.sum()
    ref = center - w @ center
    fitted = []
    for X in members:
        Xc = X - w @ X
        U, S, Vt = np.linalg.svd((Xc * w[:, None]).T @ ref)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        fitted.append(Xc @ R.T)
    fitted = np.array(fitted)
    rmsf = np.sqrt(((fitted - fitted.mean(0)) ** 2).sum(-1).mean(0))
    return BFACTOR * rmsf ** 2


def test_matches_independent_fit(tmp_path):
    rng = np.random.default_rng(0)
    n = 25
    els = [ELEMENTS[k % 5] for k in range(n)]
    base = np.cumsum(rng.normal(0, 1.5, (n, 3)), 0)
    spread = 0.4 + 0.05 * np.arange(n)[:, None]
    frames = []
    for k in range(12):
        rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rot *= np.sign(np.linalg.det(rot))
        X = (base + rng.normal(0, spread, (n, 3))) @ rot.T + \
            rng.normal(0, 5, 3)
        frames.append(write_pdb(tmp_path / ('m%d.pdb' % k), X, els))
    out = str(tmp_path / 'out.pdb')

    bfac = cluster_bfactors(frames[0], frames, out)

    coords = [mda.Universe(f).atoms.positions.astype(np.float64)
              for f in frames]
    masses = mda.Universe(frames[0]).atoms.masses
    assert len(set(np.round(masses, 2))) > 1   # the fit really is weighted
    expected = reference_bfactors(coords[0], coords, masses)
    np.testing.assert_allclose(bfac, expected, rtol=1e-3, atol=1e-3)
    written = mda.Universe(out)
    np.testing.assert_allclose(written.atoms.tempfactors, expected,
                               atol=0.01)
    np.testing.assert_allclose(written.atoms.positions, coords[0],
                               atol=1e-3)


def test_unknown_masses_fall_back_to_unweighted(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(0, 5, (10, 3))
    files = [write_pdb(tmp_path / ('x%d.pdb' % k),
                       X + rng.normal(0, 0.3, X.shape), ['X'] * 10)
             for k in range(4)]
    with pytest.warns(UserWarning, match='unweighted fit'):
        bfac = cluster_bfactors(files[0], files, str(tmp_path / 'out.pdb'))
    assert np.all(np.isfinite(bfac))


@pytest.mark.parametrize('end', ['ENDMDL', 'END', ''])
def test_copy_connects(tmp_path, end):
    X = np.zeros((3, 3)) + np.arange(3)[:, None]
    top = write_pdb(tmp_path / 'top.pdb', X, ['C'] * 3,
                    conect=[(1, 2), (2, 3)])
    dst = write_pdb(tmp_path / 'dst.pdb', X, ['C'] * 3, end=end)
    copy_connects(top, dst)
    lines = open(dst).read().splitlines()
    conect = [k for k, l in enumerate(lines) if l.startswith('CONECT')]
    assert len(conect) == 2
    if end:
        assert lines[conect[-1] + 1] == end
        assert lines[-1] == end
    else:
        assert lines[-1].startswith('CONECT')


def test_copy_connects_without_conect(tmp_path):
    X = np.zeros((3, 3))
    top = write_pdb(tmp_path / 'top.pdb', X, ['C'] * 3)
    dst = write_pdb(tmp_path / 'dst.pdb', X, ['C'] * 3)
    before = open(dst).read()
    copy_connects(top, dst)
    assert open(dst).read() == before


def test_cluster_to_trj(small_run, tmp_path):
    sfn = str(small_run / 'm.hdf5')
    with h5py.File(sfn, 'r') as f:
        labels = f['tier1/aff_labels'][:]
    out = str(tmp_path / 'c0.pdb')
    cluster_to_trj(sfn, index=0, output=out, mpi=init_mpi())
    u = mda.Universe(out)
    assert u.trajectory.n_frames == np.sum(labels == 0)
    assert u.atoms.n_atoms == 214


def test_cluster_to_trj_needs_index(small_run):
    with pytest.raises(AffBioError, match='--index'):
        cluster_to_trj(str(small_run / 'm.hdf5'), output='x.pdb',
                       mpi=init_mpi())
