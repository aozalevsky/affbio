import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from affbio import rmsd
from affbio.rmsd import rmsd_block


def ensemble(k, n=60, seed=0):
    """k chain-like structures in random poses; some identical up to pose."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 2.2, (n, 3)), axis=0)
    X = base[None] + rng.normal(0, 1.5, (k, n, 3))
    X[:k // 5] = X[0]
    rot = np.linalg.qr(rng.normal(size=(k, 3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))[:, None, None]  # proper rotations
    return np.einsum('kij,knj->kni', rot, X) + rng.normal(0, 20, (k, 1, 3))


def qcp_reference(a, b):
    a = np.ascontiguousarray(a - a.mean(0), dtype=np.float64)
    b = np.ascontiguousarray(b - b.mean(0), dtype=np.float64)
    return qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)


def test_matches_mdanalysis_qcprot():
    E = ensemble(55, seed=1)
    A, B = E[:30], E[30:]
    ref = np.array([[qcp_reference(a, b) for b in B] for a in A])
    np.testing.assert_allclose(rmsd_block(A, B), ref, atol=1e-6)


def test_known_cases():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 5, (40, 3))
    th = 0.7
    rz = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0],
                   [0, 0, 1]])
    cases = np.array([a, a + [3, 4, 0], a @ rz.T, a * [-1, 1, 1]])
    r = rmsd_block(cases[:1], cases)[0]
    assert r[0] < 1e-5   # identical
    assert r[1] < 1e-5   # translated
    assert r[2] < 1e-5   # rotated
    assert r[3] > 1.0    # a mirror image cannot be superposed


def test_noalign_is_raw_rmsd():
    A, B = ensemble(6), ensemble(5, seed=4)
    raw = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1).mean(-1))
    np.testing.assert_allclose(rmsd_block(A, B, superpose=False), raw,
                               rtol=1e-12, atol=1e-9)
    # without superposition a translation is not removed
    shifted = rmsd_block(A[:1], A[:1] + [3, 4, 0], superpose=False)
    assert shifted[0, 0] == pytest.approx(5.0)


def test_tiles_and_out(monkeypatch):
    E = ensemble(23, n=17)
    full = rmsd_block(E, E)
    monkeypatch.setattr(rmsd, 'MAX_TILE', 4)
    out = np.zeros((23, 23), dtype=np.float32)
    assert rmsd_block(E, E, out=out) is out
    np.testing.assert_allclose(out, full, atol=1e-5)


def test_single_atom_structures():
    A = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    assert np.all(rmsd_block(A, A) == 0)


def svd_reference(a, b):
    """Kabsch RMSD from the singular values of the covariance matrix."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    M = a.T @ b
    s = np.linalg.svd(M, compute_uv=False)
    d = np.sign(np.linalg.det(M))
    msd = ((a ** 2).sum() + (b ** 2).sum()
           - 2 * (s[0] + s[1] + d * s[2])) / len(a)
    return np.sqrt(max(msd, 0.0))


def random_poses(X, seed):
    rng = np.random.default_rng(seed)
    rot = np.linalg.qr(rng.normal(size=(len(X), 3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))[:, None, None]
    return np.einsum('kij,knj->kni', rot, X) + rng.normal(0, 10, (len(X), 1, 3))


def test_two_atom_structures():
    """Collinear structures make the largest eigenvalue of K degenerate."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 3, (300, 2, 3))
    b = random_poses(a, 6)          # same structures, new poses
    assert np.diag(rmsd_block(a, b)).max() < 1e-5
    ref = np.array([[svd_reference(x, y) for y in b[:40]] for x in a[:40]])
    np.testing.assert_allclose(rmsd_block(a[:40], b[:40]), ref, atol=1e-6)


def test_near_collinear_chains():
    rng = np.random.default_rng(7)
    line = np.linspace(0, 30, 20)[:, None] * np.array([1.0, 0.0, 0.0])
    E = random_poses(line[None] + rng.normal(0, 1e-3, (12, 20, 3)), 8)
    ref = np.array([[svd_reference(x, y) for y in E] for x in E])
    np.testing.assert_allclose(rmsd_block(E, E), ref, atol=1e-6)


def test_memory_stays_bounded_for_large_structures():
    """Temporaries must not grow with the number of structures per block."""
    import tracemalloc
    rng = np.random.default_rng(9)
    n = 20000
    A = rng.normal(0, 10, (120, n, 3)).astype(np.float32)
    B = rng.normal(0, 10, (120, n, 3)).astype(np.float32)
    out = np.empty((120, 120), dtype=np.float32)
    tracemalloc.start()
    rmsd_block(A, B, out=out)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # inputs are 27 MiB each; copies of whole blocks took ~220 MiB
    assert peak < 120 * 2 ** 20, peak / 2 ** 20
    ref = rmsd_block(A[:3].astype(np.float64), B[:2].astype(np.float64))
    np.testing.assert_allclose(out[:3, :2], ref, rtol=1e-5)
