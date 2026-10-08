import h5py
import numpy as np
import pytest

from affbio.median import streaming_median


def lower_median(M):
    return np.median(M[np.tril_indices(len(M), -1)].astype(np.float64))


@pytest.mark.parametrize('n', [2, 3, 5, 6, 50, 51])
def test_random_mixed_signs(n):
    M = np.random.default_rng(n).normal(0, 10, (n, n)).astype(np.float32)
    assert streaming_median(M) == lower_median(M)


def test_duplicates_and_zeros():
    M = np.random.default_rng(7).integers(-3, 4, (40, 40)).astype(np.float32)
    M[::3] = -0.0
    assert streaming_median(M) == lower_median(M)


def test_constant():
    M = np.full((10, 10), -2.5, dtype=np.float32)
    assert streaming_median(M) == -2.5


def test_small_blocks_from_hdf5(tmp_path):
    rng = np.random.default_rng(1)
    M = -((rng.random((300, 300)) * 100).astype(np.float32) ** 2)
    with h5py.File(tmp_path / 'm.hdf5', 'w') as f:
        f['cluster'] = M
        assert streaming_median(f['cluster'], block_bytes=1) == \
            lower_median(M)
        assert streaming_median(f['cluster']) == lower_median(M)


def test_needs_two_rows():
    with pytest.raises(ValueError):
        streaming_median(np.zeros((1, 1), dtype=np.float32))
