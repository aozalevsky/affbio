import pytest

from helpers import TOP, TRJ, run_affbio, split_frames


@pytest.fixture(scope='session')
def adk_frames(tmp_path_factory):
    """All AdK C-alpha frames, one PDB file each, in frame order."""
    return split_frames(TOP, TRJ, str(tmp_path_factory.mktemp('frames')))


@pytest.fixture(scope='session')
def small_run(tmp_path_factory, adk_frames):
    """Directory with m.hdf5 after `affbio -t cluster` on 120 frames."""
    d = tmp_path_factory.mktemp('small_run')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames[:120],
                   cwd=d)
    assert r.returncode == 0, r.stderr
    return d
