import pytest

from helpers import TOP, TRJ, split_frames


@pytest.fixture(scope='session')
def adk_frames(tmp_path_factory):
    """All AdK C-alpha frames, one PDB file each, in frame order."""
    return split_frames(TOP, TRJ, str(tmp_path_factory.mktemp('frames')))
