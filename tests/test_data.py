import MDAnalysis as mda
import numpy as np

from helpers import TOP, TRJ


def test_trajectory_shape():
    u = mda.Universe(TOP, TRJ)
    assert u.atoms.n_atoms == 214
    assert u.trajectory.n_frames == 1047
    assert set(u.atoms.names) == {'CA'}


def test_frames_written(adk_frames):
    assert len(adk_frames) == 1047
    first = mda.Universe(adk_frames[0]).atoms.positions
    np.testing.assert_allclose(first, mda.Universe(TOP).atoms.positions,
                               atol=0.01)
