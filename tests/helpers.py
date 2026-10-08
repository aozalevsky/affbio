import os
import warnings

import MDAnalysis as mda

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TOP = os.path.join(DATA, 'adk_ca.pdb')
TRJ = os.path.join(DATA, 'adk_ca.xtc')


def split_frames(top, trj, outdir, stop=None):
    """One PDB file per frame - the README's preparation snippet."""
    u = mda.Universe(top, trj)
    paths = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for ts in u.trajectory[:stop]:
            path = os.path.join(outdir, 'frame%04d.pdb' % ts.frame)
            u.atoms.write(path)
            paths.append(path)
    return paths
