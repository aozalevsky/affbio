import os
import subprocess
import sys
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


def run_affbio(args, cwd, env=None, launcher=()):
    """Run `affbio ARGS` (optionally under an MPI launcher) in cwd."""
    cmd = list(launcher) + [sys.executable, '-m', 'affbio.cli'] + list(args)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                          text=True)
