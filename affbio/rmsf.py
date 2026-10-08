#
# This file is part of the AffBio package for clustering of
# biomolecular structures.
#
# Copyright (c) 2015-2016, by Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

"""Per-atom B-factors of a cluster, computed as `gmx rmsf -fit -oq` does."""

import warnings

import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from MDAnalysis.coordinates.memory import MemoryReader

from .structures import read_coords

# gmx rmsf -oq writes B = 8 pi^2 / 3 * RMSF^2
BFACTOR = 8.0 * np.pi ** 2 / 3.0


def cluster_bfactors(center_pdb, member_pdbs, out_pdb):
    """Write the cluster center to out_pdb with RMSF-based B-factors.

    Every member is fitted onto the center (mass-weighted, as
    gmx rmsf -fit does); RMSF is the fluctuation about the average fitted
    position. Returns the B-factors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = mda.Universe(center_pdb)
        # Read members one at a time: a cluster can have more members
        # than files that may be open at once
        n_atoms = ref.atoms.n_atoms
        idx = np.arange(n_atoms)
        coords = np.array([read_coords(f, n_atoms, idx)
                           for f in member_pdbs], dtype=np.float32)
        mobile = mda.Universe(center_pdb)
        mobile.load_new(coords, format=MemoryReader)

    weights = 'mass'
    if np.any(ref.atoms.masses <= 0):
        warnings.warn('Unknown atomic masses in %s; using an unweighted fit '
                      'for B-factors.' % center_pdb)
        weights = None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        align.AlignTraj(mobile, ref, select='all', weights=weights,
                        in_memory=True).run()
        rmsf = rms.RMSF(mobile.atoms).run().results.rmsf

        bfac = BFACTOR * rmsf ** 2
        ref.atoms.tempfactors = bfac
        ref.atoms.write(out_pdb, bonds=None)

    return bfac
