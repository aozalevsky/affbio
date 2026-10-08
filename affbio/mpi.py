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

"""MPI communicator, or a single-process stand-in when mpi4py is missing."""

import os

import numpy as np

from .checks import AffBioError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Buffer datatypes for the Gather/Bcast/Reduce calls in aff_cluster
INT = MPI.INT if MPI is not None else None
FLOAT = MPI.FLOAT if MPI is not None else None

# Number of processes, set by mpirun/mpiexec (Open MPI, MPICH),
# srun --mpi=pmi2 and srun in general
LAUNCHER_SIZE_VARS = ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE',
                      'SLURM_STEP_NUM_TASKS')
# Rank of this process under PMIx (srun --mpi=pmix), which sets no size
LAUNCHER_RANK_VARS = ('PMIX_RANK',)


def launched_in_parallel():
    """True if an MPI launcher started this as one of several processes."""
    return (any(int(os.environ.get(v, '1')) > 1 for v in LAUNCHER_SIZE_VARS)
            or any(int(os.environ.get(v, '0')) > 0
                   for v in LAUNCHER_RANK_VARS))


class SerialComm(object):
    """The part of mpi4py's COMM_WORLD that affbio uses, for one process."""

    size = 1
    rank = 0

    def Barrier(self):
        pass

    def bcast(self, obj, root=0):
        return obj

    def Bcast(self, buf, root=0):
        pass

    def Gather(self, sendbuf, recvbuf, root=0):
        _copy(sendbuf, recvbuf)

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        _copy(sendbuf, recvbuf)


def _array(buf):
    """mpi4py accepts an array or an [array, datatype] list."""
    return buf[0] if isinstance(buf, (list, tuple)) else buf


def _copy(sendbuf, recvbuf):
    src = np.ravel(_array(sendbuf))
    _array(recvbuf).flat[:src.size] = src


def get_comm():
    """COMM_WORLD from mpi4py, or SerialComm when mpi4py is not installed."""
    if MPI is not None:
        return MPI.COMM_WORLD
    if launched_in_parallel():
        raise AffBioError(
            'affbio was started by an MPI launcher, but mpi4py is not '
            "installed. Install it with: pip install 'affbio[mpi]'")
    return SerialComm()
