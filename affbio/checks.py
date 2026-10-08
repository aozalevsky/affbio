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

"""Up-front validation of problem size, process count, disk and memory."""

import math
import os
import shutil

import psutil

# One float32 l x l block must stay below HDF5's 4 GiB chunk limit
MAX_BLOCK = 32767
GIB = 2.0 ** 30


class AffBioError(ValueError):
    """A problem with the input or setup, reported without a traceback."""


def effective_n(n, nprocs):
    """Number of structures used so that every stage splits evenly."""
    if nprocs == 1:
        return n
    return n - n % (4 * nprocs)


def check_decomposition(n, nprocs):
    """Fail early if n structures cannot be split across nprocs processes."""
    if nprocs == 1:
        if n < 2:
            raise AffBioError(
                'Need at least 2 structures to cluster, got %d.' % n)
    elif n < 4 * nprocs:
        raise AffBioError(
            'Need at least %d structures for %d processes, got %d. '
            'Use at most %d processes.'
            % (4 * nprocs, nprocs, n, max(1, n // 4)))

    l = effective_n(n, nprocs) // nprocs
    if l > MAX_BLOCK:
        raise AffBioError(
            '%d structures on %d processes give %d x %d matrix blocks; '
            'one float32 block would exceed the 4 GiB HDF5 chunk limit. '
            'Use at least %d processes.'
            % (n, nprocs, l, l, math.ceil(n / MAX_BLOCK)))


def check_stage(stage, n, nprocs, stored_chunk=None, stored_nprocs=None):
    """Fail if a matrix stage cannot split this file's n structures."""
    if stage == 'calc_rmsd':
        ok = n % nprocs == 0
    elif stage == 'prepare_matrix':
        ok = n % nprocs == 0 and (
            stored_chunk is None or stored_chunk % (n // nprocs) == 0)
    elif stage == 'aff_cluster':
        ok = nprocs == 1 or n % (4 * nprocs) == 0
    else:
        raise ValueError('Unknown stage %r' % stage)

    if not ok:
        if stored_nprocs:
            hint = ('this file was prepared with %d processes; rerun this '
                    'stage with %d processes'
                    % (stored_nprocs, stored_nprocs))
        else:
            hint = 'rerun load_pdb with %d processes first' % nprocs
        raise AffBioError(
            '%s cannot split %d structures across %d processes: %s.'
            % (stage, n, nprocs, hint))


def disk_needs(sfn, n, tasks, existing=(), overwrite=False):
    """Bytes each directory still has to hold for the requested tasks."""
    matrix = 4 * n * n  # one float32 N x N dataset
    sdir = os.path.dirname(os.path.abspath(sfn))
    needs = {sdir: 0}
    if 'calc_rmsd' in tasks and 'rmsd' not in existing:
        needs[sdir] += matrix
    if 'prepare_matrix' in tasks and 'cluster' not in existing:
        needs[sdir] += matrix
    if 'aff_cluster' in tasks:
        # aff_cluster keeps two more N x N matrices in the working directory
        cwd = os.getcwd()
        needs[cwd] = needs.get(cwd, 0) + 2 * matrix
    if overwrite and os.path.exists(sfn):
        # load_pdb recreates the file, so its current size is freed
        needs[sdir] = max(0, needs[sdir] - os.path.getsize(sfn))
    return needs


def check_disk(needs):
    """Fail if a filesystem lacks space; needs maps directory -> bytes."""
    per_fs = {}
    for d, nbytes in needs.items():
        if nbytes <= 0:
            continue
        if not os.path.isdir(d):
            raise AffBioError('Directory %s does not exist.' % d)
        dev = os.stat(d).st_dev
        total, dirs = per_fs.get(dev, (0, []))
        per_fs[dev] = (total + nbytes, dirs + [d])

    for total, dirs in per_fs.values():
        free = shutil.disk_usage(dirs[0]).free
        if total > free:
            raise AffBioError(
                'Not enough disk space in %s: need %.1f GiB, %.1f GiB free.'
                % (', '.join(sorted(set(dirs))), total / GIB, free / GIB))


def local_procs():
    """Number of MPI processes on this node (Open MPI or MPICH), else 1."""
    for var in ('OMPI_COMM_WORLD_LOCAL_SIZE', 'MPI_LOCALNRANKS'):
        if var in os.environ:
            return int(os.environ[var])
    return 1


def estimate_memory(stage, n, nprocs, natoms=0):
    """Approximate peak bytes per process for a matrix stage."""
    l = effective_n(n, nprocs) // nprocs
    if stage == 'calc_rmsd':
        # result block, two coordinate blocks and their centered copies,
        # RMSD tile temporaries
        return 4 * l * l + 96 * l * natoms + 128 * 2 ** 20
    if stage == 'prepare_matrix':
        # float32 blocks plus float64 noise temporaries
        return 40 * l * l
    raise ValueError('Unknown stage %r' % stage)


def memory_warning(stage, n, nprocs, natoms=0):
    """Warning text if a stage likely needs more memory than is free."""
    procs = local_procs()
    need = estimate_memory(stage, n, nprocs, natoms) * procs
    avail = psutil.virtual_memory().available
    if need <= avail:
        return None
    return ('%s needs about %.1f GiB of memory on this node (%d processes), '
            'but only %.1f GiB is available; use more processes or nodes.'
            % (stage, need / GIB, procs, avail / GIB))


def check_parallel_io(nprocs, h5py_mpi):
    """Fail if several processes would share an HDF5 file without MPI-IO."""
    if nprocs > 1 and not h5py_mpi:
        raise AffBioError(
            'Parallel runs need h5py built with MPI support '
            '(h5py.get_config().mpi is False); see "Parallel runs" in the '
            'README.')
