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

# General modules
import argparse as ag
import os
import sys
import traceback
from collections import OrderedDict as OD

import h5py

from affbio.utils import init_mpi, init_logging, finish_logging, dummy, \
    init_debug, finish_debug
from affbio.checks import AffBioError, check_decomposition, check_disk, \
    check_parallel_io, disk_needs, effective_n, memory_warning
from affbio.structures import load_pdb_coords, calc_rmsd_matrix, \
    expand_pdb_list, selection_indices
from affbio.prepare import prepare_cluster_matrix, calc_median, set_preference
from affbio.aff_cluster import aff_cluster, print_stat
from affbio.misc import render_b_factor, cluster_to_trj

# Tasks that create or split the N x N matrices
MATRIX_TASKS = ('load_pdb', 'calc_rmsd', 'prepare_matrix', 'aff_cluster')


def get_args(choices):
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Parallel affinity propagation for biomolecules')

    parser.add_argument('-m',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('--tier',
                        dest='tier',
                        metavar='TIER',
                        type=int,
                        default=1,
                        help='Round of clusterization')

    parser.add_argument('-t', '--task',
                        nargs='+',
                        required=True,
                        choices=choices,
                        metavar='TASK',
                        help='Task to do. Available options \
                        are: %s' % ", ".join(choices))

    parser.add_argument('-o', '--output',
                        dest='output',
                        metavar='OUTPUT',
                        help='For "render" and "cluster_to_trj" tasks \
                        name of output PNG image or multiframe PDB file')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    load_pdb = parser.add_argument_group('load_pdb')

    load_pdb.add_argument('-f',
                          nargs='*',
                          type=str,
                          dest='pdb_list',
                          metavar='FILE',
                          help='PDB files')

    load_pdb.add_argument('-s',
                          type=str,
                          dest='topology',
                          help='Topology PDB file')

    load_pdb.add_argument('--nopbc',
                          action='store_false',
                          dest='pbc',
                          help='Do not check for PBC artifacts')

    load_pdb.add_argument('--pbc_threshold',
                          type=float,
                          dest='threshold',
                          metavar='THRESHOLD',
                          default=10.0,
                          help='Threshold in Angstroms to check PBC \
                          artifacts. Default is 10.0 A')

    load_pdb.add_argument('--noalign',
                          action='store_true',
                          dest='noalign',
                          help='Do not superpose structures')

    load_pdb.add_argument('--selection',
                          default='all',
                          dest='selection',
                          help='Atom selection string in MDAnalysis syntax')

    preference = parser.add_argument_group('calculate_preference')

    preference.add_argument('--factor',
                            type=float,
                            dest='factor',
                            metavar='FACTOR',
                            default=1.0,
                            help='Multiplier for median')
    preference.add_argument('--preference',
                            type=float,
                            dest='preference',
                            metavar='PREFERENCE',
                            help='Override computed preference')

    aff = parser.add_argument_group('aff_cluster')

    aff.add_argument('--conv_iter',
                     type=int,
                     dest='conv_iter',
                     metavar='ITERATIONS',
                     default=15,
                     help='Iterations to converge')

    aff.add_argument('--max_iter',
                     type=int,
                     dest='max_iter',
                     metavar='ITERATIONS',
                     default=2000,
                     help='Maximum iterations')

    aff.add_argument('--damping',
                     type=float,
                     dest='damping',
                     metavar='DAMPING',
                     default=0.95,
                     help='Damping factor')

    stat = parser.add_argument_group('print_stat')

    stat.add_argument('--merged_labels',
                      action='store_true',
                      dest='merged',
                      help='In case of tiers > 1 print labels merged \
                        according to hierarchy')

    render = parser.add_argument_group('render')

    render.add_argument('--draw_nums',
                        action='store_true',
                        help='Draw numerical labels')

    render.add_argument('--bcolor',
                        action='store_true',
                        help='Color according to computed bfactors')

    render.add_argument('--noclear',
                        dest='clear',
                        action='store_false',
                        help='Do not clear intermidiate files')

    render.add_argument('--width',
                        nargs='?', type=int, default=640,
                        help='Width of individual image')

    render.add_argument('--height',
                        nargs='?', type=int, default=480,
                        help='Height of individual image')

    render.add_argument('--moltype',
                        nargs='?', type=str, default="general",
                        choices=["general", "origami"],
                        help='Type of molecule to draw')

    export = parser.add_argument_group('cluster_to_trj')

    export.add_argument('-i', '--index',
                        metavar='INDEX',
                        type=int,
                        dest='index',
                        help='Index of cluster to be exported')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


def main_tasks():

    tasks = OD([
        ('load_pdb', load_pdb_coords),
        ('calc_rmsd', calc_rmsd_matrix),
        ('prepare_matrix', prepare_cluster_matrix),
        ('calc_median', calc_median),
        ('set_preference', set_preference),
        ('aff_cluster', aff_cluster),
        ('print_stat', print_stat)])

    return tasks


def misc_tasks():
    tasks = OD([
        ('cluster_to_trj', cluster_to_trj),
        ('render', render_b_factor)])
    return tasks


def wrapper_tasks():
    tasks = OD([
        ('cluster', dummy),
        ('all', dummy)])
    return tasks


def expand_tasks(tasks):
    """Replace the 'cluster' and 'all' shortcuts by the tasks they run."""
    expanded = []
    for t in tasks:
        if t == 'all':
            expanded += list(main_tasks()) + list(misc_tasks())
        elif t == 'cluster':
            expanded += list(main_tasks())
        else:
            expanded.append(t)
    return expanded


def tier_size(sfn, tier, from_previous):
    """(structures, atoms, existing datasets) of a tier already on disk."""
    source = tier - 1 if from_previous else tier
    try:
        with h5py.File(sfn, 'r') as f:
            g = f['tier%d' % source]
            natoms = g['struct'].shape[1]
            if from_previous:
                n = len(g['aff_centers'])
            else:
                n = g['struct'].shape[0]
            current = f.get('tier%d' % tier)
            existing = tuple(current) if current is not None else ()
    except (OSError, KeyError) as e:
        raise AffBioError(
            'Cannot read tier %d data from %s (%s); run the earlier tasks '
            'first.' % (source, sfn, e))
    return n, natoms, existing


def preflight(tasks, args, nprocs):
    """Check sizes, process count and disk space before any work.

    Returns a list of warnings; raises AffBioError on hard errors.
    """
    if not set(tasks) & set(MATRIX_TASKS):
        return []

    sfn, tier = args['Sfn'], args['tier']

    if 'load_pdb' in tasks and tier == 1:
        pdb_list = args['pdb_list'] or []
        if not pdb_list:
            raise AffBioError('No structures given; use -f FILE ...')
        n = len(pdb_list)
        topology = args['topology'] or pdb_list[0]
        if not os.path.exists(topology):
            raise AffBioError('No such file: %s' % topology)
        natoms = len(selection_indices(topology, args['selection'])[1])
        existing = ()
    else:
        n, natoms, existing = tier_size(sfn, tier, 'load_pdb' in tasks)

    check_decomposition(n, nprocs)
    n = effective_n(n, nprocs)
    overwrite = 'load_pdb' in tasks and tier == 1
    check_disk(disk_needs(sfn, n, tasks, existing, overwrite))

    warnings = []
    for stage in ('calc_rmsd', 'prepare_matrix'):
        if stage in tasks:
            w = memory_warning(stage, n, nprocs, natoms)
            if w:
                warnings.append(w)
    return warnings


def run_tasks(tasks, args):

    comm, NPROCS, rank = args['mpi']

    for t in tasks:
        run_task(t, args)
        comm.Barrier()


def run_task(task, args):

    comm, NPROCS, rank = args['mpi']

    # Init logging
    if rank == 0:
        t0 = init_logging(task, args['verbose'])
        pr = init_debug(args['debug'])

    tasks = main_tasks()
    tasks.update(misc_tasks())
    fn = tasks[task]
    fn(**args)

    if rank == 0:
        finish_logging(task, t0, args['verbose'])
        finish_debug(pr, args['debug'])

    comm.Barrier()


def run():
    try:
        mpi = init_mpi()
    except AffBioError as e:
        sys.exit('affbio: error: %s' % e)

    comm, NPROCS, rank = mpi

    try:
        check_parallel_io(NPROCS, h5py.get_config().mpi)
    except AffBioError as e:
        if rank == 0:
            print('affbio: error: %s' % e, file=sys.stderr)
        sys.exit(1)

    args = None
    exit_code = None

    if rank == 0:
        try:
            tasks = list(main_tasks()) + list(misc_tasks()) + \
                list(wrapper_tasks())
            args = get_args(tasks)
        except SystemExit as e:
            exit_code = e.code
    exit_code = comm.bcast(exit_code)

    if exit_code is not None:
        sys.exit(exit_code)

    error = None
    if rank == 0:
        try:
            if args['pdb_list']:
                args['pdb_list'] = expand_pdb_list(args['pdb_list'])
            args['task'] = expand_tasks(args['task'])
            for warning in preflight(args['task'], args, NPROCS):
                print('affbio: warning: %s' % warning)
        except AffBioError as e:
            error = str(e)
        except Exception as e:
            # Report it to every process; otherwise they wait forever
            traceback.print_exc()
            error = 'unexpected error during checks: %s' % e
    error = comm.bcast(error)

    if error:
        if rank == 0:
            print('affbio: error: %s' % error, file=sys.stderr)
        sys.exit(1)

    args = comm.bcast(args)

    args['mpi'] = mpi

    try:
        run_tasks(args['task'], args)
    except AffBioError as e:
        where = ' (rank %d)' % rank if NPROCS > 1 else ''
        print('affbio: error%s: %s' % (where, e), file=sys.stderr)
        if NPROCS > 1:
            comm.Abort(1)
        sys.exit(1)
    except Exception:
        # Without this, the other processes would wait forever
        if NPROCS > 1:
            traceback.print_exc()
            comm.Abort(1)
        raise


if __name__ == "__main__":
    run()
