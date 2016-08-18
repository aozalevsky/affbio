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
import time
import cProfile
import pstats
import StringIO


# MPI parallelism
from mpi4py import MPI


def dummy(*args, **kwargs):
    pass


def init_mpi():
    # Get MPI info
    comm = MPI.COMM_WORLD
    # Get number of processes
    NPROCS = comm.size
    # Get rank
    rank = comm.rank

    return (comm, NPROCS, rank)


def master(fn):
    comm, NPROCS, rank = init_mpi()

    if rank == 0:
        return fn
    else:
        return dummy


class Bunch(object):

    def __init__(self, adict):
        self.__dict__.update(adict)


def task(N, NPROCS, rank):
    l = N / NPROCS
    b = rank * l
    return (b, b + l)


def init_logging(task, verbose=False):
    if verbose:

        print 'Starting task: %s' % task

    # Get current time
    t0 = time.time()

    return t0


def finish_logging(task, t0, verbose=False):
    if verbose:
        print "Task: %s execution time is %f" % (task, time.time() - t0)


def init_debug(debug=False):
    if debug is True:
        pr = cProfile.Profile()
        pr.enable()
    else:
        pr = None
    return pr


def finish_debug(pr, debug=False):
    if debug is True:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
