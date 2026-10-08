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

"""Exact median of the strict lower triangle of a big square matrix.

Radix select on float32 bit patterns: one pass histograms the top 16 bits
of every value, a second pass the low 16 bits inside the bins that hold
the middle ranks. Memory stays bounded by the block size, whatever the
matrix size.
"""

import numpy as np

_BINS = 1 << 16


def _keys(v):
    """Order-preserving map of float32 values to uint32 keys."""
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return np.where(u & 0x80000000, ~u, u | 0x80000000).astype(np.uint32)


def _value(key):
    """Inverse of _keys for one key."""
    key = np.uint32(key)
    if key & np.uint32(0x80000000):
        u = key & np.uint32(0x7FFFFFFF)
    else:
        u = ~key
    return float(np.array([u], dtype=np.uint32).view(np.float32)[0])


def _lower_keys(dset, n, rows):
    """Keys of dset[i, j], j < i, read a few rows at a time."""
    for b in range(0, n, rows):
        e = min(b + rows, n)
        block = dset[b:e, :e]
        mask = np.arange(e)[None, :] < np.arange(b, e)[:, None]
        yield _keys(block[mask])


def streaming_median(dset, block_bytes=32 * 2 ** 20):
    """Exact median of dset[i, j] for j < i of a square float32 matrix."""
    n = dset.shape[0]
    total = n * (n - 1) // 2
    if total == 0:
        raise ValueError('Need at least a 2 x 2 matrix')

    rows = max(1, block_bytes // (4 * n))
    ranks = [(total - 1) // 2, total // 2]

    # Pass 1: top 16 bits
    hist = np.zeros(_BINS, dtype=np.int64)
    for keys in _lower_keys(dset, n, rows):
        hist += np.bincount(keys >> 16, minlength=_BINS)
    cum = np.cumsum(hist)
    highs = [int(np.searchsorted(cum, r, side='right')) for r in ranks]
    offsets = [r - (int(cum[h - 1]) if h else 0)
               for r, h in zip(ranks, highs)]

    # Pass 2: low 16 bits inside the bins of the middle ranks
    low_hist = {h: np.zeros(_BINS, dtype=np.int64) for h in set(highs)}
    for keys in _lower_keys(dset, n, rows):
        top = keys >> 16
        for h, counts in low_hist.items():
            counts += np.bincount(keys[top == h] & 0xFFFF, minlength=_BINS)

    values = []
    for h, off in zip(highs, offsets):
        low = int(np.searchsorted(np.cumsum(low_hist[h]), off, side='right'))
        values.append(_value((h << 16) | low))

    return (values[0] + values[1]) / 2.0
