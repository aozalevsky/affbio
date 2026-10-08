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

"""Pairwise RMSD between two sets of structures.

Optimal superposition uses the quaternion characteristic polynomial (QCP)
method of Theobald (2005), vectorized over all pairs: one GEMM gives every
3x3 covariance matrix, then a few Newton steps find the largest eigenvalue
of each 4x4 key matrix.
"""

import numpy as np

# Structures per side of a tile; bounds temporaries to ~100 MiB
TILE = 512


def rmsd_block(A, B, superpose=True, out=None):
    """RMSD between every structure of A (la, n, 3) and B (lb, n, 3).

    superpose=True removes translation and rotation (as pyRMSD's KABSCH
    calculator); superpose=False compares raw coordinates (--noalign).
    Results go to `out` (la, lb) if given, else to a new float64 array.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if superpose:
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)

    la, n = A.shape[:2]
    lb = B.shape[0]
    if out is None:
        out = np.empty((la, lb))

    ga = np.einsum('anx,anx->a', A, A)
    gb = np.einsum('bnx,bnx->b', B, B)

    for a0 in range(0, la, TILE):
        a1 = min(a0 + TILE, la)
        for b0 in range(0, lb, TILE):
            b1 = min(b0 + TILE, lb)
            if superpose:
                msd = _qcp_msd(A[a0:a1], B[b0:b1], ga[a0:a1], gb[b0:b1])
            else:
                cross = (A[a0:a1].reshape(a1 - a0, -1)
                         @ B[b0:b1].reshape(b1 - b0, -1).T)
                msd = (ga[a0:a1, None] + gb[None, b0:b1] - 2.0 * cross) / n
            out[a0:a1, b0:b1] = np.sqrt(np.maximum(msd, 0.0))

    return out


def _qcp_msd(A, B, ga, gb):
    """Mean squared deviation after optimal superposition (centered input)."""
    la, n = A.shape[:2]
    lb = B.shape[0]

    # All 3x3 covariance matrices with one GEMM -> (la, lb, 3, 3)
    M = (A.transpose(0, 2, 1).reshape(la * 3, n)
         @ B.transpose(1, 0, 2).reshape(n, lb * 3))
    M = M.reshape(la, 3, lb, 3).transpose(0, 2, 1, 3)

    sxx, sxy, sxz = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    syx, syy, syz = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    szx, szy, szz = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]

    # Symmetric, traceless 4x4 key matrix K
    k00 = sxx + syy + szz
    k01 = syz - szy
    k02 = szx - sxz
    k03 = sxy - syx
    k11 = sxx - syy - szz
    k12 = sxy + syx
    k13 = szx + sxz
    k22 = -sxx + syy - szz
    k23 = syz + szy
    k33 = -sxx - syy + szz

    # Characteristic polynomial x^4 + c2 x^2 + c1 x + c0
    c2 = -2.0 * (M * M).sum(axis=(-2, -1))
    c1 = -8.0 * (sxx * (syy * szz - syz * szy)
                 - sxy * (syx * szz - syz * szx)
                 + sxz * (syx * szy - syy * szx))

    # c0 = det(K) by 2x2 minors of rows (0, 1) and (2, 3)
    s0 = k00 * k11 - k01 * k01
    s1 = k00 * k12 - k01 * k02
    s2 = k00 * k13 - k01 * k03
    s3 = k01 * k12 - k11 * k02
    s4 = k01 * k13 - k11 * k03
    s5 = k02 * k13 - k12 * k03
    t5 = k22 * k33 - k23 * k23
    t4 = k12 * k33 - k13 * k23
    t3 = k12 * k23 - k13 * k22
    t2 = k02 * k33 - k03 * k23
    t1 = k02 * k23 - k03 * k22
    t0 = k02 * k13 - k03 * k12
    c0 = s0 * t5 - s1 * t4 + s2 * t3 + s3 * t2 - s4 * t1 + s5 * t0

    # Newton iterations for the largest root, starting from its upper bound
    e0 = 0.5 * (ga[:, None] + gb[None, :])
    x = e0.copy()
    for _ in range(50):
        x2 = x * x
        p = (x2 + c2) * x2 + c1 * x + c0
        dp = 2.0 * (2.0 * x2 + c2) * x + c1
        step = np.divide(p, dp, out=np.zeros_like(p), where=dp != 0)
        x -= step
        done = np.abs(step) <= 1e-12 * np.abs(x)
        if done.all():
            break

    # Near a multiple largest root (collinear or nearly collinear
    # structures) Newton is unreliable: use the exact eigenvalue instead.
    # The largest root is at least s1 >= ||M||_F / sqrt(3) = sqrt(-c2 / 6).
    x2 = x * x
    dp = 2.0 * (2.0 * x2 + c2) * x + c1
    bad = (~done | (dp <= 1e-6 * np.abs(x) ** 3)
           | (x < np.sqrt(-c2 / 6.0) * (1.0 - 1e-10)))
    if bad.any():
        rows = ((k00, k01, k02, k03), (k01, k11, k12, k13),
                (k02, k12, k22, k23), (k03, k13, k23, k33))
        K = np.array([[k[bad] for k in row] for row in rows])
        x[bad] = np.linalg.eigvalsh(K.transpose(2, 0, 1))[:, -1]

    return 2.0 * (e0 - x) / n
