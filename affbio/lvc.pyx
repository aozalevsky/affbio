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

# This code is optimized version of Livestats package
# https://bitbucket.org/scassidy/livestats
# which implements "P-Square Algorithm for Dynamic Calculation
# of Quantiles and Histograms without Storing Observations"

#cython: cdivision=True
#cython: boundscheck=False
#cython profile=True
from libc.math cimport copysign, fabs, sqrt
import random, sys
import numpy as np
cimport numpy as np
from cython.parallel import *


#DTYPE = np.float32
#ctypedef np.int_t DTYPE_t

cdef float calcP2(float qp1, float q, float qm1, float d, float np1, float n, float nm1):
    cdef float outer, inner_left, inner_right
    outer = d / (np1 - nm1)
    inner_left = (n - nm1 + d) * (qp1 - q ) / (np1 - n)
    inner_right = (np1 - n - d) * (q - qm1 ) / (n - nm1)

    return q + outer * (inner_left + inner_right)

cdef sort(long c, float[:] hv):
    cdef:
        int i, j
        float x

    for i in xrange(1, c):
        x = hv[i]
        j = i
        while j > 0 and hv[j-1] > x:
            hv[j] = hv[j-1]
            j -= 1
        hv[j] = x

cdef class Quantile:
    cdef:
        long LEN, c
        bint initialized
        float p
        float[:] hv, nv, dv
        long[:] pv

    def __init__(self, float p):
        """ Constructs a single quantile object """
        cdef:
            long i
        self.pv = np.arange(1, self.LEN + 1)
        self.LEN = 5
        self.dv = np.array([0, p/2, p, (1 + p)/2, 1], dtype=np.float32)
        self.nv = np.array([1, 1 + 2*p, 1 + 4*p, 3 + 2*p, 5], dtype=np.float32)
        self.pv = np.arange(1, self.LEN + 1, dtype=np.int)
        self.hv = np.empty((self.LEN,), dtype=np.float32)
        self.c = 0
        self.initialized = False
        self.p = p


    cpdef add(self, float[:] item):
        cdef long l = len(item)
        for i in xrange(l):
            self.add_e(item[i])

    cdef add_e(self, float item):
        """ Adds another datum """
        cdef:
            long i, j, k
            float[:] hv, nv, dv
            long[:] pv

        hv = self.hv
        nv = self.nv
        dv = self.dv
        pv = self.pv

        if self.c < self.LEN:
            hv[self.c] = item
            self.c += 1
        else:
            if self.initialized == False:
                sort(self.LEN, hv)
                self.initialized = True

            # find cell k
            if item < hv[0]:
                hv[0] = item
                k = 1
            else:
                for i in xrange(1, self.LEN):
                    if hv[i - 1] <= item and item < hv[i]:
                        k = i
                        break
                else:
                    k = 4
                    if hv[k] < item:
                        hv[k] = item

            # increment all positions greater than k
            with nogil:
                for i in prange(k, self.LEN):
                    pv[i] += 1
                for i in prange(self.LEN):
                    nv[i] += dv[i]

        cdef:
            long d
            float n, q, qp1, qm1, np1, nm1, qn

        for i in xrange(1, self.LEN - 1):
            n = pv[i]
            q = hv[i]

            d = <long>(nv[i] - n)

            if (d >= 1 and pv[i + 1] - n > 1) or (d <= -1 and pv[i - 1] - n < -1):
                d = <int>copysign(1,d)

                qp1 = hv[i + 1]
                qm1 = hv[i - 1]
                np1 = pv[i + 1]
                nm1 = pv[i - 1]
                qn = calcP2(qp1, q, qm1, d, np1, n, nm1)

                if qm1 < qn and qn < qp1:
                    hv[i] = qn
                else:
                    # use linear form
                    hv[i] = q + d * (hv[i + d] - q) / (pv[i + d] - n)

                pv[i] = <long>(n + d)

    cpdef float quantile(self):
        cdef float[:] hv = self.hv
        if self.initialized:
            return hv[2]
        else:
            sort(self.LEN, hv)
            # make sure we don't overflow on p == 1 or underflow on p == 0
            i = <long>(min(max(self.c - 1, 0)), self.c * self.p)
            return hv[i]


cdef class LiveStats:
    cdef:
        float min_val, max_val
        float average, var_m2, skew_m3, kurt_m4
        long count
        dict tiles
        bint initialized

    def __init__(self, list p = [0.5]):
        cdef float i
        """ Constructs a LiveStream object

        Keyword arguments:

        p -- A list of quantiles to track, by default, [0.5]

        """
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.var_m2 = 0.0
        self.kurt_m4 = 0.0
        self.skew_m3 = 0.0
        self.average = 0.0
        self.count = 0
        self.tiles = {}
        self.initialized = False
        for i in p:
            self.tiles[i] = Quantile(i)

    cpdef add(self, np.ndarray[np.float32_t, ndim=1]  item):
        cdef long l = item.shape[0]
        for i in xrange(l):
            self.add_e(item[i])



    def add_e(self, float item):
        """ Adds another datum """
        cdef float delta

        delta = item - self.average

        self.min_val = min(self.min_val, item)
        self.max_val = max(self.max_val, item)

        # Average
        self.average = (self.count * self.average + item) / (self.count + 1)
        self.count = self.count + 1

        # Variance (except for the scale)
        self.var_m2 = self.var_m2 + delta * (item - self.average)

        # tiles
        for perc in list(self.tiles.values()):
            perc.add(item)

        # Kurtosis
        self.kurt_m4 = self.kurt_m4 + (item - self.average)**4.0

        # Skewness
        self.skew_m3 = self.skew_m3 + (item - self.average)**3.0


    cpdef list quantiles(self):
        """ Returns a list of tuples of the quantile and its location """
        return [(key, val.quantile()) for key, val in self.tiles.items()]

    cpdef float maximum(self):
        """ Returns the maximum value given """
        return self.max_val

    cpdef float mean(self):
        """ Returns the cumulative moving average of the data """
        return self.average

    cpdef float minimum(self):
        """ Returns the minimum value given """
        return self.min_val

    cpdef long num(self):
        """ Returns the number of items added so far"""
        return self.count

    cpdef float variance(self):
        """ Returns the sample variance of the data given so far"""
        if self.count > 1:
            return self.var_m2 / (self.count - 1)
        else:
            return float('NaN')

    cpdef float kurtosis(self):
        """ Returns the sample kurtosis of the data given so far"""
        if self.count > 1:
            return self.kurt_m4 / (self.count * self.variance()**2.0) - 3.0
        else:
            return float('NaN')

    cpdef float skewness(self):
        """ Returns the sample skewness of the data given so far"""
        if self.count > 1:
            return self.skew_m3 / (self.count * self.variance()**1.5)
        else:
            return float('NaN')


def bimodal( low1, high1, mode1, low2, high2, mode2 ):
    toss = random.choice( (1, 2) )
    if toss == 1:
        return random.triangular( low1, high1, mode1 )
    else:
        return random.triangular( low2, high2, mode2 )

def output (tiles, data, stats, name):
    data.sort()
    tuples = [x[1] for x in stats.quantiles()]
    med = [data[int(len(data) * x)] for x in tiles]
    pe = 0
    for approx, exact in zip(tuples, med):
        pe = pe + (fabs(approx - exact)/fabs(exact))
    pe = 100.0 * pe / len(data)
    avg = sum(data)/len(data)

    s2 = 0
    for x in data:
        s2 = s2 + (x - avg)**2
    var = s2 / len(data)

    v_pe = 100.0*fabs(stats.variance() - var)/fabs(var)
    avg_pe = 100.0*fabs(stats.mean() - avg)/fabs(avg)

    print("{}: Avg%E {} Var%E {} Quant%E {}, Kurtosis {}, Skewness {}".format(
            name, avg_pe, v_pe, pe, stats.kurtosis(), stats.skewness()));


if __name__ == '__main__':
    count = int(sys.argv[1])
    random.seed()

    tiles = [0.25, 0.5, 0.75]

    median = LiveStats(tiles)
    test = [0.02, 0.15, 0.74, 3.39, 0.83, 22.37, 10.15, 15.43, 38.62, 15.92, 34.60,
            10.28, 1.47, 0.40, 0.05, 11.39, 0.27, 0.42, 0.09, 11.37]
    for i in test:
        median.add(i)

    output(tiles, test, median, "Test")

    median = LiveStats(tiles)
    x = list(xrange(count))
    random.shuffle(x)
    for i in x:
        median.add(i)

    output(tiles, x, median, "Uniform")

    median = LiveStats(tiles)
    for i in xrange(count):
        x[i] = random.expovariate(1.0/435)
        median.add(x[i])

    output(tiles, x, median, "Expovar")

    median = LiveStats(tiles)
    for i in xrange(count):
        x[i] = random.triangular(-1000*count/10, 1000*count/10, 100)
        median.add(x[i])

    output(tiles, x, median, "Triangular")

    median = LiveStats(tiles)
    for i in xrange(count):
        x[i] = bimodal(0, 1000, 500, 500, 1500, 1400)
        median.add(x[i])

    output(tiles, x, median, "Bimodal")



