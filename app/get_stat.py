#!/usr/bin/python

import numpy as np
import h5py

s = h5py.File('aff_struct.hdf5', 'r')
c = np.load('aff.centers')
l = np.load('aff.labels')


with open('aff_centers.out', 'w') as f:
    for i in range(len(c)):
        f.write("%s\t%d\n" % (s['labels'][c[i]], i))


with open('aff_labels.out', 'w') as f:
    for i in range(len(l)):
        f.write("%s\t%d\n" % (s['labels'][i], l[i]))


with open('aff_stat.out', 'w') as f:
    f.write("NUMBER OF CLUSTERS: %d\n" % len(c))
    f.write("SIZE OF CLUSTERS:\n")
    cs = np.bincount(l)
    for i in range(len(c)):
        f.write("%d\t%s\t%d\n" % (i, s['labels'][c[i]], cs[i]))
