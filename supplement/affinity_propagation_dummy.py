import h5py
import numpy as np
import tempfile
import os.path as osp
import shutil

from mpi4py import MPI


preference = None
convergence_iter = 5
max_iter = 200
damping = 0.5
verbose = True

#Init storage for matrices
pdb = 'aff.list'
matf = 'aff.raw'
#HDF5 file
f = h5py.File(matf, 'r')
#Table for RMSD
#rmsd_matrix = f['rmsd']
#Table for clusterization
s = f['cluster']
#Reread structures by every process

if s.shape[0] != s.shape[1]:
    raise ValueError("S must be a square array (shape=%s)" % repr(s.shape))

if preference is None:
    preference = np.median(s)
if damping < 0.5 or damping >= 1:
    raise ValueError('damping must be >= 0.5 and < 1')


tmpdir = tempfile.mkdtemp(dir=osp.abspath(osp.dirname(s.file.filename)))
D = h5py.File(osp.join(tmpdir, 'tmp.hdf5'))

N = s.len()
S = D.create_dataset('S', (N, N), dtype='float32')

random_state = np.random.RandomState(0)
x = np.finfo(np.double).eps
y = np.finfo(np.double).tiny * 100

A = D.create_dataset('A', (N, N), dtype='float32')
Aold = D.create_dataset('Aold', (N, N), dtype='float32')

R = D.create_dataset('R', (N, N), dtype='float32')
Rold = D.create_dataset('Rold', (N, N), dtype='float32')
Rp = D.create_dataset('Rp', (N, N), dtype='float32')

C = D.create_dataset('C', (N,), dtype='int')

e = D.create_dataset('e', (N, convergence_iter), dtype='bool')

# Remove degeneracies
for i in range(N):
    # Place preference on the diagonal of S
    for j in range(N):
        S[i, j] = (x * s[i, j] + y) * random_state.randn() + s[i, j]
    S[i, i] = (x * preference + y) * random_state.randn() + preference


for it in range(max_iter):

    # Compute responsibilities
    for i in range(N):
        mj, m, m2 = None, None, None
        for j in range(N):
            n = A[i, j] + S[i, j]
            if n > m:
                mj = j
                m2 = m
                m = n
            elif n > m2:
                m2 = n

        for j in range(N):
            n = S[i, j] - m
            if j == mj:
                n = S[i, mj] - m2

            o = R[i, j]
            Rold[i, j] = o

            n = (1 - damping) * n + damping * o
            R[i, j] = n

            if n > 0:
                pass
            else:
                if i == j:
                    pass
                else:
                    n = 0
            Rp[i, j] = n

    # Compute availabilities
    for j in range(N):

        s = 0
        for i in range(N):
            s = s + Rp[i, j]

        for i in range(N):
            n = s - Rp[i, j]

            if n < 0:
                pass
            else:
                if j == i:
                    pass
                else:
                    n = 0

            o = A[i, j]
            Aold[i, j] = o
            A[i, j] = (1 - damping) * n + damping * o

    # Check for convergence
    itc = it % convergence_iter
    K = 0
    for i in range(N):
        if (A[i, i] + R[i, i]) > 0:
            n = True
            K = K + 1
        else:
            n = False
        e[i, itc] = n

    if it >= convergence_iter:
        st = 0
        for i in range(N):
            s = 0
            for j in range(convergence_iter):
                s = s + e[i, j]
            if s == convergence_iter or s == 0:
                st = st + 1
        unconverged = st != N

        if (not unconverged and (K > 0)) or (it == max_iter):
            if verbose:
                print("Converged after %d iterations." % it)
            break
else:
    if verbose:
        print("Did not converge")

I = []
for i in range(N):
    n = A[i, i] + R[i, i]
    if n > 0:
        I.append(i)

K = len(I)

if K > 0:

    for i in range(N):
        m, mj = None, None
        for j in range(K):
            jj = I[j]
            n = S[i, jj]
            if n >= m:
                m = n
                mj = j
        C[i] = mj

    for i in range(K):
        C[I[i]] = i

    clt = [[] for k in range(K)]
    for i in range(N):
        clt[C[i]].append(i)

    for k in range(K):
        m, mj = None, None
        for j in clt[k]:
            s = 0
            for i in clt[k]:
                s = s + S[i, j]
            if s > m:
                m = s
                mj = j
        I[k] = mj

    for i in range(N):
        m, mj = None, None
        for j in range(K):
            jj = I[j]
            n = S[i, jj]
            if n >= m:
                m = n
                mj = j
        C[i] = mj

    for i in range(K):
        C[I[i]] = i

else:
    I = [np.nan]
    for i in ((N, )):
        C[i] = np.nan

shutil.rmtree(tmpdir)

print I, C[:]
print S[0,9], S[0,20]
