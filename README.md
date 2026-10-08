# AffBio

AffBio clusters sets of biomolecular structures, such as snapshots of a
molecular dynamics trajectory, with [Affinity Propagation][1]. It is built
for large sets: the RMSD and similarity matrices are kept on disk in HDF5,
and every stage can run in parallel with MPI. The clustering code is inspired
by the implementation in [scikit-learn][2]. AffBio was first developed for
clustering DNA origami structures ([Zalevsky et al., Nucleic Acids Research
2018][3]).

## Installation

AffBio needs Python 3.10 or newer:

```
pip install affbio
```

This is enough to cluster structures on one process. Two optional extras:

```
pip install "affbio[render]"   # PyMOL, for the render task
pip install "affbio[mpi]"      # mpi4py, for parallel runs (see below)
```

PyMOL wheels on PyPI currently cover Python 3.9 to 3.13. On Python 3.14,
install `pymol-open-source-whl` instead of the `render` extra.

### Parallel runs

Parallel runs need MPI, an HDF5 library built with MPI, and h5py built
against it; the h5py wheels on PyPI are serial only. Build h5py before
installing AffBio.

Debian/Ubuntu:

```
sudo apt install python3-dev openmpi-bin libopenmpi-dev libhdf5-openmpi-dev
pip install mpi4py
CC=mpicc HDF5_MPI=ON HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
    pip install --no-binary=h5py h5py
pip install "affbio[mpi]"
```

Fedora:

```
sudo dnf install python3-devel openmpi-devel hdf5-openmpi-devel
export PATH=/usr/lib64/openmpi/bin:$PATH
pip install mpi4py
CC=mpicc HDF5_MPI=ON HDF5_INCLUDEDIR=/usr/include/openmpi-x86_64 \
    HDF5_LIBDIR=/usr/lib64/openmpi/lib pip install --no-binary=h5py h5py
pip install "affbio[mpi]"
```

With conda, MPI-enabled h5py comes ready-made:

```
conda install -c conda-forge "h5py=*=mpi_openmpi*" mpi4py
pip install affbio
```

`python -c "import h5py; print(h5py.get_config().mpi)"` should print `True`.
Then run AffBio with `mpirun -n 8 affbio ...`. Set `OMP_NUM_THREADS=1` so
that the processes do not compete for cores.

## Usage

### Prepare

AffBio reads one PDB file per structure. All files must contain the same
atoms in the same order. To split a trajectory into frames with MDAnalysis,
which is installed with AffBio:

```python
import os
import MDAnalysis as mda

u = mda.Universe("topol.tpr", "traj.xtc")
protein = u.select_atoms("protein")
os.makedirs("snapshots", exist_ok=True)
for ts in u.trajectory[::5]:
    protein.write("snapshots/frame%d.pdb" % ts.frame)
```

Frames written by `gmx trjconv -sep` work as well.

### Cluster

```
affbio -m aff_matrix.hdf5 -t cluster -f snapshots/*.pdb --verbose
```

This writes `aff_centers.out`, `aff_labels.out` and `aff_stat.out` with the
cluster centers, the label of every structure and the cluster sizes. For
very many files, quote the pattern (`-f 'snapshots/*.pdb'`) and AffBio
expands it itself, in natural order.

`--selection` chooses the atoms used for RMSD, in
[MDAnalysis selection syntax][4], for example `--selection "name CA"`.

`affbio --help` lists all options and the individual tasks that `cluster`
runs.

### Visualize

To draw the cluster centers with their sizes, colored by how much each atom
moves within its cluster (needs `affbio[render]`):

```
affbio -m aff_matrix.hdf5 -t render --draw_nums --bcolor -o clusters.png
```

### Limits

Under MPI, the number of structures is rounded down to a multiple of
4 x the number of processes, and AffBio lists the files it leaves out.
Each process holds blocks of N / processes structures, which must not
exceed 32,767 (an HDF5 limit), so 200,000 structures need at least
7 processes.

Disk space, for N structures:

- the RMSD and similarity matrices take 8 x N² bytes next to the HDF5 file;
- under MPI, clustering takes another 8 x N² bytes in the working directory;
- when the matrix does not fit in memory, clustering also keeps up to
  8 x N² bytes in the temporary directory (`$TMPDIR`), and a single process
  then needs the 8 x N² bytes in the working directory as well.

AffBio checks the task order, the number of processes and the free disk
space, and warns about memory, before it starts.

## Changes since 0.0.x

- Python 3 only; installs with pip, no compiler needed.
- pyRMSD is replaced by built-in NumPy RMSD (same values, faster).
- ProDy is replaced by MDAnalysis, so `--selection` uses MDAnalysis syntax:
  `chain A` becomes `chainID A` and `within 5 of X` becomes
  `X or around 5 X` (MDAnalysis' `around` leaves X itself out);
  `all`, `protein`, `backbone`, `name CA`, `resname`, `resnum` and
  `and`/`or`/`not` work as before. MDAnalysis keeps every alternate
  location of an atom, ProDy kept only the first.
- `render` no longer needs GROMACS or ImageMagick. Per-atom fluctuations are
  computed with MDAnalysis the way `gmx rmsf -fit -oq` did (atomic masses are
  guessed slightly differently, so colors can differ marginally), and images
  are assembled with Pillow.
- The median used as the default preference is now exact. The streaming
  estimate in 0.0.x was wrong above about 5,800 structures.
- `--noalign` now always compares raw coordinates. In 0.0.x, MPI runs
  compared some pairs after centering.
- Under MPI, every stage now uses the same structures (0.0.x dropped a few
  more before clustering), and a wrong process count or missing disk space
  is reported before any work starts.

## Test data

The tests use the C-alpha atoms of every 4th frame of a 1 µs adenylate
kinase simulation by Sean Seyler and Oliver Beckstein
([doi:10.6084/m9.figshare.5108170][5], CC BY 4.0); see
`tests/data/README.md`.

[1]: https://doi.org/10.1126/science.1136800
[2]: https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
[3]: https://doi.org/10.1093/nar/gkx1262
[4]: https://docs.mdanalysis.org/stable/documentation_pages/selections.html
[5]: https://doi.org/10.6084/m9.figshare.5108170.v1
