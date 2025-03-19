# AffBio

This is a set of tools for clustering structures of biomolecules built around **[Affinity Propagation]**[1] clustering algorithm. Inspired by python implementation from **scikit-learn**[2] package.

## Installation

### Python2

AffBio was developed using Python2 and is not compatible with Python3. I'm planning to upgrade it, but I have no specific timeline. 

### Prerequisites

Before installing affbio with pip install prerequisites. For Ubuntu LTS you can use following commands:
`apt-get install build-essential python-pip python-dev libopenmpi-dev libhdf5-dev`

Affbio also depends on **Numpy**, it can be installed with `apt-get install python-numpy` or `pip install numpy`

Visualization will also require **[GROMACS]**[3] and **imagemagick**. Usually they can also can be installed from repositories:
`apt-get install gromacs pymol imagemagick`

### Installation

AffBio is available in PyPi and can be install with `pip install affbio`

### Parallel version

AffBio supports parallel execution with **OpenMPI**. To be able to use this features you will need mpi enabled version of **[h5py]**[4] package.

## Usage

### Prepare

Obtain a set of **identical** PDB structures or generate trajectory snapshots:
```
mkdir snapshots
trjconv -f md.trr -s md.tpr -o snapshots/frame.pdb -conect -skip 5 -sep
cd snapshots
```

### Run

The most typical usage would be:
```
affbio -m aff_matrix.hdf5 -t cluster  -f *.pdb --verbose
```
This run will generate a set of output files: aff_centers.out, aff_labels.out and aff_stat.out which contain all the necessary information about the resulting clusters.

### Visualization

To create pictures with exemplars and (optionally) with clustering stats execute following command:

```
affbio -m aff_matrix.hdf5 -t render --draw_nums --merged_labels --bcolor -o clusters.png
```
## Q\&A


[1][http://www.psi.toronto.edu/index.php?q=affinity%20propagation]

[2][http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html]

[3][http://www.gromacs.org/]

[4][http://docs.h5py.org/en/latest/mpi.html]
