# AffBio

This is a set of tools for clustering structures of biomolecules built around **[Affinity Propagation]**[1] clustering algorithm

## Installation

### Prerequisites

Before installing affbio with pip install prerequisites. For Ubuntu LTS you can use following commands:
`apt-get install build-essential python-pip python-dev libopenmpi-dev libhdf5-dev`

Affbio also depends on **Numpy**, it can be installed with `apt-get install python-numpy` or `pip install numpy`

Visualization will also require **[GROMACS]**[2] and **imagemagick**. Usually they can also can be installed from repositories:
`apt-get install gromacs imagemagick`

### Installation

AffBio is available in PyPi and can be install with `pip install affbio`

### Parallel version

AffBio supports parallel execution with **OpenMPI**. To be able to use this features you will need mpi enabled version of **[h5py]**[3] package.

[1][http://www.psi.toronto.edu/index.php?q=affinity%20propagation]
[2][http://www.gromacs.org/]
[3][http://docs.h5py.org/en/latest/mpi.html]
