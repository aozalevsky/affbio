# AffBio revival: Python 3, modern packaging, conda environment

Date: 2026-10-08
Status: draft, awaiting review

## Goal

Make AffBio installable and runnable on a current system again, with the
original CLI, algorithms and HDF5 layout intact.

Success means:

1. `conda env create -f environment.yml` produces a working environment on
   Linux; affbio itself comes from PyPI via the env file's `pip:` section,
   everything else from conda-forge.
2. In that environment, the README workflow works end to end on a real MD
   trajectory: split into PDB frames -> `affbio -t cluster` ->
   `affbio -t render`. GROMACS is no longer needed.
3. The same clustering runs in parallel with `mpirun -n 4 affbio ...`.
4. A new release (0.1.0) is ready to upload to PyPI. Uploading happens only on
   the maintainer's explicit go.

## Decisions (with reasons)

| Topic | Decision | Why |
|---|---|---|
| Deployment model | conda env file installs non-pip deps from conda-forge, then `pip install affbio` | Maintainer's choice; PyMOL, ImageMagick and pyRMSD are not (usefully) on PyPI |
| RMSD engine | Keep pyRMSD, use the maintained salilab fork (4.3.3, conda-forge) | Same numbers as before; PyPI only has the dead 2015 Py2 release 4.2.1 |
| Median for preference | Replace Cython `lvc.pyx` (P-square) with an exact out-of-core NumPy radix select | Measured: exact, ~25x faster from HDF5, bounded memory. P-square in `lvc.pyx` is wrong beyond 2^24 values (float32 counters stop incrementing), i.e. for N > ~5,800 structures. Approved by the maintainer after the benchmark. |
| Render | Replace `gmx rmsf` with MDAnalysis (same math, see component 3a); keep PyMOL and ImageMagick | Maintainer's choice; drops GROMACS, the heaviest prerequisite. MDAnalysis is on PyPI and conda-forge (2.10.0 verified in the env solve) |
| MPI | Keep mpi4py as a required dependency; env ships MPI-enabled h5py + OpenMPI | Matches original design; serial runs work in the same env |
| Python | `requires-python >=3.10`; env pins `python=3.12` | prody on conda-forge is built up to 3.12 only (verified by dry-run solve) |
| Build system | Single PEP 621 `pyproject.toml`, setuptools backend, pure-Python wheel | No compiled code left in affbio |
| Legacy dirs | `app/`, `old/`, `supplement/` stay in the repo untouched, excluded from the package | Not part of the installed tool; kept for history |

## Components

### 1. Packaging (`pyproject.toml`)

- Remove `setup.py`, `setup.cfg`, `MANIFEST.in`, `README.rst`, `affbio/lvc.pyx`.
- `[build-system]`: `setuptools>=77`, backend `setuptools.build_meta`.
- `[project]`:
  - `name = "affbio"`, `version = "0.1.0"`, `requires-python = ">=3.10"`
  - `readme = "README.md"`, `license = "GPL-3.0-or-later"`,
    `license-files = ["LICENSE.txt"]` (file headers say "version 3 or later")
  - author Arthur Zalevsky, `aozalevsky@gmail.com` (the old `fbb.msu.ru`
    address and homepage are dead)
  - URLs: GitHub repo, paper DOI `10.1093/nar/gkx1262`
  - dependencies: `numpy`, `h5py`, `mpi4py`, `prody`, `pyRMSD>=4.3`,
    `MDAnalysis`, `bottleneck`, `natsort`, `psutil`. The `pyRMSD>=4.3` floor makes a
    plain-pip install outside conda fail fast with "no matching distribution"
    instead of trying to compile the 2015 sdist. The old `natsort<=7.0.0` cap
    (a Python 2 constraint) is dropped.
  - `[project.scripts] affbio = "affbio.cli:run"`
  - classifiers: Python 3 only, 3.10-3.12, GPLv3+, Bio-Informatics.
- `[tool.setuptools] packages = ["affbio"]`.

### 2. Conda environment (`environment.yml`)

Name `affbio`, channel `conda-forge` only:

```
python=3.12, numpy, h5py=*=mpi_openmpi*, mpi4py, openmpi, pyrmsd>=4.3,
prody, mdanalysis, bottleneck, natsort, psutil, pymol-open-source,
imagemagick, pip
pip:
  - affbio>=0.1
```

Every runtime dependency of affbio is listed on the conda side, so pip only
installs affbio itself and can never replace the MPI build of h5py with a
PyPI wheel. The `>=0.1` floor prevents pip from picking the old Py2 sdist.

Development/CI install: before creating the env, rewrite the `affbio>=0.1`
pip line to the checkout (`-e <repo path>`). Once 0.1.0 is on PyPI,
`pip install --no-deps -e .` on top of the normal env also works.

### 3. Python 3 port (behavior unchanged)

Module by module, the issues found by reading the code:

- `utils.py`: `print` statements; `StringIO` -> `io.StringIO`;
  `task()` uses `/` -> `//`.
- `cli.py`: `dict.keys() + dict.keys()` (3 places) -> list concatenation;
  bare `exit(0)` -> `sys.exit(0)`; the `'-i, --index'` option string is a
  single malformed flag -> `'-i', '--index'` (needed by `cluster_to_trj`).
- `structures.py`: `np.float` -> `np.float64`;
  `h5py.special_dtype(vlen=str)` -> `h5py.string_dtype()`;
  `partition()` `lN = ... / 2` -> `// 2`; labels copied between tiers are
  read with `.asstr()`.
- `prepare.py`: `print` statements; `lN` integer division; `calc_median`
  drops `pyximport`/`lvc` and calls the new median (component 4).
  The small-matrix path (`N*N <= 10000`: `np.median` of the whole matrix)
  is kept as is.
- `aff_cluster.py`: `print` statements; `np.int` -> `np.int64` (same width
  as before on 64-bit Linux, so the existing MPI `Gather`/`Bcast` byte
  semantics are unchanged); label datasets read with `.asstr()` so the
  `.out` files contain `frame1.pdb`, not `b'frame1.pdb'`.
- `misc.py`: the `gmx rmsf` subprocess in `render_b_factor` is replaced by
  component 3a (the temporary `cluster_N_trj.pdb` and `.xvg` files are no
  longer needed); `map(os.remove, ...)` (lazy in Py3, so files were silently
  not removed) -> loops; `np.array(map(...), dtype=np.bool)` -> list
  comprehension, `bool`; labels/topology read as `str`. `copy_connects`
  inserts the topology's CONECT records before the trailing `ENDMDL` if
  present, otherwise before `END` (frames written by MDAnalysis have no
  `ENDMDL`; trjconv frames do).
- `AffRender.py`: `map(int, ...)` -> list; `map(os.remove, ...)` -> loops;
  PyMOL start via `pymol.finish_launching(['pymol', '-qc'])`;
  ImageMagick 7: call `magick montage` / `magick` (instead of `convert`)
  when `magick` is on PATH, falling back to the IM6 names otherwise.

### 3a. Cluster B-factors without GROMACS (`affbio/rmsf.py`)

`cluster_bfactors(center_pdb, member_pdbs, out_pdb)` reproduces
`gmx rmsf -s center -f members -fit -oq out` (checked against
`gmx_rmsf.cpp`):

1. `ref = Universe(center_pdb)`, `mobile = Universe(center_pdb, member_pdbs)`
   (each member PDB is one frame).
2. `AlignTraj(mobile, ref, select="all", weights="mass", in_memory=True)`:
   per-frame center-of-mass removal + mass-weighted least-squares fit onto
   the center, as gmx does with `-fit`.
3. `RMSF(mobile.atoms)`: fluctuation about the average fitted position.
4. B = 8*pi^2/3 * RMSF^2 (A^2) - gmx writes `800*pi^2/3 * msf[nm^2]`, the
   same value.
5. Write the center's coordinates with these B-factors to `out_pdb`, then
   `copy_connects(topology, out_pdb)` as before.

If any guessed atomic mass is zero (e.g. coarse-grained origami beads),
fall back to an unweighted fit and log a warning.

pyRMSD calls (`KABSCH_SERIAL_CALCULATOR`, `NOSUP_SERIAL_CALCULATOR`,
`pairwiseRMSDMatrix`, `oneVsFollowing`, `condensedMatrix`) are unchanged.

No other refactoring.

### 4. Exact out-of-core median (`affbio/median.py`)

`streaming_median(dataset, block_bytes=32 MiB) -> float` over the strict
lower triangle of a square float32 HDF5 dataset, used by `calc_median` for
the large-matrix path.

Algorithm (radix select on float32 bit patterns):

1. Map each float32 to a uint32 key with an order-preserving transform
   (flip all bits of negatives, set the sign bit of non-negatives).
2. Pass 1: stream row blocks `D[b:e, :e]`, keep the strict-lower-triangle
   values, histogram the top 16 bits of the keys (65,536 int64 counters).
   Locate the bins holding the two middle ranks.
3. Pass 2: stream again, histogram the low 16 bits of keys falling in those
   bins. This fully determines the two middle values.
4. Return their mean (same definition as `np.median`).

Rows per block = `max(1, block_bytes // (4 * N))`, so memory does not depend
on N. Measured on the prototype (N = 30,000, 3.35 GiB matrix): exact, 15 s,
~290 MiB extra RSS with 32 MiB blocks; Cython P-square took 366 s and was off
by 3.8x. Projected for N = 200,000 (2e10 values): ~11 min compute + two
sequential reads of ~75 GiB.

Input validation stays in `calc_median` (square matrix, positive chunk).
The `median` attribute written to HDF5 is unchanged.

### 5. Test data (`tests/data/`)

Source: "Molecular dynamics trajectory for benchmarking MDAnalysis",
figshare article 5108170, CC BY 4.0 (Seyler & Beckstein): 1 us equilibrium
MD of adenylate kinase (AdK, 4AKE), `1ake_007-nowater-core-dt240ps.dcd`
(4,187 frames, 3,341 atoms) + `adk4AKE.psf`.

Derived files committed to the repo (~1 MB):

- `adk_ca.pdb`: C-alpha atoms (214), first frame.
- `adk_ca.xtc`: C-alpha atoms, every 4th frame (1,047 frames).
- `make_adk_ca.py`: the script that produced them (MDAnalysis; needed only
  to regenerate, not to run tests).
- `README.md`: provenance, license, citation.

### 6. Tests (`tests/`, pytest)

Fixture: split `adk_ca.xtc` into one PDB per frame with MDAnalysis into a
temp dir once per session - the same snippet the README shows.

- `test_median.py` (no external tools): `streaming_median` equals
  `np.median` of the strict lower triangle for odd/even counts, mixed signs,
  heavy duplicates, and a block size forcing many blocks.
- `test_cluster.py`: `affbio -m m.hdf5 -t cluster -f
  frames/*.pdb` on the AdK frames. Asserts: RMSD matrix has zero diagonal and
  one spot-checked pair matches an independent NumPy Kabsch RMSD; median and
  preference attributes present; 1 < number of clusters < N; every frame
  labeled; each exemplar belongs to its own cluster; `.out` files contain
  plain paths; a second run yields identical labels.
- `test_mpi.py` (skipped if `mpirun` absent): same run with
  `mpirun -n 4 --oversubscribe` on the first 1,040 frames (divisible by
  NPROCS * 4, so no truncation) gives the same exemplars and labels as a
  serial run on the same 1,040 frames.
- `test_rmsf.py`: `cluster_bfactors` on 50 AdK frames equals an independent
  NumPy computation (mass-weighted Kabsch fit + RMSF + 8*pi^2/3 factor), and
  CONECT records are copied.
- `test_render.py` (needs PyMOL, ImageMagick): `affbio -t render
  --draw_nums --bcolor -o clusters.png` after clustering produces non-empty
  `clusters.png` and `clusters_color.png`.
- `test_cli.py`: `affbio --help` exits 0 and lists all tasks.

Tests that need a missing external tool are skipped, not failed.

### 7. CI (`.github/workflows/tests.yml`)

Ubuntu, `mamba-org/setup-micromamba` with `environment.yml` (pip line
rewritten to install the checkout), then `pytest`. Runs on push and PR.
No publishing workflow.

### 8. README

Rewrite Installation: conda env (one command), plus "plain pip works only if
you provide PyMOL, ImageMagick and pyRMSD>=4.3 yourself". Remove the
"Python 2 only" notice and the GROMACS prerequisite. Prepare section: a short
MDAnalysis snippet to split a trajectory into PDB frames, noting that
`gmx trjconv -sep` output works too. Rest of Usage unchanged except fixed
typos. Add test
data attribution. Note the median change and the P-square bug for users of
0.0.x with more than ~5,800 structures.

## Out of scope

- Algorithmic changes other than the median.
- Refactoring the MPI decomposition, the `MPI.INT` buffer typing, or the
  HDF5 chunking (`(l, l)` chunks exceed HDF5's 4 GiB chunk limit for
  serial runs with N > ~32,700; unchanged, documented).
- conda-forge recipe for affbio itself.
- Publishing to PyPI (prepared, uploaded only on explicit go).
- `app/`, `old/`, `supplement/`.

## Risks

- Mass guessing differs slightly between MDAnalysis and GROMACS, so
  B-factors (and render colors) can differ marginally from 0.0.x output;
  zero masses handled by the unweighted fallback (component 3a).
- PDB frames with and without `ENDMDL` must both get CONECT records;
  covered by `test_rmsf.py` and `test_render.py`.
- ImageMagick `caption:` needs a usable font in the conda env; verified by
  `test_render.py`.
- OpenMPI in CI containers may need `--oversubscribe` and
  `OMPI_ALLOW_RUN_AS_ROOT*`; handled in the workflow if needed.
- MPI vs serial results could differ by floating-point reduction order; if
  `test_mpi.py` shows this, investigate before relaxing the assertion.
