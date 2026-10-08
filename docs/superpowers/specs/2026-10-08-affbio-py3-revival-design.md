# AffBio revival: Python 3, pip-only install

Date: 2026-10-08
Status: draft, awaiting review

## Goal

Make AffBio installable with plain `pip` and runnable on current Python
again, keeping the original CLI, workflow and HDF5 layout.

Success means:

1. `pip install affbio` on Python 3.10-3.14 (Linux, no conda, no compiler)
   gives a working single-process `affbio`.
2. The README workflow works end to end on a real MD trajectory:
   split into PDB frames -> `affbio -t cluster` -> `affbio -t render`
   (render after `pip install "affbio[render]"`).
3. The same clustering runs in parallel with `mpirun -n 4 affbio ...` after
   the documented MPI setup, with results identical to a serial run.
4. Release 0.1.0 is ready to upload to PyPI. Uploading happens only on the
   maintainer's explicit go.

## Decisions (with reasons)

| Topic | Decision | Why |
|---|---|---|
| Deployment | pip only; no conda, no `environment.yml` | Maintainer's choice; every remaining dependency has PyPI wheels |
| RMSD | Replace pyRMSD with `affbio/rmsd.py`: block-vs-block covariance via one BLAS GEMM + vectorized QCP (float64) | pyRMSD's working fork is not on PyPI (PyPI has only the 2015 Py2 release). Measured on 1 core vs affbio's current pyRMSD use: 5.7x faster at 214 atoms, 14.6x at 3,341 atoms, agreement 1e-11 A. Matches mdtraj (fastest CPU library checked) without its float32 error (up to 1e-3 A) or an extra dependency |
| `--noalign` | Raw RMSD: no centering, no rotation | Maintainer: match original. The original was inconsistent: pyRMSD NOSUP `pairwiseRMSDMatrix` (diagonal blocks) is raw, `oneVsFollowing` (off-diagonal blocks) is centered. Single-process runs were therefore raw everywhere; MPI runs mixed both. Raw is the single-process behavior and is now used everywhere |
| Median for preference | Replace Cython `lvc.pyx` (P-square) with an exact out-of-core NumPy radix select | Measured: exact, ~25x faster from HDF5, bounded memory. `lvc.pyx` is wrong beyond 2^24 values (float32 counters stop incrementing), i.e. N > ~5,800 structures |
| Structure parsing | MDAnalysis replaces prody (topology parsed once, coordinates per file) | Maintainer's choice. Same coordinates (max diff 1.7e-6 A); 27 vs 20 ms per 3,341-atom file, spread over MPI ranks |
| Cluster B-factors | MDAnalysis replaces `gmx rmsf` (same math) | Maintainer's choice; drops GROMACS |
| Image tiling/labels | Pillow replaces ImageMagick | Maintainer's choice; Pillow 10.1+ has a built-in scalable font, so no system fonts needed |
| PyMOL | Optional extra `affbio[render]` = `pymol-open-source` (official, pre-release-only on PyPI; pip resolves `3.2.0a0` without flags) | Only `render` needs it. Official wheels cover 3.9-3.13; on 3.14 the README points to `pymol-open-source-whl` |
| MPI | Optional extra `affbio[mpi]` = `mpi4py`; serial fallback when mpi4py is absent | PyPI h5py has no parallel HDF5, so parallel runs need extra setup anyway; single-process users should not need an MPI library |
| Python | `requires-python >=3.10`, classifiers 3.10-3.14 | All core dependencies ship wheels for 3.10-3.14 |
| Build system | Single PEP 621 `pyproject.toml`, setuptools backend, pure-Python wheel | No compiled code left in affbio |
| Legacy dirs | `app/`, `old/`, `supplement/` stay untouched, excluded from the package | Kept for history |

## Components

### 1. Packaging (`pyproject.toml`)

- Remove `setup.py`, `setup.cfg`, `MANIFEST.in`, `README.rst`,
  `affbio/lvc.pyx`.
- `[build-system]`: `setuptools>=77`, backend `setuptools.build_meta`.
- `[project]`:
  - `name = "affbio"`, `version = "0.1.0"`, `requires-python = ">=3.10"`
  - `readme = "README.md"`, `license = "GPL-3.0-or-later"`,
    `license-files = ["LICENSE.txt"]` (file headers say "version 3 or later")
  - author Arthur Zalevsky, `aozalevsky@gmail.com` (the old `fbb.msu.ru`
    address and homepage are dead)
  - URLs: GitHub repo, paper DOI `10.1093/nar/gkx1262`
  - dependencies: `numpy>=1.23`, `h5py>=3.0` (`.asstr()`),
    `MDAnalysis>=2.0`, `pillow>=10.1` (sized default font), `bottleneck`,
    `natsort`, `psutil`
  - optional: `mpi = ["mpi4py"]`, `render = ["pymol-open-source"]`,
    `test = ["pytest"]`
  - `[project.scripts] affbio = "affbio.cli:run"`
  - classifiers: Python 3 only, 3.10-3.14, GPLv3+, Bio-Informatics
- `[tool.setuptools] packages = ["affbio"]`.

### 2. MPI made optional (`affbio/mpi.py`)

Exports `COMM`, `INT`, `FLOAT`:

- mpi4py importable -> `MPI.COMM_WORLD`, `MPI.INT`, `MPI.FLOAT`.
- mpi4py not importable (`ImportError`) -> `SerialComm` with `size = 1`,
  `rank = 0` and the methods affbio calls: `Barrier()`,
  `bcast(obj, root=0) -> obj`, `Gather(send, recv)` and `Reduce(send, recv)`
  copy the send buffer into the receive buffer (buffers may be arrays or
  `[array, type]` lists), `Bcast(buf)` does nothing. `INT`/`FLOAT` are
  placeholders.
- mpi4py not importable but the process was launched by an MPI launcher
  (`OMPI_COMM_WORLD_SIZE`, `PMI_SIZE` or `PMIX_RANK` set) -> exit with an
  error instead of running N independent serial copies that overwrite each
  other's files.

`utils.init_mpi()` returns `(COMM, COMM.size, COMM.rank)` as before;
`aff_cluster.py` uses `INT`/`FLOAT` from this module instead of `MPI.*`.
When `NPROCS > 1` and `h5py.get_config().mpi` is false, exit with an error
pointing to the README's parallel setup.

### 3. Python 3 port (behavior unchanged)

Module by module, the issues found by reading the code:

- `utils.py`: `print` statements; `StringIO` -> `io.StringIO`;
  `task()` uses `/` -> `//`; `init_mpi()` via component 2.
- `cli.py`: `dict.keys() + dict.keys()` (3 places) -> list concatenation;
  bare `exit(0)` -> `sys.exit(0)`; the `'-i, --index'` option string is a
  single malformed flag -> `'-i', '--index'` (needed by `cluster_to_trj`);
  `--selection` help says MDAnalysis syntax.
- `structures.py`:
  - prody -> MDAnalysis. Each rank builds `Universe(topology)` once
    (topology = `-s` or the first PDB, as before), evaluates `--selection`
    once to atom indices (empty selection still raises `ValueError`), then
    reads each file with
    `MDAnalysis.coordinates.PDB.PDBReader(f).ts.positions[idx]`. A file whose
    atom count differs from the topology raises the existing
    `Broken structure` error. PBC check unchanged.
  - pyRMSD -> component 4. `calc_diag_chunk` writes the strict lower
    triangle of `rmsd_block(ic, ic)` and zeros elsewhere (as before);
    `calc_chunk` writes `rmsd_block(ic, jc)`. Storage stays float32; the
    MPI block decomposition is unchanged.
  - `np.float` -> `np.float64`; `h5py.special_dtype(vlen=str)` ->
    `h5py.string_dtype()`; `partition()` `lN = ... / 2` -> `// 2`; labels
    copied between tiers are read with `.asstr()`.
- `prepare.py`: `print` statements; `lN` integer division; `calc_median`
  drops `pyximport`/`lvc` and calls component 5. The small-matrix path
  (`N*N <= 10000`: `np.median` of the whole matrix) is kept as is.
- `aff_cluster.py`: `print` statements; `np.int` -> `np.int64` (same width
  as before on 64-bit Linux, so the existing MPI `Gather`/`Bcast` byte
  semantics are unchanged); label datasets read with `.asstr()` so the
  `.out` files contain `frame1.pdb`, not `b'frame1.pdb'`.
- `misc.py`: the `gmx rmsf` subprocess in `render_b_factor` is replaced by
  component 6 (the temporary `cluster_N_trj.pdb` and `.xvg` files are no
  longer needed); `map(os.remove, ...)` (lazy in Py3, so files were silently
  not removed) -> loops; `np.array(map(...), dtype=np.bool)` -> list
  comprehension, `bool`; labels/topology read as `str`. `copy_connects`
  inserts the topology's CONECT records before the trailing `ENDMDL` if
  present, otherwise before `END` (frames written by MDAnalysis have no
  `ENDMDL`; trjconv frames do).
- `AffRender.py`: component 7 for images; `map(int, ...)` -> list;
  `map(os.remove, ...)` -> loops; PyMOL imported lazily with a clear
  error ("render requires PyMOL: pip install 'affbio[render]'") and
  started headless.

No other refactoring.

### 4. RMSD (`affbio/rmsd.py`)

`rmsd_block(A, B, superpose=True) -> ndarray (la, lb) float64`, where `A` is
`(la, n, 3)` and `B` is `(lb, n, 3)`.

Superposed (default):

1. Center every structure on its geometric center (unweighted, as pyRMSD).
2. `Ga`, `Gb`: per-structure sums of squared coordinates.
3. All 3x3 covariance matrices at once with one GEMM:
   `A^T (la*3, n) @ B (n, lb*3)` reshaped to `(la, lb, 3, 3)`.
4. QCP (Theobald 2005): from each covariance build the 4x4 key matrix's
   characteristic polynomial `x^4 + C2 x^2 + C1 x + C0`
   (`C2 = -2 * ||M||_F^2`, `C1 = -8 * det(M)`, `C0 = det(K)` from 2x2
   minors), vectorized over all pairs.
5. Newton iterations from `E0 = (Ga + Gb) / 2` for the largest root,
   until `max|step| < 1e-11 * max|x|` (cap 50 iterations).
6. `RMSD = sqrt(max(0, 2 * (E0 - x)) / n)`.

`--noalign`: raw coordinates, `RMSD^2 = (Ga + Gb - 2 * A.B) / n` via one GEMM,
clamped at 0.

All arithmetic in float64. `B` is processed in column chunks so temporaries
stay under ~64 MiB regardless of block size. BLAS threading follows the
usual environment variables; the README recommends `OMP_NUM_THREADS=1`
under MPI.

Measured on the prototype, 1 core, vs affbio's current pyRMSD calls:
0.77 vs 4.34 us/pair (214 atoms), 4.85 vs 70.9 us/pair (3,341 atoms),
max difference 1.6e-11 A.

### 5. Exact out-of-core median (`affbio/median.py`)

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

### 6. Cluster B-factors without GROMACS (`affbio/rmsf.py`)

`cluster_bfactors(center_pdb, member_pdbs, out_pdb, topology)` reproduces
`gmx rmsf -s center -f members -fit -oq out` (checked against
`gmx_rmsf.cpp`):

1. `ref = Universe(center_pdb)`, `mobile = Universe(center_pdb, member_pdbs)`
   (each member PDB is one frame).
2. `AlignTraj(mobile, ref, select="all", weights="mass", in_memory=True)`:
   per-frame center-of-mass removal + mass-weighted least-squares fit onto
   the center, as gmx does with `-fit`.
3. `RMSF(mobile.atoms)`: fluctuation about the average fitted position.
4. B = 8*pi^2/3 * RMSF^2 (A^2); gmx writes `800*pi^2/3 * msf[nm^2]`, the
   same value.
5. Write the center's coordinates with these B-factors to `out_pdb`, then
   `copy_connects(topology, out_pdb)` as before.

If any guessed atomic mass is zero (e.g. coarse-grained origami beads),
fall back to an unweighted fit and log a warning.

### 7. Images with Pillow (`AffRender.py`)

- `gen_label`: was `convert -transparent white -size {lw}x{h} -gravity East
  -pointsize {lw/3} caption:"N%"`. Now a transparent RGBA image `lw x h`
  with black text `"N%"`, `ImageFont.load_default(size=lw // 3)`, anchored
  right-middle.
- `tile`: was `montage -mode Concatenate -background none -tile Nx1`
  (horizontal) or `1xN` (vertical). Now paste images edge to edge on a
  transparent canvas: horizontal for the label + three poses of one
  cluster, vertical for stacking clusters. Pixel-identical output is not a
  goal; layout and transparency are.

### 8. Test data (`tests/data/`)

Source: "Molecular dynamics trajectory for benchmarking MDAnalysis",
figshare article 5108170, CC BY 4.0 (Seyler & Beckstein): 1 us equilibrium
MD of adenylate kinase (AdK, 4AKE), `1ake_007-nowater-core-dt240ps.dcd`
(4,187 frames, 3,341 atoms) + `adk4AKE.psf`.

Derived files committed to the repo (~1 MB):

- `adk_ca.pdb`: C-alpha atoms (214), first frame.
- `adk_ca.xtc`: C-alpha atoms, every 4th frame (1,047 frames).
- `make_adk_ca.py`: the script that produced them (MDAnalysis).
- `README.md`: provenance, license, citation.

### 9. Tests (`tests/`, pytest)

Fixture: split `adk_ca.xtc` into one PDB per frame with MDAnalysis into a
temp dir once per session (the same snippet the README shows).

- `test_rmsd.py`: `rmsd_block` vs MDAnalysis `lib.qcprot` per pair on 50
  AdK frames (tolerance 1e-6 A); known cases: identical -> 0, translated ->
  0, rotated -> 0, mirror image -> > 0; `superpose=False` vs the direct raw
  formula; chunking gives the same result as one block.
- `test_median.py`: `streaming_median` equals `np.median` of the strict
  lower triangle for odd/even counts, mixed signs, heavy duplicates, and a
  block size forcing many blocks.
- `test_load.py`: `--selection "resid 1:100"` stores `struct` with shape
  `(N, 100, 3)`; a frame with a missing atom raises `Broken structure`.
- `test_cluster.py`: `affbio -m m.hdf5 -t cluster -f frames/*.pdb` on the
  AdK frames. Asserts: zero RMSD diagonal; one spot-checked pair matches
  `qcprot`; median and preference attributes present; 1 < clusters < N;
  every frame labeled; each exemplar is in its own cluster; `.out` files
  contain plain paths; a second run yields identical labels.
- `test_mpi.py` (skipped unless mpi4py imports, `h5py.get_config().mpi`, and
  `mpirun` is on PATH): `mpirun -n 4` on the first 1,040 frames (divisible
  by NPROCS * 4, so no truncation) gives the same exemplars and labels as a
  serial run on the same 1,040 frames.
- `test_rmsf.py`: `cluster_bfactors` on 50 AdK frames equals an independent
  NumPy computation (mass-weighted Kabsch fit + RMSF + 8*pi^2/3 factor);
  CONECT records are copied into files with and without `ENDMDL`.
- `test_render.py` (skipped without PyMOL): `affbio -t render --draw_nums
  --bcolor -o clusters.png` after clustering produces non-empty
  `clusters.png` and `clusters_color.png` of the expected size.
- `test_cli.py`: `affbio --help` exits 0 and lists all tasks; with mpi4py
  hidden, a run works serially, and with `OMPI_COMM_WORLD_SIZE` set it
  exits with the launcher error.

### 10. CI (`.github/workflows/tests.yml`)

On push and PR, Ubuntu:

- `pip` job, Python 3.10-3.14: `pip install ".[render,test]"` (without
  `render` on 3.14), `pytest`.
- `mpi` job, one Python version: `apt install openmpi-bin libopenmpi-dev
  libhdf5-openmpi-dev`; `pip install mpi4py`;
  `CC=mpicc HDF5_MPI=ON HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
  pip install --no-binary=h5py h5py`; `pip install ".[mpi,test]"`; `pytest tests/test_mpi.py`. These are
  the README's parallel-setup commands, so CI keeps them correct.

No publishing workflow.

### 11. README

- Installation: `pip install affbio`; `pip install "affbio[render]"` for
  render (Python 3.14: use `pymol-open-source-whl`); parallel setup as in
  the CI `mpi` job, with Fedora and Debian/Ubuntu package names, and a
  conda-forge alternative (`h5py=*=mpi_openmpi*`, `mpi4py`).
- Remove the "Python 2 only" notice and the GROMACS/ImageMagick
  prerequisites.
- Prepare: a short MDAnalysis snippet to split a trajectory into PDB frames;
  `gmx trjconv -sep` output works too.
- `--selection` now uses MDAnalysis syntax, with common translations
  (`chain A` -> `chainID A`, `resnum 10 to 20` -> `resid 10:20`,
  `within 5 of X` -> `around 5 X`; `all`, `protein`, `backbone`, `nucleic`,
  `name CA`, `resname X`, `and/or/not` are unchanged).
- Usage otherwise unchanged except fixed typos.
- "Changes since 0.0.x": P-square median bug (N > ~5,800), `--noalign`
  MPI inconsistency fixed, pyRMSD/prody/GROMACS/ImageMagick replaced,
  selection syntax.
- Test data attribution.

## Out of scope

- Algorithmic changes beyond the median, RMSD implementation and the
  `--noalign` fix.
- Refactoring the MPI decomposition, the `MPI.INT` buffer typing, or the
  HDF5 chunking (`(l, l)` chunks exceed HDF5's 4 GiB chunk limit for
  serial runs with N > ~32,700; unchanged, documented).
- GPU RMSD.
- conda-forge recipe for affbio.
- Publishing to PyPI (prepared, uploaded only on explicit go).
- `app/`, `old/`, `supplement/`.

## Risks

- Mass guessing differs slightly between MDAnalysis and GROMACS, so
  B-factors (and render colors) can differ marginally from 0.0.x output;
  zero masses handled by the unweighted fallback (component 6).
- `--selection` syntax changes from ProDy to MDAnalysis; scripts using
  ProDy-only keywords break (documented with translations).
- Alternate locations: ProDy kept only the first altloc, MDAnalysis keeps
  all. MD snapshots have none; crystallographic PDBs with altlocs would get
  extra atoms (documented).
- PyMOL on PyPI is pre-release only; if pip's pre-release handling changes,
  the `render` extra may need `pymol-open-source>=3.2.0a0`.
- The mpi4py PyPI wheel must load the system Open MPI that h5py was built
  against; if it does not, the documented setup switches to
  `pip install --no-binary=mpi4py mpi4py` (the CI `mpi` job decides).
- OpenMPI in CI containers may need `--oversubscribe` and
  `OMPI_ALLOW_RUN_AS_ROOT*`.
- MPI vs serial results could differ by floating-point reduction order; if
  `test_mpi.py` shows this, investigate before relaxing the assertion.
