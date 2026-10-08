# AffBio Python 3 Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AffBio pip-installable and runnable on Python 3.10-3.14 with the original CLI, workflow and HDF5 layout.

**Architecture:** Port the existing `affbio` package to Python 3 module by module. Replace the dead or non-pip dependencies with small focused modules: `rmsd.py` (NumPy QCP), `median.py` (out-of-core radix select), `rmsf.py` (MDAnalysis B-factors), `mpi.py` (optional mpi4py), `checks.py` (up-front validation). MDAnalysis replaces prody for parsing. Pillow replaces ImageMagick in `AffRender.py`. The MPI block decomposition and the HDF5 layout stay as they are.

**Tech Stack:** Python >=3.10, NumPy, h5py, MDAnalysis, Pillow, bottleneck, natsort, psutil; optional mpi4py, PyMOL (`pymol2`); pytest; setuptools (PEP 621); GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-08-affbio-py3-revival-design.md`

## Global Constraints

- `requires-python = ">=3.10"`; tests pass on 3.10-3.14 (PyMOL render test only on <=3.13).
- Core dependencies exactly: `numpy>=1.23`, `h5py>=3.0`, `MDAnalysis>=2.0`, `pillow>=10.1`, `bottleneck`, `natsort`, `psutil`. Extras: `mpi = ["mpi4py"]`, `render = ["pymol-open-source"]`, `test = ["pytest"]`.
- No compiled code in `affbio`; no conda needed by users; no pyRMSD, prody, Cython, GROMACS or ImageMagick anywhere.
- CLI options, task names, HDF5 group/dataset names, dtypes and chunk shapes stay as they are; the only new HDF5 content is an `nprocs` attribute on `struct`, `rmsd` and `cluster`.
- RMSD and median arithmetic in float64; matrices stored as float32 (as before).
- `--noalign` = raw RMSD (no centering, no rotation) everywhere.
- Every new module in `affbio/` starts with the 21-line license header copied verbatim from lines 1-21 of `affbio/__init__.py`, followed by a blank line. Modified modules keep their existing header.
- Match the surrounding code: 4-space indent, `%` formatting, short comments, `print()` calls.
- Do not touch `app/`, `old/`, `supplement/`.
- Work on branch `py3-revival`. Never push, open a PR, or upload to PyPI without the user's explicit approval.
- Every commit message ends with these two trailer lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL`
- Dev environment: `uv venv -p 3.13 .venv && uv pip install -p .venv -e ".[render,test]"`; run tests with `.venv/bin/pytest`.

## Review Focus

1. `affbio -t render` after a plain `pip install affbio` (no PyMOL): expect a one-line error naming `affbio[render]`, no traceback. (Task 11, `test_missing_pymol`)
2. An old ProDy selection such as `--selection "chain A"`: expect exit 1 with a message explaining MDAnalysis syntax, no traceback. (Task 7 `test_old_prody_selection_is_explained`, Task 9 `test_bad_selection_is_explained`)
3. Frames written by `gmx trjconv -sep` (`MODEL`/`TER`/`CONECT`/`ENDMDL`) instead of MDAnalysis: expect identical coordinates and correct CONECT copying. (Task 7 `test_trjconv_style_frames`, Task 10 `test_copy_connects`)
4. A quoted glob `-f 'dir/f*.pdb'` over `f1..f12`: expect natural order (`f2` before `f10`). (Task 9 `test_quoted_glob_in_natural_order`)
5. A second tier, `--tier 2 -t cluster --merged_labels`: expect labels for every tier-1 frame. (Task 9 `test_tier2_merged_labels`)

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (new) | PEP 621 metadata, dependencies, extras, entry point, pytest config |
| `.gitignore` (new) | venv, build artifacts |
| `affbio/checks.py` (new) | `AffBioError`; size, process-count, stage, disk and memory checks |
| `affbio/mpi.py` (new) | mpi4py `COMM_WORLD` or `SerialComm`; `INT`/`FLOAT` datatypes |
| `affbio/rmsd.py` (new) | `rmsd_block`: tiled GEMM + vectorized QCP; raw mode for `--noalign` |
| `affbio/median.py` (new) | `streaming_median`: exact, two-pass, bounded memory |
| `affbio/rmsf.py` (new) | `cluster_bfactors`: `gmx rmsf -fit -oq` equivalent |
| `affbio/utils.py` | Py3 port; `init_mpi` via `mpi.get_comm` |
| `affbio/structures.py` | MDAnalysis loading, trimming, RMSD matrix via `rmsd_block` |
| `affbio/prepare.py` | Py3 port, stage checks, `streaming_median` |
| `affbio/aff_cluster.py` | Py3 port, stage/disk checks, no-exemplar error, `str` labels |
| `affbio/misc.py` | `copy_connects`, `cluster_to_trj`, `render_b_factor` without GROMACS |
| `affbio/AffRender.py` | Pillow labels/tiling, PyMOL via `pymol2` |
| `affbio/cli.py` | Py3 port, task expansion, preflight, error reporting |
| `tests/helpers.py`, `tests/conftest.py` | data paths, frame splitting, CLI runner, shared fixtures |
| `tests/data/*` | AdK C-alpha test trajectory + provenance |
| `tests/test_*.py` | one file per unit (see tasks) |
| `.github/workflows/tests.yml` | pip matrix + MPI job |
| `README.md`, `tests/data/README.md` | user docs, data attribution |
| removed | `setup.py`, `setup.cfg`, `MANIFEST.in`, `README.rst`, `affbio/lvc.pyx` |

---

### Task 1: Packaging and dev environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `tests/test_packaging.py`
- Delete: `setup.py`, `setup.cfg`, `MANIFEST.in`, `README.rst`

**Interfaces:**
- Produces: installable `affbio` 0.1.0 with console script `affbio = affbio.cli:run`; `.venv` dev environment used by every later task.

- [ ] **Step 1: Write the failing test** — `tests/test_packaging.py`:

```python
import re
from importlib.metadata import entry_points, requires, version


def test_version():
    assert version('affbio') == '0.1.0'


def test_console_script():
    scripts = entry_points(group='console_scripts')
    assert any(ep.name == 'affbio' and ep.value == 'affbio.cli:run'
               for ep in scripts)


def test_dependencies():
    reqs = requires('affbio')
    core = {re.split(r'[<>=!~;\[ ]', r)[0].lower()
            for r in reqs if 'extra ==' not in r}
    assert core == {'numpy', 'h5py', 'mdanalysis', 'pillow', 'bottleneck',
                    'natsort', 'psutil'}
    extras = ' '.join(r for r in reqs if 'extra ==' in r)
    assert 'mpi4py' in extras and 'pymol-open-source' in extras
```

- [ ] **Step 2: Create the dev environment and run the test to verify it fails**

The old `setup.py` cannot build on Python 3 (it imports numpy at build time and lists pyRMSD), so the install itself is expected to fail:

Run: `uv venv -p 3.13 .venv && uv pip install -p .venv -e ".[test]"`
Expected: FAIL (build error from `setup.py`, e.g. `ModuleNotFoundError: No module named 'numpy'` or a pyRMSD build failure).

- [ ] **Step 3: Write `pyproject.toml`, `.gitignore`; delete the old build files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "affbio"
version = "0.1.0"
description = "Affinity Propagation for structures of biomolecules. First developed for clustering of DNA origami structures (see https://doi.org/10.1093/nar/gkx1262 for details)"
readme = "README.md"
requires-python = ">=3.10"
license = "GPL-3.0-or-later"
license-files = ["LICENSE.txt"]
authors = [{ name = "Arthur Zalevsky", email = "aozalevsky@gmail.com" }]
keywords = ["clustering", "bioinformatics", "affinity propagation", "molecular dynamics"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
]
dependencies = [
    "numpy>=1.23",
    "h5py>=3.0",
    "MDAnalysis>=2.0",
    "pillow>=10.1",
    "bottleneck",
    "natsort",
    "psutil",
]

[project.optional-dependencies]
mpi = ["mpi4py"]
render = ["pymol-open-source"]
test = ["pytest"]

[project.urls]
Homepage = "https://github.com/aozalevsky/affbio"
Paper = "https://doi.org/10.1093/nar/gkx1262"

[project.scripts]
affbio = "affbio.cli:run"

[tool.setuptools]
packages = ["affbio"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

(No `License ::` classifier: setuptools >=77 rejects it together with a `license` expression.)

`.gitignore`:

```
__pycache__/
*.egg-info/
build/
dist/
.venv/
```

Run: `git rm -q setup.py setup.cfg MANIFEST.in README.rst`

- [ ] **Step 4: Install and run the test to verify it passes**

Run: `uv pip install -p .venv -e ".[render,test]" && .venv/bin/pytest tests/test_packaging.py -v`
Expected: 3 passed. (`pymol-open-source` resolves to the pre-release `3.2.0a0`; that is expected.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore tests/test_packaging.py
git commit -q -F - <<'EOF'
Switch packaging to pyproject.toml (PEP 621), version 0.1.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 2: Preflight checks (`affbio/checks.py`)

**Files:**
- Create: `affbio/checks.py`
- Test: `tests/test_checks.py`

**Interfaces:**
- Produces:
  - `class AffBioError(ValueError)`
  - `MAX_BLOCK = 32767`, `GIB = 2.0 ** 30`
  - `effective_n(n: int, nprocs: int) -> int`
  - `check_decomposition(n: int, nprocs: int) -> None` (raises `AffBioError`)
  - `check_stage(stage: str, n: int, nprocs: int, stored_chunk=None, stored_nprocs=None) -> None`; `stage` in `'calc_rmsd' | 'prepare_matrix' | 'aff_cluster'`
  - `disk_needs(sfn: str, n: int, tasks, existing=(), overwrite=False) -> dict[str, int]`
  - `check_disk(needs: dict[str, int]) -> None`
  - `local_procs() -> int`
  - `estimate_memory(stage: str, n: int, nprocs: int, natoms: int = 0) -> int`
  - `memory_warning(stage: str, n: int, nprocs: int, natoms: int = 0) -> str | None`
  - `check_parallel_io(nprocs: int, h5py_mpi: bool) -> None`

- [ ] **Step 1: Write the failing tests** — `tests/test_checks.py`:

```python
import shutil
import types

import pytest

from affbio import checks
from affbio.checks import (AffBioError, check_decomposition, check_disk,
                           check_parallel_io, check_stage, disk_needs,
                           effective_n, estimate_memory, local_procs,
                           memory_warning)

GIB = 2 ** 30


def test_error_is_value_error():
    assert issubclass(AffBioError, ValueError)


def test_effective_n():
    assert effective_n(1047, 1) == 1047
    assert effective_n(1047, 3) == 1044
    assert effective_n(1040, 4) == 1040
    assert effective_n(200000, 7) == 199992


@pytest.mark.parametrize('n, nprocs', [(2, 1), (16, 4), (200000, 7)])
def test_decomposition_ok(n, nprocs):
    check_decomposition(n, nprocs)


@pytest.mark.parametrize('n, nprocs, message', [
    (1, 1, 'Need at least 2 structures'),
    (15, 4, 'Need at least 16 structures for 4 processes'),
    (200000, 6, 'Use at least 7 processes'),
    (40000, 1, 'Use at least 2 processes'),
])
def test_decomposition_fails(n, nprocs, message):
    with pytest.raises(AffBioError, match=message):
        check_decomposition(n, nprocs)


def test_stage_ok():
    check_stage('calc_rmsd', 240, 4)
    check_stage('prepare_matrix', 240, 4, stored_chunk=60)
    check_stage('prepare_matrix', 240, 8, stored_chunk=60)
    check_stage('aff_cluster', 240, 4)
    check_stage('aff_cluster', 1047, 1)


@pytest.mark.parametrize('stage, n, nprocs, chunk', [
    ('calc_rmsd', 240, 7, None),
    ('prepare_matrix', 240, 3, 60),
    ('prepare_matrix', 240, 1, 60),
    ('aff_cluster', 240, 7, None),
])
def test_stage_mismatch_names_original_nprocs(stage, n, nprocs, chunk):
    with pytest.raises(AffBioError, match='prepared with 4 processes'):
        check_stage(stage, n, nprocs, stored_chunk=chunk, stored_nprocs=4)


def test_disk_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 1000
    needs = disk_needs(str(tmp_path / 'm.hdf5'), n,
                       ['load_pdb', 'calc_rmsd', 'prepare_matrix',
                        'aff_cluster'])
    # rmsd + cluster next to the file, Rp + A in the working directory
    assert needs == {str(tmp_path): 4 * 4 * n * n}


def test_disk_needs_skips_existing_and_reclaims(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sfn = tmp_path / 'm.hdf5'
    sfn.write_bytes(b'x' * 1000)
    needs = disk_needs(str(sfn), 100, ['calc_rmsd', 'prepare_matrix'],
                       existing=('rmsd',), overwrite=True)
    assert needs == {str(tmp_path): 4 * 100 * 100 - 1000}


def test_check_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(checks.shutil, 'disk_usage',
                        lambda path: types.SimpleNamespace(free=10 * GIB))
    check_disk({str(tmp_path): 9 * GIB})
    with pytest.raises(AffBioError, match='need 11.0 GiB, 10.0 GiB free'):
        check_disk({str(tmp_path): 11 * GIB})
    sub = tmp_path / 'sub'
    sub.mkdir()
    # two directories on one filesystem add up
    with pytest.raises(AffBioError, match='need 12.0 GiB'):
        check_disk({str(tmp_path): 6 * GIB, str(sub): 6 * GIB})
    with pytest.raises(AffBioError, match='does not exist'):
        check_disk({str(tmp_path / 'missing'): 1})


def test_local_procs(monkeypatch):
    monkeypatch.delenv('OMPI_COMM_WORLD_LOCAL_SIZE', raising=False)
    monkeypatch.delenv('MPI_LOCALNRANKS', raising=False)
    assert local_procs() == 1
    monkeypatch.setenv('MPI_LOCALNRANKS', '6')
    assert local_procs() == 6


def test_memory(monkeypatch):
    monkeypatch.delenv('OMPI_COMM_WORLD_LOCAL_SIZE', raising=False)
    monkeypatch.delenv('MPI_LOCALNRANKS', raising=False)
    monkeypatch.setattr(checks.psutil, 'virtual_memory',
                        lambda: types.SimpleNamespace(available=4 * GIB))
    assert estimate_memory('prepare_matrix', 10000, 1) == 40 * 10000 ** 2
    assert estimate_memory('calc_rmsd', 1000, 1, natoms=214) == \
        4 * 1000 ** 2 + 96 * 1000 * 214 + 128 * 2 ** 20
    assert memory_warning('prepare_matrix', 1000, 1) is None
    w = memory_warning('prepare_matrix', 20000, 1)
    assert 'prepare_matrix needs about 14.9 GiB' in w


def test_parallel_io():
    check_parallel_io(1, False)
    check_parallel_io(4, True)
    with pytest.raises(AffBioError, match='h5py built with MPI'):
        check_parallel_io(4, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_checks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'affbio.checks'`

- [ ] **Step 3: Write `affbio/checks.py`** (license header first, see Global Constraints):

```python
"""Up-front validation of problem size, process count, disk and memory."""

import math
import os
import shutil

import psutil

# One float32 l x l block must stay below HDF5's 4 GiB chunk limit
MAX_BLOCK = 32767
GIB = 2.0 ** 30


class AffBioError(ValueError):
    """A problem with the input or setup, reported without a traceback."""


def effective_n(n, nprocs):
    """Number of structures used so that every stage splits evenly."""
    if nprocs == 1:
        return n
    return n - n % (4 * nprocs)


def check_decomposition(n, nprocs):
    """Fail early if n structures cannot be split across nprocs processes."""
    if nprocs == 1:
        if n < 2:
            raise AffBioError(
                'Need at least 2 structures to cluster, got %d.' % n)
    elif n < 4 * nprocs:
        raise AffBioError(
            'Need at least %d structures for %d processes, got %d. '
            'Use at most %d processes.'
            % (4 * nprocs, nprocs, n, max(1, n // 4)))

    l = effective_n(n, nprocs) // nprocs
    if l > MAX_BLOCK:
        raise AffBioError(
            '%d structures on %d processes give %d x %d matrix blocks; '
            'one float32 block would exceed the 4 GiB HDF5 chunk limit. '
            'Use at least %d processes.'
            % (n, nprocs, l, l, math.ceil(n / MAX_BLOCK)))


def check_stage(stage, n, nprocs, stored_chunk=None, stored_nprocs=None):
    """Fail if a matrix stage cannot split this file's n structures."""
    if stage == 'calc_rmsd':
        ok = n % nprocs == 0
    elif stage == 'prepare_matrix':
        ok = n % nprocs == 0 and (
            stored_chunk is None or stored_chunk % (n // nprocs) == 0)
    elif stage == 'aff_cluster':
        ok = nprocs == 1 or n % (4 * nprocs) == 0
    else:
        raise ValueError('Unknown stage %r' % stage)

    if not ok:
        if stored_nprocs:
            hint = ('this file was prepared with %d processes; rerun this '
                    'stage with %d processes'
                    % (stored_nprocs, stored_nprocs))
        else:
            hint = 'rerun load_pdb with %d processes first' % nprocs
        raise AffBioError(
            '%s cannot split %d structures across %d processes: %s.'
            % (stage, n, nprocs, hint))


def disk_needs(sfn, n, tasks, existing=(), overwrite=False):
    """Bytes each directory still has to hold for the requested tasks."""
    matrix = 4 * n * n  # one float32 N x N dataset
    sdir = os.path.dirname(os.path.abspath(sfn))
    needs = {sdir: 0}
    if 'calc_rmsd' in tasks and 'rmsd' not in existing:
        needs[sdir] += matrix
    if 'prepare_matrix' in tasks and 'cluster' not in existing:
        needs[sdir] += matrix
    if 'aff_cluster' in tasks:
        # aff_cluster keeps two more N x N matrices in the working directory
        cwd = os.getcwd()
        needs[cwd] = needs.get(cwd, 0) + 2 * matrix
    if overwrite and os.path.exists(sfn):
        # load_pdb recreates the file, so its current size is freed
        needs[sdir] = max(0, needs[sdir] - os.path.getsize(sfn))
    return needs


def check_disk(needs):
    """Fail if a filesystem lacks space; needs maps directory -> bytes."""
    per_fs = {}
    for d, nbytes in needs.items():
        if nbytes <= 0:
            continue
        if not os.path.isdir(d):
            raise AffBioError('Directory %s does not exist.' % d)
        dev = os.stat(d).st_dev
        total, dirs = per_fs.get(dev, (0, []))
        per_fs[dev] = (total + nbytes, dirs + [d])

    for total, dirs in per_fs.values():
        free = shutil.disk_usage(dirs[0]).free
        if total > free:
            raise AffBioError(
                'Not enough disk space in %s: need %.1f GiB, %.1f GiB free.'
                % (', '.join(sorted(set(dirs))), total / GIB, free / GIB))


def local_procs():
    """Number of MPI processes on this node (Open MPI or MPICH), else 1."""
    for var in ('OMPI_COMM_WORLD_LOCAL_SIZE', 'MPI_LOCALNRANKS'):
        if var in os.environ:
            return int(os.environ[var])
    return 1


def estimate_memory(stage, n, nprocs, natoms=0):
    """Approximate peak bytes per process for a matrix stage."""
    l = effective_n(n, nprocs) // nprocs
    if stage == 'calc_rmsd':
        # result block, two coordinate blocks and their centered copies,
        # RMSD tile temporaries
        return 4 * l * l + 96 * l * natoms + 128 * 2 ** 20
    if stage == 'prepare_matrix':
        # float32 blocks plus float64 noise temporaries
        return 40 * l * l
    raise ValueError('Unknown stage %r' % stage)


def memory_warning(stage, n, nprocs, natoms=0):
    """Warning text if a stage likely needs more memory than is free."""
    procs = local_procs()
    need = estimate_memory(stage, n, nprocs, natoms) * procs
    avail = psutil.virtual_memory().available
    if need <= avail:
        return None
    return ('%s needs about %.1f GiB of memory on this node (%d processes), '
            'but only %.1f GiB is available; use more processes or nodes.'
            % (stage, need / GIB, procs, avail / GIB))


def check_parallel_io(nprocs, h5py_mpi):
    """Fail if several processes would share an HDF5 file without MPI-IO."""
    if nprocs > 1 and not h5py_mpi:
        raise AffBioError(
            'Parallel runs need h5py built with MPI support '
            '(h5py.get_config().mpi is False); see "Parallel runs" in the '
            'README.')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_checks.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add affbio/checks.py tests/test_checks.py
git commit -q -F - <<'EOF'
Add preflight checks for process count, chunk limit, disk and memory

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 3: Optional MPI (`affbio/mpi.py`) and `utils.py` port

**Files:**
- Create: `affbio/mpi.py`
- Modify: `affbio/utils.py` (everything after the license header)
- Test: `tests/test_mpi_fallback.py`

**Interfaces:**
- Consumes: `AffBioError` (Task 2).
- Produces:
  - `affbio.mpi.MPI` (mpi4py module or `None`), `INT`, `FLOAT` (mpi4py datatypes or `None`)
  - `class SerialComm` with `size = 1`, `rank = 0`, `Barrier()`, `bcast(obj, root=0)`, `Bcast(buf, root=0)`, `Gather(sendbuf, recvbuf, root=0)`, `Reduce(sendbuf, recvbuf, op=None, root=0)`
  - `get_comm() -> COMM_WORLD | SerialComm` (raises `AffBioError` under an MPI launcher without mpi4py)
  - `affbio.utils.init_mpi() -> (comm, NPROCS, rank)`, `task(N, NPROCS, rank) -> (begin, end)` (integers), `Bunch`, `dummy`, `init_logging`, `finish_logging`, `init_debug`, `finish_debug`

- [ ] **Step 1: Write the failing tests** — `tests/test_mpi_fallback.py`:

```python
import numpy as np
import pytest

from affbio import mpi
from affbio.checks import AffBioError
from affbio.mpi import SerialComm, get_comm
from affbio.utils import finish_debug, init_debug, init_mpi, task


def test_serial_comm():
    comm = SerialComm()
    assert (comm.size, comm.rank) == (1, 0)
    assert comm.bcast({'a': 1}) == {'a': 1}
    src = np.arange(6, dtype=np.int64)
    dst = np.zeros(6, dtype=np.int64)
    comm.Gather([src, None], [dst, None])
    assert np.array_equal(dst, src)
    dst2 = np.zeros(6, dtype=np.int64)
    comm.Reduce(src, dst2)
    assert np.array_equal(dst2, src)
    comm.Barrier()
    comm.Bcast([dst, None])


def test_without_mpi4py(monkeypatch):
    monkeypatch.setattr(mpi, 'MPI', None)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    monkeypatch.delenv('PMI_SIZE', raising=False)
    assert isinstance(get_comm(), SerialComm)


@pytest.mark.parametrize('var', ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'])
def test_launcher_without_mpi4py(monkeypatch, var):
    monkeypatch.setattr(mpi, 'MPI', None)
    monkeypatch.setenv(var, '4')
    with pytest.raises(AffBioError, match=r"affbio\[mpi\]"):
        get_comm()


def test_utils():
    comm, nprocs, rank = init_mpi()
    assert (nprocs, rank) == (1, 0)
    assert task(10, 2, 1) == (5, 10)
    assert isinstance(task(10, 3, 0)[1], int)
    finish_debug(init_debug(True), True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mpi_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'affbio.mpi'`

- [ ] **Step 3: Write `affbio/mpi.py`** (license header first):

```python
"""MPI communicator, or a single-process stand-in when mpi4py is missing."""

import os

import numpy as np

from .checks import AffBioError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Buffer datatypes for the Gather/Bcast/Reduce calls in aff_cluster
INT = MPI.INT if MPI is not None else None
FLOAT = MPI.FLOAT if MPI is not None else None

# Set by mpirun/mpiexec (Open MPI, MPICH, srun --mpi=pmi2)
LAUNCHER_SIZE_VARS = ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE')


class SerialComm(object):
    """The part of mpi4py's COMM_WORLD that affbio uses, for one process."""

    size = 1
    rank = 0

    def Barrier(self):
        pass

    def bcast(self, obj, root=0):
        return obj

    def Bcast(self, buf, root=0):
        pass

    def Gather(self, sendbuf, recvbuf, root=0):
        _copy(sendbuf, recvbuf)

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        _copy(sendbuf, recvbuf)


def _array(buf):
    """mpi4py accepts an array or an [array, datatype] list."""
    return buf[0] if isinstance(buf, (list, tuple)) else buf


def _copy(sendbuf, recvbuf):
    src = np.ravel(_array(sendbuf))
    _array(recvbuf).flat[:src.size] = src


def get_comm():
    """COMM_WORLD from mpi4py, or SerialComm when mpi4py is not installed."""
    if MPI is not None:
        return MPI.COMM_WORLD
    if any(int(os.environ.get(v, '1')) > 1 for v in LAUNCHER_SIZE_VARS):
        raise AffBioError(
            'affbio was started by an MPI launcher, but mpi4py is not '
            "installed. Install it with: pip install 'affbio[mpi]'")
    return SerialComm()
```

- [ ] **Step 4: Replace `affbio/utils.py` after the license header (keep lines 1-21) with:**

```python

# General modules
import io
import time
import cProfile
import pstats

from .mpi import get_comm


def dummy(*args, **kwargs):
    pass


def init_mpi():
    # Get MPI info (a single-process stand-in without mpi4py)
    comm = get_comm()
    # Get number of processes
    NPROCS = comm.size
    # Get rank
    rank = comm.rank

    return (comm, NPROCS, rank)


def master(fn):
    comm, NPROCS, rank = init_mpi()

    if rank == 0:
        return fn
    else:
        return dummy


class Bunch(object):

    def __init__(self, adict):
        self.__dict__.update(adict)


def task(N, NPROCS, rank):
    l = N // NPROCS
    b = rank * l
    return (b, b + l)


def init_logging(task, verbose=False):
    if verbose:

        print('Starting task: %s' % task)

    # Get current time
    t0 = time.time()

    return t0


def finish_logging(task, t0, verbose=False):
    if verbose:
        print("Task: %s execution time is %f" % (task, time.time() - t0))


def init_debug(debug=False):
    if debug is True:
        pr = cProfile.Profile()
        pr.enable()
    else:
        pr = None
    return pr


def finish_debug(pr, debug=False):
    if debug is True:
        pr.disable()
        s = io.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mpi_fallback.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add affbio/mpi.py affbio/utils.py tests/test_mpi_fallback.py
git commit -q -F - <<'EOF'
Make mpi4py optional with a single-process fallback; port utils

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 4: RMSD (`affbio/rmsd.py`)

**Files:**
- Create: `affbio/rmsd.py`
- Test: `tests/test_rmsd.py`

**Interfaces:**
- Produces: `TILE = 512`; `rmsd_block(A, B, superpose=True, out=None) -> ndarray (la, lb)`. `A`: `(la, n, 3)`, `B`: `(lb, n, 3)`, any float dtype. `out`: optional `(la, lb)` array (may be a float32 view) that receives the result and is returned.

- [ ] **Step 1: Write the failing tests** — `tests/test_rmsd.py`:

```python
import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from affbio import rmsd
from affbio.rmsd import rmsd_block


def ensemble(k, n=60, seed=0):
    """k chain-like structures in random poses; some identical up to pose."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 2.2, (n, 3)), axis=0)
    X = base[None] + rng.normal(0, 1.5, (k, n, 3))
    X[:k // 5] = X[0]
    rot = np.linalg.qr(rng.normal(size=(k, 3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))[:, None, None]  # proper rotations
    return np.einsum('kij,knj->kni', rot, X) + rng.normal(0, 20, (k, 1, 3))


def qcp_reference(a, b):
    a = np.ascontiguousarray(a - a.mean(0), dtype=np.float64)
    b = np.ascontiguousarray(b - b.mean(0), dtype=np.float64)
    return qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)


def test_matches_mdanalysis_qcprot():
    E = ensemble(55, seed=1)
    A, B = E[:30], E[30:]
    ref = np.array([[qcp_reference(a, b) for b in B] for a in A])
    np.testing.assert_allclose(rmsd_block(A, B), ref, atol=1e-6)


def test_known_cases():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 5, (40, 3))
    th = 0.7
    rz = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0],
                   [0, 0, 1]])
    cases = np.array([a, a + [3, 4, 0], a @ rz.T, a * [-1, 1, 1]])
    r = rmsd_block(cases[:1], cases)[0]
    assert r[0] < 1e-5   # identical
    assert r[1] < 1e-5   # translated
    assert r[2] < 1e-5   # rotated
    assert r[3] > 1.0    # a mirror image cannot be superposed


def test_noalign_is_raw_rmsd():
    A, B = ensemble(6), ensemble(5, seed=4)
    raw = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1).mean(-1))
    np.testing.assert_allclose(rmsd_block(A, B, superpose=False), raw,
                               rtol=1e-12, atol=1e-9)
    # without superposition a translation is not removed
    shifted = rmsd_block(A[:1], A[:1] + [3, 4, 0], superpose=False)
    assert shifted[0, 0] == pytest.approx(5.0)


def test_tiles_and_out(monkeypatch):
    E = ensemble(23, n=17)
    full = rmsd_block(E, E)
    monkeypatch.setattr(rmsd, 'TILE', 4)
    out = np.zeros((23, 23), dtype=np.float32)
    assert rmsd_block(E, E, out=out) is out
    np.testing.assert_allclose(out, full, atol=1e-5)


def test_single_atom_structures():
    A = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    assert np.all(rmsd_block(A, A) == 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rmsd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'affbio.rmsd'`

- [ ] **Step 3: Write `affbio/rmsd.py`** (license header first):

```python
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
        if np.abs(step).max() <= 1e-11 * np.abs(x).max():
            break

    return 2.0 * (e0 - x) / n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rmsd.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add affbio/rmsd.py tests/test_rmsd.py
git commit -q -F - <<'EOF'
Add vectorized QCP RMSD to replace pyRMSD

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 5: Exact out-of-core median (`affbio/median.py`)

**Files:**
- Create: `affbio/median.py`
- Delete: `affbio/lvc.pyx`
- Test: `tests/test_median.py`

**Interfaces:**
- Produces: `streaming_median(dset, block_bytes=32 * 2 ** 20) -> float` for any 2-D square float32 array-like supporting `dset.shape` and `dset[b:e, :e]` (NumPy array or h5py dataset); median of the strict lower triangle; raises `ValueError` for a 1x1 matrix.

- [ ] **Step 1: Write the failing tests** — `tests/test_median.py`:

```python
import h5py
import numpy as np
import pytest

from affbio.median import streaming_median


def lower_median(M):
    return np.median(M[np.tril_indices(len(M), -1)].astype(np.float64))


@pytest.mark.parametrize('n', [2, 3, 5, 6, 50, 51])
def test_random_mixed_signs(n):
    M = np.random.default_rng(n).normal(0, 10, (n, n)).astype(np.float32)
    assert streaming_median(M) == lower_median(M)


def test_duplicates_and_zeros():
    M = np.random.default_rng(7).integers(-3, 4, (40, 40)).astype(np.float32)
    M[::3] = -0.0
    assert streaming_median(M) == lower_median(M)


def test_constant():
    M = np.full((10, 10), -2.5, dtype=np.float32)
    assert streaming_median(M) == -2.5


def test_small_blocks_from_hdf5(tmp_path):
    rng = np.random.default_rng(1)
    M = -((rng.random((300, 300)) * 100).astype(np.float32) ** 2)
    with h5py.File(tmp_path / 'm.hdf5', 'w') as f:
        f['cluster'] = M
        assert streaming_median(f['cluster'], block_bytes=1) == \
            lower_median(M)
        assert streaming_median(f['cluster']) == lower_median(M)


def test_needs_two_rows():
    with pytest.raises(ValueError):
        streaming_median(np.zeros((1, 1), dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_median.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'affbio.median'`

- [ ] **Step 3: Write `affbio/median.py`** (license header first) and delete `lvc.pyx`:

```python
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
```

Run: `git rm -q affbio/lvc.pyx`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_median.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add affbio/median.py tests/test_median.py
git commit -q -F - <<'EOF'
Replace the Cython P-square median with an exact out-of-core median

The float32 counters in lvc.pyx stop incrementing past 2^24 values, so
the old estimate was wrong for more than ~5,800 structures.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 6: AdK test data and frame fixture

**Files:**
- Create: `tests/data/make_adk_ca.py`, `tests/data/README.md`, `tests/data/adk_ca.pdb`, `tests/data/adk_ca.xtc` (generated), `tests/helpers.py`, `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `tests/helpers.py`: `DATA`, `TOP` (path to `adk_ca.pdb`), `TRJ` (path to `adk_ca.xtc`), `split_frames(top, trj, outdir, stop=None) -> list[str]` (absolute paths `outdir/frame%04d.pdb`)
  - `tests/conftest.py`: session fixture `adk_frames -> list[str]` (1,047 PDB paths in frame order)

- [ ] **Step 1: Write the generator script** — `tests/data/make_adk_ca.py`:

```python
"""Regenerate the AdK C-alpha test trajectory.

Source: Seyler, Sean; Beckstein, Oliver (2017). Molecular dynamics
trajectory for benchmarking MDAnalysis. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.5108170.v1 (CC BY 4.0)

Usage: python make_adk_ca.py DOWNLOAD_DIR   (downloads ~170 MB)
"""

import json
import os
import sys
import urllib.request

import MDAnalysis as mda

ARTICLE = 'https://api.figshare.com/v2/articles/5108170'
FILES = ('adk4AKE.psf', '1ake_007-nowater-core-dt240ps.dcd')
STRIDE = 4


def download(dest):
    meta = json.load(urllib.request.urlopen(ARTICLE))
    urls = {f['name']: f['download_url'] for f in meta['files']}
    paths = []
    for name in FILES:
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(urls[name], path)
        paths.append(path)
    return paths


def main(dest):
    os.makedirs(dest, exist_ok=True)
    psf, dcd = download(dest)
    here = os.path.dirname(os.path.abspath(__file__))

    u = mda.Universe(psf, dcd)
    ca = u.select_atoms('name CA')
    ca.write(os.path.join(here, 'adk_ca.pdb'))  # first frame

    frames = u.trajectory[::STRIDE]
    with mda.Writer(os.path.join(here, 'adk_ca.xtc'), ca.n_atoms) as w:
        for ts in frames:
            w.write(ca)
    print('%d frames, %d atoms' % (len(frames), ca.n_atoms))


if __name__ == '__main__':
    main(sys.argv[1])
```

- [ ] **Step 2: Generate the data**

Run: `.venv/bin/python tests/data/make_adk_ca.py /tmp/claude-11130/-home-arthur-work-affbio/b1077477-25d6-4aba-be6a-db48a74b64d8/scratchpad/adk_download && ls -la tests/data`
Expected: prints `1047 frames, 214 atoms`; `adk_ca.xtc` about 1 MB, `adk_ca.pdb` about 20 KB.

- [ ] **Step 3: Write `tests/data/README.md`**

```markdown
# AdK test trajectory

`adk_ca.pdb` and `adk_ca.xtc` hold the 214 C-alpha atoms of every 4th frame
(1,047 frames) of a 1 µs equilibrium simulation of adenylate kinase (4AKE).

Source: Seyler, Sean; Beckstein, Oliver (2017). Molecular dynamics trajectory
for benchmarking MDAnalysis. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.5108170.v1

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: C-alpha atoms only, every 4th frame, converted to PDB (first frame)
and XTC.

Regenerate with `python make_adk_ca.py /path/to/download/dir`
(downloads ~170 MB).
```

- [ ] **Step 4: Write the failing test** — `tests/test_data.py`:

```python
import MDAnalysis as mda
import numpy as np

from helpers import TOP, TRJ


def test_trajectory_shape():
    u = mda.Universe(TOP, TRJ)
    assert u.atoms.n_atoms == 214
    assert u.trajectory.n_frames == 1047
    assert set(u.atoms.names) == {'CA'}


def test_frames_written(adk_frames):
    assert len(adk_frames) == 1047
    first = mda.Universe(adk_frames[0]).atoms.positions
    np.testing.assert_allclose(first, mda.Universe(TOP).atoms.positions,
                               atol=0.01)
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'helpers'`

- [ ] **Step 6: Write `tests/helpers.py` and `tests/conftest.py`**

`tests/helpers.py`:

```python
import os
import warnings

import MDAnalysis as mda

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TOP = os.path.join(DATA, 'adk_ca.pdb')
TRJ = os.path.join(DATA, 'adk_ca.xtc')


def split_frames(top, trj, outdir, stop=None):
    """One PDB file per frame - the README's preparation snippet."""
    u = mda.Universe(top, trj)
    paths = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for ts in u.trajectory[:stop]:
            path = os.path.join(outdir, 'frame%04d.pdb' % ts.frame)
            u.atoms.write(path)
            paths.append(path)
    return paths
```

`tests/conftest.py`:

```python
import pytest

from helpers import TOP, TRJ, split_frames


@pytest.fixture(scope='session')
def adk_frames(tmp_path_factory):
    """All AdK C-alpha frames, one PDB file each, in frame order."""
    return split_frames(TOP, TRJ, str(tmp_path_factory.mktemp('frames')))
```

- [ ] **Step 7: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add tests/data tests/helpers.py tests/conftest.py tests/test_data.py
git commit -q -F - <<'EOF'
Add AdK C-alpha test trajectory (1,047 frames, CC BY 4.0)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 7: Structure loading and RMSD matrix (`affbio/structures.py`)

**Files:**
- Modify: `affbio/structures.py` (everything after the license header)
- Test: `tests/test_load.py`

**Interfaces:**
- Consumes: `AffBioError`, `check_decomposition`, `check_stage`, `effective_n` (Task 2); `rmsd_block` (Task 4); `task` (Task 3); `adk_frames` fixture (Task 6).
- Produces:
  - `expand_pdb_list(pdb_list: list[str]) -> list[str]`
  - `selection_indices(topology: str, selection='all') -> (n_atoms: int, indices: ndarray)` (raises `AffBioError`)
  - `read_coords(fname: str, n_atoms: int, idx) -> ndarray (len(idx), 3) float64`
  - `load_pdb_coords(Sfn, pdb_list, tier=1, topology=None, pbc=True, threshold=10.0, mpi=None, verbose=False, selection='all', *args, **kwargs)` (same signature as before)
  - `calc_rmsd_matrix(Sfn, tier=1, mpi=None, verbose=False, noalign=False, *args, **kwargs)` (same signature as before)
  - `DIAG_ROWS = 256`
  - HDF5: `tierN/struct` float64 `(N, natoms, 3)` with attr `nprocs`; `tierN/labels` str with attr `topology`; `tierN/rmsd` float32 `(N, N)` strict lower triangle, attrs `chunk`, `nprocs`.

- [ ] **Step 1: Write the failing tests** — `tests/test_load.py`:

```python
import h5py
import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from affbio.checks import AffBioError
from affbio.structures import (calc_rmsd_matrix, expand_pdb_list,
                               load_pdb_coords, read_coords,
                               selection_indices)
from affbio.utils import init_mpi


@pytest.fixture
def mpi():
    return init_mpi()


def qcp(a, b):
    a = np.ascontiguousarray(a - a.mean(0))
    b = np.ascontiguousarray(b - b.mean(0))
    return qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)


def test_load_with_selection(tmp_path, adk_frames, mpi):
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:40], mpi=mpi, selection='resid 1:100')
    with h5py.File(sfn, 'r') as f:
        g = f['tier1']
        assert g['struct'].shape == (40, 100, 3)
        assert g['struct'].attrs['nprocs'] == 1
        assert list(g['labels'].asstr()[:]) == adk_frames[:40]
        assert g['labels'].attrs['topology'] == adk_frames[0]
        np.testing.assert_allclose(
            g['struct'][0], read_coords(adk_frames[0], 214, np.arange(100)))


def test_broken_structure(tmp_path, adk_frames, mpi):
    lines = open(adk_frames[1]).readlines()
    del lines[next(i for i, l in enumerate(lines) if l.startswith('ATOM'))]
    broken = tmp_path / 'broken.pdb'
    broken.write_text(''.join(lines))
    with pytest.raises(AffBioError, match='Broken structure .*broken.pdb'):
        load_pdb_coords(str(tmp_path / 'm.hdf5'),
                        [adk_frames[0], str(broken)], mpi=mpi)


def test_old_prody_selection_is_explained(adk_frames):
    with pytest.raises(AffBioError, match='MDAnalysis selection syntax'):
        selection_indices(adk_frames[0], 'chain A')


def test_empty_selection(adk_frames):
    with pytest.raises(AffBioError, match='Empty selection'):
        selection_indices(adk_frames[0], 'resname XYZ')


def test_trjconv_style_frames(tmp_path, adk_frames):
    """gmx trjconv -sep frames have MODEL/TER/ENDMDL and may have CONECT."""
    atoms = [l for l in open(adk_frames[0]) if l.startswith('ATOM')]
    frame = tmp_path / 'frame0.pdb'
    frame.write_text('TITLE     Protein\nMODEL        1\n' + ''.join(atoms)
                     + 'TER\nCONECT    1    2\nENDMDL\n')
    idx = np.arange(214)
    np.testing.assert_allclose(read_coords(str(frame), 214, idx),
                               read_coords(adk_frames[0], 214, idx))


def test_expand_glob_natural_order(tmp_path):
    for k in (1, 2, 10, 9):
        (tmp_path / ('f%d.pdb' % k)).write_text('')
    assert expand_pdb_list([str(tmp_path / 'f*.pdb')]) == \
        [str(tmp_path / ('f%d.pdb' % k)) for k in (1, 2, 9, 10)]
    assert expand_pdb_list(['b.pdb', 'a.pdb']) == ['b.pdb', 'a.pdb']


@pytest.mark.parametrize('noalign', [False, True])
def test_rmsd_matrix(tmp_path, adk_frames, mpi, noalign):
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:300], mpi=mpi)
    calc_rmsd_matrix(sfn, mpi=mpi, noalign=noalign)
    with h5py.File(sfn, 'r') as f:
        X = f['tier1/struct'][:]
        R = f['tier1/rmsd'][:]
        assert f['tier1/rmsd'].attrs['chunk'] == 300
        assert f['tier1/rmsd'].attrs['nprocs'] == 1
    assert np.all(np.triu(R) == 0)
    # pairs inside and across the DIAG_ROWS = 256 row tiles
    for i, j in [(1, 0), (299, 0), (257, 256), (299, 298), (150, 3)]:
        if noalign:
            expected = np.sqrt(((X[i] - X[j]) ** 2).sum(-1).mean())
        else:
            expected = qcp(X[i], X[j])
        assert R[i, j] == pytest.approx(expected, abs=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_load.py -v`
Expected: FAIL at import: `ModuleNotFoundError: No module named 'prody'` (or `pyRMSD`).

- [ ] **Step 3: Replace `affbio/structures.py` after the license header (keep lines 1-21) with:**

```python

# General modules
import glob
import time
import warnings

# H5PY for storage
import h5py
from h5py import h5s

# NumPy
import numpy as np

# MDAnalysis for reading structures
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBReader
from MDAnalysis.exceptions import SelectionError

from natsort import natsorted

from .checks import AffBioError, check_decomposition, check_stage, \
    effective_n
from .rmsd import rmsd_block
from .utils import task

# Rows of a diagonal block per RMSD call, so only its lower half is computed
DIAG_ROWS = 256


def expand_pdb_list(pdb_list):
    """Expand a single quoted glob pattern into a naturally sorted list."""
    if len(pdb_list) == 1:
        ptrn = pdb_list[0]
        if '*' in ptrn or '?' in ptrn:
            return natsorted(glob.glob(ptrn))
    return list(pdb_list)


def selection_indices(topology, selection='all'):
    """Atom count of the topology and indices of the selected atoms."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = mda.Universe(topology)
    try:
        sel = u.select_atoms(selection)
    except (SelectionError, ValueError) as e:
        raise AffBioError(
            'Invalid --selection "%s": %s. AffBio uses MDAnalysis selection '
            'syntax, e.g. "chainID A" instead of ProDy\'s "chain A".'
            % (selection, e))
    if sel.n_atoms == 0:
        raise AffBioError('Empty selection "%s"' % selection)
    return u.atoms.n_atoms, sel.indices


def read_coords(fname, n_atoms, idx):
    """Coordinates of the selected atoms in the first model of a PDB file."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with PDBReader(fname) as reader:
            ts = reader.ts
            if ts.n_atoms != n_atoms:
                raise ValueError('has %d atoms, the topology has %d'
                                 % (ts.n_atoms, n_atoms))
            return ts.positions[idx].astype(np.float64)


def load_pdb_coords(
        Sfn,
        pdb_list,
        tier=1,
        topology=None,
        pbc=True,
        threshold=10.0,
        mpi=None,
        verbose=False,
        selection='all',
        *args, **kwargs):

    def check_pbc(coords, threshold=10.0):
        dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        bad = np.nonzero(dist >= threshold)[0]
        if bad.size:
            i = bad[0]
            raise ValueError('atoms %d and %d are %.1f A apart '
                             '(PBC artifact?)' % (i, i + 1, dist[i]))

    def parse_pdb(fname, n_atoms, idx):
        """Parse PDB files"""
        coords = read_coords(fname, n_atoms, idx)
        if pbc:
            check_pbc(coords, threshold)
        return coords

    def load_pdb_names(Sfn, pdb_list, topology):
        N = len(pdb_list)

        Sf = h5py.File(Sfn, 'w', driver='sec2')

        Gn = 'tier%d' % tier
        G = Sf.require_group(Gn)
        L = G.create_dataset(
            'labels',
            (N,),
            dtype=h5py.string_dtype())

        L[:] = pdb_list
        L.attrs['topology'] = topology

        Sf.close()

    def load_from_previous_tier(Sfn, tier, NPROCS):
        Sf = h5py.File(Sfn, 'r+', driver='sec2')

        PG = Sf['tier%d' % (tier - 1)]
        PS = PG['struct']
        nstruct, natoms, ncoords = PS.shape
        PNL = PG['labels'].asstr()

        PC = PG['aff_centers'][:]
        check_decomposition(len(PC), NPROCS)
        nstruct = effective_n(len(PC), NPROCS)
        if nstruct < len(PC):
            print('Using %d of %d centers of tier %d to split evenly across '
                  '%d processes; dropped: %s'
                  % (nstruct, len(PC), tier - 1, NPROCS,
                     ', '.join(PNL[c] for c in PC[nstruct:])))

        shape = (nstruct, natoms, ncoords)
        chunk = (1, natoms, ncoords)

        G = Sf.require_group('tier%d' % tier)
        S = G.require_dataset(
            'struct',
            shape,
            dtype=np.float64,
            chunks=chunk)
        S.attrs['nprocs'] = NPROCS

        L = G.require_dataset(
            'labels',
            (nstruct,),
            dtype=h5py.string_dtype())

        for i in range(nstruct):
            S[i] = PS[PC[i]][:]
            L[i] = PNL[PC[i]]

        Sf.close()

    comm, NPROCS, rank = mpi

    if tier > 1:
        if rank == 0:
            load_from_previous_tier(Sfn, tier, NPROCS)
        return

    pdb_list = expand_pdb_list(pdb_list)
    check_decomposition(len(pdb_list), NPROCS)
    N = effective_n(len(pdb_list), NPROCS)

    if not topology:
        topology = pdb_list[0]
    n_atoms, idx = selection_indices(topology, selection)

    shape = (N, len(idx), 3)
    chunk = (1, len(idx), 3)

    if rank == 0:
        if N < len(pdb_list):
            print('Using %d of %d structures to split evenly across %d '
                  'processes; dropped: %s'
                  % (N, len(pdb_list), NPROCS, ', '.join(pdb_list[N:])))
        load_pdb_names(Sfn, pdb_list[:N], topology)

    # Wait until the file exists
    comm.Barrier()

    # Init storage for matrices
    # HDF5 file
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    # Table for RMSD
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    S = G.require_dataset(
        'struct',
        shape,
        dtype=np.float64,
        chunks=chunk)
    S.attrs['nprocs'] = NPROCS

    # A little bit of dark magic for faster io
    Ss = S.id.get_space()
    ms = h5s.create_simple(chunk)

    tb, te = task(N, NPROCS, rank)

    for i in range(tb, te):
        try:
            tS = parse_pdb(pdb_list[i], n_atoms, idx)
        except Exception as e:
            raise AffBioError('Broken structure %s: %s' % (pdb_list[i], e))

        if verbose:
            print('Parsed %s' % pdb_list[i])

        Ss.select_hyperslab((i, 0, 0), chunk)
        S.id.write(ms, Ss, tS)

    # Wait for all processes
    comm.Barrier()

    Sf.close()


def calc_rmsd_matrix(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        noalign=False,
        *args, **kwargs):

    # --noalign compares raw coordinates: no centering, no rotation
    superpose = not noalign

    def calc_diag_chunk(ic, tS):
        # Strict lower triangle only, DIAG_ROWS rows at a time
        ln = len(ic)
        for r0 in range(0, ln, DIAG_ROWS):
            r1 = min(r0 + DIAG_ROWS, ln)
            rmsd_block(ic[r0:r1], ic[:r1], superpose, out=tS[r0:r1, :r1])
        for i in range(ln):
            tS[i, i:] = 0

    def calc_chunk(ic, jc, tS):
        rmsd_block(ic, jc, superpose, out=tS)

    def partition(N, NPROCS, rank):
        # Partiotioning
        l = N // NPROCS

        lN = (NPROCS + 1) * NPROCS // 2

        m = lN // NPROCS
        mr = lN % NPROCS

        if mr > 0:
            m = m + 1 if rank % 2 == 0 else m

        return (l, m)

    comm, NPROCS, rank = mpi

    # Reread structures by every process
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    S = G['struct']
    # Count number of structures
    N = S.len()

    try:
        check_decomposition(N, NPROCS)
        check_stage('calc_rmsd', N, NPROCS,
                    stored_nprocs=S.attrs.get('nprocs'))
    except AffBioError:
        Sf.close()
        raise

    l, m = partition(N, NPROCS, rank)

    # HDF5 file
    # Table for RMSD
    RM = G.require_dataset(
        'rmsd',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    RM.attrs['chunk'] = l
    RM.attrs['nprocs'] = NPROCS
    RMs = RM.id.get_space()

    # Init calculations
    tS = np.zeros((l, l), dtype=np.float32)
    ms = h5s.create_simple((l, l))

    i, j = rank, rank
    ic = S[i * l: (i + 1) * l]
    jc = ic

    for c in range(0, m):
        if rank == 0:
            tit = time.time()

        if i == j:
            calc_diag_chunk(ic, tS)
        else:
            calc_chunk(ic, jc, tS)

        RMs.select_hyperslab((i * l, j * l), (l, l))
        RM.id.write(ms, RMs, tS)

        if rank == 0:
            teit = time.time()
            if verbose:
                print("Step %d of %d T %s" % (c, m, teit - tit))

        # Dark magic of task assingment

        if 0 < (rank - c):
            j = j - 1
            jc = S[j * l: (j + 1) * l]
        elif rank - c == 0:
            i = NPROCS - rank - 1
            ic = S[i * l: (i + 1) * l]
        else:
            j = j + 1
            jc = S[j * l: (j + 1) * l]

    # Wait for all processes
    comm.Barrier()

    # Cleanup
    # Close matrix file
    Sf.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_load.py -v`
Expected: all passed. If `test_old_prody_selection_is_explained` fails because MDAnalysis raises a different exception type for `chain A`, add that type to the `except` clause in `selection_indices` (the probe on MDAnalysis 2.10 raised `SelectionError: Unknown selection token: 'chain'`).

- [ ] **Step 5: Commit**

```bash
git add affbio/structures.py tests/test_load.py
git commit -q -F - <<'EOF'
Port structure loading to MDAnalysis and RMSD matrix to rmsd_block

--noalign now gives raw RMSD everywhere (pyRMSD's NOSUP calculator
centered some pairs under MPI).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 8: Cluster matrix, median, preference and Affinity Propagation

**Files:**
- Modify: `affbio/prepare.py`, `affbio/aff_cluster.py` (everything after the license headers)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `AffBioError`, `check_decomposition`, `check_stage`, `check_disk`, `local_procs` (Task 2); `INT`, `FLOAT` (Task 3); `streaming_median` (Task 5); `load_pdb_coords`, `calc_rmsd_matrix` (Task 7).
- Produces (signatures unchanged from 0.0.x):
  - `prepare_cluster_matrix(Sfn, tier=1, mpi=None, verbose=False, *args, **kwargs)`: `tierN/cluster` float32 with attrs `chunk`, `nprocs`
  - `calc_median(Sfn, tier=1, mpi=None, verbose=False, debug=False, *args, **kwargs)`: attr `median`
  - `set_preference(Sfn, tier=1, preference=None, factor=1.0, mpi=None, verbose=False, debug=False, *args, **kwargs)`: attr `preference`
  - `aff_cluster(Sfn, tier=1, conv_iter=15, max_iter=2000, damping=0.95, mpi=None, verbose=False, debug=False, *args, **kwargs)`: `aff_labels`, `aff_centers` (int64), `aff_labels_merged` for tier > 1; raises `AffBioError` matching `'no exemplars'` when K == 0
  - `print_stat(Sfn, tier=1, merged=False, mpi=None, verbose=False, debug=False, *args, **kwargs)`: writes `aff_centers.out`, `aff_labels.out`, `aff_stat.out` in the working directory

- [ ] **Step 1: Write the failing tests** — `tests/test_pipeline.py`:

```python
import h5py
import numpy as np
import pytest

from affbio.aff_cluster import aff_cluster, print_stat
from affbio.checks import AffBioError
from affbio.prepare import calc_median, prepare_cluster_matrix, \
    set_preference
from affbio.structures import calc_rmsd_matrix, load_pdb_coords
from affbio.utils import init_mpi


@pytest.fixture
def matrix(tmp_path, adk_frames):
    """Cluster matrix with preference for 150 AdK frames."""
    mpi = init_mpi()
    sfn = str(tmp_path / 'm.hdf5')
    load_pdb_coords(sfn, adk_frames[:150], mpi=mpi)
    calc_rmsd_matrix(sfn, mpi=mpi)
    prepare_cluster_matrix(sfn, mpi=mpi)
    calc_median(sfn, mpi=mpi)
    set_preference(sfn, mpi=mpi)
    return sfn, mpi


def test_cluster_matrix_and_preference(matrix):
    sfn, mpi = matrix
    il = np.tril_indices(150, -1)
    with h5py.File(sfn, 'r') as f:
        R = f['tier1/rmsd'][:].astype(np.float64)
        C = f['tier1/cluster'][:]
        attrs = dict(f['tier1/cluster'].attrs)
    assert attrs['nprocs'] == 1
    # similarity = -RMSD^2 plus tiny noise, symmetric
    np.testing.assert_allclose(C[il], -(R[il] ** 2), rtol=1e-5)
    np.testing.assert_allclose(C[il], C.T[il], rtol=1e-5)
    median = np.median(C[il].astype(np.float64))
    assert attrs['median'] == pytest.approx(median)
    assert attrs['preference'] == pytest.approx(median, rel=1e-6)
    np.testing.assert_allclose(np.diag(C), attrs['preference'], rtol=1e-5)


def test_aff_cluster_and_stat(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    aff_cluster(sfn, mpi=mpi)
    print_stat(sfn, mpi=mpi)
    with h5py.File(sfn, 'r') as f:
        centers = f['tier1/aff_centers'][:]
        labels = f['tier1/aff_labels'][:]
        names = f['tier1/labels'].asstr()[:]
    assert labels.dtype == np.int64
    assert 1 < len(centers) < 150
    assert len(labels) == 150
    assert np.array_equal(labels[centers], np.arange(len(centers)))
    stat = (work / 'aff_stat.out').read_text()
    assert 'NUMBER OF CLUSTERS: %d' % len(centers) in stat
    lines = (work / 'aff_labels.out').read_text().splitlines()
    assert lines[0] == '%s\t%d' % (names[0], labels[0])
    assert "b'" not in (work / 'aff_centers.out').read_text()
    # only the three .out files remain; the temporary HDF5 file is gone
    assert sorted(p.name for p in work.iterdir()) == \
        ['aff_centers.out', 'aff_labels.out', 'aff_stat.out']


def test_no_exemplars_is_reported(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(AffBioError, match='no exemplars'):
        aff_cluster(sfn, mpi=mpi, conv_iter=1, max_iter=2, damping=0.99)
    assert list(work.iterdir()) == []


def test_bad_damping(matrix, tmp_path, monkeypatch):
    sfn, mpi = matrix
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AffBioError, match='damping'):
        aff_cluster(sfn, mpi=mpi, damping=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL at import: `SyntaxError: Missing parentheses in call to 'print'` (prepare.py / aff_cluster.py).

- [ ] **Step 3: Replace `affbio/prepare.py` after the license header (keep lines 1-21) with:**

```python

#General modules
import time

#NumPy for arrays
import numpy as np

#H5PY for storage
import h5py
from h5py import h5s

from .checks import AffBioError, check_decomposition, check_stage
from .median import streaming_median


def prepare_cluster_matrix(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        *args, **kwargs):

    def calc_chunk(l, tRM, tCM):
        ttCM = tRM + tCM * random_state.randn(l, l)
        # transposed inputs may give a Fortran-ordered result
        return np.ascontiguousarray(ttCM)

    def calc_chunk_diag(l, tRM, tCM):
        ttCM = tCM + tCM.transpose()
        ttRM = tRM + tRM.transpose()
        ttCM = calc_chunk(l, ttRM, ttCM)
        return ttCM

    comm, NPROCS, rank = mpi

    #Init RMSD matrix
    #Open matrix file in parallel mode
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    RM = G['rmsd']
    RMs = RM.id.get_space()

    N, N1 = RM.shape

    try:
        if N != N1:
            raise AffBioError(
                "S must be a square array (shape=%s)" % repr(RM.shape))
        check_decomposition(N, NPROCS)
        check_stage('prepare_matrix', N, NPROCS,
                    stored_chunk=RM.attrs['chunk'],
                    stored_nprocs=RM.attrs.get('nprocs'))
    except AffBioError:
        Sf.close()
        raise

    l = N // NPROCS

    CM = G.require_dataset(
        'cluster',
        (N, N),
        dtype=np.float32,
        chunks=(l, l))
    CM.attrs['chunk'] = l
    CM.attrs['nprocs'] = NPROCS
    CMs = CM.id.get_space()

    random_state = np.random.RandomState(0)
    x = np.finfo(np.float32).eps
    y = np.finfo(np.float32).tiny * 100

    #Partiotioning
    lN = (NPROCS + 1) * NPROCS // 2

    m = lN // NPROCS
    mr = lN % NPROCS

    if mr > 0:
        m = m + 1 if rank % 2 == 0 else m

    #Init calculations
    tRM = np.zeros((l, l), dtype=np.float32)
    tCM = np.zeros((l, l), dtype=np.float32)
    ttCM = np.zeros((l, l), dtype=np.float32)
    ms = h5s.create_simple((l, l))

    i, j = rank, rank

    for c in range(m):
        if rank == 0:
            tit = time.time()
        RMs.select_hyperslab((i * l, j * l), (l, l))
        RM.id.read(ms, RMs, tRM)

        #tRM = -1 * tRM ** 2
        tRM **= 2
        tRM *= -1
        tCM = tRM * x + y

        if i == j:
            ttCM = calc_chunk_diag(l, tRM[:], tCM[:])
            CMs.select_hyperslab((i * l, j * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

        else:
            ttCM = calc_chunk(l, tRM[:], tCM[:])
            CMs.select_hyperslab((i * l, j * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

            ttCM = calc_chunk(l, tRM.transpose(), tCM.transpose())
            CMs.select_hyperslab((j * l, i * l), (l, l))
            CM.id.write(ms, CMs, ttCM)

        if rank == 0:
            teit = time.time()
            if verbose:
                print("Step %d of %d T %s" % (c, m, teit - tit))

        if (rank - c) > 0:
            j = j - 1
        elif (rank - c) == 0:
            i = NPROCS - rank - 1
        else:
            j = j + 1

    #Wait for all processes
    comm.Barrier()

    Sf.close()


def calc_median(
        Sfn,
        tier=1,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    #Init cluster matrix
    #Open matrix file in single mode
    Sf = h5py.File(Sfn, 'r+', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    CM = G['cluster']

    l = CM.attrs['chunk']

    N, N1 = CM.shape

    if N != N1:
        raise ValueError(
            "S must be a square array (shape=%s)" % repr(CM.shape))

    if l <= 0:
        raise ValueError(
            "Wrong chunk size in RMSD matrix")

    if N * N1 > 10000:
        # Exact median of the lower triangle, streamed from disk
        median = streaming_median(CM)
    else:
        median = np.median(CM[:])

    if verbose:
        print('Median: %f' % median)

    CM.attrs['median'] = median

    Sf.close()


def set_preference(
        Sfn,
        tier=1,
        preference=None,
        factor=1.0,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    #Init storage for matrices
    #Get file name
    #Open matrix file in parallel mode
    Sf = h5py.File(Sfn, 'r+', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    #Open table with data for clusterization
    SS = G['cluster']
    SSs = SS.id.get_space()
    ms = h5s.create_simple((1, 1))
    tS = np.zeros((1,), dtype=np.float32)

    ft = np.float32

    N, N1 = SS.shape

    if N != N1:
        raise ValueError("S must be a square array \
            (shape=%s)" % repr((N, N1)))

    if not preference:
        try:
            preference = SS.attrs['median']
        except:
            raise ValueError(
                'Unable to get preference from cluster matrix')

    preference = ft(preference * factor)

    #Copy input data and
    #place preference on diagonal
    random_state = np.random.RandomState(0)
    x = np.finfo(ft).eps
    y = np.finfo(ft).tiny * 100

    for i in range(N):
        tS[0] = preference + (preference * x + y) * random_state.randn()
        SSs.select_hyperslab((i, i), (1, 1))
        SS.id.write(ms, SSs, tS)

    SS.attrs['preference'] = preference

    if verbose:
        print('Preference: %f' % preference)

    Sf.close()
```

- [ ] **Step 4: Replace `affbio/aff_cluster.py` after the license header (keep lines 1-21) with:**

```python

# This code is heavily relies on the Affiniti Propagation
# code from Scikit-learn package
# see http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation


# General modules
import os
import time
import uuid
import shutil
import psutil
import tempfile
from os.path import join as osp

# NumPy for arrays
import numpy as np
import bottleneck as bn

# H5PY for storage
import h5py
from h5py import h5s

from .checks import AffBioError, check_decomposition, check_disk, \
    check_stage, local_procs
from .mpi import INT, FLOAT
from .utils import Bunch, task


def aff_cluster(
        Sfn,
        tier=1,
        conv_iter=15,
        max_iter=2000,
        damping=0.95,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    NPROCS_LOCAL = local_procs()

    # Init storage for matrices
    # Get file name
    # Open matrix file in parallel mode
    if NPROCS == 1:
        Sf = h5py.File(Sfn, 'r+', driver='sec2')
    else:
        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=comm)
        Sf.atomic = True

    Gn = 'tier%d' % tier
    G = Sf.require_group(Gn)
    # Open table with data for clusterization
    SS = G['cluster']
    SSs = SS.id.get_space()

    try:
        check_decomposition(SS.shape[0], NPROCS)
        check_stage('aff_cluster', SS.shape[0], NPROCS,
                    stored_nprocs=SS.attrs.get('nprocs'))
    except AffBioError:
        Sf.close()
        raise

    params = {
        'N': 0,
        'l': 0,
        'll': 0,
        'TMfn': '',
        'disk': False,
        'preference': 0.0,
        'error': None}

    P = Bunch(params)

    ft = np.float32

    if rank == 0:

        N, N1 = SS.shape

        try:
            if N != N1:
                raise AffBioError("S must be a square array \
                    (shape=%s)" % repr((N, N1)))

            try:
                preference = SS.attrs['preference']
            except KeyError:
                raise AffBioError(
                    'Unable to get preference from cluster matrix')

            if max_iter < 0:
                raise AffBioError('max_iter must be > 0')

            if not 0 < conv_iter < max_iter:
                raise AffBioError('conv_iter must lie in \
                    interval between 0 and max_iter')

            if damping < 0.5 or damping >= 1.0:
                raise AffBioError(
                    'damping must lie in interval between 0.5 and 1')
        except AffBioError as e:
            P.error = str(e)

    if rank == 0 and not P.error:

        P.N = N

        print('#' * 10, 'Main params', '#' * 10)
        print('preference: %.3f' % preference)
        print('damping: %.3f' % damping)
        print('conv_iter: %d' % conv_iter)
        print('max_iter: %d' % max_iter)
        print('#' * 31)

        P.TMbfn = str(uuid.uuid1())
        P.TMfn = P.TMbfn + '.hdf5'

        # N is a multiple of 4 * NPROCS (checked above), as Gather needs
        l = N // NPROCS

        # Fit to memory
        MEM = psutil.virtual_memory().available / NPROCS_LOCAL
        # MEM = 500 * 10 ** 6
        ts = np.dtype(ft).itemsize * N  # Python give bits
        ts *= 8 * 1.1  # Allocate memory for e, tE, and ...
        # MEM -= ts  # ----
        tl = int(MEM // ts)  # Allocate memory for tS, tA, tR....

        def adjust_cache(tl, l):
            while float(l) % float(tl) > 0:
                tl -= 1
            return tl

        if tl < l:
            P.disk = True
            try:
                cache = 0
#                cache = int(sys.argv[1])
                assert cache < l
            except:
                cache = tl
            tl = adjust_cache(tl, l)
            P.l = l
            P.ll = tl
            # Out-of-memory mode keeps S and R of every process on disk
            try:
                check_disk({tempfile.gettempdir(): 8 * l * N * NPROCS_LOCAL})
            except AffBioError as e:
                P.error = str(e)
        else:
            P.l = l
            P.ll = l

        if verbose:
            print("Available memory per process: %.2fG" % (MEM / 10.0 ** 9))
            print("Memory per row: %.2fM" % (ts / 10.0 ** 6))
            print("Estimated memory per process: %.2fG"
                  % (ts * P.ll / 10.0 ** 9))
            print('Cache size is %d of %d' % (P.ll, P.l))

    P = comm.bcast(P)

    if P.error:
        Sf.close()
        raise AffBioError(P.error)

    N = P.N
    l = P.l
    ll = P.ll

    ms = h5s.create_simple((ll, N))
    ms_l = h5s.create_simple((N,))

    tb, te = task(N, NPROCS, rank)

    tS = np.zeros((ll, N), dtype=ft)
    tSl = np.zeros((N,), dtype=ft)

    disk = P.disk

    if disk is True:
        TMLfd = tempfile.mkdtemp()
        TMLfn = osp(TMLfd, P.TMbfn + '_' + str(rank) + '.hdf5')
        TMLf = h5py.File(TMLfn, 'w')
        TMLf.atomic = True

        S = TMLf.create_dataset('S', (l, N), dtype=ft)
        Ss = S.id.get_space()

    # Copy input data and
    # place preference on diagonal
    z = - np.finfo(ft).max

    for i in range(tb, te, ll):
        SSs.select_hyperslab((i, 0), (ll, N))
        SS.id.read(ms, SSs, tS)

        if disk is True:
            Ss.select_hyperslab((i - tb, 0), (ll, N))
            S.id.write(ms, Ss, tS)

    if disk is True:
        R = TMLf.create_dataset('R', (l, N), dtype=ft)
        Rs = R.id.get_space()

    tRold = np.zeros((ll, N), dtype=ft)
    tR = np.zeros((ll, N), dtype=ft)
    tdR = np.zeros((l,), dtype=ft)

    # Shared storage
    if NPROCS == 1:
        TMf = h5py.File(P.TMfn, 'w', driver='sec2')
    else:
        TMf = h5py.File(P.TMfn, 'w', driver='mpio', comm=comm)
        TMf.atomic = True

    Rp = TMf.create_dataset('Rp', (N, N), dtype=ft)
    Rps = Rp.id.get_space()

    tRp = np.zeros((ll, N), dtype=ft)
    tRpa = np.zeros((N, ll), dtype=ft)

    A = TMf.create_dataset('A', (N, N), dtype=ft)
    As = A.id.get_space()

    tAS = np.zeros((ll, N), dtype=ft)
    tAold = np.zeros((N, ll), dtype=ft)
    tA = np.zeros((N, ll), dtype=ft)
    tdA = np.zeros((l,), dtype=ft)

    e = np.zeros((N, conv_iter), dtype=np.int8)
    tE = np.zeros((N,), dtype=np.int8)
    ttE = np.zeros((l,), dtype=np.int8)

    converged = False
    cK = 0
    K = 0
    ind = np.arange(ll)

    comm.Barrier()


    # Starting clustering
    for it in range(max_iter):
        if rank == 0:
            if verbose is True:
                print('=' * 10 + 'It %d' % (it) + '=' * 10)
                tit = time.time()

        # Compute responsibilities
        for i in range(tb, te, ll):
            if disk is True:
                il = i - tb
                Ss.select_hyperslab((il, 0), (ll, N))
                S.id.read(ms, Ss, tS)
            # tS = S[i, :]
                Rs.select_hyperslab((il, 0), (ll, N))
                R.id.read(ms, Rs, tRold)
            else:
                tRold = tR.copy()

            if NPROCS > 1:
                As.select_hyperslab((i, 0), (ll, N))
                A.id.read(ms, As, tAS)
            else:
                tAS = tA.copy()

            # Tas = a[I, :]
            tAS += tS
            # tRold = R[i, :]

            tI = bn.nanargmax(tAS, axis=1)
            tY = tAS[ind, tI]
            tAS[ind, tI[ind]] = z
            tY2 = bn.nanmax(tAS, axis=1)

            tR = tS - tY[:, np.newaxis]
            tR[ind, tI[ind]] = tS[ind, tI[ind]] - tY2[ind]
            tR = (1 - damping) * tR + damping * tRold

            tRp = np.maximum(tR, 0)

            tRp[:ll, i:i + ll].flat[::ll + 1] = tR[:ll, i:i + ll].flat[::ll + 1]
            tdR[i - tb: i - tb + ll] = tR[:ll, i:i + ll].flat[::ll + 1]

            if disk is True:
                R.id.write(ms, Rs, tR)
                # R[i, :] = tR

            if NPROCS > 1:
                Rps.select_hyperslab((i, 0), (ll, N))
                Rp.id.write(ms, Rps, tRp)

            # Rp[i, :] = tRp
        if rank == 0:
            if verbose is True:
                teit1 = time.time()
                print('R T %s' % (teit1 - tit))

        comm.Barrier()

        # Compute availabilities
        for j in range(tb, te, ll):

            if NPROCS > 1 or disk is True:
                As.select_hyperslab((0, j), (N, ll))

            if disk is True:
                A.id.read(ms, As, tAold)
            else:
                tAold = tA.copy()

            if NPROCS > 1:
                Rps.select_hyperslab((0, j), (N, ll))
                Rp.id.read(ms, Rps, tRpa)
            else:
                tRpa = tRp.copy()
            # tRp = Rp[:, j]

            tA = bn.nansum(tRpa, axis=0)[np.newaxis, :] - tRpa
            tdA[j - tb: j - tb + ll] = tA[j: j + ll, :ll].flat[::ll + 1]

            tA = np.minimum(tA, 0)

            tA[j:j + ll, :ll].flat[::ll + 1] = tdA[j - tb: j - tb + ll]

            tA *= (1.0 - damping)

            tA += damping * tAold

            for jl in range(ll):
                tdA[j - tb + jl] = tA[j + jl, jl]

            if NPROCS > 1:
                A.id.write(ms, As, tA)

        if rank == 0:
            if verbose is True:
                teit2 = time.time()
                print('A T %s' % (teit2 - teit1))

        ttE = np.array(((tdA + tdR) > 0), dtype=np.int8)

        if NPROCS > 1:
            comm.Gather([ttE, INT], [tE, INT])
            comm.Bcast([tE, INT])
        else:
            tE = ttE

        e[:, it % conv_iter] = tE
        pK = K
        K = bn.nansum(tE)

        if rank == 0:
            if verbose is True:
                teit = time.time()
                cc = ''
                if K == pK:
                    if cK == 0:
                        cK += 1
                    elif cK > 1:
                        cc = ' Conv %d of %d' % (cK, conv_iter)
                else:
                    cK = 0

                print('Total K %d T %s%s' % (K, teit - tit, cc))

        if it >= conv_iter:

            if rank == 0:
                se = bn.nansum(e, axis=1)
                converged = (bn.nansum((se == conv_iter) + (se == 0)) == N)

                if (converged == np.bool_(True)) and (K > 0):
                    if verbose is True:
                        print("Converged after %d iterations." % (it))
                    converged = True
                else:
                    converged = False

            converged = comm.bcast(converged, root=0)

        if converged is True:
            break

    if not converged and verbose and rank == 0:
        print("Failed to converge after %d iterations." % (max_iter))

    if K > 0:

        I = np.nonzero(e[:, 0])[0]
        C = np.zeros((N,), dtype=np.int64)
        tC = np.zeros((l,), dtype=np.int64)

        for i in range(l):
            if disk is True:
                Ss.select_hyperslab((i, 0), (1, N))
                S.id.read(ms_l, Ss, tSl)
            else:
                tSl = tS[i]

            tC[i] = bn.nanargmax(tSl[I])

        comm.Gather([tC, INT], [C, INT])

        if rank == 0:
            C[I] = np.arange(K)

        comm.Bcast([C, INT])

        for k in range(K):
            if NPROCS > 1:
                ii = np.where(C == k)[0]
                tN = ii.shape[0]

                tI = np.zeros((tN, ), dtype=np.float32)
                ttI = np.zeros((tN, ), dtype=np.float32)
                tttI = np.zeros((tN, ), dtype=np.float32)
                ms_k = h5s.create_simple((tN,))

                j = rank
                while j < tN:
                    ind = [(ii[i], ii[j]) for i in range(tN)]
                    SSs.select_elements(ind)
                    SS.id.read(ms_k, SSs, tttI)

                    ttI[j] = bn.nansum(tttI)
                    j += NPROCS

                comm.Reduce([ttI, FLOAT], [tI, FLOAT])

            else:
                ii = np.where(C == k)[0]
                tI = bn.nansum(tS[ii[:, np.newaxis], ii], axis=0)

            if rank == 0:
                I[k] = ii[bn.nanargmax(tI)]

        I.sort()
        comm.Bcast([I, INT])

        for i in range(l):
            if disk is True:
                Ss.select_hyperslab((i, 0), (1, N))
                S.id.read(ms_l, Ss, tSl)
            else:
                tSl = tS[i]

            tC[i] = bn.nanargmax(tSl[I])

        comm.Gather([tC, INT], [C, INT])

        if rank == 0:
            C[I] = np.arange(K)

    # Cleanup
    Sf.close()
    TMf.close()

    if disk is True:
        TMLf.close()
        shutil.rmtree(TMLfd)

    comm.Barrier()

    if rank == 0:

        os.remove(P.TMfn)

        if verbose:
            print('APN: %d' % K)

        if K > 0:

            Sf = h5py.File(Sfn, 'r+', driver='sec2')
            Gn = 'tier%d' % tier
            G = Sf.require_group(Gn)

            if 'aff_labels' in G.keys():
                del G['aff_labels']

            L = G.require_dataset(
                'aff_labels',
                shape=C.shape,
                dtype=np.int64)

            L[:] = C[:]

            if tier > 1:
                PGn = 'tier%d' % (tier - 1)
                PG = Sf.require_group(PGn)
                PL = PG['aff_labels'][:]
                NL = np.copy(PL)

                for i in range(len(C)):
                    ind = np.where(PL == i)
                    NL[ind] = C[i]

                if 'aff_labels_merged' in G.keys():
                    del G['aff_labels_merged']

                LM = G.require_dataset(
                    'aff_labels_merged',
                    shape=PL.shape,
                    dtype=np.int64)

                LM[:] = NL[:]

            if 'aff_centers' in G.keys():
                del G['aff_centers']

            CM = G.require_dataset(
                'aff_centers',
                shape=I.shape,
                dtype=np.int64)
            CM[:] = I[:]
            Sf.close()

    # K is the same on every process
    if K == 0:
        raise AffBioError(
            'Affinity propagation found no exemplars after %d iterations; '
            'try a larger --max_iter or a different --preference/--factor.'
            % max_iter)


def print_stat(
        Sfn,
        tier=1,
        merged=False,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    Sf = h5py.File(Sfn, 'r', driver='sec2')
    Gn = 'tier%d' % tier
    G = Sf[Gn]

    C = G['aff_centers'][:]
    NC = len(C)

    I = G['aff_labels'][:]

    LC = G['labels'].asstr()[:]
    L = LC
    if tier > 1 and merged:
        I = G['aff_labels_merged'][:]
        L = Sf['tier1']['labels'].asstr()[:]
    NI = len(I)

    Sf.close()

    with open('aff_centers.out', 'w') as f:
        for i in range(NC):
            f.write("%s\t%d\n" % (LC[C[i]], i))

    with open('aff_labels.out', 'w') as f:
        for i in range(NI):
            f.write("%s\t%d\n" % (L[i], I[i]))

    with open('aff_stat.out', 'w') as f:
        f.write("AFF tier: %d\n" % tier)
        f.write("NUMBER OF CLUSTERS: %d\n" % NC)

        cs = np.bincount(I)
        pcs = cs * 100.0 / NI

        f.write('INDEX CENTER SIZE PERCENTAGE\n')

        for i in range(NC):
            f.write("%d\t%s\t%d\t%.3f\n" % (i, LC[C[i]], cs[i], pcs[i]))
```

Changes from 0.0.x in this file, for the record: `MPI.*` -> `INT`/`FLOAT`; `np.int` -> `np.int64`; parameter errors are `AffBioError` and are broadcast so every process stops; the "magic 4" truncation is replaced by `check_stage`; a disk check guards the out-of-memory mode; K == 0 raises instead of writing scalar datasets; a stale `aff_labels_merged` is deleted before rewriting; `print_stat` reads labels as `str`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: all passed. If `test_no_exemplars_is_reported` fails because two iterations already produce an exemplar, the K == 0 path was not reached: confirm with `aff_cluster(..., verbose=True)` that `Total K` is 0 at both iterations before changing anything, and report it rather than weakening the assertion.

- [ ] **Step 6: Commit**

```bash
git add affbio/prepare.py affbio/aff_cluster.py tests/test_pipeline.py
git commit -q -F - <<'EOF'
Port cluster matrix, median, preference and AP to Python 3

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 9: CLI, preflight and end-to-end clustering

**Files:**
- Modify: `affbio/cli.py` (everything after the license header)
- Modify: `tests/helpers.py` (add `run_affbio`), `tests/conftest.py` (add `small_run`)
- Test: `tests/test_cli.py`, `tests/test_cluster.py`

**Interfaces:**
- Consumes: everything from Tasks 2-8 (including `check_parallel_io`).
- Produces:
  - `affbio.cli.run()`; `get_args(choices)`; `main_tasks()`, `misc_tasks()`, `wrapper_tasks()`; `expand_tasks(tasks: list[str]) -> list[str]`; `preflight(tasks, args: dict, nprocs: int) -> list[str]` (warnings); `tier_size(sfn, tier, from_previous) -> (n, natoms, existing)`; `python -m affbio.cli` works.
  - `tests/helpers.py`: `run_affbio(args: list[str], cwd, env=None, launcher=()) -> subprocess.CompletedProcess` (text mode).
  - `tests/conftest.py`: session fixture `small_run -> pathlib.Path` (directory with `m.hdf5` after `-t cluster` on the first 120 frames).
- Note: `render_b_factor` still calls GROMACS until Task 10; no test in this task uses `render`.

- [ ] **Step 1: Add the CLI runner and shared fixture**

Append to `tests/helpers.py`:

```python
import subprocess
import sys


def run_affbio(args, cwd, env=None, launcher=()):
    """Run `affbio ARGS` (optionally under an MPI launcher) in cwd."""
    cmd = list(launcher) + [sys.executable, '-m', 'affbio.cli'] + list(args)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                          text=True)
```

(Move the two new imports to the top of the file with the others.)

Replace `tests/conftest.py` with:

```python
import pytest

from helpers import TOP, TRJ, run_affbio, split_frames


@pytest.fixture(scope='session')
def adk_frames(tmp_path_factory):
    """All AdK C-alpha frames, one PDB file each, in frame order."""
    return split_frames(TOP, TRJ, str(tmp_path_factory.mktemp('frames')))


@pytest.fixture(scope='session')
def small_run(tmp_path_factory, adk_frames):
    """Directory with m.hdf5 after `affbio -t cluster` on 120 frames."""
    d = tmp_path_factory.mktemp('small_run')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames[:120],
                   cwd=d)
    assert r.returncode == 0, r.stderr
    return d
```

- [ ] **Step 2: Write the failing tests**

`tests/test_cli.py`:

```python
import os
import subprocess
import sys

import h5py

from affbio.cli import expand_tasks
from helpers import run_affbio

HIDE_MPI = ("import sys; sys.modules['mpi4py'] = None; "
            "from affbio.cli import run; run()")


def run_without_mpi4py(args, cwd, env=None):
    return subprocess.run([sys.executable, '-c', HIDE_MPI] + list(args),
                          cwd=cwd, env=env, capture_output=True, text=True)


def test_help_lists_tasks(tmp_path):
    r = run_affbio(['--help'], cwd=tmp_path)
    assert r.returncode == 0
    for name in ('load_pdb', 'calc_rmsd', 'prepare_matrix', 'calc_median',
                 'set_preference', 'aff_cluster', 'print_stat',
                 'cluster_to_trj', 'render', 'cluster', 'all'):
        assert name in r.stdout


def test_expand_tasks():
    assert expand_tasks(['cluster'])[0] == 'load_pdb'
    assert expand_tasks(['cluster'])[-1] == 'print_stat'
    assert expand_tasks(['cluster', 'render'])[-1] == 'render'
    assert expand_tasks(['calc_median']) == ['calc_median']


def test_single_structure_is_rejected(tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f', adk_frames[0]],
                   cwd=tmp_path)
    assert r.returncode == 1
    assert 'Need at least 2 structures' in r.stderr
    assert 'Traceback' not in r.stderr


def test_bad_selection_is_explained(tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '--selection', 'chain A',
                    '-f'] + adk_frames[:5], cwd=tmp_path)
    assert r.returncode == 1
    assert 'MDAnalysis selection syntax' in r.stderr
    assert 'Traceback' not in r.stderr


def test_missing_matrix_file_is_explained(tmp_path):
    r = run_affbio(['-m', 'nothing.hdf5', '-t', 'calc_rmsd'], cwd=tmp_path)
    assert r.returncode == 1
    assert 'run the earlier tasks first' in r.stderr


def test_runs_without_mpi4py(tmp_path, adk_frames):
    r = run_without_mpi4py(['-m', 'm.hdf5', '-t', 'load_pdb', 'calc_rmsd',
                            '-f'] + adk_frames[:20], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / 'm.hdf5', 'r') as f:
        assert f['tier1/rmsd'].shape == (20, 20)


def test_mpi_launcher_without_mpi4py(tmp_path):
    env = dict(os.environ, OMPI_COMM_WORLD_SIZE='2')
    r = run_without_mpi4py(['--help'], cwd=tmp_path, env=env)
    assert r.returncode == 1
    assert 'affbio[mpi]' in r.stderr
```

`tests/test_cluster.py`:

```python
import shutil

import h5py
import numpy as np
import pytest
from MDAnalysis.lib import qcprot

from helpers import run_affbio


@pytest.fixture(scope='module')
def full_run(tmp_path_factory, adk_frames):
    d = tmp_path_factory.mktemp('full_run')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames,
                   cwd=d)
    assert r.returncode == 0, r.stderr
    return d


def read(run_dir, name, tier=1):
    with h5py.File(run_dir / 'm.hdf5', 'r') as f:
        return f['tier%d' % tier][name][:]


def test_rmsd_matrix(full_run):
    R = read(full_run, 'rmsd')
    X = read(full_run, 'struct')
    assert R.shape == (1047, 1047)
    assert np.all(np.diag(R) == 0)
    a = np.ascontiguousarray(X[700] - X[700].mean(0))
    b = np.ascontiguousarray(X[3] - X[3].mean(0))
    expected = qcprot.CalcRMSDRotationalMatrix(a, b, len(a), None, None)
    assert R[700, 3] == pytest.approx(expected, abs=1e-5)


def test_clusters(full_run):
    centers = read(full_run, 'aff_centers')
    labels = read(full_run, 'aff_labels')
    with h5py.File(full_run / 'm.hdf5', 'r') as f:
        attrs = dict(f['tier1/cluster'].attrs)
    assert 'median' in attrs and 'preference' in attrs
    assert 1 < len(centers) < 1047
    assert len(labels) == 1047
    assert np.array_equal(labels[centers], np.arange(len(centers)))


def test_out_files(full_run, adk_frames):
    labels = read(full_run, 'aff_labels')
    lines = (full_run / 'aff_labels.out').read_text().splitlines()
    assert lines == ['%s\t%d' % (f, k) for f, k in zip(adk_frames, labels)]
    assert "b'" not in (full_run / 'aff_centers.out').read_text()
    stat = (full_run / 'aff_stat.out').read_text()
    assert 'NUMBER OF CLUSTERS: %d' % len(read(full_run, 'aff_centers')) \
        in stat


def test_rerun_gives_identical_clusters(full_run, tmp_path, adk_frames):
    r = run_affbio(['-m', 'm.hdf5', '-t', 'cluster', '-f'] + adk_frames,
                   cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert np.array_equal(read(tmp_path, 'aff_labels'),
                          read(full_run, 'aff_labels'))
    assert np.array_equal(read(tmp_path, 'aff_centers'),
                          read(full_run, 'aff_centers'))


def test_quoted_glob_in_natural_order(tmp_path, adk_frames):
    for k in range(12):
        shutil.copy(adk_frames[k], tmp_path / ('f%d.pdb' % (k + 1)))
    r = run_affbio(['-m', 'm.hdf5', '-t', 'load_pdb', '-f',
                    str(tmp_path / 'f*.pdb')], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / 'm.hdf5', 'r') as f:
        names = list(f['tier1/labels'].asstr()[:])
    assert names == [str(tmp_path / ('f%d.pdb' % (k + 1)))
                     for k in range(12)]


def test_tier2_merged_labels(full_run, tmp_path):
    shutil.copy(full_run / 'm.hdf5', tmp_path / 'm.hdf5')
    r = run_affbio(['-m', 'm.hdf5', '--tier', '2', '-t', 'cluster',
                    '--merged_labels'], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    k1 = len(read(tmp_path, 'aff_centers', tier=1))
    k2 = len(read(tmp_path, 'aff_centers', tier=2))
    merged = read(tmp_path, 'aff_labels_merged', tier=2)
    assert 0 < k2 <= k1
    assert len(merged) == 1047
    assert merged.min() >= 0 and merged.max() < k2
    assert len((tmp_path / 'aff_labels.out').read_text().splitlines()) == 1047
    assert 'AFF tier: 2' in (tmp_path / 'aff_stat.out').read_text()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_cluster.py -v`
Expected: FAIL; `test_cli.py` at import (`ImportError: cannot import name 'expand_tasks'` or Py2 syntax in `misc.py`), `test_cluster.py` with non-zero `affbio` exit codes.

- [ ] **Step 4: Port `affbio/misc.py` just enough to import** (Task 10 rewrites it). In `affbio/misc.py`:
  - `g_rmsf.communicate(input='0')` -> `g_rmsf.communicate(input=b'0')`
  - `map(os.remove, centers)` -> `for c in centers: os.remove(c)` (two lines)
  - in `copy_connects`, `np.array(map(lambda x: re.match('CONECT', x), inpdb), dtype=np.bool)` -> `np.array([bool(re.match('CONECT', x)) for x in inpdb], dtype=bool)`

Also in `affbio/AffRender.py`: `re.sub('\s+', ...)` -> `re.sub(r'\s+', ...)`, `re.search('aff_(\d+)', model)` -> `re.search(r'aff_(\d+)', model)` (silences `SyntaxWarning`).

- [ ] **Step 5: Replace `affbio/cli.py` after the license header (keep lines 1-21) with:**

```python

# General modules
import argparse as ag
import os
import sys
import traceback
from collections import OrderedDict as OD

import h5py

from affbio.utils import init_mpi, init_logging, finish_logging, dummy, \
    init_debug, finish_debug
from affbio.checks import AffBioError, check_decomposition, check_disk, \
    check_parallel_io, disk_needs, effective_n, memory_warning
from affbio.structures import load_pdb_coords, calc_rmsd_matrix, \
    expand_pdb_list, selection_indices
from affbio.prepare import prepare_cluster_matrix, calc_median, set_preference
from affbio.aff_cluster import aff_cluster, print_stat
from affbio.misc import render_b_factor, cluster_to_trj

# Tasks that create or split the N x N matrices
MATRIX_TASKS = ('load_pdb', 'calc_rmsd', 'prepare_matrix', 'aff_cluster')


def get_args(choices):
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Parallel affinity propagation for biomolecules')

    parser.add_argument('-m',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('--tier',
                        dest='tier',
                        metavar='TIER',
                        type=int,
                        default=1,
                        help='Round of clusterization')

    parser.add_argument('-t', '--task',
                        nargs='+',
                        required=True,
                        choices=choices,
                        metavar='TASK',
                        help='Task to do. Available options \
                        are: %s' % ", ".join(choices))

    parser.add_argument('-o', '--output',
                        dest='output',
                        metavar='OUTPUT',
                        help='For "render" and "cluster_to_trj" tasks \
                        name of output PNG image or multiframe PDB file')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    load_pdb = parser.add_argument_group('load_pdb')

    load_pdb.add_argument('-f',
                          nargs='*',
                          type=str,
                          dest='pdb_list',
                          metavar='FILE',
                          help='PDB files')

    load_pdb.add_argument('-s',
                          type=str,
                          dest='topology',
                          help='Topology PDB file')

    load_pdb.add_argument('--nopbc',
                          action='store_false',
                          dest='pbc',
                          help='Do not check for PBC artifacts')

    load_pdb.add_argument('--pbc_threshold',
                          type=float,
                          dest='threshold',
                          metavar='THRESHOLD',
                          default=10.0,
                          help='Threshold in Angstroms to check PBC \
                          artifacts. Default is 10.0 A')

    load_pdb.add_argument('--noalign',
                          action='store_true',
                          dest='noalign',
                          help='Do not superpose structures')

    load_pdb.add_argument('--selection',
                          default='all',
                          dest='selection',
                          help='Atom selection string in MDAnalysis syntax')

    preference = parser.add_argument_group('calculate_preference')

    preference.add_argument('--factor',
                            type=float,
                            dest='factor',
                            metavar='FACTOR',
                            default=1.0,
                            help='Multiplier for median')
    preference.add_argument('--preference',
                            type=float,
                            dest='preference',
                            metavar='PREFERENCE',
                            help='Override computed preference')

    aff = parser.add_argument_group('aff_cluster')

    aff.add_argument('--conv_iter',
                     type=int,
                     dest='conv_iter',
                     metavar='ITERATIONS',
                     default=15,
                     help='Iterations to converge')

    aff.add_argument('--max_iter',
                     type=int,
                     dest='max_iter',
                     metavar='ITERATIONS',
                     default=2000,
                     help='Maximum iterations')

    aff.add_argument('--damping',
                     type=float,
                     dest='damping',
                     metavar='DAMPING',
                     default=0.95,
                     help='Damping factor')

    stat = parser.add_argument_group('print_stat')

    stat.add_argument('--merged_labels',
                      action='store_true',
                      dest='merged',
                      help='In case of tiers > 1 print labels merged \
                        according to hierarchy')

    render = parser.add_argument_group('render')

    render.add_argument('--draw_nums',
                        action='store_true',
                        help='Draw numerical labels')

    render.add_argument('--bcolor',
                        action='store_true',
                        help='Color according to computed bfactors')

    render.add_argument('--noclear',
                        dest='clear',
                        action='store_false',
                        help='Do not clear intermidiate files')

    render.add_argument('--width',
                        nargs='?', type=int, default=640,
                        help='Width of individual image')

    render.add_argument('--height',
                        nargs='?', type=int, default=480,
                        help='Height of individual image')

    render.add_argument('--moltype',
                        nargs='?', type=str, default="general",
                        choices=["general", "origami"],
                        help='Type of molecule to draw')

    export = parser.add_argument_group('cluster_to_trj')

    export.add_argument('-i', '--index',
                        metavar='INDEX',
                        type=int,
                        dest='index',
                        help='Index of cluster to be exported')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


def main_tasks():

    tasks = OD([
        ('load_pdb', load_pdb_coords),
        ('calc_rmsd', calc_rmsd_matrix),
        ('prepare_matrix', prepare_cluster_matrix),
        ('calc_median', calc_median),
        ('set_preference', set_preference),
        ('aff_cluster', aff_cluster),
        ('print_stat', print_stat)])

    return tasks


def misc_tasks():
    tasks = OD([
        ('cluster_to_trj', cluster_to_trj),
        ('render', render_b_factor)])
    return tasks


def wrapper_tasks():
    tasks = OD([
        ('cluster', dummy),
        ('all', dummy)])
    return tasks


def expand_tasks(tasks):
    """Replace the 'cluster' and 'all' shortcuts by the tasks they run."""
    expanded = []
    for t in tasks:
        if t == 'all':
            expanded += list(main_tasks()) + list(misc_tasks())
        elif t == 'cluster':
            expanded += list(main_tasks())
        else:
            expanded.append(t)
    return expanded


def tier_size(sfn, tier, from_previous):
    """(structures, atoms, existing datasets) of a tier already on disk."""
    source = tier - 1 if from_previous else tier
    try:
        with h5py.File(sfn, 'r') as f:
            g = f['tier%d' % source]
            natoms = g['struct'].shape[1]
            if from_previous:
                n = len(g['aff_centers'])
            else:
                n = g['struct'].shape[0]
            current = f.get('tier%d' % tier)
            existing = tuple(current) if current is not None else ()
    except (OSError, KeyError) as e:
        raise AffBioError(
            'Cannot read tier %d data from %s (%s); run the earlier tasks '
            'first.' % (source, sfn, e))
    return n, natoms, existing


def preflight(tasks, args, nprocs):
    """Check sizes, process count and disk space before any work.

    Returns a list of warnings; raises AffBioError on hard errors.
    """
    if not set(tasks) & set(MATRIX_TASKS):
        return []

    sfn, tier = args['Sfn'], args['tier']

    if 'load_pdb' in tasks and tier == 1:
        pdb_list = args['pdb_list'] or []
        if not pdb_list:
            raise AffBioError('No structures given; use -f FILE ...')
        n = len(pdb_list)
        topology = args['topology'] or pdb_list[0]
        if not os.path.exists(topology):
            raise AffBioError('No such file: %s' % topology)
        natoms = len(selection_indices(topology, args['selection'])[1])
        existing = ()
    else:
        n, natoms, existing = tier_size(sfn, tier, 'load_pdb' in tasks)

    check_decomposition(n, nprocs)
    n = effective_n(n, nprocs)
    overwrite = 'load_pdb' in tasks and tier == 1
    check_disk(disk_needs(sfn, n, tasks, existing, overwrite))

    warnings = []
    for stage in ('calc_rmsd', 'prepare_matrix'):
        if stage in tasks:
            w = memory_warning(stage, n, nprocs, natoms)
            if w:
                warnings.append(w)
    return warnings


def run_tasks(tasks, args):

    comm, NPROCS, rank = args['mpi']

    for t in tasks:
        run_task(t, args)
        comm.Barrier()


def run_task(task, args):

    comm, NPROCS, rank = args['mpi']

    # Init logging
    if rank == 0:
        t0 = init_logging(task, args['verbose'])
        pr = init_debug(args['debug'])

    tasks = main_tasks()
    tasks.update(misc_tasks())
    fn = tasks[task]
    fn(**args)

    if rank == 0:
        finish_logging(task, t0, args['verbose'])
        finish_debug(pr, args['debug'])

    comm.Barrier()


def run():
    try:
        mpi = init_mpi()
    except AffBioError as e:
        sys.exit('affbio: error: %s' % e)

    comm, NPROCS, rank = mpi

    try:
        check_parallel_io(NPROCS, h5py.get_config().mpi)
    except AffBioError as e:
        if rank == 0:
            print('affbio: error: %s' % e, file=sys.stderr)
        sys.exit(1)

    args = None
    exit_code = None

    if rank == 0:
        try:
            tasks = list(main_tasks()) + list(misc_tasks()) + \
                list(wrapper_tasks())
            args = get_args(tasks)
        except SystemExit as e:
            exit_code = e.code
    exit_code = comm.bcast(exit_code)

    if exit_code is not None:
        sys.exit(exit_code)

    error = None
    if rank == 0:
        if args['pdb_list']:
            args['pdb_list'] = expand_pdb_list(args['pdb_list'])
        args['task'] = expand_tasks(args['task'])
        try:
            for warning in preflight(args['task'], args, NPROCS):
                print('affbio: warning: %s' % warning)
        except AffBioError as e:
            error = str(e)
    error = comm.bcast(error)

    if error:
        if rank == 0:
            print('affbio: error: %s' % error, file=sys.stderr)
        sys.exit(1)

    args = comm.bcast(args)

    args['mpi'] = mpi

    try:
        run_tasks(args['task'], args)
    except AffBioError as e:
        where = ' (rank %d)' % rank if NPROCS > 1 else ''
        print('affbio: error%s: %s' % (where, e), file=sys.stderr)
        if NPROCS > 1:
            comm.Abort(1)
        sys.exit(1)
    except Exception:
        # Without this, the other processes would wait forever
        if NPROCS > 1:
            traceback.print_exc()
            comm.Abort(1)
        raise


if __name__ == "__main__":
    run()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_cluster.py -v`
Expected: all passed. Then run the whole suite: `.venv/bin/pytest -q`; expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add affbio/cli.py affbio/misc.py affbio/AffRender.py tests/helpers.py tests/conftest.py tests/test_cli.py tests/test_cluster.py
git commit -q -F - <<'EOF'
Port CLI to Python 3 with preflight checks and clean error reporting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 10: B-factors without GROMACS (`affbio/rmsf.py`, `affbio/misc.py`)

**Files:**
- Create: `affbio/rmsf.py`
- Modify: `affbio/misc.py` (everything after the license header)
- Test: `tests/test_rmsf.py`

**Interfaces:**
- Consumes: `AffBioError` (Task 2); `small_run` fixture, `init_mpi` (Tasks 3, 9).
- Produces:
  - `affbio.rmsf.BFACTOR = 8 * pi**2 / 3`; `cluster_bfactors(center_pdb: str, member_pdbs, out_pdb: str) -> ndarray` (B-factors; writes `out_pdb` without CONECT). Deviation from the spec: `copy_connects` is called by `render_b_factor`, as in 0.0.x, which keeps `rmsf.py` free of a circular import with `misc.py`.
  - `affbio.misc.copy_connects(src, dst)`, `model_lines(fname, number) -> list[str]`, `cluster_members(Sf, tier=1, merged=False) -> (labels, names)`, `cluster_to_trj(...)`, `render_b_factor(...)` (signatures as in 0.0.x).

- [ ] **Step 1: Write the failing tests** — `tests/test_rmsf.py`:

```python
import h5py
import MDAnalysis as mda
import numpy as np
import pytest

from affbio.checks import AffBioError
from affbio.misc import cluster_to_trj, copy_connects
from affbio.rmsf import BFACTOR, cluster_bfactors
from affbio.utils import init_mpi

ELEMENTS = ['N', 'C', 'C', 'O', 'S']


def write_pdb(path, coords, elements, conect=(), end='END'):
    lines = ['ATOM  %5d %-4s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00'
             '          %2s\n' % (k, el, k, x, y, z, el)
             for k, ((x, y, z), el) in enumerate(zip(coords, elements), 1)]
    lines += ['CONECT%5d%5d\n' % pair for pair in conect]
    if end:
        lines.append(end + '\n')
    path.write_text(''.join(lines))
    return str(path)


def reference_bfactors(center, members, masses):
    """Mass-weighted Kabsch fit onto the center, then RMSF (gmx rmsf)."""
    w = masses / masses.sum()
    ref = center - w @ center
    fitted = []
    for X in members:
        Xc = X - w @ X
        U, S, Vt = np.linalg.svd((Xc * w[:, None]).T @ ref)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        fitted.append(Xc @ R.T)
    fitted = np.array(fitted)
    rmsf = np.sqrt(((fitted - fitted.mean(0)) ** 2).sum(-1).mean(0))
    return BFACTOR * rmsf ** 2


def test_matches_independent_fit(tmp_path):
    rng = np.random.default_rng(0)
    n = 25
    els = [ELEMENTS[k % 5] for k in range(n)]
    base = np.cumsum(rng.normal(0, 1.5, (n, 3)), 0)
    spread = 0.4 + 0.05 * np.arange(n)[:, None]
    frames = []
    for k in range(12):
        rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rot *= np.sign(np.linalg.det(rot))
        X = (base + rng.normal(0, spread, (n, 3))) @ rot.T + \
            rng.normal(0, 5, 3)
        frames.append(write_pdb(tmp_path / ('m%d.pdb' % k), X, els))
    out = str(tmp_path / 'out.pdb')

    bfac = cluster_bfactors(frames[0], frames, out)

    coords = [mda.Universe(f).atoms.positions.astype(np.float64)
              for f in frames]
    masses = mda.Universe(frames[0]).atoms.masses
    assert len(set(np.round(masses, 2))) > 1   # the fit really is weighted
    expected = reference_bfactors(coords[0], coords, masses)
    np.testing.assert_allclose(bfac, expected, rtol=1e-3, atol=1e-3)
    written = mda.Universe(out)
    np.testing.assert_allclose(written.atoms.tempfactors, expected,
                               atol=0.01)
    np.testing.assert_allclose(written.atoms.positions, coords[0],
                               atol=1e-3)


def test_unknown_masses_fall_back_to_unweighted(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(0, 5, (10, 3))
    files = [write_pdb(tmp_path / ('x%d.pdb' % k),
                       X + rng.normal(0, 0.3, X.shape), ['X'] * 10)
             for k in range(4)]
    with pytest.warns(UserWarning, match='unweighted fit'):
        bfac = cluster_bfactors(files[0], files, str(tmp_path / 'out.pdb'))
    assert np.all(np.isfinite(bfac))


@pytest.mark.parametrize('end', ['ENDMDL', 'END', ''])
def test_copy_connects(tmp_path, end):
    X = np.zeros((3, 3)) + np.arange(3)[:, None]
    top = write_pdb(tmp_path / 'top.pdb', X, ['C'] * 3,
                    conect=[(1, 2), (2, 3)])
    dst = write_pdb(tmp_path / 'dst.pdb', X, ['C'] * 3, end=end)
    copy_connects(top, dst)
    lines = open(dst).read().splitlines()
    conect = [k for k, l in enumerate(lines) if l.startswith('CONECT')]
    assert len(conect) == 2
    if end:
        assert lines[conect[-1] + 1] == end
        assert lines[-1] == end
    else:
        assert lines[-1].startswith('CONECT')


def test_copy_connects_without_conect(tmp_path):
    X = np.zeros((3, 3))
    top = write_pdb(tmp_path / 'top.pdb', X, ['C'] * 3)
    dst = write_pdb(tmp_path / 'dst.pdb', X, ['C'] * 3)
    before = open(dst).read()
    copy_connects(top, dst)
    assert open(dst).read() == before


def test_cluster_to_trj(small_run, tmp_path):
    sfn = str(small_run / 'm.hdf5')
    with h5py.File(sfn, 'r') as f:
        labels = f['tier1/aff_labels'][:]
    out = str(tmp_path / 'c0.pdb')
    cluster_to_trj(sfn, index=0, output=out, mpi=init_mpi())
    u = mda.Universe(out)
    assert u.trajectory.n_frames == np.sum(labels == 0)
    assert u.atoms.n_atoms == 214


def test_cluster_to_trj_needs_index(small_run):
    with pytest.raises(AffBioError, match='--index'):
        cluster_to_trj(str(small_run / 'm.hdf5'), output='x.pdb',
                       mpi=init_mpi())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rmsf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'affbio.rmsf'`

- [ ] **Step 3: Write `affbio/rmsf.py`** (license header first):

```python
"""Per-atom B-factors of a cluster, computed as `gmx rmsf -fit -oq` does."""

import warnings

import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

# gmx rmsf -oq writes B = 8 pi^2 / 3 * RMSF^2
BFACTOR = 8.0 * np.pi ** 2 / 3.0


def cluster_bfactors(center_pdb, member_pdbs, out_pdb):
    """Write the cluster center to out_pdb with RMSF-based B-factors.

    Every member is fitted onto the center (mass-weighted, as
    gmx rmsf -fit does); RMSF is the fluctuation about the average fitted
    position. Returns the B-factors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = mda.Universe(center_pdb)
        mobile = mda.Universe(center_pdb, list(member_pdbs))

    weights = 'mass'
    if np.any(ref.atoms.masses <= 0):
        warnings.warn('Unknown atomic masses in %s; using an unweighted fit '
                      'for B-factors.' % center_pdb)
        weights = None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        align.AlignTraj(mobile, ref, select='all', weights=weights,
                        in_memory=True).run()
        rmsf = rms.RMSF(mobile.atoms).run().results.rmsf

        bfac = BFACTOR * rmsf ** 2
        ref.atoms.tempfactors = bfac
        ref.atoms.write(out_pdb, bonds=None)

    return bfac
```

- [ ] **Step 4: Replace `affbio/misc.py` after the license header (keep lines 1-21) with:**

```python

# General modules
import os

# NumPy for arrays
import numpy as np

# H5PY for storage
import h5py

from .AffRender import AffRender
from .checks import AffBioError
from .rmsf import cluster_bfactors

# Records kept when a frame is wrapped into a MODEL block
COORD_RECORDS = ('ATOM', 'HETATM', 'ANISOU', 'TER', 'CONECT')


def cluster_members(Sf, tier=1, merged=False):
    """Cluster label of every structure and the structure file names."""
    G = Sf['tier%d' % tier]
    if tier > 1 and merged:
        return G['aff_labels_merged'][:], Sf['tier1']['labels'].asstr()[:]
    return G['aff_labels'][:], G['labels'].asstr()[:]


def model_lines(fname, number):
    """Records of one frame as a MODEL ... ENDMDL block, without END."""
    with open(fname, 'r') as f:
        lines = [l for l in f if l[:6].strip() != 'END']
    if any(l.startswith('MODEL') for l in lines):
        return lines
    body = [l for l in lines if l[:6].strip() in COORD_RECORDS]
    return ['MODEL     %4d\n' % number] + body + ['ENDMDL\n']


def cluster_to_trj(
        Sfn,
        tier=1,
        index=None,
        merged=False,
        output=None,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    if index is None or output is None:
        raise AffBioError(
            'cluster_to_trj needs a cluster --index and an -o output file.')

    with h5py.File(Sfn, 'r', driver='sec2') as Sf:
        I, L = cluster_members(Sf, tier, merged)
        top = Sf['tier1']['labels'].attrs['topology']

    frames = L[I == index]
    if len(frames) == 0:
        raise AffBioError('There is no cluster %d; clusters are numbered '
                          '0 to %d.' % (index, I.max()))

    with open(output, 'w') as fout:
        fout.writelines(model_lines(frames[0], 1))

    copy_connects(top, output)

    with open(output, 'a') as fout:
        for k, frame in enumerate(frames[1:], 2):
            fout.writelines(model_lines(frame, k))
        fout.write('END\n')


def render_b_factor(
        Sfn,
        tier=1,
        merged=False,
        mpi=None,
        verbose=False,
        debug=False,
        *args, **kwargs):

    comm, NPROCS, rank = mpi

    if rank != 0:
        return

    with h5py.File(Sfn, 'r', driver='sec2') as Sf:
        top = Sf['tier1']['labels'].attrs['topology']
        G = Sf['tier%d' % tier]
        C = G['aff_centers'][:]
        LC = G['labels'].asstr()[:]
        I, L = cluster_members(Sf, tier, merged)

    NI = len(I)

    cs = np.bincount(I)
    pcs = cs * 100.0 / NI

    centers = []

    for i in range(len(C)):
        TMbfac = 'cluster_%d_bfac.pdb' % i
        cluster_bfactors(LC[C[i]], L[I == i], TMbfac)
        copy_connects(top, TMbfac)
        centers.append(TMbfac)

    kwargs['pdb_list'] = centers
    kwargs['nums'] = pcs

    AffRender(**kwargs)

    for c in centers:
        os.remove(c)


def copy_connects(src, dst):
    """Copy the CONECT records of src into dst before ENDMDL or END."""
    with open(src, 'r') as fin:
        con = [l for l in fin if l.startswith('CONECT')]
    if not con:
        return

    with open(dst, 'r') as fout:
        lines = fout.readlines()

    records = [l[:6].strip() for l in lines]
    for marker in ('ENDMDL', 'END'):
        if marker in records:
            pos = len(records) - 1 - records[::-1].index(marker)
            break
    else:
        pos = len(lines)

    lines[pos:pos] = con

    with open(dst, 'w') as fout:
        fout.write(''.join(lines))
```

Note: `render_b_factor` now uses the given tier's labels for every cluster. 0.0.x called `cluster_to_trj` without `tier`, so tier > 1 renders used tier-1 labels.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rmsf.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add affbio/rmsf.py affbio/misc.py tests/test_rmsf.py
git commit -q -F - <<'EOF'
Compute cluster B-factors with MDAnalysis instead of gmx rmsf

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 11: Rendering with Pillow and pymol2 (`affbio/AffRender.py`)

**Files:**
- Modify: `affbio/AffRender.py` (everything after the license header)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `AffBioError` (Task 2); `small_run`, `run_affbio` (Task 9); `render_b_factor` (Task 10).
- Produces: `AffRender(pdb_list=None, output=None, nums=list(), draw_nums=False, guess_nums=False, bcolor=False, lowt=10, width=640, height=480, moltype="general", clear=False, *args, **kwargs)` (unchanged); `AffRender.init_pymol()` returns a started `pymol2.PyMOL` session (has `.cmd`) or raises `AffBioError`; `AffRender.tile(images, out, direction='h'|'v')`; `AffRender.gen_label(basename, num, width, height) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_render.py`:

```python
import shutil
import sys

import h5py
import numpy as np
import pytest
from PIL import Image

from affbio.AffRender import AffRender
from affbio.checks import AffBioError
from helpers import run_affbio


def test_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = AffRender.gen_label('c', 42, width=640, height=480)
    im = Image.open(name)
    assert im.size == (128, 480) and im.mode == 'RGBA'
    alpha = np.asarray(im)[..., 3]
    assert alpha[:, :40].max() == 0      # right-aligned text
    assert alpha[:150].max() == 0        # vertically centered
    assert alpha[:, 64:].max() == 255


def test_tile(tmp_path):
    a, b = tmp_path / 'a.png', tmp_path / 'b.png'
    Image.new('RGBA', (10, 20), (255, 0, 0, 255)).save(a)
    Image.new('RGBA', (30, 5), (0, 0, 255, 128)).save(b)
    AffRender.tile([str(a), str(b)], str(tmp_path / 'h.png'), 'h')
    AffRender.tile([str(a), str(b)], str(tmp_path / 'v.png'), 'v')
    h = Image.open(tmp_path / 'h.png')
    v = Image.open(tmp_path / 'v.png')
    assert h.size == (40, 20) and v.size == (30, 25)
    assert h.getpixel((35, 15))[3] == 0   # empty area stays transparent
    assert h.getpixel((15, 2)) == (0, 0, 255, 128)


def test_missing_pymol(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pymol2', None)
    with pytest.raises(AffBioError, match=r"affbio\[render\]"):
        AffRender.init_pymol()


def test_render_end_to_end(small_run, tmp_path):
    pytest.importorskip('pymol2')
    shutil.copy(small_run / 'm.hdf5', tmp_path / 'm.hdf5')
    r = run_affbio(['-m', 'm.hdf5', '-t', 'render', '--draw_nums',
                    '--bcolor', '--width', '160', '--height', '120',
                    '-o', 'clusters.png'], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / 'm.hdf5', 'r') as f:
        k = len(f['tier1/aff_centers'])
    for name in ('clusters.png', 'clusters_color.png'):
        im = Image.open(tmp_path / name)
        # label (20% of width) + 3 poses per row, one row per cluster
        assert im.size == (32 + 3 * 160, 120 * k)
    assert not list(tmp_path.glob('cluster_*'))   # intermediates removed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: `test_missing_pymol` FAILS (the old `init_pymol` imports `pymol`, not `pymol2`, and never raises `AffBioError`). `test_label`/`test_tile` may pass or fail depending on whether ImageMagick is installed; either is fine at this step.

- [ ] **Step 3: Replace `affbio/AffRender.py` after the license header (keep lines 1-21) with:**

```python

# -*- coding: UTF-8 -*-

import os
import re

from PIL import Image, ImageDraw, ImageFont

from .checks import AffBioError


class AffRender(object):

    def __init__(
            self,
            pdb_list=None,
            output=None,
            nums=list(),
            draw_nums=False,
            guess_nums=False,
            bcolor=False,
            lowt=10,
            width=640, height=480,
            moltype="general",
            clear=False,
            *args, **kwargs):

        self.models = pdb_list

        if output is None:
            self.out = 'out.png'
        else:
            self.out = output

        self.nums = nums
        self.draw_nums = draw_nums
        self.guess_nums = guess_nums

        if self.draw_nums and not self.guess_nums:
            if len(self.nums) != len(self.models):
                raise(Exception("Numbers of models and nums are different"))

        self.bcolor = False

        self.width = width
        self.height = height

        self.moltype = moltype

        self.clear = clear

        self.pymol = self.init_pymol()

        self.process_models()

        if bcolor is True:
            self.bcolor = bcolor
            filename, file_extension = os.path.splitext(self.out)
            self.out = filename + '_color.png'
            self.process_models()

        self.pymol.stop()

    @staticmethod
    def init_pymol():
        try:
            import pymol2
        except ImportError:
            raise AffBioError(
                "The render task needs PyMOL: pip install 'affbio[render]'")
        session = pymol2.PyMOL()
        session.start()
        return session

    def setup_scene(self):

        self.pymol.cmd.set("ambient", '0.00000')
        self.pymol.cmd.set("antialias", 4)
        self.pymol.cmd.set("light_count", 1)
        self.pymol.cmd.set("ray_shadow", 'off')
        self.pymol.cmd.set("reflect_power", '0.10000')
        self.pymol.cmd.set("spec_power", '0.00000')
        self.pymol.cmd.set("specular", '0.00000')
        self.pymol.cmd.set("orthoscopic", 1)

        # self.pymol.cmd.bg_color("white")
        self.pymol.cmd.set("opaque_background", 0)

    @staticmethod
    def tile(images, out, direction="h"):
        """Put images edge to edge on a transparent canvas."""
        opened = [Image.open(i).convert('RGBA') for i in images]

        if direction == 'h':
            size = (sum(i.width for i in opened),
                    max(i.height for i in opened))
        else:
            size = (max(i.width for i in opened),
                    sum(i.height for i in opened))

        canvas = Image.new('RGBA', size, (0, 0, 0, 0))
        offset = 0
        for im in opened:
            if direction == 'h':
                canvas.paste(im, (offset, 0))
                offset += im.width
            else:
                canvas.paste(im, (0, offset))
                offset += im.height

        canvas.save(out)

    # DNA origami specific part ###

    @staticmethod
    def is_backbone(i, j):
        return True if abs(i - j) == 1 else False

    @staticmethod
    def is_crossover(i, j):
        return True if abs(i - j) > 1 else False

    def draw_backbone(self):
        # Get total number of atoms
        # N = self.pymol.cmd.count_atoms()

        # Show backbone with sticks
        # for i in range(1, N):
        #   self.pymol.cmd.select('bck', 'resi %d+%d' % (i, i + 1))
        #   self.pymol.cmd.show('sticks', 'bck')

        # Set sticks width
        self.pymol.cmd.show("sticks")
        self.pymol.cmd.set("stick_radius", 1.5)

        # Find single stranded regions
        space = {"single": []}
        self.pymol.cmd.iterate("resn S*", "single.append(resi)", space=space)
        single = [int(s) for s in space["single"]]

        # Set stich width for single stranded region
        # Only change stick radius for contigious regions in backbone
        # and do not touch crossovers
        if len(single) >= 2:
            for s in range(len(single) - 1):
                i = single[s]
                j = single[s + 1]
                if self.is_backbone(i, j):
                    self.pymol.cmd.set_bond(
                        "stick_radius", 0.5,
                        "i. %d" % i, "i. %d" % j)

        self.pymol.cmd.color('black')
        # Color bacbone according to B-factors
        if self.bcolor is True:
            self.pymol.cmd.spectrum("b")

    def draw_crossovers(self, fname):
        """ Read all CONECT from model file and draw all non-backbone
        bonds with dashes"""

        def get_ind(line):
            line = line.strip()
            line = re.sub(r'\s+', ';', line)
            i, j = map(int, line.split(";")[1:3])
            return i, j

        def is_bond(line):
            return True if re.match('CONECT', line) else False

        with open(fname, 'r') as f:
            bonds = f.readlines()

        crossover = []
        backbone = []

        for b in bonds:
            if is_bond(b):
                i, j = get_ind(b)
                if self.is_crossover(i, j):
                    crossover.append((i, j))
                elif self.is_backbone(i, j):
                    backbone.append((i, j))

        for b in crossover:
            i, j = b
            self.pymol.cmd.unbond("i. %d" % i, "i. %d" % j)
            self.pymol.cmd.distance("i. %d" % i, "i. %d" % j)
        self.pymol.cmd.hide("labels")

        self.pymol.cmd.set("dash_color", "gray80")
        self.pymol.cmd.set("dash_gap", 0)
        self.pymol.cmd.set("dash_length", 4)
        self.pymol.cmd.set("dash_radius", 1.0)

    def draw_nucleic_acid(self):
        self.pymol.cmd.hide("everything")
        self.pymol.cmd.show("lines")
        # self.pymol.cmd.show("cartoon")
        # self.pymol.cmd.set("cartoon_nucleic_acid_mode", 1)
        # self.pymol.cmd.set("cartoon_tube_radius", 0.1)
        # self.pymol.cmd.set("cartoon_ring_finder", 2)
        # self.pymol.cmd.set("cartoon_ring_mode", 2)
        # self.pymol.cmd.set("cartoon_flat_sheets", 0)

        # Color bacbone according to B-factors
        self.pymol.cmd.spectrum("b")

    @classmethod
    def gen_label(cls, basename="gg", num=100, width=640, height=480):
        """Transparent image with "N%" right-aligned, as wide as 20% of
        a pose image."""

        lwidth = int(0.2 * width)  # 20% - empirically

        name = cls.gen_name(basename, 0)

        image = Image.new('RGBA', (lwidth, height), (0, 0, 0, 0))
        font = ImageFont.load_default(size=max(1, lwidth // 3))
        ImageDraw.Draw(image).text(
            (lwidth - 1, height // 2), "%d%%" % num,
            font=font, fill=(0, 0, 0, 255), anchor='rm')
        image.save(name)

        return name

    def ray(self, name, width=640, height=480):
        # pymol.cmd.zoom("all", 100)
        self.pymol.cmd.zoom("all", 20)
        self.pymol.cmd.ray(width, height)
        self.pymol.cmd.save(name)

    @staticmethod
    def gen_name(basename, number):
        return "%s_%d.png" % (basename, number)

    def ray_poses(self, basename="gg", width=640, height=480):

        images = []

        name = self.gen_name(basename, 1)
        self.pymol.cmd.orient()
        self.ray(name, width, height)
        images.append(name)

        name = self.gen_name(basename, 2)
        self.pymol.cmd.rotate("x", 90)
        self.ray(name, width, height)
        images.append(name)

        name = self.gen_name(basename, 3)
        self.pymol.cmd.rotate("y", 90)
        self.ray(name, width, height)
        images.append(name)

        return images

    def process_model(self, index):

        model = self.models[index]

        # basename = model.replace('.pdb', '.png')
        basename = model[:-4] + '_tmp'

        if self.bcolor:
            basename += '_color'

        self.pymol.cmd.reinitialize()
        # Desired pymol commands here to produce and save figures
        self.setup_scene()

        self.pymol.cmd.load(model)

        if self.moltype == "general":
            self.draw_nucleic_acid()
        elif self.moltype == "origami":
            self.draw_crossovers(model)
            self.draw_backbone()

        images = list()

        if self.draw_nums:
            if self.guess_nums:
                num = self.get_num(model)
            else:
                num = self.nums[index]

            label = self.gen_label(
                basename, num, width=self.width, height=self.height)
            images.append(label)

        poses = self.ray_poses(basename, width=self.width, height=self.height)

        images.extend(poses)

        name = basename + '.png'
        self.tile(images=images, out=name)

        if self.clear:
            for image in images:
                os.remove(image)

        return name

    @staticmethod
    def get_num(model):
        """ Get representativeness of current model """
        try:
            # try to parse filename like frameXXXX_aff_YY.pdb
            num = re.search(r'aff_(\d+)', model).groups()[0]
        except:
            # if no, set default num
            raise(Exception("Unable to get num from filename"))
        return int(num)

    def process_models(self):

        images = list()

        for i in range(len(self.models)):

            # now you can call it directly with basename
            image = self.process_model(i)
            images.append(image)

        self.tile(images, self.out, direction='v')

        if self.clear:
            for image in images:
                os.remove(image)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: all passed (the end-to-end test needs the `render` extra, installed in Task 1). Open `clusters_color.png` once and check that it shows 3 poses per cluster with a percentage label: `xdg-open` it, or read it with the Read tool.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add affbio/AffRender.py tests/test_render.py
git commit -q -F - <<'EOF'
Render with Pillow and pymol2 instead of ImageMagick and pymol_argv

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 12: MPI tests and local MPI verification

**Files:**
- Test: `tests/test_mpi.py`

**Interfaces:**
- Consumes: `run_affbio` (Task 9), `adk_frames` (Task 6), the full CLI.
- Produces: tests that run only where mpi4py, MPI-enabled h5py and `mpirun` exist (skipped otherwise).

- [ ] **Step 1: Write the tests** — `tests/test_mpi.py`:

```python
import os
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from helpers import run_affbio

pytest.importorskip('mpi4py')
if not h5py.get_config().mpi:
    pytest.skip('h5py is built without MPI', allow_module_level=True)
MPIRUN = shutil.which('mpirun') or shutil.which('mpiexec')
if MPIRUN is None:
    pytest.skip('mpirun not found', allow_module_level=True)

ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def launcher(nprocs):
    cmd = [MPIRUN, '-n', str(nprocs)]
    version = subprocess.run([MPIRUN, '--version'], capture_output=True,
                             text=True).stdout
    if 'Open MPI' in version or 'OpenRTE' in version:
        cmd.append('--oversubscribe')
    return cmd


def affbio(args, cwd, nprocs=1):
    return run_affbio(args, cwd=cwd, env=ENV,
                      launcher=launcher(nprocs) if nprocs > 1 else ())


def tier1(path, name):
    with h5py.File(path / 'm.hdf5', 'r') as f:
        return f['tier1'][name][:]


def rmsd_args(frames, noalign):
    return (['-m', 'm.hdf5', '-t', 'load_pdb', 'calc_rmsd', '-f'] + frames
            + (['--noalign'] if noalign else []))


@pytest.fixture(scope='module')
def serial_rmsd(tmp_path_factory, adk_frames):
    out = {}
    for noalign in (False, True):
        d = tmp_path_factory.mktemp('serial_rmsd')
        r = affbio(rmsd_args(adk_frames[:240], noalign), d)
        assert r.returncode == 0, r.stderr
        out[noalign] = tier1(d, 'rmsd')
    return out


@pytest.mark.parametrize('nprocs', [2, 3, 4])
@pytest.mark.parametrize('noalign', [False, True])
def test_rmsd_matrix_matches_serial(serial_rmsd, adk_frames, tmp_path,
                                    nprocs, noalign):
    # 240 is a multiple of 4 * nprocs for 2, 3 and 4: nothing is dropped
    r = affbio(rmsd_args(adk_frames[:240], noalign), tmp_path, nprocs)
    assert r.returncode == 0, r.stdout + r.stderr
    R = tier1(tmp_path, 'rmsd')
    il = np.tril_indices(240, -1)
    np.testing.assert_allclose(R[il], serial_rmsd[noalign][il],
                               rtol=1e-6, atol=1e-5)
    assert np.all(np.triu(R) == 0)


def test_cluster_matches_serial(adk_frames, tmp_path):
    frames = adk_frames[:1040]  # multiple of 4 * 4
    serial, parallel = tmp_path / 's', tmp_path / 'p'
    serial.mkdir()
    parallel.mkdir()
    args = ['-m', 'm.hdf5', '-t', 'cluster', '-f'] + frames
    r = affbio(args, serial)
    assert r.returncode == 0, r.stderr
    r = affbio(args, parallel, 4)
    assert r.returncode == 0, r.stdout + r.stderr
    assert np.array_equal(tier1(parallel, 'aff_centers'),
                          tier1(serial, 'aff_centers'))
    assert np.array_equal(tier1(parallel, 'aff_labels'),
                          tier1(serial, 'aff_labels'))


def test_trimming_is_logged(adk_frames, tmp_path):
    r = affbio(['-m', 'm.hdf5', '-t', 'load_pdb', '-f'] + adk_frames,
               tmp_path, 3)
    assert r.returncode == 0, r.stderr
    assert tier1(tmp_path, 'struct').shape[0] == 1044
    assert 'Using 1044 of 1047 structures' in r.stdout
    for f in adk_frames[1044:]:
        assert f in r.stdout


def test_other_process_count_is_rejected(adk_frames, tmp_path):
    r = affbio(rmsd_args(adk_frames[:240], False), tmp_path, 4)
    assert r.returncode == 0, r.stderr
    r = affbio(['-m', 'm.hdf5', '-t', 'prepare_matrix'], tmp_path, 3)
    assert r.returncode != 0
    assert 'prepared with 4 processes' in r.stderr
```

- [ ] **Step 2: Confirm the tests skip in the default environment**

Run: `.venv/bin/pytest tests/test_mpi.py -v`
Expected: `SKIPPED` (no mpi4py in `.venv`).

- [ ] **Step 3: Create a throwaway MPI environment (developer machine only, not user-facing)**

This machine has no system MPI, so use conda-forge's MPI-enabled h5py for local verification:

```bash
SCRATCH=/tmp/claude-11130/-home-arthur-work-affbio/b1077477-25d6-4aba-be6a-db48a74b64d8/scratchpad
/home/arthur/miniconda3/bin/mamba create -y -q -p $SCRATCH/mpienv --override-channels -c conda-forge \
    python=3.13 "h5py=*=mpi_openmpi*" mpi4py openmpi mdanalysis bottleneck natsort psutil pillow pytest
$SCRATCH/mpienv/bin/python -m pip install --no-deps -e .
$SCRATCH/mpienv/bin/python -c "import h5py, mpi4py; assert h5py.get_config().mpi; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Run the MPI tests**

Run: `PATH=$SCRATCH/mpienv/bin:$PATH $SCRATCH/mpienv/bin/pytest tests/test_mpi.py -v`
Expected: all passed.

If `test_cluster_matches_serial` fails, do not relax it yet. First compare the inputs that feed AP: diff `tier1/rmsd` and `tier1/cluster` attrs `median`/`preference` between the serial and parallel files. Known source of difference: `prepare_cluster_matrix` draws its noise per block, so serial and MPI noise differ at the 1e-7 relative level. Report the measured differences and the label disagreement count to the user before changing the assertion.

If an off-diagonal block write fails with an h5py contiguity error, check that `calc_chunk` in `prepare.py` returns `np.ascontiguousarray(...)` (Task 8).

- [ ] **Step 5: Run the full suite in the MPI environment too**

Run: `PATH=$SCRATCH/mpienv/bin:$PATH $SCRATCH/mpienv/bin/pytest -q`
Expected: all passed; `test_render_end_to_end` skipped (no PyMOL in that env).

- [ ] **Step 6: Commit**

```bash
git add tests/test_mpi.py
git commit -q -F - <<'EOF'
Test MPI runs against serial for 2, 3 and 4 processes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 13: CI workflow

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `pyproject.toml` extras (Task 1), the test suite.
- Produces: `pip` job (Python 3.10-3.14) and `mpi` job (README's parallel-setup commands).

- [ ] **Step 1: Write `.github/workflows/tests.yml`**

```yaml
name: tests

on: [push, pull_request]

permissions:
  contents: read

jobs:
  pip:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          extras=render,test
          # Official PyMOL wheels stop at Python 3.13
          if [ "${{ matrix.python-version }}" = "3.14" ]; then extras=test; fi
          python -m pip install ".[$extras]"
      - name: Test
        run: python -m pytest -v

  mpi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
      - name: Install MPI and parallel HDF5
        run: |
          sudo apt-get update
          sudo apt-get install -y openmpi-bin libopenmpi-dev libhdf5-openmpi-dev
      - name: Install mpi4py, parallel h5py and affbio (as in the README)
        run: |
          python -m pip install mpi4py
          CC=mpicc HDF5_MPI=ON HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
            python -m pip install --no-binary=h5py h5py
          python -m pip install ".[mpi,test]"
          python -c "import h5py; assert h5py.get_config().mpi"
      - name: Test
        run: python -m pytest -v tests/test_mpi.py
```

- [ ] **Step 2: Lint the workflow**

Run: `uvx --from actionlint-py actionlint .github/workflows/tests.yml`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/tests.yml
git commit -q -F - <<'EOF'
Add CI: pip matrix on Python 3.10-3.14 and an MPI job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

- [ ] **Step 4: Ask before running CI**

CI runs only on GitHub. Ask the user whether to push `py3-revival`; do not push without a yes. After pushing, if the `mpi` job fails while building h5py because `mpi4py` is missing at build time, change that step to:

```bash
python -m pip install mpi4py Cython numpy pkgconfig setuptools
CC=mpicc HDF5_MPI=ON HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
  python -m pip install --no-build-isolation --no-binary=h5py h5py
```

and update the README's parallel-setup commands (Task 14) to match.

---

### Task 14: README

**Files:**
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: final CLI behavior and install commands (Tasks 1-13).

- [ ] **Step 1: Replace `README.md` with:**

~~~~markdown
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
sudo apt install openmpi-bin libopenmpi-dev libhdf5-openmpi-dev
pip install mpi4py
CC=mpicc HDF5_MPI=ON HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
    pip install --no-binary=h5py h5py
pip install "affbio[mpi]"
```

Fedora:

```
sudo dnf install openmpi-devel hdf5-openmpi-devel
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
import MDAnalysis as mda

u = mda.Universe("topol.tpr", "traj.xtc")
protein = u.select_atoms("protein")
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

Under MPI, AffBio uses a multiple of 4 x the number of processes of the
structures and lists the files it leaves out. Each process holds blocks of
N / processes structures, which must not exceed 32,767 (an HDF5 limit), so
200,000 structures need at least 7 processes. The two N x N float32
matrices need 8 x N² bytes of disk next to the HDF5 file, and clustering
needs as much again in the working directory. AffBio checks all of this,
and warns about memory, before it starts.

## Changes since 0.0.x

- Python 3 only; installs with pip, no compiler needed.
- pyRMSD is replaced by built-in NumPy RMSD (same values, faster).
- ProDy is replaced by MDAnalysis, so `--selection` uses MDAnalysis syntax:
  `chain A` becomes `chainID A` and `within 5 of X` becomes `around 5 X`;
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
~~~~

- [ ] **Step 2: Check the README examples against the code**

Run: `.venv/bin/affbio --help | grep -E 'selection|noalign|merged_labels|draw_nums|bcolor'`
Expected: every option the README mentions is listed.

Run: `cd /tmp && /home/arthur/work/affbio/.venv/bin/python -c "import MDAnalysis as mda; u = mda.Universe('/home/arthur/work/affbio/tests/data/adk_ca.pdb'); print(u.select_atoms('chainID A').n_atoms, u.select_atoms('resnum 1 to 10').n_atoms)"`
Expected: two numbers, no exception (the README's translations are valid MDAnalysis syntax).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -q -F - <<'EOF'
Rewrite README for the Python 3 / pip release

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013GKrXY73C6QGMYx2VUrATL
EOF
```

---

### Task 15: Release build check (no upload)

**Files:** none changed (fixes found here go into the task that owns the code).

**Interfaces:**
- Consumes: everything.
- Produces: verified `dist/affbio-0.1.0.tar.gz` and `dist/affbio-0.1.0-py3-none-any.whl` (git-ignored).

- [ ] **Step 1: Build**

Run: `rm -rf dist && uv build`
Expected: `dist/affbio-0.1.0.tar.gz` and `dist/affbio-0.1.0-py3-none-any.whl`.

- [ ] **Step 2: Inspect the contents**

Run: `unzip -l dist/affbio-0.1.0-py3-none-any.whl | awk '{print $4}' | grep -v '^$'`
Expected: `affbio/__init__.py`, `AffRender.py`, `aff_cluster.py`, `checks.py`, `cli.py`, `median.py`, `misc.py`, `mpi.py`, `prepare.py`, `rmsd.py`, `rmsf.py`, `structures.py`, `utils.py`, and the `affbio-0.1.0.dist-info/` files (including `licenses/LICENSE.txt`); no `lvc.pyx`, no `.so`.

Run: `tar tzf dist/affbio-0.1.0.tar.gz | grep -E '/(app|old|supplement|docs)/' ; echo "exit=$?"`
Expected: `exit=1` (no matches).

Run: `uvx twine check dist/*`
Expected: `PASSED` for both files.

- [ ] **Step 3: Install the wheel into clean Python 3.10 and 3.14 environments and run the tests**

```bash
SCRATCH=/tmp/claude-11130/-home-arthur-work-affbio/b1077477-25d6-4aba-be6a-db48a74b64d8/scratchpad
for py in 3.10 3.14; do
  uv venv -q -p $py $SCRATCH/wheel-$py
  uv pip install -q -p $SCRATCH/wheel-$py dist/affbio-0.1.0-py3-none-any.whl pytest
  (cd /tmp && $SCRATCH/wheel-$py/bin/pytest /home/arthur/work/affbio/tests -q)
done
```

Expected: all passed for both; render end-to-end and MPI tests skipped (no PyMOL / mpi4py). Running from `/tmp` with the `pytest` executable makes the tests import the installed wheel, not the checkout.

- [ ] **Step 4: Report to the user**

Summarize: test results per environment, MPI results from Task 12, CI status (if pushed), wheel/sdist contents. Ask whether to upload 0.1.0 to PyPI (`uvx twine upload dist/*` with their token). Do not upload without an explicit yes.
