import os
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
        # every pose shows the molecule as a trace (C-alpha atoms have no
        # bonds); the poses share one scale, and the largest view fills
        # most of the image
        alpha = np.asarray(im)[..., 3]
        for row in range(k):
            fills = []
            for pose in range(3):
                x0 = 32 + pose * 160
                tile = alpha[row * 120:(row + 1) * 120, x0:x0 + 160] > 0
                assert tile.mean() > 0.02, (name, row, pose, tile.mean())
                cols = np.nonzero(tile.any(axis=0))[0]
                rows = np.nonzero(tile.any(axis=1))[0]
                fills.append(max((cols.max() - cols.min() + 1) / 160,
                                 (rows.max() - rows.min() + 1) / 120))
            assert max(fills) > 0.55, (name, row, fills)
    assert not list(tmp_path.glob('cluster_*'))   # intermediates removed


def test_ray_tracing_uses_all_available_cores():
    pytest.importorskip('pymol2')
    renderer = AffRender.__new__(AffRender)   # skip rendering in __init__
    renderer.pymol = AffRender.init_pymol()
    try:
        renderer.setup_scene()
        threads = int(float(renderer.pymol.cmd.get('max_threads')))
    finally:
        renderer.pymol.stop()
    assert threads == len(os.sched_getaffinity(0))
