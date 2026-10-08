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
