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
