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
