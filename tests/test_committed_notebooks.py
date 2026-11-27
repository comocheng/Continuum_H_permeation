"""The notebooks .gitattributes marks filter=strip-outputs are staged without their outputs.

The clean filter (tools/strip_notebook_outputs.py) runs only in a clone that has enabled it
(``git config --local include.path ../.gitconfig``). A clone that has not would commit the
outputs, 4-25 MB per notebook, without warning. This test reads what is staged for each of those
notebooks, so it fails in exactly that case, before the commit is pushed.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from strip_notebook_outputs import strip  # noqa: E402


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def filtered_notebooks():
    try:
        files = git('ls-files', '-z', '--', '*.ipynb').split('\0')
    except (OSError, subprocess.CalledProcessError):
        return []                                    # not a git checkout (e.g. an unpacked archive)
    attrs = git('check-attr', 'filter', '--', *filter(None, files)).splitlines()
    return [line.split(': ')[0] for line in attrs if line.endswith(': strip-outputs')]


NOTEBOOKS = filtered_notebooks()


@pytest.mark.skipif(not NOTEBOOKS, reason='not a git checkout')
@pytest.mark.parametrize('path', NOTEBOOKS)
def test_staged_notebook_has_no_outputs(path):
    nb = json.loads(git('show', f':{path}'))         # the index copy: what the next commit holds
    dirty = [i for i, c in enumerate(nb['cells'])
             if c.get('outputs') or c.get('execution_count') is not None]
    assert not dirty, (f'{path} is staged with outputs in cells {dirty}. Enable the clean filter '
                       f'(git config --local include.path ../.gitconfig) and re-add it.')


def test_strip_clears_outputs_and_timing_and_keeps_the_rest():
    nb = {'cells': [{'cell_type': 'code', 'execution_count': 3, 'source': ['x = 1'],
                     'outputs': [{'output_type': 'stream', 'text': ['1']}],
                     'metadata': {'execution': {'iopub.status.idle': 't'}, 'tags': ['keep']}},
                    {'cell_type': 'markdown', 'source': ['# t'], 'metadata': {}}],
          'metadata': {'kernelspec': {'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out = strip(json.loads(json.dumps(nb)))
    code, md = out['cells']
    assert code['outputs'] == [] and code['execution_count'] is None
    assert code['metadata'] == {'tags': ['keep']} and code['source'] == ['x = 1']
    assert md == nb['cells'][1] and out['metadata'] == nb['metadata']
    assert strip(json.loads(json.dumps(out))) == out


def test_filter_passes_non_notebooks_through():
    script = ROOT / 'tools' / 'strip_notebook_outputs.py'
    for raw in (b'not json at all', b'{"no": "cells"}', b'{"cells": [}'):
        r = subprocess.run([sys.executable, str(script)], input=raw, capture_output=True, check=True)
        assert r.stdout == raw
