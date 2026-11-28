"""Every notebook in a study folder checks that the active study is that folder's.

model_config resolves the study from $PERMEATION_STUDY before it looks in the working directory, so
a notebook launched with that variable inherited, or with its kernel started in another folder,
would compute another alloy's results and write them into this folder's results directories.
assert_study_is_local() turns that into an error before any physics runs, but only if it is called
before the first import of the study's configuration.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(nb for cfg in ROOT.glob('*/study_config.py') for nb in cfg.parent.glob('*.ipynb'))


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_asserts_its_study_before_loading_one(path):
    code = '\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                     if c['cell_type'] == 'code')
    guard = code.find('assert_study_is_local()')
    first_config = min((i for i in (code.find('model_config import'), code.find('import model_config'))
                        if i >= 0), default=len(code))
    assert guard >= 0, f'{path.name} never calls assert_study_is_local()'
    assert guard < first_config, f'{path.name} loads a study config before calling assert_study_is_local()'
