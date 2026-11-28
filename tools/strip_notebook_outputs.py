#!/usr/bin/env python
"""Git clean filter: a notebook on stdin, the same notebook without its outputs on stdout.

The notebooks that .gitattributes marks ``filter=strip-outputs`` are committed without their
outputs; their working copies keep them. Git runs this on every ``git add`` of such a notebook.
It clears each code cell's outputs and execution count, and the per-cell execution timestamps
that Jupyter and nbconvert write, so re-running a notebook changes nothing that is committed.

The standard library only, so it runs in any Python 3 with no environment active. The output
is formatted the way nbformat writes a notebook (sorted keys, one-space indent), so a notebook
saved by Jupyter and then cleaned differs only where the content does.

Input that is not a notebook passes through unchanged, so a half-saved file is never mangled.
Enable it once per clone with ``git config --local include.path ../.gitconfig`` (see README).
"""
import json
import sys

TIMING_KEYS = ('execution', 'ExecuteTime')


def strip(nb):
    """Clear outputs, execution counts and execution timing from every cell, in place."""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        for key in TIMING_KEYS:
            cell.get('metadata', {}).pop(key, None)
    return nb


def main():
    raw = sys.stdin.buffer.read()
    try:
        nb = json.loads(raw.decode('utf-8'))
        if not isinstance(nb, dict) or 'cells' not in nb:
            raise ValueError('not a notebook')
    except ValueError:                       # includes JSON and Unicode decoding errors
        sys.stdout.buffer.write(raw)
        return
    out = json.dumps(strip(nb), sort_keys=True, indent=1, ensure_ascii=False) + '\n'
    sys.stdout.buffer.write(out.encode('utf-8'))


if __name__ == '__main__':
    main()
