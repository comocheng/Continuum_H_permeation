"""A model evaluation that fails must be visible as a failure.

The wrappers used to catch every exception, print a traceback and return flux = 1e-20,
which is -20 on the log10 scale the indices use. The row was labelled 'undefined' and
quietly dropped from every cluster. A NaN total flux in L5L6 was worse: every flux
weight became 0 and argmax over (0, 0, 0) labelled the row 'surface'.
"""
import numpy as np
import pytest

from calculations.sensitivity import (
    level5_model_wrapper, level5L6_model_wrapper, run_global_lhs_scan,
    SCAN_OUTPUT_FIELDS_L5, SCAN_LABEL_FIELDS_L5,
)
import calculations.surface_kinetics as sk

# Defect area fractions summing past 1 make both models raise.
IMPOSSIBLE = {'f_pinhole': 0.6, 'f_crack': 0.6, 'f_gb_defect': 0.1}


@pytest.mark.parametrize("wrapper", [level5_model_wrapper, level5L6_model_wrapper])
def test_failure_returns_nan_and_undefined(wrapper):
    with pytest.warns(RuntimeWarning, match="failed"):
        rec = wrapper(IMPOSSIBLE, return_full_record=True)
    assert np.isnan(rec['flux']) and np.isnan(rec['permeability'])
    assert rec['regime'] == 'undefined'
    assert rec['error']


def test_nan_total_flux_is_not_labelled_surface(monkeypatch):
    real = sk.calculate_path_flux_L346_v2

    def broken(*a, **k):
        out = real(*a, **k)
        if k.get('path_type') == 'crack':
            out = dict(out, flux=np.nan)
        return out
    monkeypatch.setattr(sk, 'calculate_path_flux_L346_v2', broken)
    with pytest.warns(RuntimeWarning):
        rec = level5L6_model_wrapper({'f_crack': 1e-3})
    assert rec['regime'] == 'undefined'
    assert 'crack' in rec['error']


def test_scan_counts_failures(capsys):
    ranges = {'f_pinhole': [1e-3, 0.99], 'f_crack': [0.2, 0.5]}
    with pytest.warns(RuntimeWarning):
        df, _ = run_global_lhs_scan(
            ranges, 50, wrapper=level5_model_wrapper,
            output_fields=SCAN_OUTPUT_FIELDS_L5, label_fields=SCAN_LABEL_FIELDS_L5,
            verbose=False)
    failed = df['error'] != ''
    assert failed.any() and (~failed).any()
    assert (df.loc[failed, 'regime'] == 'undefined').all()
    assert df.loc[failed, 'flux'].isna().all()
    assert f"{failed.sum()}/50 draws failed" in capsys.readouterr().out
