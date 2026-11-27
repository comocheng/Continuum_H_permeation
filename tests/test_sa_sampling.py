"""How the sensitivity scan places its samples.

Most SA ranges span several decades (f_pinhole 1e-5..0.1, P_upstream 1e-7..1e7 Pa),
so scale parameters are drawn log-uniformly: each decade of a stated range gets the
same number of draws. (Drawn uniformly in linear space, 99.9% of f_pinhole draws
would land above 1e-4.) Energies and temperature stay linear.
"""
import numpy as np
import pytest

from calculations.config.model_config import (
    SUGGESTED_RANGES_LEVEL5, SUGGESTED_RANGES_LEVEL5L6,
)
from calculations.sensitivity import (
    LINEAR_SAMPLED_PARAMS, lhs_sample, sampling_scales, run_global_lhs_scan,
    level5_model_wrapper, SCAN_OUTPUT_FIELDS_L5, SCAN_LABEL_FIELDS_L5,
)


def test_every_shipped_range_has_a_sampling_scale():
    # raises if a log-sampled range reaches zero
    for ranges in (SUGGESTED_RANGES_LEVEL5, SUGGESTED_RANGES_LEVEL5L6):
        scales = sampling_scales(ranges)
        assert set(scales) == set(ranges)


def test_energies_and_temperature_are_linear_everything_else_log():
    scales = sampling_scales(SUGGESTED_RANGES_LEVEL5L6)
    for name, s in scales.items():
        assert s == ('linear' if name in LINEAR_SAMPLED_PARAMS else 'log'), name
    for name in ('f_pinhole', 'P_upstream', 'D_ref', 'oxide_thickness', 'k_diss_ref'):
        assert scales[name] == 'log'


def test_log_uniform_gives_each_decade_equal_weight():
    ranges = {'f': [1e-5, 1e-1], 'E_D': [1e4, 5e4]}
    X, _ = lhs_sample(ranges, 4000, seed=1)
    f = X[:, 0]
    counts = np.histogram(np.log10(f), bins=4, range=(-5, -1))[0]
    # Latin hypercube in log space: the four decades hold exactly N/4 each
    assert counts.tolist() == [1000, 1000, 1000, 1000]
    # linear stays linear: half the draws below the arithmetic midpoint
    assert abs(np.mean(X[:, 1] < 3e4) - 0.5) < 0.01


def test_samples_stay_inside_their_stated_ranges():
    X, _ = lhs_sample(SUGGESTED_RANGES_LEVEL5L6, 500, seed=3)
    for j, (lo, hi) in enumerate(SUGGESTED_RANGES_LEVEL5L6.values()):
        assert X[:, j].min() >= lo and X[:, j].max() <= hi


def test_nonpositive_log_range_is_refused():
    with pytest.raises(ValueError, match="LINEAR_SAMPLED_PARAMS"):
        sampling_scales({'f_new': [0.0, 1.0]})


def test_scan_records_physical_values_and_scales():
    ranges = {'f_pinhole': [1e-5, 1e-1], 'H_s': [1000, 50000]}
    df, problem = run_global_lhs_scan(
        ranges, 200, wrapper=level5_model_wrapper,
        output_fields=SCAN_OUTPUT_FIELDS_L5, label_fields=SCAN_LABEL_FIELDS_L5,
        verbose=False)
    assert problem['scales'] == {'f_pinhole': 'log', 'H_s': 'linear'}
    assert np.median(df['f_pinhole']) == pytest.approx(10 ** -3, rel=0.1)


def test_both_noise_floors_are_reported():
    """Dummy inputs give a delta floor and a PAWN floor; real inputs clear them."""
    import pandas as pd
    from calculations.sensitivity import (
        givendata_sensitivity_by_regime, summarize_givendata)
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(1500, 3))
    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    df['flux'] = np.exp(4 * df['a'])            # only 'a' matters
    res = givendata_sensitivity_by_regime({'r': df}, ['a', 'b', 'c'],
                                          output_metrics=['flux'], verbose=False)
    e = res['r']['flux']
    assert 0 < e['floor'] < 0.2 and 0 < e['floor_pawn'] < 0.2
    assert len(e['pawn']['median']) == 3        # dummies trimmed off
    t = summarize_givendata(res, ['a', 'b', 'c']).set_index('parameter')
    assert t.loc['a', 'pawn_over_floor'] > 3 and t.loc['a', 'delta_over_floor'] > 3


def test_scan_stamp_records_the_scan_version(tmp_path):
    from calculations.sensitivity import SCAN_VERSION, scan_version
    assert scan_version(str(tmp_path)) is None
    run_global_lhs_scan({'f_pinhole': [1e-5, 1e-1]}, 5, wrapper=level5_model_wrapper,
                        output_fields=SCAN_OUTPUT_FIELDS_L5,
                        label_fields=SCAN_LABEL_FIELDS_L5,
                        save_path=str(tmp_path / 'scan_x.csv'), verbose=False)
    assert scan_version(str(tmp_path)) == SCAN_VERSION


def test_interrupted_batch_is_not_stamped_current(tmp_path):
    """Until a batch's last scan is written the stamp must not claim SCAN_VERSION,
    or a cache check would accept old scans left beside the new ones."""
    from calculations.sensitivity import SCAN_VERSION, scan_version, _stamp_study
    _stamp_study(str(tmp_path), complete=False)
    assert scan_version(str(tmp_path)) == 0
    _stamp_study(str(tmp_path))
    assert scan_version(str(tmp_path)) == SCAN_VERSION

