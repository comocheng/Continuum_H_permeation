"""Running the SA stages in worker processes must not change a single number."""
import numpy as np
import pandas as pd
import pytest

from calculations.sensitivity import (
    run_global_lhs_scan, givendata_sensitivity_by_regime, level5_model_wrapper,
    SCAN_OUTPUT_FIELDS_L5, SCAN_LABEL_FIELDS_L5, PARALLEL_MIN_DRAWS,
)
from calculations.config.model_config import SUGGESTED_RANGES_LEVEL5

pytestmark = pytest.mark.slow   # starts worker processes: ~10 s


def test_parallel_scan_is_identical_to_serial():
    ranges = {k: v for k, v in SUGGESTED_RANGES_LEVEL5.items() if k != 'temperature'}
    kw = dict(wrapper=level5_model_wrapper, output_fields=SCAN_OUTPUT_FIELDS_L5,
              label_fields=SCAN_LABEL_FIELDS_L5, fixed_params={'temperature': 900.0},
              verbose=False)
    a, _ = run_global_lhs_scan(ranges, PARALLEL_MIN_DRAWS, n_jobs=1, **kw)
    b, _ = run_global_lhs_scan(ranges, PARALLEL_MIN_DRAWS, n_jobs=3, **kw)
    pd.testing.assert_frame_equal(a, b)


def test_parallel_indices_are_identical_to_serial():
    rng = np.random.default_rng(1)
    clusters = {}
    for r in ('x', 'y'):
        df = pd.DataFrame(rng.uniform(size=(300, 3)), columns=['a', 'b', 'c'])
        df['flux'] = np.exp(3 * df['a'] + df['b'])
        df['theta'] = df['c'] ** 2
        clusters[r] = df
    kw = dict(output_metrics=['flux', 'theta'], log_metrics=('flux',), n_dummy=4,
              delta_resamples=10, verbose=False)
    s = givendata_sensitivity_by_regime(clusters, ['a', 'b', 'c'], n_jobs=1, **kw)
    p = givendata_sensitivity_by_regime(clusters, ['a', 'b', 'c'], n_jobs=3, **kw)
    for r in s:
        for m in s[r]:
            assert s[r][m]['floor'] == p[r][m]['floor']
            assert s[r][m]['floor_pawn'] == p[r][m]['floor_pawn']
            np.testing.assert_array_equal(s[r][m]['delta']['delta'], p[r][m]['delta']['delta'])
            np.testing.assert_array_equal(s[r][m]['pawn']['median'], p[r][m]['pawn']['median'])
