"""The inclusive design-length endpoint must contribute to comparisons."""

import numpy as np
import pandas as pd
from analysis import figures


def test_composition_bins_include_both_endpoints():
    result = figures._binned_means(
        pd.Series([0.2, 0.8, 1.0]),
        pd.Series([60, 200, 201]),
        np.linspace(60, 200, 6),
    )
    assert result.n.sum() == 2
    assert result.iloc[0]["mean"] == 0.2
    assert result.iloc[-1]["mean"] == 0.8
