import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from FastGaussianPuff import GridMode, SensorMode


@pytest.mark.parametrize(
    "GPClass, mode_params",
    [
        (GridMode, grid_params),
        (SensorMode, sensor_params),
    ],
)
def test_wind_skip(GPClass, mode_params):
    test_params = params.copy()

    gp = GPClass(**test_params, **mode_params)
    gp.simulate()
    ch4_nonzero = gp.ch4_obs.copy()

    ws = np.array([0.2]*61)
    test_params["wind_speeds"] = ws
    test_params["skip_low_wind"] = True
    test_params["low_wind_cutoff"] = 0.5
    gp = GPClass(**test_params, **mode_params)
    gp.simulate()
    ch4_zero = gp.ch4_obs.copy()

    assert np.all(ch4_zero == 0)
    assert not np.all(ch4_nonzero == 0)


@pytest.mark.parametrize(
    "GPClass, mode_params",
    [
        (GridMode, grid_params),
        (SensorMode, sensor_params),
    ],
)
def test_zero_wind(GPClass, mode_params):
    test_params = params.copy()
    ws = np.array([0.5]*61)
    ws[20] = 0
    test_params["wind_speeds"] = ws
    test_params["skip_low_wind"] = False
    with pytest.raises(ValueError, match="[FastGaussianPuff]*"):
        gp = GPClass(**test_params, **mode_params)
