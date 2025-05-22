import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from FastGaussianPuff import GridMode, SensorMode


def test_init():
    gp = GridMode(**params, **grid_params)
    sp = SensorMode(**params, **sensor_params)

def test_list_init():
    ws = [3]*61
    wd = [200]*61

    test_params = params.copy()
    test_grid_params = grid_params.copy()
    test_sensor_params = sensor_params.copy()

    # this should still work if we pass lists instead of numpy arrays
    test_params["wind_speeds"] = ws
    test_params["wind_directions"] = wd
    test_params["emission_rates"] = [3]
    test_grid_params["grid_coordinates"] = [0, 0, 0, 50, 50, 10]
    test_sensor_params["sensor_coordinates"] = [[5, 5, 5], [16, 19, 4], [47, 4, 1]]
    test_params["source_coordinates"] = [[25, 25, 5]]

    gp = GridMode(**params, **grid_params)
    sp = SensorMode(**params, **sensor_params)

def test_bad_list_init():
    ws = [3]*61

    test_params = params.copy()

    test_params["wind_speeds"] = pd.DataFrame(ws)

    with pytest.raises(TypeError):
        gp = GridMode(**test_params, **grid_params)
    with pytest.raises(TypeError):
        sp = SensorMode(**test_params, **sensor_params)


@pytest.mark.parametrize(
    "GPClass, mode_params",
    [
        (GridMode, grid_params),
        (SensorMode, sensor_params),
    ],
)
def test_source_coordinate_formats(GPClass, mode_params):
    test_params = params.copy()

    # Case 1: Single source coordinate (as nested list)
    test_params["source_coordinates"] = [[25, 25, 5]]
    gp = GPClass(**test_params, **mode_params)
    gp.simulate()
    ans = np.linalg.norm(gp.ch4_obs)
    assert ans > 0

    # Case 2: Single source coordinate (as flat list)
    test_params["source_coordinates"] = [25, 25, 5]
    gp = GPClass(**test_params, **mode_params)
    gp.simulate()
    assert np.linalg.norm(gp.ch4_obs) == ans

    # Case 3: Multiple source coordinates (not yet supported)
    test_params["source_coordinates"] = [[25, 25, 5], [30, 30, 5]]
    with pytest.raises(NotImplementedError):
        gp = GPClass(**test_params, **mode_params)


@pytest.mark.parametrize(
    "GPClass, mode_params",
    [
        (GridMode, grid_params),
        (SensorMode, sensor_params),
    ],
)
def test_naive_tz(GPClass, mode_params):
    test_params = params.copy()
    test_params["simulation_start"] = pd.to_datetime("2022-01-01 12:00:00")
    with pytest.raises(ValueError):
        gp = GPClass(**test_params, **mode_params)

    test_params = params.copy()
    test_params["simulation_end"] = pd.to_datetime("2022-01-01 13:00:00")
    with pytest.raises(ValueError):
        gp = GPClass(**test_params, **mode_params)

    test_params = params.copy()
    test_params["time_zone"] = "bad_tz"
    with pytest.raises(ValueError):
        gp = GPClass(**test_params, **mode_params)
