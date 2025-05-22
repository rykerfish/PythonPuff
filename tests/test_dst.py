import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from PythonPuff import GridMode, SensorMode

@pytest.mark.parametrize(
    "GPClass, mode_params",
    [
        (GridMode, grid_params),
        (SensorMode, sensor_params),
    ],
)
@pytest.mark.parametrize(
    ("dst_start", "dst_end", "expected_length"), [
        ("2024-03-09 12:00:00-07:00", "2024-03-10 12:00:00-06:00", 23*2),
        ("2024-11-02 12:00:00-06:00", "2024-11-03 12:00:00-07:00", 25*2),
        ("2024-03-09 12:00:00+00:00", "2024-03-11 11:00:00+00:00", 47*2),
        ("2024-11-02 12:00:00+00:00", "2024-11-04 12:00:00+00:00", 48*2)]
)
def test_dst_lengths_grid(GPClass, mode_params, dst_start, dst_end, expected_length):
    test_params = params.copy()

    # dst_start = pd.to_datetime("2024-03-09 12:00:00-07:00")
    # dst_end = pd.to_datetime("2024-03-10 12:00:00-06:00")

    test_params["simulation_start"] = dst_start
    test_params["simulation_end"] = dst_end
    test_params["wind_speeds"] = [2.0]*expected_length  # every half hour
    test_params["wind_directions"] = [180.0]*expected_length
    test_params["obs_dt"] = 60*30
    test_params["sim_dt"] = 60
    test_params["puff_dt"] = 60

    gp = GPClass(**test_params, **mode_params)
    gp.simulate()

    length = len(gp.ch4_obs)

    assert(length == expected_length + 1) # TODO fix the off by one error in the interface
