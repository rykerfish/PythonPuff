import numpy as np
import pandas as pd
import datetime


def _ensure_utc(dt):
    """
    Ensures the input datetime is timezone-aware and in UTC.
    Converts to UTC if it's in a different timezone.
    """
    ts = pd.Timestamp(dt)

    if ts.tz is None:
        raise ValueError(
            f"[FastGaussianPuff] Naive datetime detected: {dt}. Please provide a timezone-aware datetime."
        )

    if ts.tz != datetime.timezone.utc:
        ts = ts.tz_convert("UTC")

    return ts


def _interpolate_wind_data(
    wind_speeds, wind_directions, puff_dt, sim_start, sim_end, n_obs
):
    """
    Resample wind_speeds and wind_directions to the simulation resolution by interpolation.
    Inputs:
        sim_dt [s] (int):
            the target time resolution to resample to
        sim_start, sim_end (pd.DateTime)
            DateTimes the simulation start and ends at
        n_obs (int)
            number of observation data points across the simulation time range
    """

    # creates a timeseries at obs_dt resolution
    time_stamps = pd.date_range(sim_start, sim_end, n_obs)

    # technically need a final value to interpolate between, so just extend timeseries
    if len(wind_speeds) == n_obs - 1:
        wind_speeds = np.append(wind_speeds, wind_speeds[-1])

    if len(wind_directions) == n_obs - 1:
        wind_directions = np.append(wind_directions, wind_directions[-1])

    # interpolate for wind_speeds and wind_directions:
    ## 1. convert ws & wd to x and y component of wind (denoted by u, v, respectively)
    ## 2. interpolate u and v
    ## 3. bring resampled u and v back to resampled ws and wd
    wind_u, wind_v = _wind_vector_convert(
        wind_speeds, wind_directions, direction="ws_wd_2_u_v"
    )

    # resamples wind data to sim_dt resolution
    wind_df = pd.DataFrame(data={"wind_u": wind_u, "wind_v": wind_v}, index=time_stamps)

    wind_df = wind_df.resample(str(puff_dt) + "s").interpolate()
    wind_u = wind_df["wind_u"].to_list()
    wind_v = wind_df["wind_v"].to_list()

    wind_speeds_sim, wind_directions_sim = _wind_vector_convert(
        wind_u, wind_v, direction="u_v_2_ws_wd"
    )

    return wind_speeds_sim, wind_directions_sim


def _wind_vector_convert(input_1, input_2, direction="ws_wd_2_u_v"):
    """
    Convert between (ws, wd) and (u,v)
    Inputs:
        input_1:
            - list of wind speed [m/s] if direction = 'ws_wd_2_u_v'
            - list of the x component (denoted by u) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
        input_2:
            - list of wind direction [degree] if direction = 'ws_wd_2_u_v'
            - list of the y component (denoted by v) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
        direction:
            - 'ws_wd_2_u_v': convert wind speed and wind direction to x,y components of a wind vector
            - 'u_v_2_ws_wd': convert x,y components of a wind vector to wind speed and wind directin

    Outputs:
        quantities corresponding to the conversion direction
    """
    if direction == "ws_wd_2_u_v":
        ws, wd = input_1, input_2
        thetas = [
            270 - x for x in wd
        ]  # convert wind direction to the angle between the wind vector and positive x-axis
        thetas = np.radians(thetas)
        u = [np.cos(theta) * l for l, theta in zip(ws, thetas)]
        v = [np.sin(theta) * l for l, theta in zip(ws, thetas)]
        output_1, output_2 = u, v
    elif direction == "u_v_2_ws_wd":
        u, v = input_1, input_2
        ws = [(x**2 + y**2) ** (1 / 2) for x, y in zip(u, v)]
        thetas = [
            np.arctan2(y, x) for x, y in zip(u, v)
        ]  # the angles between the wind vector and positive x-axis
        thetas = [x * 180 / np.pi for x in thetas]
        wd = [270 - x for x in thetas]  # convert back to cardinal definition
        for i, x in enumerate(wd):
            if x < 0:
                wd[i] = wd[i] + 360
            elif x >= 360:
                wd[i] = wd[i] - 360
            else:
                pass
        output_1, output_2 = ws, wd
    else:
        raise NotImplementedError(">>>>> wind vector conversion direction")

    return output_1, output_2


def _check_wind_data(ws, skip_low_wind):
    if skip_low_wind:
        return

    if np.any(ws <= 0):
        raise (ValueError("[FastGaussianPuff] wind speeds must be greater than 0"))
    if np.any(ws < 1e-2):
        print(
            "[FastGaussianPuff] WARNING: There's a wind speed < 0.01 m/s. This is likely a mistake and will cause slow performance. The simulation will continue, but results will be poor as the puff model is degenerate in low wind speeds."
        )


def _check_array_dtypes(**kwargs):
    casted_arrays = {}

    for name, var in kwargs.items():
        if var is not None:
            try:
                casted_arrays[name] = np.asarray(var, dtype=float)
            except Exception as e:
                raise ValueError(
                    f"[fGP] Error: Failed to cast '{name}' to a NumPy float array: {e}. "
                    f"Try using an array-like with numeric dtype."
                )
        else:
            casted_arrays[name] = None

    return casted_arrays


def _check_timestep_parameters(sim_dt, puff_dt, out_dt):

    # rationale: negative time seems bad. maybe ask a physicist.
    if sim_dt <= 0:
        print("[fGP] Error: sim_dt must be a positive value")
        exit(-1)

    # rationale: breaking this would mean multiple puffs are emitted in between simulation time steps. this constraint
    # decouples sim_dt and puff_dt so that sim_dt can be scaled according to wind speed and puff_dt can be scaled
    # according to the change in wind direction over time to guarantee accuracy of the simulation (i.e. no skipping)
    if puff_dt < sim_dt:
        print("[fGP] Error: puff_dt must be greater than or equal to sim_dt")
        exit(-1)

    # rationale: concentration arrays are build at sim_dt resolution. puff_dt = n*sim_dt for an integer n > 0
    # ensure that puffs are emitted on simulation time steps and prevents the need for a weird interpolation/rounding.
    # this constaint could likely be avoided, but it isn't that strong and makes the code easier.
    eps = 1e-5
    ratio = puff_dt / sim_dt
    if abs(ratio - round(ratio)) > eps:
        print("[fGP] Error: puff_dt needs to be a positive integer multiple of sim_dt")
        exit(-1)

    # rationale: we don't have simulation data at a resolution less than sim_dt, so you'll have blank
    # concentration fields if this condition isn't met
    if out_dt < sim_dt:
        print("[fGP] Error: output_dt must be greater than or equal to sim_dt")
        exit(-1)


def _parse_source_coords(source_coordinates):
    size = np.shape(source_coordinates)
    if len(size) == 1:
        if size[0] == 3:
            source_coordinates = np.array(
                [source_coordinates]
            )  # now a nested array- C++ code expects this format
        else:
            print(
                "[fGP] Error: source_coordinates must be a 3-element array, e.g. [x0, y0, z0]."
            )
            exit(-1)
    else:
        if size[0] == 1 and size[1] == 3:
            return source_coordinates
        elif size[0] > 1 and size[1] == 3:
            raise (
                NotImplementedError(
                    "[fGP] Error: Multi-source currently isn't implemented. Only provide coordinates for a single source, e.g. [x0, y0, z0] or [[x0, y0, z0]]."
                )
            )

    return source_coordinates
