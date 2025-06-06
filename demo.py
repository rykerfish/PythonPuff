import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import utils
from PythonPuff import SensorMode


def main():
    dir = "./data/"
    source_dat = pd.read_csv(dir + "source_locations.csv")
    sensor_dat = pd.read_csv(dir + "sensor_locations.csv")

    source_x, source_y = utils.latlon_to_utm_array(
        source_dat["lat"].values, source_dat["lon"].values
    )
    source_z = source_dat["height"].values
    source_names = source_dat["name"].values

    sensor_x, sensor_y = utils.latlon_to_utm_array(
        sensor_dat["lat"].values, sensor_dat["lon"].values
    )
    sensor_z = sensor_dat["height"].values
    sensor_names = sensor_dat["name"].values

    # there are some constraints on timestep parameters. all units are in seconds.
    # obs_dt needs to come from the delta between the timestamps in the wind data, so 60 s = 1 observation per minute.
    # sim_dt and puff_dt determine how well the plume is resolved. we require sim_dt <= puff_dt.
    # sim_dt = puff_dt = 1 is a good place to start, but you can sometimes increase puff_dt to speed up the simulation.
    obs_dt, sim_dt, puff_dt = 60, 1, 1

    # these can be specified in any timezone, but timezone-aware timestamps are required.
    start = pd.to_datetime("2022-04-15 16:42:00-06:00")
    end = pd.to_datetime("2022-04-15 18:42:00-06:00")

    # this is the timezone of the location being simulated. The below works for METEC
    time_zone = "America/Denver"  # alternative: "US/Mountain"

    # wind data is in m/s and degrees.
    # each needs to be a single timeseries and the same length as the simulation.
    n_mins = int((end - start).total_seconds() / 60) + 1
    wind_speeds = np.ones(n_mins) * 3.0 + np.cos(np.arange(n_mins) / 10.0) * 0.75
    wind_directions = np.ones(n_mins) * 180.0 + np.sin(np.arange(n_mins) / 10.0) * 80.0


    # coordinate format: [[x1, y1, z1], [x2, y2, z2], ...]
    # currently, only single source is supported, but you can have as many sensors as you want.
    source_coordinates = np.column_stack((source_x, source_y, source_z))
    sensor_coordinates = np.column_stack(
        (sensor_x, sensor_y, sensor_z)
    )  #  it is assumed that these encase the source coordinates.

    emission_rate = np.array([1.0])  # emission rate for sources, [kg/hr]

    output_dir = "./output/"

    plotting_dir = "./img/"

    n_sources = len(source_coordinates)
    for i in range(n_sources):
        print("Simulating source " + str(i + 1) + " of " + str(n_sources))

        source_i = source_coordinates[i]

        # construct the simulation
        sp = SensorMode(
            obs_dt=obs_dt,
            sim_dt=sim_dt,
            puff_dt=puff_dt,
            simulation_start=start,
            simulation_end=end,
            time_zone=time_zone,
            source_coordinates=source_i,
            emission_rates=emission_rate,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            sensor_coordinates=sensor_coordinates,
        )

        # run the simulation
        ch4 = sp.simulate()
        plot(sp, sensor_names, plotting_dir, source_names[i])

        # save the results with timestamps
        timestamps = pd.date_range(
            start=start, end=end, freq=pd.DateOffset(seconds=obs_dt)
        )  # create timestamps for the observations
        df = pd.DataFrame(ch4, index=timestamps, columns=sensor_names)
        df.index.name = "timestamp"
        df.to_csv(output_dir + source_names[i] + ".csv", index=True)


def plot(sp, sensor_names, plotting_dir, source_name):
    t, n_sensors = np.shape(sp.ch4_obs)

    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    m = sp.ch4_obs.max()
    fig.supxlabel("Time from emission start (minutes)")
    fig.supylabel("Methane concentration (ppm)")

    n_sensors = len(sensor_names)
    for i in range(0, n_sensors):

        if i < 4:
            row = 0
            col = i
        else:
            row = 1
            col = i - 4

        times = np.arange(0, t)

        sensor_ch4 = sp.ch4_obs[:, i]

        ax[row][col].plot(times, sensor_ch4, label="Simulated", color="blue")
        ax[row][col].set_ylim(-1, m + 2)
        ax[row][col].set_title(sensor_names[i])

    fname = plotting_dir + source_name + ".png"
    fig.savefig(fname, format="png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    main()
