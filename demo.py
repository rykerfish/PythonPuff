import pandas as pd
import numpy as np

import utils
from PythonPuff import SensorMode

dir = "./data/"
source_dat = pd.read_csv(dir + "source_locations.csv")
sensor_dat = pd.read_csv(dir + "sensor_locations.csv")

source_x, source_y = utils.latlon_to_utm_array(source_dat["lat"].values, source_dat["lon"].values)
source_z = source_dat["height"].values
source_names = source_dat["name"].values

sensor_x, sensor_y = utils.latlon_to_utm_array(sensor_dat["lat"].values, sensor_dat["lon"].values)
sensor_z = sensor_dat["height"].values
sensor_names = sensor_dat["name"].values

# there are some constraints on timestep parameters. all units are in seconds.
# obs_dt needs to come from the delta between the timestamps in the wind data, so 60 s = 1 observation per minute.
# sim_dt and puff_dt determine how well the plume is resolved. we require sim_dt <= puff_dt.
# sim_dt = puff_dt = 1 is a good place to start, but you can sometimes increase puff_dt to speed up the simulation.
obs_dt, sim_dt, puff_dt = 60, 1, 1

# these can be specified in any timezone
start = pd.to_datetime("2022-04-15 16:42:00-06:00")
end = pd.to_datetime("2022-04-15 18:42:00-06:00")

# this is the timezone of the location being simulated. The below works for METEC
time_zone = "America/Denver"  # alternative: "US/Mountain"

wind_dat = pd.read_csv(dir + "wind_median.csv")

wind_dat["timestamp"] = pd.to_datetime(wind_dat["timestamp"], utc=True)
filtered_wind_dat = wind_dat[(wind_dat["timestamp"] >= start) & (wind_dat["timestamp"] <= end)]

# wind data is in m/s and degrees.
# each needs to be a single timeseries and the same length as the simulation.
wind_speeds = filtered_wind_dat["wind_speed"].values
wind_directions = filtered_wind_dat["wind_dir"].values
wind_speeds = np.array(wind_speeds)
wind_directions = np.array(wind_directions)

# coordinate format: [[x1, y1, z1], [x2, y2, z2], ...]
# currently, only single source is supported, but you can have as many sensors as you want.
source_coordinates = np.column_stack((source_x, source_y, source_z))
sensor_coordinates = np.column_stack(
    (sensor_x, sensor_y, sensor_z)
)  #  it is assumed that these encase the source coordinates.

emission_rate = np.array([1.0])  # emission rate for sources, [kg/hr]

n_sources = len(source_coordinates)
output_dir = "./output/"
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

    # save the results with timestamps
    timestamps = pd.date_range(
        start=start, end=end, freq=pd.DateOffset(seconds=obs_dt)
    )  # create timestamps for the observations
    df = pd.DataFrame(ch4, index=timestamps, columns=sensor_names)
    df.index.name = "timestamp"
    df.to_csv(output_dir + source_names[i] + ".csv", index=True)
