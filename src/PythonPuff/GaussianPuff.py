import datetime
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from numba import njit, prange

from . import interface_helpers as ih

from abc import abstractmethod


"""
Inputs:
    obs_dt [s] (scalar, double):
        time interval (dt) for the observations
        NOTE: must be larger than sim_dt. This should be the resolution of the wind data.
    sim_dt [s] (scalar, double):
        time interval (dt) for the simulation results
    puff_dt [s] (scalar, double):
        time interval (dt) between two successive puffs' creation
        NOTE: must also be a positive integer multiple of sim_dt, e.g. puff_dt = n*sim_dt for integer n > 0
    output_dt [s] (scalar, double):
        resolution to resample the concentration to at the end of the sismulation. By default,
        resamples to the resolution of the wind observations obs_dt.
    simulation_start, simulation_end (pd.DateTime values)
        start and end times for the emission to be simulated. Must be timezone-aware.
    time_zone (timezone string):
        timezone to use for the simulation. This should be a string that can be parsed by
        the zoneinfo module, e.g. "America/New_York".
    source_coordinates (array, size=(n_sources, 3)) [m]:
        holds source coordinate (x0,y0,z0) in meters for each source.
    emission_rates: (array, length=n_sources) [kg/hr]:
        rate that each source is emitting at in kg/hr.
    wind_speeds [m/s] (list of floats):
        wind speed at each time stamp, in obs_dt resolution
    wind_directions [degree] (list of floats):
        wind direction at each time stamp, in obs_dt resolution.
        follows the conventional definition:
        0 -> wind blowing from North, 90 -> E, 180 -> S, 270 -> W
    using_sensors (boolean):
        If True, ignores grid-related input parameters and only simulates at sensor coordinates.
        True inputs:
            - sensor_coordinates
        False inputs:
            - grid_coordinates,
            - nx, ny, nz
    sensor_coordinates: (array, size=(n_sensors, 3)) [m]
        coordinates of the sensors in (x,y,z) format.
    grid_coordinates: (array, length=6) [m]
        holds the coordinates for the corners of the rectangular grid to be created.
        format is grid_coordinates=[min_x, min_y, min_z, max_x, max_y, max_z]
    nx, ny, ny (scalar, int):
        Number of points for the grid the x, y, and z directions
    puff_duration (double) [seconds] :
        how many seconds a puff can 'live'; we assume a puff will fade away after a certain time.
        Depending on the grid size wind speed, this parameter will never come into play as the simulation
        for the puff stops when the plume has moved far away. In low wind speeds, however, this cutoff will
        halt the simulation of a puff early. This may be desirable as exceedingly long (and likely unphysical)
        plume departure times can be computed for wind speeds << 1 m/s
    skip_low_wind (boolean), low_wind_cutoff [m/s] (float):
        if True, the simulation will skip any time step where the wind speed is below low_wind_cutoff.
        This is useful to avoid zero-values or situations where the wind is so slow that it'd create unreasonable predictions.
        Default is False.
    exp_threshold_tolerance (scalar, float):
        the tolerance used to threshold the exponentials when evaluating the Gaussian equation.
        If, for example, exp_tol = 1e-9, the concentration at a single point for an individual time step
        will have error less than 1e-9. Upsampling to different time resolutions may introduce extra error.
        Default is 1e-7, which passess all safe-mode tests with less than 0.1% error.
    conversion_factor (scalar, float):
        convert from kg/m^3 to ppm, this factor is for ch4 only
    unsafe (boolean):
        if True, will use unsafe evaluations for some operations. This mode is faster but introduces some
        error. If you're unsure about results, set to False and compare error between the two methods.
    quiet (boolean):
        if True, outputs extra information about the simulation and its progress.
"""


class GaussianPuff:
    def __init__(
        self,
        obs_dt,
        sim_dt,
        puff_dt,
        simulation_start,
        simulation_end,
        time_zone,
        X,
        Y,
        Z,
        source_coordinates,
        emission_rates,
        wind_speeds,
        wind_directions,
        output_dt=None,
        puff_duration=1200,
        skip_low_wind=False,
        low_wind_cutoff=-1,
        exp_threshold_tolerance=None,
        conversion_factor=1e6 * 1.524,
        unsafe=False,
        quiet=True,
    ):
        ih._check_timestep_parameters(sim_dt, puff_dt, obs_dt)
        self.obs_dt = obs_dt
        self.sim_dt = sim_dt
        self.puff_dt = puff_dt
        self.output_dt = output_dt or obs_dt

        self.sim_start = ih._ensure_utc(simulation_start)
        self.sim_end = ih._ensure_utc(simulation_end)

        ns = (self.sim_end - self.sim_start).total_seconds()
        self.n_obs = (
            math.floor(ns / obs_dt) + 1
        )  # number of observed data points we have

        try:
            time_zone = ZoneInfo(time_zone)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone: {time_zone}")
        self.time_zone = time_zone

        self.X = X
        self.Y = Y
        self.Z = Z
        self.N_points = len(self.X)

        result = ih._check_array_dtypes(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            source_coordinates=source_coordinates,
            emission_rates=emission_rates,
        )
        wind_speeds = result["wind_speeds"]
        wind_directions = result["wind_directions"]
        source_coordinates = result["source_coordinates"]
        self.emission_rates = result["emission_rates"]

        ih._check_wind_data(wind_speeds, skip_low_wind)
        # resample the wind data from obs_dt to the simulation resolution sim_dt
        self.ws, self.wd = ih._interpolate_wind_data(
            wind_speeds,
            wind_directions,
            puff_dt,
            self.sim_start,
            self.sim_end,
            self.n_obs,
        )

        self.source_coordinates = ih._parse_source_coords(source_coordinates)

        self.puff_duration = puff_duration
        self.skip_low_wind = skip_low_wind
        self.low_wind_cutoff = low_wind_cutoff
        self.exp_threshold_tolerance = exp_threshold_tolerance
        self.conversion_factor = conversion_factor
        self.unsafe = unsafe
        self.quiet = quiet

        self.one_over_two_pi_three_halves = 1 / ((2 * math.pi) ** (3 / 2))

        utc_total_time_series = pd.date_range(
            start=self.sim_start, end=self.sim_end, freq=f"{puff_dt}s", tz="UTC"
        )
        local_ts = utc_total_time_series.tz_convert(time_zone)
        self.hours_arr = local_ts.hour.values
        self.n_puffs = len(self.hours_arr)

        self.quiet = quiet

        # allow unsafe mode to have coarser thresholding
        if exp_threshold_tolerance is None:
            if unsafe:
                self.exp_threshold_tol = 1e-5
            else:
                self.exp_threshold_tol = 1e-7
        else:
            self.exp_threshold_tol = exp_threshold_tolerance

        if skip_low_wind:
            if low_wind_cutoff <= 0:
                raise ValueError(
                    "[FastGaussianPuff] low wind cutoff must be greater than 0"
                )
            self.skip_low_wind = True
            self.low_wind_thresh = low_wind_cutoff

        # save timeseries of simulation resolution so we can resample back to observation later
        self.time_stamps_sim = pd.date_range(
            self.sim_start, self.sim_end, freq=str(self.sim_dt) + "s"
        )
        self.n_sim = len(self.time_stamps_sim)  # number of simulation time steps

        self.puff_duration = puff_duration
        if self.puff_duration == None:
            self.puff_duration = self.n_sim  # ensures we don't overflow time index

        # initialize the final simulated concentration array
        self.ch4_result = np.zeros(
            (self.n_sim, self.N_points)
        )  # simulation in sim_dt resolution, flattened

    def simulate(self):
        self.setSourceCoordinates(0)
        q = self.emission_rates[0] / 3600  # convert to kg/s
        emission_per_puff = q * self.puff_dt
        report_ratio = 0.1

        puff_lifetime = math.ceil(self.puff_duration / self.sim_dt)
        puff_to_sim_ratio = round(self.puff_dt / self.sim_dt)

        for p in range(self.n_puffs):
            if self.skip_low_wind and self.ws[p] < self.low_wind_thresh:
                continue

            if p * puff_to_sim_ratio + puff_lifetime >= self.ch4_result.shape[0]:
                puff_lifetime = self.ch4_result.shape[0] - p * puff_to_sim_ratio

            theta = self.windDirectionToAngle(self.wd[p])

            # this slice shifts the starting index which is useful in concentrationPerPuff. does not copy.
            ch4_slice = self.ch4_result[
                p * puff_to_sim_ratio : p * puff_to_sim_ratio + puff_lifetime, :
            ]

            self.concentrationPerPuff(
                emission_per_puff, theta, self.ws[p], self.hours_arr[p], ch4_slice
            )

            if not self.quiet and math.floor(self.n_puffs * report_ratio) == p:
                print(f"Simulation is {int(report_ratio * 100)}% done")
                report_ratio += 0.1

        self.ch4_obs = self._resample_simulation(self.ch4_result, self.output_dt)

        return self.ch4_obs

    def setSourceCoordinates(self, source_index):

        self.x0 = self.source_coordinates[source_index, 0]
        self.y0 = self.source_coordinates[source_index, 1]
        self.z0 = self.source_coordinates[source_index, 2]

        # center the x, y grids at the source
        self.X_centered = self.X - self.x0
        self.Y_centered = self.Y - self.y0

        self.x_min = np.min(self.X) - self.x0
        self.y_min = np.min(self.Y) - self.y0
        self.x_max = np.max(self.X) - self.x0
        self.y_max = np.max(self.Y) - self.y0

    def windDirectionToAngle(self, wd):
        deg_to_rad_factor = np.pi / 180  # or use np.deg2rad(1)
        theta = wd - 270
        theta *= deg_to_rad_factor
        return theta

    def concentrationPerPuff(self, q, theta, ws, hour, ch4_slice):
        self.cosine = math.cos(theta)  # Cache cosine and sine
        self.sine = math.sin(theta)

        stability_class = self.stabilityClassifier(ws, hour)

        self.GaussianPuffEquation(q, ws, stability_class, ch4_slice)

    def stabilityClassifier(self, wind_speed, hour, day_start=7, day_end=18):
        is_day = day_start <= hour <= day_end

        if wind_speed < 2:
            return ["A", "B"] if is_day else ["E", "F"]
        elif wind_speed < 3:
            return ["B"] if is_day else ["E", "F"]
        elif wind_speed < 5:
            return ["B", "C"] if is_day else ["D", "E"]
        elif wind_speed < 6:
            return ["C", "D"] if is_day else ["D"]
        else:
            return ["D"]

    def GaussianPuffEquation(self, q, ws, stability_class, ch4):
        exit_location = self.calculateExitLocation()
        max_downwind_dist = math.sqrt(exit_location[0] ** 2 + exit_location[1] ** 2)

        temp_y, temp_z = self.getSigmaCoefficients(
            stability_class, [max_downwind_dist], 0
        )
        sigma_y_max = temp_y[0]
        sigma_z_max = temp_z[0]

        prefactor = (q * self.conversion_factor * self.one_over_two_pi_three_halves) / (
            sigma_y_max**2 * sigma_z_max
        )
        threshold = math.log(self.exp_threshold_tol / (2 * prefactor))
        thresh_constant = math.sqrt(-2 * threshold)

        self.thresh_xy_max = sigma_y_max * thresh_constant
        self.thresh_z_max = sigma_z_max * thresh_constant

        t = self.calculatePlumeTravelTime(self.thresh_xy_max, ws)
        n_time_steps = math.ceil(t / self.sim_dt)
        if n_time_steps >= ch4.shape[0]:
            n_time_steps = ch4.shape[0] - 1

        sigma_y, sigma_z = self.getPuffCenteredSigmas(n_time_steps, ws, stability_class)

        shift_per_step = ws * self.sim_dt
        x_shift_per_step = self.cosine * shift_per_step
        y_shift_per_step = -self.sine * shift_per_step
        wind_shift = 0
        x_shift = 0
        y_shift = 0

        prefactor = q * self.conversion_factor * self.one_over_two_pi_three_halves

        for i in range(1, n_time_steps + 1):
            one_over_sig_y = 1 / sigma_y[i]
            one_over_sig_z = 1 / sigma_z[i]
            local_prefactor = prefactor * one_over_sig_y**2 * one_over_sig_z

            temp_log = math.log(self.exp_threshold_tol / (2 * local_prefactor))
            local_thresh = math.sqrt(-2 * temp_log)

            wind_shift += shift_per_step
            x_shift += x_shift_per_step
            y_shift += y_shift_per_step

            indices = self.coarseSpatialThreshold(
                wind_shift, local_thresh, sigma_y[i], sigma_z[i]
            )

            gaussian_puff_inner_loop(
                ch4,
                indices,
                i,
                sigma_y[i],
                sigma_z[i],
                x_shift,
                y_shift,
                self.z0,
                self.X_centered,
                self.Y_centered,
                self.Z,
                self.exp_threshold_tol,
                local_prefactor,
            )

    def calculatePlumeTravelTime(self, threshXY, windSpeed):
        boxMin = np.array([-threshXY, -threshXY])
        boxMax = np.array([threshXY, threshXY])

        gridMin = np.array([self.x_min, self.y_min])
        gridMax = np.array([self.x_max, self.y_max])

        origin = np.array([0.0, 0.0])
        rayDir = np.array([self.cosine, -self.sine])
        invRayDir = 1.0 / rayDir

        # Step 1: Where the ray intersects the threshold box (backwards)
        boxTimes = self.AABB(boxMin, boxMax, origin, invRayDir)
        backwardCollision = boxTimes[0] * rayDir
        boxCorner = self.findNearestCorner(boxMin, boxMax, backwardCollision)

        # Step 2: Where the ray intersects the grid box (forwards)
        gridMiddle = 0.5 * (gridMax - gridMin) + gridMin
        gridTimes = self.AABB(gridMin, gridMax, gridMiddle, invRayDir)
        forwardCollision = gridTimes[1] * rayDir + gridMiddle
        gridCorner = self.findNearestCorner(gridMin, gridMax, forwardCollision)

        # Step 3: Compute travel distance and time
        distance = np.abs(gridCorner - boxCorner)
        invRayDirAbs = np.abs(invRayDir)
        travelDistance = np.min(distance * invRayDirAbs)
        travelTime = travelDistance / windSpeed

        return travelTime

    def findNearestCorner(self, minCorner, maxCorner, point):
        corner = np.empty(2)

        corner[0] = (
            minCorner[0]
            if abs(minCorner[0] - point[0]) < abs(maxCorner[0] - point[0])
            else maxCorner[0]
        )
        corner[1] = (
            minCorner[1]
            if abs(minCorner[1] - point[1]) < abs(maxCorner[1] - point[1])
            else maxCorner[1]
        )

        return corner

    def calculateExitLocation(self):
        box_min = np.array([self.x_min, self.y_min])
        box_max = np.array([self.x_max, self.y_max])
        origin = np.array([0.0, 0.0])
        ray_dir = np.array([self.cosine, -self.sine])
        inv_ray_dir = 1.0 / ray_dir

        exit_times = self.AABB(box_min, box_max, origin, inv_ray_dir)

        return exit_times[1] * ray_dir

    def AABB(self, boxMin, boxMax, origin, invRayDir):
        t0 = (boxMin - origin) * invRayDir
        t1 = (boxMax - origin) * invRayDir

        tmax = np.maximum(t0, t1).min()
        tmin = np.minimum(t0, t1).max()

        return np.array([tmin, tmax])

    def getSigmaCoefficients(self, stability_class, downwind_dists, n_time_steps):
        sigmaY = np.zeros(n_time_steps + 1)
        sigmaZ = np.zeros(n_time_steps + 1)

        nStab = len(stability_class)

        for i in range(len(downwind_dists)):
            x = downwind_dists[i] * 0.001  # km to m

            if x <= 0:
                sigmaY[i] = -1
                sigmaZ[i] = -1
                continue

            sigmaYTempTotal = 0.0
            sigmaZTempTotal = 0.0

            for stab in stability_class:
                flag = 0
                a = b = c = d = 0.0

                if stab == "A":
                    if x < 0.1:
                        a, b = 122.800, 0.94470
                    elif x < 0.15:
                        a, b = 158.080, 1.05420
                    elif x < 0.20:
                        a, b = 170.220, 1.09320
                    elif x < 0.25:
                        a, b = 179.520, 1.12620
                    elif x < 0.30:
                        a, b = 217.410, 1.26440
                    elif x < 0.40:
                        a, b = 258.890, 1.40940
                    elif x < 0.50:
                        a, b = 346.750, 1.72830
                    elif x < 3.11:
                        a, b = 453.850, 2.11660
                    else:
                        flag = 1
                    c, d = 24.1670, 2.5334

                elif stab == "B":
                    if x < 0.2:
                        a, b = 90.673, 0.93198
                    elif x < 0.4:
                        a, b = 98.483, 0.98332
                    else:
                        a, b = 109.300, 1.09710
                    c, d = 18.3330, 1.8096

                elif stab == "C":
                    a, b = 61.141, 0.91465
                    c, d = 12.5000, 1.0857

                elif stab == "D":
                    if x < 0.3:
                        a, b = 34.459, 0.86974
                    elif x < 1:
                        a, b = 32.093, 0.81066
                    elif x < 3:
                        a, b = 32.093, 0.64403
                    elif x < 10:
                        a, b = 33.504, 0.60486
                    elif x < 30:
                        a, b = 36.650, 0.56589
                    else:
                        a, b = 44.053, 0.51179
                    c, d = 8.3330, 0.72382

                elif stab == "E":
                    if x < 0.1:
                        a, b = 24.260, 0.83660
                    elif x < 0.3:
                        a, b = 23.331, 0.81956
                    elif x < 1:
                        a, b = 21.628, 0.75660
                    elif x < 2:
                        a, b = 21.628, 0.63077
                    elif x < 4:
                        a, b = 22.534, 0.57154
                    elif x < 10:
                        a, b = 24.703, 0.50527
                    elif x < 20:
                        a, b = 26.970, 0.46173
                    elif x < 40:
                        a, b = 35.420, 0.37615
                    else:
                        a, b = 47.618, 0.29592
                    c, d = 6.2500, 0.54287

                elif stab == "F":
                    if x < 0.2:
                        a, b = 15.209, 0.81558
                    elif x < 0.7:
                        a, b = 14.457, 0.78407
                    elif x < 1:
                        a, b = 13.953, 0.68465
                    elif x < 2:
                        a, b = 13.953, 0.63227
                    elif x < 3:
                        a, b = 14.823, 0.54503
                    elif x < 7:
                        a, b = 16.187, 0.46490
                    elif x < 15:
                        a, b = 17.836, 0.41507
                    elif x < 30:
                        a, b = 22.651, 0.32681
                    elif x < 60:
                        a, b = 27.074, 0.27436
                    else:
                        a, b = 34.219, 0.21716
                    c, d = 4.1667, 0.36191

                else:
                    raise ValueError(f"Invalid stability class: {stab}")

                theta = 0.017453293 * (c - d * np.log(x))  # in radians
                sigmaYTemp = 465.11628 * x * np.tan(theta)

                if flag == 0:
                    sigmaZTemp = a * (x**b)
                    sigmaZTemp = min(sigmaZTemp, 5000.0)
                else:
                    sigmaZTemp = 5000.0

                sigmaYTempTotal += sigmaYTemp
                sigmaZTempTotal += sigmaZTemp

            sigmaY[i] = sigmaYTempTotal / nStab
            sigmaZ[i] = sigmaZTempTotal / nStab

        return sigmaY, sigmaZ

    def getPuffCenteredSigmas(self, n_time_steps, wind_speed, stability_class):
        downwind_dists = np.zeros(n_time_steps + 1)
        shift_per_step = wind_speed * self.sim_dt

        for i in range(1, n_time_steps + 1):
            downwind_dists[i] = downwind_dists[i - 1] + shift_per_step

        sigma_y, sigma_z = self.getSigmaCoefficients(stability_class, downwind_dists, n_time_steps)

        return sigma_y, sigma_z

    @abstractmethod
    def coarseSpatialThreshold(self, wind_shift, local_thresh, sigma_y_i, sigma_z_i):
        return np.array([], dtype=np.int32)

    def exp(self, val):
        return math.exp(val)

    def _resample_simulation(self, c_matrix, resample_dt):

        df = pd.DataFrame(c_matrix, index=self.time_stamps_sim)
        df = df.resample(str(resample_dt) + "s").mean()
        c_matrix_res = df.to_numpy()

        self.n_out = np.shape(c_matrix_res)[0]

        return c_matrix_res


class GridMode(GaussianPuff):

    def __init__(
        self,
        obs_dt,
        sim_dt,
        puff_dt,
        simulation_start,
        simulation_end,
        time_zone,
        source_coordinates,
        emission_rates,
        wind_speeds,
        wind_directions,
        grid_coordinates,
        nx,
        ny,
        nz,
        output_dt=None,
        puff_duration=1200,
        skip_low_wind=False,
        low_wind_cutoff=-1,
        exp_threshold_tolerance=None,
        conversion_factor=1e6 * 1.524,
        unsafe=False,
        quiet=True,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.N_points = self.nx * self.ny * self.nz

        result = ih._check_array_dtypes(grid_coordinates=grid_coordinates)
        grid_coordinates = result["grid_coordinates"]

        x_min = grid_coordinates[0]
        y_min = grid_coordinates[1]
        z_min = grid_coordinates[2]
        x_max = grid_coordinates[3]
        y_max = grid_coordinates[4]
        z_max = grid_coordinates[5]

        x, y, z = (
            np.linspace(x_min, x_max, self.nx),
            np.linspace(y_min, y_max, self.ny),
            np.linspace(z_min, z_max, self.nz),
        )

        X, Y, Z = np.meshgrid(x, y, z)  # x-y-z grid across site in utm
        # self.grid_dims = np.shape(self.X)

        # work with the flattened grids
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()

        self.computeGridSpacing(X,Y,Z)

        self.map_table = np.empty((self.ny, self.nx, self.nz), dtype=int)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.map_table[j][i][k] = self.map(i, j, k)

        super().__init__(
            obs_dt=obs_dt,
            sim_dt=sim_dt,
            puff_dt=puff_dt,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            time_zone=time_zone,
            X=X,
            Y=Y,
            Z=Z,
            source_coordinates=source_coordinates,
            emission_rates=emission_rates,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            output_dt=output_dt,
            puff_duration=puff_duration,
            skip_low_wind=skip_low_wind,
            low_wind_cutoff=low_wind_cutoff,
            exp_threshold_tolerance=exp_threshold_tolerance,
            conversion_factor=conversion_factor,
            unsafe=unsafe,
            quiet=quiet,
        )

    def map(self, i, j, k):
        return j * self.nz * self.nx + i * self.nz + k

    # TODO check the arguments to the function
    def coarseSpatialThreshold(self, wind_shift, local_thresh, sigma_y_i, sigma_z_i):
        indices = self.getValidIndices(
            self.thresh_xy_max, self.thresh_z_max, wind_shift
        )
        return indices.astype(np.int32)

    def getValidIndices(self, thresh_xy, thresh_z, wind_shift):
        index_bounds = self.computeIndexBounds(thresh_xy, thresh_z, wind_shift)

        i_lower = math.floor(index_bounds[0])
        i_upper = math.ceil(index_bounds[1])
        j_lower = math.floor(index_bounds[2])
        j_upper = math.ceil(index_bounds[3])
        k_lower = math.floor(index_bounds[4])
        k_upper = math.ceil(index_bounds[5])

        # Clamp index ranges to grid dimensions
        i_lower = max(i_lower, 0)
        i_upper = min(i_upper, self.nx - 1)

        j_lower = max(j_lower, 0)
        j_upper = min(j_upper, self.ny - 1)

        k_lower = max(k_lower, 0)
        k_upper = min(k_upper, self.nz - 1)

        if i_upper < i_lower or j_upper < j_lower or k_upper < k_lower:
            return np.array([])

        # Slice the map_table to get the sub-block
        sub_block = self.map_table[
            j_lower : j_upper + 1, i_lower : i_upper + 1, k_lower : k_upper + 1
        ]

        # Flatten it into 1D array of indices
        indices = sub_block.ravel()
        return indices

    def computeIndexBounds(self, thresh_xy, thresh_z, wind_shift):
        R = np.array([[self.cosine, -self.sine], [self.sine, self.cosine]])

        X0 = np.array([self.x_min, self.y_min])

        v = R[:, 0]
        vp = R[:, 1]

        tw = np.array([wind_shift, 0])

        X0_r = R @ X0
        X0_rt = X0_r - tw

        Xrt_dot_v = np.dot(X0_rt, v)
        Xrt_dot_vp = np.dot(X0_rt, vp)

        one_over_dx = 1 / self.dx
        one_over_dy = 1 / self.dy
        one_over_dz = 1 / self.dz

        i_lower = (-Xrt_dot_v - thresh_xy - 1) * one_over_dx
        i_upper = (-Xrt_dot_v + thresh_xy + 1) * one_over_dx

        j_lower = (-Xrt_dot_vp - thresh_xy - 1) * one_over_dy
        j_upper = (-Xrt_dot_vp + thresh_xy + 1) * one_over_dy

        k_lower = (-thresh_z + self.z0) * one_over_dz
        k_upper = (thresh_z + self.z0) * one_over_dz

        return [i_lower, i_upper, j_lower, j_upper, k_lower, k_upper]

    def computeGridSpacing(self,X,Y,Z):
        self.dx = abs(X[self.nz] - X[0])  # dx
        self.dy = abs(Y[self.nz * self.nx] - Y[0])  # dy
        self.dz = abs(Z[1] - Z[0])  # dz


class SensorMode(GaussianPuff):

    def __init__(
        self,
        obs_dt,
        sim_dt,
        puff_dt,
        simulation_start,
        simulation_end,
        time_zone,
        source_coordinates,
        emission_rates,
        wind_speeds,
        wind_directions,
        sensor_coordinates,
        output_dt=None,
        puff_duration=1200,
        skip_low_wind=False,
        low_wind_cutoff=-1,
        exp_threshold_tolerance=None,
        conversion_factor=1e6 * 1.524,
        unsafe=False,
        quiet=True,
    ):
        self.using_sensors = True
        N_points = len(sensor_coordinates)

        result = ih._check_array_dtypes(sensor_coordinates=sensor_coordinates)
        sensor_coordinates = result["sensor_coordinates"]

        X, Y, Z = (
            np.empty(N_points),
            np.empty(N_points),
            np.empty(N_points),
        )
        # for sensor in sensor_coordinates:
        for i, sensor in enumerate(sensor_coordinates):
            X[i] = sensor[0]
            Y[i] = sensor[1]
            Z[i] = sensor[2]

        self.indices = np.arange(len(sensor_coordinates), dtype=np.int32)

        super().__init__(
            obs_dt=obs_dt,
            sim_dt=sim_dt,
            puff_dt=puff_dt,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            time_zone=time_zone,
            X=X,
            Y=Y,
            Z=Z,
            source_coordinates=source_coordinates,
            emission_rates=emission_rates,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            output_dt=output_dt,
            puff_duration=puff_duration,
            skip_low_wind=skip_low_wind,
            low_wind_cutoff=low_wind_cutoff,
            exp_threshold_tolerance=exp_threshold_tolerance,
            conversion_factor=conversion_factor,
            unsafe=unsafe,
            quiet=quiet,
        )

    # mostly a stub- could be used to implement a more efficient version
    def coarseSpatialThreshold(self, wind_shift, local_thresh, sigma_y_i, sigma_z_i):
        return self.indices


@njit(parallel=True)
def gaussian_puff_inner_loop(
    ch4,
    indices,
    i,
    sigma_y_i,
    sigma_z_i,
    x_shift,
    y_shift,
    z0,
    X_centered,
    Y_centered,
    Z,
    exp_threshold_tol,
    local_prefactor,
):
    one_over_sig_y = 1.0 / sigma_y_i
    one_over_sig_z = 1.0 / sigma_z_i
    one_over_sig_y_sq = one_over_sig_y**2

    local_thresh = np.sqrt(-2 * np.log(exp_threshold_tol / (2 * local_prefactor)))

    for j in prange(len(indices)):
        idx = indices[j]

        if sigma_y_i <= 0 or sigma_z_i <= 0:
            continue

        t_xy = sigma_y_i * local_thresh

        if abs(X_centered[idx] - x_shift) >= t_xy:
            continue
        if abs(Y_centered[idx] - y_shift) >= t_xy:
            continue

        t_z = sigma_z_i * local_thresh
        if abs(Z[idx] - z0) >= t_z:
            continue

        y_dist = Y_centered[idx] - y_shift
        x_dist = X_centered[idx] - x_shift
        z_minus_by_sig = (Z[idx] - z0) * one_over_sig_z
        z_plus_by_sig = (Z[idx] + z0) * one_over_sig_z

        term_3_arg = (y_dist**2 + x_dist**2) * one_over_sig_y_sq
        term_4_a_arg = z_minus_by_sig**2
        term_4_b_arg = z_plus_by_sig**2

        term_4 = np.exp(-0.5 * (term_3_arg + term_4_a_arg)) + np.exp(
            -0.5 * (term_3_arg + term_4_b_arg)
        )

        ch4[i, idx] += local_prefactor * term_4
