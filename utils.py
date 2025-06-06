import numpy as np

def latlon_to_utm_array(latitude, longitude):
    easting = np.zeros(latitude.shape)
    northing = np.zeros(latitude.shape)
    for i in range(len(latitude)):
        if np.isnan(latitude[i]) or np.isnan(longitude[i]):
            continue
        easting[i], northing[i], _, _ = latlon_to_utm(
            latitude[i], longitude[i]
        )
    return easting, northing


def latlon_to_utm(
    latitude, longitude, force_zone_number=None, R=6378137, E=0.00669438
):
    # UTM scale on the central meridian
    K0 = 0.9996
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1.0 - E)
    M1 = 1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256
    M2 = 3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024
    M3 = 15 * E2 / 256 + 45 * E3 / 1024
    M4 = 35 * E3 / 3072
    lat_rad = np.pi * (latitude) / 180
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)
    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(latitude)
    lon_rad = np.pi * (longitude) / 180
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.pi * (central_lon) / 180

    n = R / np.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2
    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a
    m = R * (
        M1 * lat_rad
        - M2 * np.sin(2 * lat_rad)
        + M3 * np.sin(4 * lat_rad)
        - M4 * np.sin(6 * lat_rad)
    )
    easting = (
        K0
        * n
        * (
            a
            + a3 / 6 * (1 - lat_tan2 + c)
            + a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)
        )
        + 500000
    )
    northing = K0 * (
        m
        + n
        * lat_tan
        * (
            a2 / 2
            + a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2)
            + a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)
        )
    )
    return easting, northing, zone_number, zone_letter

def latlon_to_zone_number( latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32
    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37
    return int((longitude + 180) / 6) + 1

def zone_number_to_central_longitude( zone_number):
    return (zone_number - 1) * 6 - 180 + 3

ZONE_LETTERS = ["C", "D", "E", "F", "G", "H", "J",
        "K", "L", "M", "N", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "X"]

def latitude_to_zone_letter(latitude):
    latitude = int(latitude)
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[(latitude + 80) >> 3]
    else:
        return None