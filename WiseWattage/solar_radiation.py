"""Import functionality required to calculate radiation values"""
from numpy import radians, degrees, cos, sin, arccos, pi
import numpy as np


def calc_declination(n_day):
    """Calculates the solar declination angle for a given day of the year.
    Args:
        n_day (int): Day of the year (1 through 365 or 366).
    Returns:
        float: Solar declination angle in degrees.
    """
    return 23.45 * sin(radians((360 / 365) * (284 + n_day)))


def calc_time_correction(n_day):
    """Calculates the equation of time correction factor.
    Args:
        n_day (int): Day of the year.
    Returns:
        float: Time correction factor in minutes.
    """
    B = radians(360 * (n_day - 1) / 365)
    return 3.82 * (
        0.000075
        + 0.001868 * cos(B)
        - 0.032077 * sin(B)
        - 0.014615 * cos(2 * B)
        - 0.04089 * sin(2 * B)
    )


def calc_solar_time(n_day, civil_time, longitude, timestep=60, tmz_hrs_east=0):
    """Calculates the solar time at a given location and time.
    Args:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        longitude (float): Longitude of the location.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Solar time in hours.
    """
    time_correction = calc_time_correction(n_day)
    return (
        (civil_time + ((timestep / 60) / 2))
        + (longitude / 15)
        - tmz_hrs_east
        + time_correction
    )


def calc_hour_angle(n_day, civil_time, longitude, timestep=60, tmz_hrs_east=0):
    """Calculates the solar hour angle at a given time and location.
    Args:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        longitude (float): Longitude of the location.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Hour angle in degrees.
    """
    solar_time = calc_solar_time(n_day, civil_time, longitude, timestep, tmz_hrs_east)

    return (solar_time - 12) * 15


def calc_aoi(
    n_day,
    civil_time,
    latitude,
    longitude,
    surface_azimuth,
    surface_pitch,
    timestep=60,
    tmz_hrs_east=0,
):
    """Calculates the angle of incidence of solar radiation on a given surface.
    Args:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from the horizontal.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Angle of incidence in degrees.
    """
    hour_angle_rad = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    declination_rad = radians(calc_declination(n_day))

    latitude_rad = radians(latitude)
    surface_pitch_rad = radians(surface_pitch)
    surface_azimuth_rad = radians(surface_azimuth)

    aoi = arccos(
        (sin(declination_rad) * sin(latitude_rad) * cos(surface_pitch_rad))
        - (
            sin(declination_rad)
            * cos(latitude_rad)
            * sin(surface_pitch_rad)
            * cos(surface_azimuth_rad)
        )
        + (
            cos(declination_rad)
            * cos(latitude_rad)
            * cos(surface_pitch_rad)
            * cos(hour_angle_rad)
        )
        + (
            cos(declination_rad)
            * sin(latitude_rad)
            * sin(surface_pitch_rad)
            * cos(surface_azimuth_rad)
            * cos(hour_angle_rad)
        )
        + (
            cos(declination_rad)
            * sin(surface_pitch_rad)
            * sin(surface_azimuth_rad)
            * sin(hour_angle_rad)
        )
    )

    return degrees(aoi)


def calc_zenith(latitude, longitude, n_day, civil_time, timestep=60, tmz_hrs_east=0):
    """Calculates the solar zenith angle at a given time and location.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Zenith angle in degrees.
    """
    latitude_rad = radians(latitude)
    declination_rad = radians(calc_declination(n_day))
    hour_angle_rad = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    return degrees(
        arccos(
            (cos(latitude_rad) * cos(declination_rad) * cos(hour_angle_rad))
            + (sin(latitude_rad) * sin(declination_rad))
        )
    )


def calc_et_normal_radiation(n_day):
    """Calculates the extraterrestrial normal radiation for a given day of the year.
    Args:
        n_day (int): Day of the year.
    Returns:
        float: Extraterrestrial normal radiation in W/m^2.
    """
    solar_constant = 1367
    return solar_constant * (1 + 0.033 * cos(radians((360 * n_day) / 365)))


def calc_et_horizontal_radiation(
    latitude, longitude, n_day, civil_time, timestep=60, tmz_hrs_east=0
):
    """Calculates the extraterrestrial horizontal radiation over a specified timestep.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours at the beginning of the timestep.
        timestep (int, optional): Duration of the timestep in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Extraterrestrial horizontal radiation in kW/m^2 for the timestep.
    """
    civil_time_2 = civil_time + (timestep / 60)

    declination_rad = radians(calc_declination(n_day))
    latitude_rad = radians(latitude)

    hour_angle_1 = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    hour_angle_2 = radians(
        calc_hour_angle(n_day, civil_time_2, longitude, timestep, tmz_hrs_east)
    )

    et_horizontal_radiation = (
        (12 / pi * calc_et_normal_radiation(n_day))
        * (
            (
                cos(latitude_rad)
                * cos(declination_rad)
                * (sin(hour_angle_2) - sin(hour_angle_1))
                + (
                    (hour_angle_2 - hour_angle_1)
                    * sin(latitude_rad)
                    * sin(declination_rad)
                )
            )
        )
    ) * (60 / timestep)

    et_horizontal_radiation = et_horizontal_radiation / 1000  # Convert to kW/m^2
    et_horizontal_radiation = np.where(
        et_horizontal_radiation > 0, et_horizontal_radiation, 0
    )  # Ensure non-negative values

    return et_horizontal_radiation


def calc_beam_radiation(
    dni,
    n_day,
    civil_time,
    latitude,
    longitude,
    surface_azimuth,
    surface_pitch,
    timestep=60,
    tmz_hrs_east=0,
):
    """Calculates the beam component of solar radiation on a tilted surface.
    Args:
        dni (float): Direct normal irradiance in W/m^2.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Beam irradiance on the surface in kW/m^2.
    """
    aoi_rad = radians(
        calc_aoi(
            n_day,
            civil_time,
            latitude,
            longitude,
            surface_azimuth,
            surface_pitch,
            timestep,
            tmz_hrs_east,
        )
    )

    # Calculate beam irradiance and ensure it is not calculated for angles > 85 degrees
    e_beam = dni * cos(aoi_rad) / 1000  # Convert to kW/m^2
    e_beam = np.where(degrees(aoi_rad) > 85, 0, e_beam)
    e_beam = np.where(e_beam < 0, 0, e_beam)  # Ensure non-negative values

    return e_beam


def calc_diffuse_radiation(
    dhi,
    ghi,
    surface_pitch,
    latitude,
    longitude,
    n_day,
    civil_time,
    timestep=60,
    tmz_hrs_east=0,
):
    """Calculates the diffuse component of solar radiation on a tilted surface.
    Args:
        dhi (float): Diffuse horizontal irradiance in W/m^2.
        ghi (float): Global horizontal irradiance in W/m^2.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Diffuse irradiance on the surface in kW/m^2.
    """
    surface_pitch_rad = radians(surface_pitch)
    zenith_rad = radians(
        calc_zenith(latitude, longitude, n_day, civil_time, timestep, tmz_hrs_east)
    )

    e_diffuse = (dhi * ((1 + cos(surface_pitch_rad)) / 2)) + (
        ghi * ((0.12 * zenith_rad) - 0.04) * (1 - cos(surface_pitch_rad)) / 2
    )
    e_diffuse = np.where(
        degrees(zenith_rad) > 85, 0, e_diffuse
    )  # Ensure irradiance is not calculated for zenith angles > 85 degrees
    return e_diffuse / 1000  # Convert to kW/m^2


def calc_ground_radiation(ghi, surface_pitch, albedo=0.2):
    """Calculates the ground-reflected component of solar radiation on a tilted surface.
    Args:
        ghi (float): Global horizontal irradiance in W/m^2.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        albedo (float, optional): Ground reflectance factor. Defaults to 0.2.
    Returns:
        float: Ground-reflected irradiance on the surface in kW/m^2.
    """
    surface_pitch_rad = radians(surface_pitch)
    e_ground = ghi * albedo * ((1 - cos(surface_pitch_rad)) / 2)
    return e_ground / 1000  # Convert to kW/m^2


def calc_poa_radiation(
    dni,
    dhi,
    ghi,
    n_day,
    civil_time,
    latitude,
    longitude,
    surface_azimuth,
    surface_pitch,
    albedo=0.2,
    timestep=60,
    tmz_hrs_east=0,
):
    """Calculates the plane of array (POA) irradiance, considering beam, diffuse, and ground-reflected components.
    Args:
        dni (float): Direct normal irradiance in W/m^2.
        dhi (float): Diffuse horizontal irradiance in W/m^2.
        ghi (float): Global horizontal irradiance in W/m^2.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        albedo (float, optional): Ground reflectance factor. Defaults to 0.2.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.
    Returns:
        float: Total irradiance on the plane of array in kW/m^2.
    """
    # Beam Radiation Calculation
    e_beam = calc_beam_radiation(
        dni,
        n_day,
        civil_time,
        latitude,
        longitude,
        surface_azimuth,
        surface_pitch,
        timestep,
        tmz_hrs_east,
    )

    # Diffuse Radiation Calculation
    e_diffuse = calc_diffuse_radiation(
        dhi,
        ghi,
        surface_pitch,
        latitude,
        longitude,
        n_day,
        civil_time,
        timestep,
        tmz_hrs_east,
    )

    # Ground-reflected Radiation Calculation
    e_ground = calc_ground_radiation(ghi, surface_pitch, albedo)

    e_poa = (
        e_beam + e_diffuse + e_ground
    ) / 1000  # Sum components and convert to kW/m^2

    return e_poa
