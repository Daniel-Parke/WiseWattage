from datetime import datetime
from numpy import radians, degrees, cos, sin, arccos, pi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import httpx

def main():
    """
    latitude = 54.95
    longitude = -5.95
    pv_kwp = 3.2
    surface_azimuth = 83
    surface_pitch = 35
    model = "Full"
    daily_profile = 10
    battery_capacity = 14.336
    import_price_day = 0.2974
    import_price_night = None
    export_tariff = 0.1422

    tmy_data = get_jrc_tmy(latitude, longitude)

    data = generate_model(tmy_data, latitude, longitude, 
                          pv_kwp, surface_azimuth, surface_pitch, model, 
                          daily_profile, battery_capacity, 
                          import_price_day, import_price_night, export_tariff)

    return data
    
    """
    """
    latitude = 54.95
    longitude = -5.95

    tmy_data = get_jrc_tmy(latitude, longitude)
    data = generate_model(tmy_data, 54.95, -5.95, 3.2, 0, 35, "Full", 10, 14.336, 0.2974, None, 0.1422)

    return data
    """
    """

    data = get_jrc_tmy(54.95, -5.95)
    data = generate_model(tmy_data, 54.95, -5.95)
    
    return data
    """

    pass


def get_day(timestamp):
    """ This function parses a timestamp string in the format "YYYYMMDD:HHMM"
    and extracts the day of the month.

    Parameters:
    timestamp (str): A timestamp string in the format "YYYYMMDD:HHMM".

    Returns:
    int: The day of the year extracted from the timestamp.
    """

    day_of_year = datetime.strptime(timestamp, "%Y%m%d:%H%M").timetuple().tm_yday
    return day_of_year


def get_hour(timestamp):
    """ This function parses a timestamp string in the format "YYYYMMDD:HHMM"
    and extracts the hour of the day in 24-hour format.

    Parameters:
    timestamp (str): A timestamp string in the format "YYYYMMDD:HHMM".

    Returns:
    int: The hour of the day (in 24-hour format) extracted from the timestamp.
    """

    hour = datetime.strptime(timestamp, "%Y%m%d:%H%M").hour
    return hour


def get_jrc_tmy(latitude, longitude, start_year=2005, end_year=2016):
    """ Fetches Typical Meteorological Year (TMY) data from the JRC database for a specific location and time range.

    This function retrieves hourly solar and meteorological data for a given latitude and longitude. 
    The data is fetched for the years specified in the range from start_year to end_year.

    Parameters:
    latitude (float): Latitude of the location for which data is to be retrieved.
    longitude (float): Longitude of the location.
    start_year (int, optional): Starting year for the data range. Defaults to 2005.
    end_year (int, optional): Ending year for the data range. Defaults to 2016.

    Returns:
    pandas.DataFrame: A DataFrame containing the TMY data, with each row representing one hour. 
                      The DataFrame includes columns for various meteorological parameters, 
                      and additional columns 'day_of_year' and 'hour_of_day' derived from the index.

    Note:
    - The function assumes the availability of an external 'get_day' and 'get_hour' function 
      for mapping index to day of the year and hour of the day.
    - The data index is formatted to '%Y%m%d:%H%M' and reset to a default range index before returning.
    """
    
    # Send a GET request to the JRC API
    request = httpx.get(f"https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&startyear={start_year}&endyear={end_year}&outputformat=json", timeout = 25)

    # Parse the JSON response
    response = request.json()
    data = pd.DataFrame(response["outputs"]["tmy_hourly"])
    
    # Reset the index with a new date range
    date_range = pd.date_range(start="2023-01-01 00:00:00", periods=8760, freq='H')
    formatted_index = date_range.strftime('%Y%m%d:%H%M')
    data.index = formatted_index

    # Map the index to day of the year and hour of the day. Reset dataframe to remove generated index
    data['day_of_year'] = data.index.map(get_day)
    data['hour_of_day'] = data.index.map(get_hour)
    data.reset_index(drop=True, inplace=True)

    return data

def calc_declination(n_day):
    """ Calculate the solar declination angle for a given day of the year.

    Parameters:
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)

    Returns:
    float: Solar declination angle in degrees
    """

    return 23.45 * sin(radians((360 / 365) * (284+n_day)))


def calc_time_correction(n_day):
    """ Calculate the Equation of Time for a given day of the year.

    Parameters:
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)

    Returns:
    float: Time correction factor
    """

    B = radians(360 * (n_day-1) / 365)
    return 3.82 * (0.000075 + 0.001868*cos(B) - 0.032077*sin(B) - 0.014615*cos(2 * B) - 0.04089*sin(2 * B))


def calc_solar_time(n_day, civil_time, longitude, timestep=60, tmz_hrs_east=0):
    """ Convert civil time to solar time.

    Parameters:
    civil_time (float): Local civil time
    longitude (float): Geographical longitude of the location
    timestep (float): The length of the time step in minutes.
    tmz_hrs_east (float): Time zone offset from UTC in hours (east is positive)

    Returns:
    float: Solar time

    Note:
    This function relies on external functions to perform intermediate calculations:
    - calc_time_correction: to calculate the time correction factor for the solar time.
    """
    time_correction = calc_time_correction(n_day)

    return (civil_time + ((timestep/60)/2)) + (longitude / 15) - tmz_hrs_east + time_correction


def calc_hour_angle(n_day, civil_time, longitude, timestep=60, tmz_hrs_east=0):
    """ Calculate the solar hour angle.

    Parameters:
    civil_time (float): Local civil time
    longitude (float): Geographical longitude of the location
    timestep (float): The length of the time step in minutes.
    tmz_hrs_east (float): Time zone offset from UTC in hours (east is positive)

    Returns:
    float: Solar hour angle in degrees

    Note:
    This function relies on external functions to perform intermediate calculations:
    - calc_time_correction: to calculate the time correction factor for the solar time.
    - calc_solar_time: to convert civil time to solar time, considering longitude and time zone.
    """

    solar_time = calc_solar_time(n_day, civil_time, longitude, timestep=60, tmz_hrs_east=0)

    return (solar_time - 12) * 15


def calc_aoi(n_day, civil_time, latitude, longitude, 
             surface_azimuth, surface_pitch, timestep=60, tmz_hrs_east=0):
    """ Calculate the Angle of Incidence (AOI) of sunlight on a surface.

    Parameters:
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)
    civil_time (float): Local civil time
    timestep (float): The length of the time step in minutes.
    tmz_hrs_east (float): Time zone offset from UTC in hours (east is positive)
    latitude (float): Geographical latitude of the location
    longitude (float): Geographical longitude of the location
    surface_azimuth (float): Azimuth angle of the surface (degrees from north)
    surface_pitch (float): Pitch angle of the surface (degrees from horizontal)

    Returns:
    float: Angle of Incidence (AOI) in degrees

    Note:
    This function relies on external functions to perform intermediate calculations:
    - calc_declination: to calculate the solar declination angle based on the day of the year.
    - calc_time_correction: to calculate the time correction factor for the solar time.
    - calc_solar_time: to convert civil time to solar time, considering longitude and time zone.
    - calc_hour_angle: to calculate the solar hour angle at the given solar time.
    """

    hour_angle_rad = radians(calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east))
    declination_rad = radians(calc_declination(n_day))

    latitude_rad = radians(latitude)
    surface_pitch_rad = radians(surface_pitch)
    surface_azimuth_rad = radians(surface_azimuth)

    aoi = arccos((sin(declination_rad) * sin(latitude_rad) * cos(surface_pitch_rad))\
        - (sin(declination_rad) * cos(latitude_rad) * sin(surface_pitch_rad) * cos(surface_azimuth_rad))\
        + (cos(declination_rad) * cos(latitude_rad) * cos(surface_pitch_rad) * cos(hour_angle_rad))\
        + (cos(declination_rad) * sin(latitude_rad) * sin(surface_pitch_rad) * cos(surface_azimuth_rad) * cos(hour_angle_rad))\
        + (cos(declination_rad) * sin(surface_pitch_rad) * sin(surface_azimuth_rad) * sin(hour_angle_rad)))

    return degrees(aoi)


def calc_zenith(latitude, longitude, n_day, civil_time, timestep=60, tmz_hrs_east=0):
    """ Calculate the zenith angle for a given location and time.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    longitude (float): Longitude of the location in degrees.
    n_day (int): Day of the year, used for calculating the solar declination.
    civil_time (float): Local civil time at the location.
    timestep (float): The length of the time step in minutes.
    tmz_hrs_east (float): Time zone offset from UTC, with east being positive.

    Returns:
    float: The zenith angle in degrees.

    Note:
    This function relies on external functions to perform intermediate calculations:
    - calc_declination: to calculate the solar declination angle based on the day of the year.
    - calc_time_correction: to calculate the time correction factor for the solar time.
    - calc_solar_time: to convert civil time to solar time, considering longitude and time zone.
    - calc_hour_angle: to calculate the solar hour angle at the given solar time.
    """

    latitude_rad = radians(latitude)

    declination_rad = radians(calc_declination(n_day))
    hour_angle_rad = radians(calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east))

    return degrees(arccos((cos(latitude_rad) * cos(declination_rad) * cos(hour_angle_rad)) + (sin(latitude_rad) * sin(declination_rad))))


def calc_et_normal_radiation(n_day):
    """ Calculate the extraterrestrial normal radiation for a given day of the year.

    Parameters:
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)

    Returns:
    float: The extraterrestrial normal radiation (G_on) in W/m^2
    """

    solar_constant = 1367
    return solar_constant * (1 + 0.033 * cos(radians((360 * n_day) / 365)))


def calc_et_horizontal_radiation(latitude, longitude, n_day, civil_time, 
                                 timestep=60, tmz_hrs_east=0):
    """ Calculate the average extraterrestrial horizontal radiation received on a horizontal surface at the top of the atmosphere over timestep.

    Parameters:
    latitude (float): The latitude of the location in degrees.
    longitude (float): The longitude of the location in degrees.
    n_day (int): The day of the year (1 through 365 or 366 for a leap year).
    civil_time (float): The local civil time at the location in hours.
    timestep (float): The length of the time step in minutes.
    tmz_hrs_east (float): The time zone offset from UTC in hours (east is positive).

    Returns:
    float: The extraterrestrial horizontal radiation in W/m^2 averaged over the timestep. Minimum return value set to 0

    Note:
    This function relies on external functions to perform intermediate calculations:
    - calc_declination: to calculate the solar declination angle based on the day of the year.
    - calc_time_correction: to calculate the time correction factor for the solar time.
    - calc_solar_time: to convert civil time to solar time, considering longitude and time zone.
    - calc_hour_angle: to calculate the solar hour angle at the given solar time.
    - calc_et_normal_radiation: to calculate the extra terrestrial normal solar radiation.
    """

    civil_time_2 = civil_time + (timestep / 60)

    declination_rad = radians(calc_declination(n_day))
    latitude_rad = radians(latitude)

    hour_angle_1 = radians(calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east))
    hour_angle_2 = radians(calc_hour_angle(n_day, civil_time_2, longitude, timestep, tmz_hrs_east))

    et_horizontal_radiation = ((12/pi*calc_et_normal_radiation(n_day)) * \
        ((cos(latitude_rad)*cos(declination_rad)*(sin(hour_angle_2) - sin(hour_angle_1)) + \
         ((hour_angle_2-hour_angle_1)*sin(latitude_rad)*sin(declination_rad))))) * (60 / timestep)

    et_horizontal_radiation = np.where(et_horizontal_radiation > 0, et_horizontal_radiation, 0)

    return et_horizontal_radiation
    

def calc_beam_radiation(dni, n_day, civil_time, latitude, longitude, 
                        surface_azimuth, surface_pitch, timestep=60, tmz_hrs_east=0):
    """ Calculate the beam radiation on a surface.

    Parameters:
    dni (float): Direct Normal Irradiance (W/m^2)
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)
    civil_time (float): Local civil time
    latitude (float): Geographical latitude of the location
    longitude (float): Geographical longitude of the location
    surface_azimuth (float): Azimuth angle of the surface (degrees from north)
    surface_pitch (float): Pitch angle of the surface (degrees from horizontal)
    timestep (float): The length of the time step in minutes
    tmz_hrs_east (float): Time zone offset from UTC in hours (east is positive)

    Returns:
    float: Beam radiation (W/m^2) incident on the surface

    Note:
    This function calculates the beam radiation based on the angle of incidence, 
    which is determined by various factors including the day of the year, 
    time of day, geographic location, and orientation of the surface.
    """

    aoi_rad = radians(calc_aoi(n_day, civil_time, latitude, longitude, surface_azimuth, 
                               surface_pitch, timestep, tmz_hrs_east))
    
    e_beam = dni * cos(aoi_rad)

    # Use numpy.where for vectorized conditional operation
    e_beam = np.where(e_beam < 0, 0, e_beam)

    return e_beam


def calc_diffuse_radiation(dhi, ghi, surface_pitch, latitude, longitude, n_day, civil_time, timestep=60, tmz_hrs_east=0):
    """ Calculate the diffuse radiation on a surface.

    Parameters:
    dhi (float): Diffuse Horizontal Irradiance (W/m^2)
    ghi (float): Global Horizontal Irradiance (W/m^2)
    surface_pitch (float): Pitch angle of the surface (degrees from horizontal)
    latitude (float): Geographical latitude of the location
    longitude (float): Geographical longitude of the location
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year)
    civil_time (float): Local civil time
    timestep (float): The length of the time step in minutes
    tmz_hrs_east (float): Time zone offset from UTC in hours (east is positive)

    Returns:
    float: Diffuse radiation (W/m^2) incident on the surface

    Note:
    This function calculates the diffuse radiation on a surface considering 
    both the sky radiation and the ground-reflected radiation components.
    """

    surface_pitch_rad = radians(surface_pitch)
    zenith_rad = radians(calc_zenith(latitude, longitude, n_day, civil_time, timestep, tmz_hrs_east))

    e_diffuse = (dhi * ((1 + cos(surface_pitch_rad)) / 2)) + (ghi * ((0.12 * zenith_rad) - 0.04) * (1 -cos(surface_pitch_rad)) / 2)
    return e_diffuse



def calc_ground_radiation(ghi, surface_pitch, albedo=0.2):
    """ Calculate the ground-reflected radiation on a surface.

    Parameters:
    ghi (float): Global Horizontal Irradiance (W/m^2)
    albedo (float): Albedo (reflection coefficient) of the ground
    surface_pitch (float): Pitch angle of the surface (degrees from horizontal)

    Returns:
    float: Ground-reflected radiation (W/m^2) incident on the surface

    Note:
    This function calculates the amount of solar radiation reflected from the ground 
    onto a tilted surface, based on the global horizontal irradiance, the ground albedo, 
    and the tilt of the surface.
    """

    surface_pitch_rad = radians(surface_pitch)
    e_ground = ghi * albedo * ((1 - cos(surface_pitch_rad)) / 2)
    return e_ground


def calc_poa_radiation(dni, dhi, ghi, n_day, civil_time, latitude, longitude, surface_azimuth, 
                       surface_pitch, albedo=0.2, timestep=60, tmz_hrs_east=0):
    """ Calculate the plane of array (POA) radiation for a solar panel.

    Parameters:
    dni (float): Direct Normal Irradiance in W/m^2.
    dhi (float): Diffuse Horizontal Irradiance in W/m^2.
    ghi (float): Global Horizontal Irradiance in W/m^2.
    n_day (int): Day of the year (number between 1 and 365 or 366 in a leap year).
    civil_time (float): Local civil time.
    latitude (float): Geographical latitude of the location.
    longitude (float): Geographical longitude of the location.
    surface_azimuth (float): Azimuth angle of the surface (degrees from north).
    surface_pitch (float): Pitch angle of the surface (degrees from horizontal).
    albedo (float, optional): Ground reflectance, default is 0.2.
    timestep (float, optional): Time step in minutes, default is 60.
    tmz_hrs_east (float, optional): Time zone offset from UTC in hours (east is positive), default is 0.

    Returns:
    float: Total plane of array radiation in W/m^2.

    Notes:
    - The function calculates the beam, diffuse, and ground-reflected components of solar radiation.
    - It assumes the availability of 'calc_aoi' and 'calc_zenith' functions for angle of incidence and zenith angle calculations.
    """
    
    # Calculate Beam Radiation
    aoi_rad = radians(calc_aoi(n_day, civil_time, latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east))
    e_beam = dni * cos(aoi_rad)
    e_beam = np.where(e_beam < 0, 0, e_beam)

    # Calculate Diffuse Radiation
    surface_pitch_rad = radians(surface_pitch)
    zenith_rad = radians(calc_zenith(latitude, longitude, n_day, civil_time, timestep, tmz_hrs_east))
    e_diffuse = (dhi * ((1 + cos(surface_pitch_rad)) / 2)) + (ghi * ((0.12 * zenith_rad) - 0.04) * (1 - cos(surface_pitch_rad)) / 2)

    # Calculate Ground Reflectance Radiation
    e_ground = ghi * albedo * ((1 - cos(surface_pitch_rad)) / 2)

    return e_beam + e_diffuse + e_ground


def calc_cell_temp(e_poa = 900, ambient_temp = 20, cell_temp_coeff = -0.0048, electrical_eff = 0.22,
                   cell_NOCT = 42, ambient_NOCT = 20, e_poa_NOCT = 800, cell_temp_STC = 25, transmittance_absorptance = 0.9):
    
    """ Calculate the temperature of a solar cell.
    
    Parameters:
    solar_radiation (float): Current solar radiation as kW/m2.                                                      (default = NONE)
    ambient_temp (float): Ambient temperature in degrees celcius.                                                   (default = NONE)
    cell_temp_coeff (float): Temperature coefficient of the cell as %/celcius.                                      (default = -0.0048)
    electrical_eff (float): Electrical efficiency of the cell as a decimal.                                         (default = 0.21)
    cell_NOCT (float): Cell temperature at Nominal Operating Cell Temperature conditions in degrees celcius.        (default = 42)
    ambient_NOCT (float): Ambient temperature at NOCT in degrees celcius.                                           (default = 20)
    solar_radiation_NOCT (float): Solar radiation at NOCT as kW/m2.                                                 (default = 0.8)
    cell_temp_STC (float): Cell temperature at Standard Test Conditions conditions in degrees celcius.              (default = 25)
    transmittance_absorptance (float): Combined transmittance and absorptance of the cell as %.                     (default = 0.9)

    Returns:
    float: Calculated cell temperature in degrees celcius.
    """

    temp_factor = ((cell_NOCT - ambient_NOCT) * (e_poa / e_poa_NOCT))
    numerator = ambient_temp + temp_factor * (1 - (electrical_eff * (1 - cell_temp_coeff * cell_temp_STC)) / transmittance_absorptance)
    denominator = 1 + temp_factor * (cell_temp_coeff * electrical_eff / transmittance_absorptance)

    return numerator / denominator


def calc_pv_power(pv_kwp, e_poa, ambient_temp, pv_derating=1,
                  cell_temp_coeff=-0.0048, e_poa_STC=1000, cell_temp_STC=25):
    """ Calculate the power output of a photovoltaic (PV) system.

    Parameters:
    pv_kwp (float): The rated capacity of the PV system in kilowatts peak (kWp)
    pv_derating (float): Derating factor accounting for losses in system efficiency (e.g., due to wiring, inverter efficiency, etc.)
    e_poa (float): The plane of array irradiance in watts per square meter (W/m^2)
    cell_temp (float): The operating cell temperature in degrees Celsius
    cell_temp_coeff (float): The temperature coefficient of power (default: -0.0048 per degree Celsius)
    e_poa_STC (float): The irradiance at Standard Test Conditions in watts per square meter (default: 1000 kW/m^2)
    cell_temp_STC (float): The cell temperature at Standard Test Conditions in degrees Celsius (default: 25°C)

    Returns:
    float: The power output of the PV system in kilowatts (W)

    Note:
    This function estimates the power output of a solar PV system based on its rated capacity, 
    environmental factors (irradiance and cell temperature), and efficiency losses.
    The formula adjusts the power output based on the actual irradiance and cell temperature, 
    compared to the Standard Test Conditions (STC) which are 1000 W/m^2 and 25°C.
    """

    cell_temp = calc_cell_temp(e_poa, ambient_temp, cell_temp_coeff)

    pv_power = pv_kwp * pv_derating * (e_poa / e_poa_STC) * (1 + cell_temp_coeff * (cell_temp - cell_temp_STC))

    return pv_power


def add_load_profile(data, daily_profile=10, path=r"C:\Users\djp12\Documents\DFE_Data\Solar_Model\load_profile_10kWh_8760.csv"):
    """ This function adds a load profile to an existing DataFrame and calculates the net power demand.

    Parameters:
    - data (DataFrame): The existing DataFrame to which the load profile will be added.
    - daily_profile (float, optional): The daily energy usage profile to scale the load profile. Default is 10.
    - path (str, optional): The file path to the load profile CSV. Default is a specified path.

    Returns:
    - DataFrame: The original DataFrame with additional columns for energy use and net power demand.
    """

    load_profile = pd.read_csv(path)

    # Scale the energy use values in the load profile based on the provided daily profile
    load_profile["Energy_Demand_kWh"] = load_profile["Energy Use (kWh)"] * (daily_profile / 10)

    # Concatenate the energy use column from the load profile to the existing DataFrame
    data = pd.concat([data, load_profile["Energy_Demand_kWh"]], axis=1)

    # Calculate the net power demand by subtracting the PV output from the energy use
    if "Net_Energy_Demand_kWh" not in data.columns:
        data["Net_Energy_Demand_kWh"] = 0

    data["Net_Energy_Demand_kWh"] = data["Energy_Demand_kWh"] + data["Net_Energy_Demand_kWh"]

    # Return the modified DataFrame
    return data


def configure_converter_model(inverter_output_amps=100, inverter_charge_amps=40, battery_nominal_voltage_v=52.8,
                              inverter_eff=0.94, battery_charging_eff=0.92, rectifier_eff=0.94, standby_factor=230):
    """ Configures and calculates key parameters for a converter model based on the given specifications.

    Parameters:
    - inverter_output_amps (float): The output current of the inverter in amperes. Default is 100 amps.
    - inverter_charge_amps (float): The charging current of the inverter in amperes. Default is 40 amps.
    - battery_nominal_voltage_v (float): The nominal voltage of the battery in volts. Default is 52.8 volts.
    - inverter_eff (float): Efficiency of the inverter. Default is 0.94 (94%).
    - battery_charging_eff (float): Efficiency of battery charging. Default is 0.92 (92%).
    - rectifier_eff (float): Efficiency of the rectifier. Default is 0.94 (94%).
    - standby_factor (float): A divisor factor for calculating standby power. Default is 230.

    Returns:
    - tuple: A tuple containing the following configured parameters of the converter model:
        - standby_power (float): The estimated standby power consumption of the system in kilowatts (kW).
        - max_output_kW (float): The maximum output power of the inverter in kilowatts (kW).
        - max_charge_kW (float): The maximum charging power of the inverter in kilowatts (kW).
        - inverter_eff (float): The efficiency of the inverter (as a decimal).
        - battery_charging_eff (float): The efficiency of battery charging (as a decimal).
        - rectifier_eff (float): The efficiency of the rectifier (as a decimal).

    The function calculates key parameters of a converter model, including maximum output power, maximum charge power, 
    and standby power, based on the input specifications. These parameters are essential for designing and analyzing 
    power systems involving battery storage and inverters.
    """

    max_output_kW = inverter_output_amps * battery_nominal_voltage_v / 1000
    max_charge_kW = inverter_charge_amps * battery_nominal_voltage_v / 1000
    standby_power = max_output_kW / standby_factor

    inverter_eff = inverter_eff                             # Modify later to scale efficiency values relative to load %
    battery_charging_eff = battery_charging_eff
    rectifier_eff = rectifier_eff
    
    return standby_power, max_output_kW, max_charge_kW, inverter_eff, battery_charging_eff, rectifier_eff



def calc_solar_model(data, latitude, longitude, surface_azimuth, surface_pitch, pv_kwp, inverter_eff=0.94, max_output_kW=100, max_charge_kW=40,
                                   electrical_eff=0.22, pv_derating=1, albedo=0.2, cell_temp_coeff=-0.0048,
                                   transmittance_absorptance=0.9, cell_NOCT=42, ambient_NOCT=20, 
                                   e_poa_NOCT=800, e_poa_STC=1000, cell_temp_STC=25, timestep=60, tmz_hrs_east=0):
    """ Calculate plane of array irradiance and PV power output for a given location and PV system configuration.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing TMY meteorological data, obtained via 'get_jrc_tmy()'.
      requires columns with names "Gb(n)", "Gd(h)", "G(h)" & "T2m" to represent 
      DNI, DHI, GHI & Temperature (C) respectively.
    - latitude (float): Geographical latitude of the location.
    - longitude (float): Geographical longitude of the location.
    - surface_azimuth (float): Azimuth angle of the PV surface (degrees from north).
    - surface_pitch (float): Pitch angle of the PV surface (degrees from horizontal).
    - pv_kwp (float): Rated capacity of the PV system in kilowatts peak (kWp).
    - inverter_eff (float, optional): Inverter efficiency. Default is 0.96.
    - electrical_eff (float, optional): Electrical efficiency of the PV system. Default is 0.22.
    - pv_derating (float, optional): Derating factor for the PV system. Default is 1.
    - albedo (float, optional): Ground reflectance value. Default is 0.2.
    - cell_temp_coeff (float, optional): Temperature coefficient of PV cells. Default is -0.0048.
    - transmittance_absorptance (float, optional): Combined transmittance and absorptance of the PV system. Default is 0.9.
    - cell_NOCT (float, optional): Nominal operating cell temperature. Default is 42 degrees Celsius.
    - ambient_NOCT (float, optional): Ambient temperature at NOCT. Default is 20 degrees Celsius.
    - e_poa_NOCT (float, optional): Irradiance at NOCT. Default is 800 W/m^2.
    - e_poa_STC (float, optional): Irradiance at standard test conditions. Default is 1000 W/m^2.
    - cell_temp_STC (float, optional): Cell temperature at standard test conditions. Default is 25 degrees Celsius.
    - timestep (int, optional): Time step in minutes for calculating solar radiation. Default is 60 minutes.
    - tmz_hrs_east (int, optional): Time zone offset from UTC in hours, to the east. Default is 0.

    Returns:
    - pandas.DataFrame: DataFrame containing the calculated plane of array irradiance ('E_POA') 
                        and PV power output ('Power_Output_kWh'), along with ambient ('T2m') 
                        and cell temperatures ('Cell_Temp_C').

    Note:
    This function relies on the 'calc_poa_radiation', 'calc_cell_temp', and 'calc_pv_power' functions 
    to compute the required values. It assumes the availability of the 'get_jrc_tmy' function 
    for fetching meteorological data if not available, and simplifies the output compared to 'calc_solar_model_radcomponents'
    by focusing on key parameters for solar power calculation.
    """


    max_dc_output_per_timestep = (max_charge_kW * timestep) / 60
    max_ac_output_per_timestep = (max_output_kW * timestep) / 60
    
    data["E_POA_Wm2"] = calc_poa_radiation(data["Gb(n)"], data["Gd(h)"], data["G(h)"], data['day_of_year'], data['hour_of_day'], 
                                       latitude, longitude, surface_azimuth, surface_pitch, albedo, timestep, tmz_hrs_east)
    
    # Extra Terrestrial Horizontal Irradiation
    data["ET_HRad_Wm2"] = calc_et_horizontal_radiation(latitude, longitude, data['day_of_year'], data['hour_of_day'], timestep, tmz_hrs_east)
    
    data["Cell_Temp_C"] = calc_cell_temp(data["E_POA_Wm2"], data["T2m"], cell_temp_coeff, electrical_eff,
                   cell_NOCT, ambient_NOCT, e_poa_NOCT, cell_temp_STC, transmittance_absorptance)

    # DC & AC Output
    data["PV_Output_DC_kWh"] = calc_pv_power(pv_kwp, data["E_POA_Wm2"], data["T2m"], pv_derating,
                                             cell_temp_coeff, e_poa_STC, cell_temp_STC)
    data["PV_Output_AC_kWh"] = data["PV_Output_DC_kWh"] * inverter_eff
    
    # DC & AC Output limits
    data["PV_Output_DC_kWh"] = np.minimum(data["PV_Output_DC_kWh"], max_dc_output_per_timestep)
    data["PV_Output_AC_kWh"] = np.minimum(data["PV_Output_AC_kWh"], max_ac_output_per_timestep)

    # Net Energy Demand
    if "Net_Energy_Demand_kWh" not in data.columns:
        data["Net_Energy_Demand_kWh"] = 0
    data["Net_Energy_Demand_kWh"] -= data["PV_Output_AC_kWh"]

    return data



def calc_solar_model_radcomponents(data, latitude, longitude, surface_azimuth, surface_pitch, pv_kwp, inverter_eff=0.94, max_output_kW=100, max_charge_kW=40,
                                   electrical_eff=0.22, pv_derating=1, albedo=0.2, cell_temp_coeff=-0.0048,
                                   transmittance_absorptance=0.9, cell_NOCT=42, ambient_NOCT=20, 
                                   e_poa_NOCT=800, e_poa_STC=1000, cell_temp_STC=25, timestep=60, tmz_hrs_east=0):
    """ Calculate detailed solar radiation components and PV power output for a given location and PV system configuration.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing TMY meteorological data, obtained via 'get_jrc_tmy()'.
      requires columns with names "Gb(n)", "Gd(h)", "G(h)" & "T2m" to represent 
      DNI, DHI, GHI & Temperature (C) respectively.
    - latitude (float): Geographical latitude of the location.
    - longitude (float): Geographical longitude of the location.
    - surface_azimuth (float): Azimuth angle of the PV surface (degrees from north).
    - surface_pitch (float): Pitch angle of the PV surface (degrees from horizontal).
    - pv_kwp (float): Rated capacity of the PV system in kilowatts peak (kWp).
    - inverter_eff (float, optional): Inverter efficiency. Default is 0.96.
    - electrical_eff (float, optional): Electrical efficiency of the PV system. Default is 0.22.
    - pv_derating (float, optional): Derating factor for the PV system. Default is 1.
    - albedo (float, optional): Ground reflectance value. Default is 0.2.
    - cell_temp_coeff (float, optional): Temperature coefficient of PV cells. Default is -0.0048.
    - transmittance_absorptance (float, optional): Combined transmittance and absorptance of the PV system. Default is 0.9.
    - cell_NOCT (float, optional): Nominal operating cell temperature. Default is 42 degrees Celsius.
    - ambient_NOCT (float, optional): Ambient temperature at NOCT. Default is 20 degrees Celsius.
    - e_poa_NOCT (float, optional): Irradiance at NOCT. Default is 800 W/m^2.
    - e_poa_STC (float, optional): Irradiance at standard test conditions. Default is 1000 W/m^2.
    - cell_temp_STC (float, optional): Cell temperature at standard test conditions. Default is 25 degrees Celsius.
    - timestep (int, optional): Time step in minutes for calculating solar radiation. Default is 60 minutes.
    - tmz_hrs_east (int, optional): Time zone offset from UTC in hours, to the east. Default is 0.

    Returns:
    - pandas.DataFrame: DataFrame containing calculated radiation components and PV power output. Includes columns for
                        beam radiation ('E_Beam'), diffuse radiation ('E_Diffuse'), ground radiation ('E_Ground'),
                        plane of array irradiance ('E_POA'), extraterrestrial radiation ('ET_HRad'), ambient temperature 
                        ('T2m'), cell temperature ('Cell_Temp_C'), and power output ('Power_Output_kWh').
                        
    Note:
    The function utilizes several external functions for specific calculations, such as 'calc_beam_radiation', 
    'calc_diffuse_radiation', 'calc_ground_radiation', 'calc_et_horizontal_radiation', 'calc_cell_temp', and 
    'calc_pv_power', and assumes the availability of the 'get_jrc_tmy' function for fetching meteorological data if not available.
    """

    max_dc_output_per_timestep = (max_output_kW * timestep) / 60
    max_ac_output_per_timestep = max_dc_output_per_timestep * inverter_eff

    # Beam Radiation
    data["E_Beam_Wm2"] = calc_beam_radiation(data["Gb(n)"], data['day_of_year'], data['hour_of_day'], 
                                             latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east)
    
    # Diffuse Radiation
    data["E_Diffuse_Wm2"] = calc_diffuse_radiation(data["Gd(h)"], data["G(h)"], surface_pitch, latitude, 
                                                   longitude, data['day_of_year'], data['hour_of_day'], timestep, tmz_hrs_east)
    
    # Ground Radiation
    data["E_Ground_Wm2"] = calc_ground_radiation(data["G(h)"], surface_pitch, albedo)
    
    # POA Irradiance
    data["E_POA_Wm2"] = data["E_Beam_Wm2"] + data["E_Diffuse_Wm2"] + data["E_Ground_Wm2"]
    
    # Extra Terrestrial Horizontal Irradiation
    data["ET_HRad_Wm2"] = calc_et_horizontal_radiation(latitude, longitude, data['day_of_year'], data['hour_of_day'], timestep, tmz_hrs_east)

    # Cell Temperature
    data["Cell_Temp_C"] = calc_cell_temp(data["E_POA_Wm2"], data["T2m"], cell_temp_coeff, electrical_eff,
                                         cell_NOCT, ambient_NOCT, e_poa_NOCT, cell_temp_STC, transmittance_absorptance)

    # DC & AC Output
    data["PV_Generation_kWh"] = calc_pv_power(pv_kwp, data["E_POA_Wm2"], data["T2m"], pv_derating,
                                             cell_temp_coeff, e_poa_STC, cell_temp_STC)
    data["PV_Output_AC_kWh"] = data["PV_Generation_kWh"] * inverter_eff
    
    # DC & AC Output limits
    data["PV_Output_DC_kWh"] = np.minimum(data["PV_Generation_kWh"], max_dc_output_per_timestep)
    data["PV_Output_AC_kWh"] = np.minimum(data["PV_Output_AC_kWh"], max_ac_output_per_timestep)

    # Net Energy Demand
    if "Net_Energy_Demand_kWh" not in data.columns:
        data["Net_Energy_Demand_kWh"] = 0
    data["Net_Energy_Demand_kWh"] -= data["PV_Output_AC_kWh"]

    return data


def configure_battery_model(battery_capacity_amphours=100, battery_nominal_voltage_v=51.2, start_charge_perc=0.2,
                            battery_charging_eff=0.92, rectifier_eff=0.94, max_output_kW=100, max_charge_kW=40,
                            min_capacity_perc=0.1, max_capacity_perc=0.9, max_charge_c_rate=0.5, max_discharge_c_rate=1):
    """ Configures and calculates key parameters for a battery model based on the given specifications.

    Parameters:
    battery_capacity (float): The total capacity of the battery in kilowatt-hours (kWh).
    start_charge_perc (float): The initial state of charge of the battery as a percentage of its capacity.
    min_capacity_perc (float): The minimum allowable state of charge as a percentage of the battery capacity.
    max_capacity_perc (float): The maximum allowable state of charge as a percentage of the battery capacity.
    max_charge_c_rate (float): The maximum charge rate, defined as a multiple of the battery capacity (C-rate).
    max_discharge_c_rate (float): The maximum discharge rate, defined as a multiple of the battery capacity (C-rate).

    Returns:
    tuple: A tuple containing the following configured parameters of the battery model:
        - useful_battery_capacity (float): The usable capacity of the battery in kWh, considering minimum and maximum state of charge limits.
        - initial_charge_kWh (float): The initial charge of the battery in kWh.
        - max_charge_power (float): The maximum power at which the battery can be charged in kilowatts (kW).
        - max_discharge_power (float): The maximum power at which the battery can be discharged in kilowatts (kW).
        - standby_power_kW (float): An estimated standby power consumption of the battery system in kilowatts (kW).
    """

    battery_capacity = battery_capacity_amphours * battery_nominal_voltage_v / 1000
    useful_battery_capacity = battery_capacity * (max_capacity_perc - min_capacity_perc)
    initial_charge_kWh = useful_battery_capacity * start_charge_perc
    max_discharge_power = np.minimum((battery_capacity * max_discharge_c_rate), max_output_kW)
    max_charge_power = np.minimum((battery_capacity * max_charge_c_rate), max_charge_kW)

    ac_charge_eff = (battery_charging_eff * rectifier_eff * 0.98)    # 98% is placeholder for internal battery cycle losses
    dc_charge_eff = (battery_charging_eff * 0.98)

    return useful_battery_capacity, initial_charge_kWh, max_charge_power, max_discharge_power, ac_charge_eff, dc_charge_eff



def calc_battery_model(data, useful_battery_capacity=11.4688, initial_charge_kWh=2.2937, max_charge_power=7.168,
                               max_discharge_power=14.336, standby_power_kW=0.05, inverter_eff=0.94, dc_charge_eff=0.92):
    """ Calculates and updates the battery capacity for an off-grid power system within a DataFrame.
    This function iteratively updates the battery capacity based on the net energy demand,
    taking into account the inverter/charger standby power, converter efficiency, and charge efficiency.
    It also calculates the change in battery capacity at each step.

    Parameters:
    data (pd.DataFrame): DataFrame containing energy demand and other related data. It must include 
                         a 'Net_Energy_Demand_kWh' column or this column will be initialized to zero.
    useful_battery_capacity (float): The usable capacity of the battery in kWh, considering the maximum and 
                                     minimum state of charge limits.
    initial_charge_kWh (float): The initial charge of the battery in kWh.
    max_charge_power (float): The maximum power at which the battery can be charged, in kW.
    max_discharge_power (float): The maximum power at which the battery can be discharged, in kW.
    standby_power_kW (float): The standby power consumption of the inverter/charger system, in kW.
    converter_eff (float): Efficiency of the energy conversion process (for charging).
    charge_eff (float): Efficiency of the battery charging process.

    Returns:
    pd.DataFrame: The input DataFrame with the following updated/added columns:
                  - 'Battery_Capacity_kWh': The updated battery capacity at each time step.
                  - 'Battery_Capacity_Change_kWh': The change in battery capacity at each time step.
                  - 'Net_Energy_Demand_kWh': The adjusted net energy demand, incorporating battery capacity changes.
    """

    # Initialize 'Battery_Capacity_kWh' column
    if "Battery_Capacity_kWh" not in data.columns:
        data["Battery_Capacity_kWh"] = initial_charge_kWh

    # Account for inverter/charger standby power
    data["Net_Energy_Demand_kWh"] += standby_power_kW
    data["Energy_Demand_kWh"] += standby_power_kW

    # Vectorized calculation for adjusted net demand
    data['Adjusted_Net_Demand'] = np.where(data['Net_Energy_Demand_kWh'] < 0,
                                           data['Net_Energy_Demand_kWh'] / inverter_eff * dc_charge_eff,
                                           data['Net_Energy_Demand_kWh'])

    # Prepare arrays for computation
    adjusted_net_demand = data['Adjusted_Net_Demand'].values
    battery_capacity = np.full(len(data), initial_charge_kWh)

    # Calculate battery capacity for each row
    for i in range(1, len(data)):
        # Calculate potential change in capacity
        delta_capacity = -adjusted_net_demand[i]

        # Ensure delta_capacity is within inverter limits
        if delta_capacity > 0:  # Charging
            delta_capacity = min(delta_capacity, max_charge_power)
        else:  # Discharging
            delta_capacity = min(delta_capacity, max_discharge_power)

        # Update battery capacity
        potential_new_capacity = battery_capacity[i - 1] + delta_capacity
        battery_capacity[i] = np.clip(potential_new_capacity, 0, useful_battery_capacity)

    # Calculate battery capacity change
    battery_capacity_change = np.diff(battery_capacity, prepend=battery_capacity[0])

    # Update DataFrame with calculated values
    data['Battery_Capacity_kWh'] = battery_capacity
    data['Battery_Capacity_Change_kWh'] = battery_capacity_change

    # Adjust 'Net_Energy_Demand_kWh' by battery capacity change
    data['Net_Energy_Demand_kWh'] += (battery_capacity_change / dc_charge_eff * inverter_eff)

    # Drop the 'Adjusted_Net_Demand' column as it's no longer needed
    data.drop(columns=['Adjusted_Net_Demand'], inplace=True)

    return data


def calc_grid_flow(data, import_price_day=0.2974, export_tariff=0.1422, 
                   import_price_night=None, night_start=1, night_end=8, grid_limit_kW=18.4):
    """ Calculates grid import and export values based on net energy demand, considering different tariffs for day and night.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Net_Energy_Demand_kWh' and 'hour_of_day' columns.
    import_price_day (float): Price per kWh for importing energy during the day. Defaults to 0.2974.
    import_price_night (float, optional): Price per kWh for importing energy during the night hours. If not provided, import_price_day is used.
    export_tariff (float): Tariff per kWh for exporting energy. Defaults to 0.1422.
    night_start (int): Starting hour of the night tariff (inclusive). Defaults to 1.
    night_end (int): Ending hour of the night tariff (exclusive). Defaults to 8.
    grid_limit_kW (float): The maximum limit for grid import or export in kW. Defaults to 18.4.

    Returns:
    pd.DataFrame: The modified DataFrame with added columns for grid imports, exports, and their respective values. 

    Notes:
    - The function assumes that the 'Net_Energy_Demand_kWh' column contains the net energy demand in kWh, 
      and 'hour_of_day' column contains the hour of the day in 24-hour format.
    - Grid imports and exports are calculated based on the positive and negative values of net energy demand, respectively.
    - The grid import and export values are limited to not exceed the specified 'grid_limit_kW'.
    - The function applies different import prices for day and night, determined by 'night_start' and 'night_end' hours.
    - The monetary value of grid imports and exports is calculated based on the respective tariffs.
    - The 'Net_Energy_Demand_kWh' column is updated to reflect the remaining power demand after accounting for grid imports and exports.
    - If the grid import or export is below the grid limit, the net energy demand is set to zero.
    """

    # Initialize columns to avoid KeyError
    data["Grid_Imports"] = data["Net_Energy_Demand_kWh"].clip(lower=0, upper=grid_limit_kW)
    data["Grid_Exports"] = -data["Net_Energy_Demand_kWh"].clip(upper=0, lower=-grid_limit_kW)

    # Pre-compute is_night if night tariff is provided
    if import_price_night is not None:
        is_night = data['hour_of_day'].between(night_start, night_end, inclusive='left')
        import_price = np.where(is_night, import_price_night, import_price_day)
        data["Grid_Imports_Value"] = data["Grid_Imports"] * import_price
    else:
        data["Grid_Imports_Value"] = data["Grid_Imports"] * import_price_day

    data["Grid_Exports_Value"] = data["Grid_Exports"] * export_tariff

    # In-place update of Net Energy Demand
    data["Net_Energy_Demand_kWh"] -= data["Grid_Imports"]
    data["Net_Energy_Demand_kWh"] += data["Grid_Exports"]

    return data


def generate_model(data, latitude, longitude, pv_kwp=3.2, surface_azimuth=0, surface_pitch=35, model="Full", daily_profile=10,
                   battery_capacity_amphours=100, battery_nominal_voltage_v=51.2, inverter_output_amps=100, inverter_charge_amps=40,
                   import_price_day=0.2974, export_tariff=0.1422, import_price_night=None,
                   inverter_eff=0.94, battery_charging_eff=0.92, rectifier_eff=0.94, standby_factor=230, 
                   electrical_eff=0.22, pv_derating=1, albedo=0.2, cell_temp_coeff=-0.0048,
                   transmittance_absorptance=0.9, cell_NOCT=42, ambient_NOCT=20, e_poa_NOCT=800, e_poa_STC=1000,
                   cell_temp_STC=25, timestep=60, tmz_hrs_east=0, start_charge_perc=0.2, min_capacity_perc=0.1, 
                   max_capacity_perc=0.9, max_charge_c_rate=0.5, max_discharge_c_rate=1,
                   night_start=1, night_end=8, grid_limit_kW=18.4, depth="", net_gen=True):
    """ Generates a solar power model based on given parameters and configurations. 
    This function integrates various components like PV generation, battery storage, and grid interaction.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing TMY meteorological data, obtained via 'get_jrc_tmy()'.
      requires columns with names "Gb(n)", "Gd(h)", "G(h)" & "T2m" to represent 
      DNI, DHI, GHI & Temperature (C) respectively.
    - latitude, longitude (float): Geographical coordinates of the location.
    - pv_kwp (float): Rated capacity of the PV system in kilowatts peak.
    - surface_azimuth, surface_pitch (float): Orientation of the solar panels.
    - model (str): Type of model to generate ('Full', 'PV_Storage_Offgrid', 'PV_Grid').
    - daily_profile (float): Daily energy usage profile for load simulation.
    - battery_capacity (float): Total capacity of the battery in kWh.
    - import_price_day, import_price_night (float): Import prices for electricity from the grid.
    - export_tariff (float): Export tariff for electricity fed into the grid.
    - inverter_eff, electrical_eff (float): Efficiency of inverter and electrical systems.
    - pv_derating (float): Derating factor for the PV system.
    - albedo (float): Ground reflectance value.
    - cell_temp_coeff (float): Temperature coefficient of PV cells.
    - transmittance_absorptance (float): Combined transmittance and absorptance of the PV system.
    - cell_NOCT, ambient_NOCT (float): Temperatures for Nominal Operating Cell Temperature.
    - e_poa_NOCT, e_poa_STC (float): Irradiance at NOCT and STC.
    - cell_temp_STC (float): Cell temperature at Standard Test Conditions.
    - timestep (int): Time step in minutes for calculations.
    - tmz_hrs_east (int): Time zone offset from UTC.
    - start_charge_perc, min_capacity_perc, max_capacity_perc (float): Battery state of charge parameters.
    - max_charge_c_rate, max_discharge_c_rate (float): Charge and discharge rates of the battery.
    - converter_eff, charge_eff (float): Converter and charge efficiency.
    - night_start, night_end (int): Start and end hours for the night tariff.
    - grid_limit_kW (float): Maximum limit for grid import or export.
    - depth (str): Depth of data cleaning ('Full', 'Short', etc.).

    Returns:
    pandas.DataFrame: DataFrame containing the results of the model simulation, with different levels of detail based on the selected model type.
    """

    data = add_load_profile(data, daily_profile)

    sp, mo_kW, mc_kW, i_eff, bc_eff, r_eff =  configure_converter_model(inverter_output_amps, inverter_charge_amps, battery_nominal_voltage_v,
                                                                        inverter_eff, battery_charging_eff, rectifier_eff, standby_factor)

    data = calc_solar_model_radcomponents(data, latitude, longitude, surface_azimuth, surface_pitch, pv_kwp, i_eff, mo_kW, mc_kW,
                                        electrical_eff, pv_derating, albedo, cell_temp_coeff,
                                        transmittance_absorptance, cell_NOCT, ambient_NOCT, 
                                        e_poa_NOCT, e_poa_STC, cell_temp_STC, timestep, tmz_hrs_east)
    

    

    if model == "Full":
        ubc, ic, mcp, mdp, acc_eff, dcc_eff = configure_battery_model(battery_capacity_amphours, battery_nominal_voltage_v, start_charge_perc, bc_eff, r_eff,
                                                    mo_kW, mc_kW, min_capacity_perc, max_capacity_perc, max_charge_c_rate, max_discharge_c_rate)
        
        data = calc_battery_model(data, ubc, ic, mcp, mdp, sp, i_eff, dcc_eff)

        data = calc_grid_flow(data, import_price_day, export_tariff, import_price_night, night_start, night_end, grid_limit_kW)

        data = data_clean(data, depth, net_gen)

        return data

    elif model == "PV_Storage_Offgrid":
        ubc, ic, mcp, mdp, acc_eff, dcc_eff = configure_battery_model(battery_capacity_amphours, battery_nominal_voltage_v, start_charge_perc, bc_eff, r_eff,
                                                    mo_kW, mc_kW, min_capacity_perc, max_capacity_perc, max_charge_c_rate, max_discharge_c_rate)
        
        data = calc_battery_model(data, ubc, ic, mcp, mdp, sp, i_eff, acc_eff, dcc_eff)

        data = data_clean(data, depth, net_gen)

        return data
    
    elif model == "PV_Grid":
        data = calc_grid_flow(data, import_price_day, export_tariff, import_price_night, night_start, night_end, grid_limit_kW)

        data = data_clean(data, depth, net_gen)
        return data
    
    data = data_clean(data, depth, net_gen)
    
    return data


def data_clean(data, depth="Model", net_gen=True):
    """ Cleans the provided DataFrame by dropping specified columns based on the selected depth of cleaning.

    Parameters:
    data (pd.DataFrame): The DataFrame to be cleaned.
    depth (str): Determines the level of cleaning to be performed on the DataFrame. 
                 Accepts two values:
                 - "Full": A more comprehensive cleaning, removing a larger set of specified columns.
                 - "Short": A less comprehensive cleaning, removing a smaller set of specified columns.

    Returns:
    pd.DataFrame: The cleaned DataFrame with the specified columns removed.

    Note:
    The function expects certain column names to be present in the DataFrame, and it will remove these
    columns based on the chosen depth. If certain expected columns are not found, they will simply be 
    ignored in the drop process.
    """
    
    if depth == "Model":
        data = data[['Net_Energy_Demand_kWh', 'Energy_Demand_kWh', "PV_Generation_kWh", 'PV_Output_DC_kWh', 'PV_Output_AC_kWh', 'Battery_Capacity_kWh', 'Battery_Capacity_Change_kWh', 
             'Grid_Imports', 'Grid_Exports', 'Grid_Imports_Value', 'Grid_Exports_Value']]
        
        if net_gen == False:
            data = data.drop(["Net_Energy_Demand_kWh"], axis=1)

        return data
        
    elif depth == "Short":
        data = data[['Net_Energy_Demand_kWh', 'Energy_Demand_kWh', "PV_Generation_kWh", 'PV_Output_DC_kWh', 'PV_Output_AC_kWh',
                     'Battery_Capacity_kWh', 'Battery_Capacity_Change_kWh', 'Grid_Imports', 'Grid_Exports', 'Grid_Imports_Value', 
                     'Grid_Exports_Value', 'E_POA_Wm2', 'E_Beam_Wm2', 'E_Diffuse_Wm2', 'E_Ground_Wm2', "ET_HRad_Wm2", 'Cell_Temp_C', 'time(UTC)']]
        
        if net_gen == False:
            data = data.drop(["Net_Energy_Demand_kWh"], axis=1)

        return data
        

    data = data[['Net_Energy_Demand_kWh', 'Energy_Demand_kWh', "PV_Generation_kWh", 'PV_Output_DC_kWh', 'PV_Output_AC_kWh', 'Battery_Capacity_kWh', 
                 'Battery_Capacity_Change_kWh', 'Grid_Imports', 'Grid_Exports', 'Grid_Imports_Value', 'Grid_Exports_Value', 
             'E_POA_Wm2', 'E_Beam_Wm2', 'E_Diffuse_Wm2', 'E_Ground_Wm2', 'ET_HRad_Wm2', 'Cell_Temp_C',
             'time(UTC)', 'T2m', 'RH', 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP', 'day_of_year', 'hour_of_day']]
    
    if net_gen == False:
        data = data.drop(["Net_Energy_Demand_kWh"], axis=1)
    
    return data


if __name__ == "__main__":
    main()
