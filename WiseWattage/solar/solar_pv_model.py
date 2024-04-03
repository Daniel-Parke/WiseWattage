"""Import functionality required to model Solar PV performance"""
from collections import OrderedDict

import pandas as pd
import numpy as np
from numpy import cos, radians, exp

import solar.solar_radiation as sr


def calc_pv_derating(n_day: int, civil_time: float, pv_eol_derating: float, 
                     lifespan: int, year: int = 1) -> float:
    """
    Calculates the derating factor for PV panels over their lifespan.

    Parameters:
    - n_day (int): Day of the year.
    - civil_time (float): Time of day in hours.
    - pv_eol_derating (float): End-of-life derating factor for the PV panel.
    - lifespan (int): Expected lifespan of the PV panel in years.
    - year (int, optional): Current year of the panel's operation. Defaults to 1.

    Returns:
    - float: The derating factor for the PV panel at the given time.
    """
    yearly_derating = (1 - pv_eol_derating) / lifespan
    hourly_derating = ((1 - pv_eol_derating) / lifespan) / 8760

    pv_derating = 1 - (
        ((civil_time * hourly_derating) + ((n_day - 1) * (24 * hourly_derating)))
        + (yearly_derating * (year - 1))
    )

    return pv_derating


def calc_array_temp_sandia(e_poa: float, ambient_temp: float, wind_speed: float, 
                           a: float = -3.47, b: float = -0.0594) -> float:
    """
    Calculate the temperature of a photovoltaic (PV) array based on the Sandia method.
    
    Parameters:
    - e_poa (float): Plane of array irradiance in W/m^2. Represents the solar irradiance incident on the PV array.
    - ambient_temp (float): Ambient temperature in degrees Celsius.
    - wind_speed (float): Wind speed in m/s at the site of the PV array.
    - a (float, optional): Coefficient a in the exponential model, defaulting to -3.47.
    - b (float, optional): Coefficient b in the exponential model, defaulting to -0.0594.
    
    Returns:
    - float: Estimated temperature of the PV array in degrees Celsius.
    """
    array_temp = e_poa * exp(a + b * wind_speed) + ambient_temp
    return array_temp


def calc_cell_temp_coeff(e_poa: float, ambient_temp: float, wind_speed: float, 
                         cell_temp_coeff: float = -0.0035, cell_temp_STC: float = 25) -> float:
    """
    Adjusts the temperature coefficient based on current conditions.

    Parameters:
    - e_poa (float): Plane of array irradiance in W/m^2.
    - ambient_temp (float): Ambient temperature in degrees Celsius.
    - wind_speed (float): Wind speed in m/s measured at site.
    - cell_temp_coeff (float, optional): Temperature coefficient of the PV cell. Defaults to -0.0035.
    - cell_temp_STC (float, optional): Cell temperature at standard test conditions in degrees Celsius. Defaults to 25.

    Returns:
    - float: Adjusted temperature coefficient.
    """
    array_temp = calc_array_temp_sandia(e_poa, ambient_temp, wind_speed)
    temp_coeff = np.minimum(1, (1 + cell_temp_coeff * (array_temp - cell_temp_STC)))

    return temp_coeff


def iam_losses(aoi: float, refraction_index: float = 0.05) -> float:
    """
    Calculates the incident angle modifier (IAM) losses for solar panels based on the angle of incidence (AOI).

    Parameters:
    - aoi (float): Angle of incidence in degrees.
    - refraction_index (float, optional): Refractive index of the panel's surface material. Defaults to 0.1.

    Returns:
    - float: IAM loss factor for each time step.

    References
    ----------
    .. [1] Souka A.F., Safwat H.H., "Determination of the optimum
       orientations for the double exposure flat-plate collector and its
       reflections". Solar Energy vol .10, pp 170-174. 1966.

    .. [2] ASHRAE standard 93-77

    .. [3] PVsyst 7 Help.
       https://www.pvsyst.com/help/index.html?iam_loss.htm retrieved on
       January 30, 2024
    """
    iam_factor = 1 - refraction_index * ((1 / cos(radians(aoi))) - 1)
    iam_factor = np.where(aoi > 85, 0, iam_factor)

    return iam_factor


def calc_low_light_losses(pv_kwp: float, e_poa: float, k: float = 0.0075, 
                          midpoint: float = 25) -> float:
    """
    Modified logistic function to calculate efficiency based on irradiance, with a minimum efficiency level.

    Parameters:
    - pv_kwp (float): The rated solar PV size (kWp).
    - e_poa (float): The irradiance incident on array (W/m^2).
    - k (float, optional): The steepness of the curve. Defaults to 0.0075.
    - midpoint (float, optional): The irradiance at which the efficiency is at its midpoint. Defaults to 25.

    Returns:
    - float: The calculated efficiency at the given irradiance.
    """
    pv_kwp_min = pv_kwp * 0.6
    eff = pv_kwp_min + (pv_kwp - pv_kwp_min) / (1 + np.exp(-k * (e_poa - midpoint)))
    return eff


def calc_pv_power(pv_kwp: float, e_poa: float, ambient_temp: float, wind_speed: float, 
                  pv_derating: float = 1, cell_temp_coeff: float = -0.004, 
                  e_poa_STC: float = 1000, cell_temp_STC: float = 25) -> tuple:
    """
    Calculates the power output of a PV panel.

    Parameters:
    - pv_kwp (float): Rated power of the PV panel in kWp.
    - e_poa (float): Plane of array irradiance in W/m^2.
    - ambient_temp (float): Ambient temperature in degrees Celsius.
    - wind_speed (float): Wind speed at the site.
    - pv_derating (float, optional): Derating factor for the PV panel. Defaults to 1.
    - cell_temp_coeff (float, optional): Temperature coefficient of the PV cell. Defaults to -0.004.
    - e_poa_STC (float, optional): Irradiance at standard test conditions in W/m^2. Defaults to 1000.
    - cell_temp_STC (float, optional): Cell temperature at standard test conditions in degrees Celsius. Defaults to 25.

    Returns:
    - tuple: A tuple of (pv_power, temp_diff), where pv_power is the power output in W, and temp_diff is the temperature difference effect on power output.
    """
    temp_coeff = calc_cell_temp_coeff(
        e_poa, ambient_temp, wind_speed, cell_temp_coeff, cell_temp_STC
    )

    pv_power = (pv_kwp * pv_derating * (e_poa / e_poa_STC) * temp_coeff)
    temp_diff = (pv_power / temp_coeff) - pv_power

    return pv_power, temp_diff


def calc_solar_model(data: pd.DataFrame, latitude: float, longitude: float, pv_kwp: float = 1, 
                     surface_pitch: float = 35, surface_azimuth: float = 0, lifespan: int = 25, 
                     pv_derating: float = 0.88, albedo: float = 0.2, cell_temp_coeff: float = -0.0035,
                     refraction_index: float = 0.05, e_poa_STC: float = 1000, cell_temp_STC: float = 25,
                     timestep: int = 60, tmz_hrs_east: float = 0) -> pd.DataFrame:
    """
    Processes TMY data to simulate solar PV system performance over a typical meteorological year.

    Parameters:
        data (pd.DataFrame): TMY data including irradiance and temperature.
        latitude (float): Site latitude.
        longitude (float): Site longitude.
        pv_kwp (float): Rated power of the PV system in kWp.
        surface_pitch (float): Tilt angle of the PV panel from the horizontal plane.
        surface_azimuth (float): Orientation of the PV panel from true north.
        lifespan (int): Estimated lifespan of the PV system in years.
        pv_derating (float): Derating factor of the PV system to account for end-of-life performance.
        albedo (float): Ground reflectance factor.
        cell_temp_coeff (float): Temperature coefficient of the PV cell per degree Celsius.
        refraction_index (float): Refractive index of the panel's cover.
        e_poa_STC (float): Plane of array irradiance under standard test conditions.
        cell_temp_STC (float): Cell temperature under standard test conditions.
        timestep (int): Time interval for calculations in minutes.
        tmz_hrs_east (float): Time zone offset from GMT in hours.

    Returns:
        pd.DataFrame: Enhanced TMY data with added columns for solar radiation calculations and PV system performance metrics.
    """
    # Convert required DataFrame columns to numpy arrays for calculations
    hour_of_day = data['Hour_of_Day'].to_numpy()
    day_of_year = data['Day_of_Year'].to_numpy()
    week_of_year = data['Week_of_Year'].to_numpy()
    month_of_year = data['Month_of_Year'].to_numpy()
    Gb_n = data['Gb(n)'].to_numpy()
    Gd_h = data['Gd(h)'].to_numpy()
    G_h = data['G(h)'].to_numpy()
    Ambient_Temperature_C = data['T2m'].to_numpy()
    wind_speed = data["WS10m"].to_numpy()

    # Perform calculations using numpy arrays
    declination_angle = sr.calc_declination(day_of_year)
    solar_time = sr.calc_solar_time(day_of_year, hour_of_day, longitude, timestep, tmz_hrs_east)
    hour_angle = sr.calc_hour_angle(day_of_year, hour_of_day, longitude, timestep, tmz_hrs_east)
    aoi = sr.calc_aoi(day_of_year, hour_of_day, latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east)
    zenith_angle = sr.calc_zenith(latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    e_beam_w_m2 = sr.calc_beam_radiation(Gb_n, day_of_year, hour_of_day, latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east)
    e_diffuse_w_m2 = sr.calc_diffuse_radiation(Gd_h, G_h, surface_pitch, latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    e_ground_w_m2 = sr.calc_ground_radiation(G_h, surface_pitch, albedo)
    e_poa_w_m2 = e_beam_w_m2 + e_diffuse_w_m2 + e_ground_w_m2

    # Assuming iam_losses function exists and can operate on numpy arrays
    panel_poa_w_m2 = e_beam_w_m2 * iam_losses(aoi, refraction_index) + e_diffuse_w_m2 + e_ground_w_m2
    iam_loss_perc = np.divide((e_poa_w_m2 - panel_poa_w_m2), e_poa_w_m2, where=e_poa_w_m2!=0)

    et_hrad_w_m2 = sr.calc_et_horizontal_radiation(latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    pv_derated_eff = calc_pv_derating(day_of_year, hour_of_day, pv_derating, lifespan, year=1)  # Assuming this function can handle numpy arrays
    Array_Temp_C = calc_array_temp_sandia(panel_poa_w_m2, Ambient_Temperature_C, wind_speed)

    # Calculate low light efficiency
    ll_pv_eff = calc_low_light_losses(1, panel_poa_w_m2)

    # Assuming calc_pv_power returns two arrays or can be adapted to do so
    pv_gen_kwh, pv_thermal_loss_kwh = calc_pv_power(pv_kwp, panel_poa_w_m2, Ambient_Temperature_C, wind_speed, pv_derated_eff, cell_temp_coeff, e_poa_STC, cell_temp_STC)
    ll_pv_gen_kWh = pv_gen_kwh * ll_pv_eff
    low_light_loss_kWh = pv_gen_kwh - ll_pv_gen_kWh
    iam_loss_kWh = pv_gen_kwh * iam_loss_perc
    total_loss_kwh = iam_loss_kWh + low_light_loss_kWh + pv_thermal_loss_kwh

    # Construct a new DataFrame from the calculated arrays
    results = pd.DataFrame({
        "Hour_of_Day": hour_of_day,
        "Day_of_Year": day_of_year,
        "Week_of_Year": week_of_year,
        "Month_of_Year": month_of_year,
        "Wind_Speed_ms": wind_speed,
        "Ambient_Temperature_C": Ambient_Temperature_C,
        "Declination_Angle": declination_angle,
        "Solar_Time": solar_time,
        "Hour_Angle": hour_angle,
        "AOI": aoi,
        "Zenith_Angle": zenith_angle,
        "E_Beam_kWm2": e_beam_w_m2 / 1000,                   # Convert to kW
        "E_Diffuse_kWm2": e_diffuse_w_m2 / 1000,             # Convert to kW
        "E_Ground_kWm2": e_ground_w_m2 / 1000,               # Convert to kW
        "E_POA_kWm2": e_poa_w_m2 / 1000,                     # Convert to kW
        "Panel_POA_kWm2": panel_poa_w_m2 / 1000,             # Convert to kW
        "IAM_Loss_kWh": iam_loss_kWh,                       # Convert to kW
        "ET_HRad_kWm2": et_hrad_w_m2 / 1000,                 # Convert to kW
        "PV_Derated_Eff": pv_derated_eff,
        "Array_Temp_C": Array_Temp_C,
        "PV_Gen_kWh": ll_pv_gen_kWh,
        "PV_Thermal_Loss_kWh": pv_thermal_loss_kwh,
        "Low_Light_Loss_kWh": low_light_loss_kWh,
        "Combined_PV_Losses_kWh": total_loss_kwh
    })

    return results


def combine_array_results(results: list) -> pd.DataFrame:
    """
    Combines array results into a single DataFrame with averaged and summed columns.

    Parameters:
        results (list): List of DataFrame results from multiple PV array simulations.

    Returns:
        pd.DataFrame: Combined results with averaged and summed values for key metrics.
    """
    columns_to_average = [
        "E_Beam_kWm2",
        "E_Diffuse_kWm2",
        "E_Ground_kWm2",
        "E_POA_kWm2",
        "Panel_POA_kWm2",
        "ET_HRad_kWm2",
        "Array_Temp_C"
    ]
    
    columns_to_sum = [
        "PV_Gen_kWh",
        "PV_Thermal_Loss_kWh",
        "Low_Light_Loss_kWh",
        "IAM_Loss_kWh",
        "Combined_PV_Losses_kWh",
    ]
    
    columns_to_add = ["AOI", "Zenith_Angle"]
    time_columns = [
        "Declination_Angle",
        "Solar_Time",
        "Hour_Angle",
        "Ambient_Temperature_C",
        "Wind_Speed_ms",
        "Hour_of_Day",
        "Day_of_Year",
        "Week_of_Year",
        "Month_of_Year",
    ]
    array_data = []

    for i, result in enumerate(results, start=1):
        df_to_combine = result["model_result"][columns_to_average + columns_to_sum].copy()
        df_to_add = result["model_result"][columns_to_add].copy()

        df_to_combine.columns = [f"{col}_Array_{i}" for col in df_to_combine.columns]
        df_to_add.columns = [f"{col}_Array_{i}" for col in columns_to_add]

        array_df = pd.concat([df_to_combine, df_to_add], axis=1)
        array_data.append(array_df)

    combined_df = pd.concat(array_data, axis=1)

    for col in columns_to_average:
        combined_df[f"{col}_Avg"] = combined_df.filter(regex=f"^{col}_Array_").mean(axis=1)

    for col in columns_to_sum:
        combined_df[f"{col}_Total"] = combined_df.filter(regex=f"^{col}_Array_").sum(axis=1)

    combined_df["Array_Temp_C_Avg"] = combined_df.filter(regex="^Array_Temp_C_Array_").mean(axis=1)

    for time_col in time_columns:
        combined_df[time_col] = results[0]["model_result"][time_col]

    return combined_df


def total_array_results(results: list) -> pd.DataFrame:
    """
    Aggregates total results from multiple PV array simulations.

    Parameters:
        results (list): List of DataFrame results from multiple PV array simulations.

    Returns:
        pd.DataFrame: Aggregated results with total sums and averages for key metrics.
    """
    columns_to_average = [
        "E_Beam_kWm2",
        "E_Diffuse_kWm2",
        "E_Ground_kWm2",
        "E_POA_kWm2",
        "Panel_POA_kWm2",
        "ET_HRad_kWm2",
    ]
    
    columns_to_sum = [
        "PV_Gen_kWh",
        "PV_Thermal_Loss_kWh",
        "Low_Light_Loss_kWh",
        "IAM_Loss_kWh",
        "Combined_PV_Losses_kWh",
    ]

    time_columns = [
        "Declination_Angle",
        "Solar_Time",
        "Hour_Angle",
        "Ambient_Temperature_C",
        "Hour_of_Day",
        "Day_of_Year",
        "Week_of_Year",
        "Month_of_Year",
    ]

    combined_df = pd.DataFrame()

    for col in columns_to_average:
        combined_df[f"{col}_Avg"] = sum(result["model_result"][col] for result in results) / len(results)

    for col in columns_to_sum:
        combined_df[f"{col}_Total"] = sum(result["model_result"][col] for result in results)

    combined_df["Array_Temp_C_Avg"] = sum(result["model_result"]["Array_Temp_C"] for result in results) / len(results)

    for time_col in time_columns:
        combined_df[time_col] = results[0]["model_result"][time_col]

    return combined_df


def pv_stats(model_results: pd.DataFrame, arrays: list) -> pd.Series:
    """
    Generates a summary of key PV performance metrics over the specified period.

    Parameters:
        model_results (pd.DataFrame): DataFrame containing model results.
        arrays (list): List of PV array configurations used in the model.

    Returns:
        pd.Series: Series with summarized PV performance metrics.
    """
    # Columns to sum
    columns_to_sum = [
        "E_Beam_kWm2_Avg",
        "E_Diffuse_kWm2_Avg",
        "E_Ground_kWm2_Avg",
        "E_POA_kWm2_Avg",
        "Panel_POA_kWm2_Avg",
        "ET_HRad_kWm2_Avg",
        "PV_Gen_kWh_Total",
        "PV_Thermal_Loss_kWh_Total",
        "Low_Light_Loss_kWh_Total",
        "IAM_Loss_kWh_Total",
        "Combined_PV_Losses_kWh_Total",
    ]

    # Columns to calculate the mean
    columns_to_mean = ["Array_Temp_C_Avg", "Ambient_Temperature_C"]

    # Initialize a dictionary to hold the summary
    summary = {}

    # Sum the specified columns
    for col in columns_to_sum:
        summary[col] = model_results[col].sum()

    # Calculate the mean for the specified columns
    for col in columns_to_mean:
        summary[col] = model_results[col].mean()

    lifespan = arrays[0].pv_panel.lifespan
    eol_derating = arrays[0].pv_panel.pv_eol_derating
    yearly_derating = (1 - eol_derating) / lifespan
    total_gen = 0

    for i in range(lifespan):
        gen = (1 - (yearly_derating * (i + 1))) * summary["PV_Gen_kWh_Total"]
        total_gen += gen

    pv_capacity = 0 
    total_area = 0
    for i in range(len(arrays)):
        pv_capacity += arrays[i].pv_panel.panel_kwp * arrays[i].num_panels
        total_area += arrays[i].pv_panel.size_m2 * arrays[i].num_panels

    pv_capacity = round(pv_capacity, 3)
    total_area = round(total_area, 3)

    kWh_kWp_annual = summary["PV_Gen_kWh_Total"] / pv_capacity
    kWh_m2_annual = summary["PV_Gen_kWh_Total"] / total_area

    summary["Total_PV_Capacity_kWp"] = pv_capacity
    summary["Total_Area_Covered_m2"] = total_area
    summary["PV_kWh_kWp_Annual"] = kWh_kWp_annual
    summary["PV_kWh_m2_Annual"] = kWh_m2_annual
    summary["PV_Gen_kWh_Lifetime"] = total_gen
    summary["PV_Gen_kWh_Annual"] = summary["PV_Gen_kWh_Total"]
    summary["PV_Thermal_Loss_kWh_Annual"] = summary["PV_Thermal_Loss_kWh_Total"]
    summary["IAM_Loss_kWh_Annual"] = summary["IAM_Loss_kWh_Total"]
    summary["Low_Light_Loss_kWh_Annual"] = summary["Low_Light_Loss_kWh_Total"]
    summary["Panel_POA_kWm2_Annual"] = summary["Panel_POA_kWm2_Avg"]
    summary["E_POA_kWm2_Annual"] = summary["E_POA_kWm2_Avg"]
    summary["E_Beam_kWm2_Annual"] = summary["E_Beam_kWm2_Avg"]
    summary["E_Diffuse_kWm2_Annual"] = summary["E_Diffuse_kWm2_Avg"]
    summary["E_Ground_kWm2_Annual"] = summary["E_Ground_kWm2_Avg"]
    summary["ET_HRad_kWm2_Annual"] = summary["ET_HRad_kWm2_Avg"]
    summary["Ambient_Temperature_C_Avg"] = summary["Ambient_Temperature_C"]
    summary["Combined_PV_Losses_kWh_Annual"] = summary["Combined_PV_Losses_kWh_Total"]

    # Define the desired order of keys
    desired_order = [
        "Total_PV_Capacity_kWp",
        "PV_kWh_kWp_Annual",
        "Total_Area_Covered_m2",
        "PV_kWh_m2_Annual",
        "PV_Gen_kWh_Annual",
        "PV_Gen_kWh_Lifetime",
        "E_POA_kWm2_Annual",
        "Panel_POA_kWm2_Annual",
        "IAM_Loss_kWh_Annual",
        "PV_Thermal_Loss_kWh_Annual",
        "Low_Light_Loss_kWh_Annual",
        "Combined_PV_Losses_kWh_Annual",
        "E_Beam_kWm2_Annual",
        "E_Diffuse_kWm2_Annual",
        "E_Ground_kWm2_Annual",
        "ET_HRad_kWm2_Annual",
        "Array_Temp_C_Avg",
        "Ambient_Temperature_C_Avg",
    ]

    # Create an OrderedDict with items in the desired order
    ordered_summary = OrderedDict((k, summary[k]) for k in desired_order)

    # Convert to pandas Series and round the values
    summary_series = pd.Series(ordered_summary).round(3)

    return summary_series


class SummaryGrouped:
    """
    Organizes grouped summary statistics of PV system performance.
    
    Parameters:
        summaries (dict): Dictionary with time grouping as keys and summary statistics DataFrames as values.
    """

    def __init__(self, summaries):
        for key, df in summaries.items():
            setattr(self, key.lower(), df)


def pv_grouped(model_results: pd.DataFrame) -> SummaryGrouped:
    """
    Generates grouped statistics of PV performance over different time frames.

    Parameters:
        model_results (pd.DataFrame): DataFrame containing model results.

    Returns:
        SummaryGrouped: Object containing DataFrames of grouped statistics.
    """
    # Define the groupings for different human timeframes
    groupings = {
        "Hourly": "Hour_of_Day",
        "Daily": "Day_of_Year",
        "Weekly": "Week_of_Year",
        "Monthly": "Month_of_Year",
        "Quarterly": model_results["Month_of_Year"].apply(lambda x: (x - 1) // 3 + 1),
    }

    # Columns to sum and to calculate the mean
    columns_to_sum = [
        "PV_Gen_kWh_Total",
        "E_POA_kWm2_Avg",
        "Panel_POA_kWm2_Avg",
        "IAM_Loss_kWh_Total",
        "PV_Thermal_Loss_kWh_Total",
        "Low_Light_Loss_kWh_Total",
        "Combined_PV_Losses_kWh_Total",
        "E_Beam_kWm2_Avg",
        "E_Diffuse_kWm2_Avg",
        "E_Ground_kWm2_Avg",
        "ET_HRad_kWm2_Avg",
    ]
    columns_to_mean = ["Array_Temp_C_Avg", "Ambient_Temperature_C"]

    summaries = {}

    # Gets Hourly and Hour of Day from .items() tuple list
    for timeframe, group_by in groupings.items():
        grouped = model_results.groupby(group_by)

        # Summing specified columns and rounding
        summed = round(grouped[columns_to_sum].sum(), 3)

        # Calculating the mean for specified columns and rounding
        meaned = round(grouped[columns_to_mean].mean(), 3)

        # Combine the summed and meaned results into a single DataFrame
        summary_df = pd.concat([summed, meaned], axis=1)

        # Adds summary dataframe to dictionary with timeframe key
        summaries[timeframe] = summary_df

    # Return an instance of SummaryGrouped with summaries as attributes
    return SummaryGrouped(summaries)
