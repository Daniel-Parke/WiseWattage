"""Import functionality required to model Solar PV performance"""
from collections import OrderedDict

import pandas as pd
import numpy as np
from numpy import cos, radians

import WiseWattage.solar_radiation as sr


def iam_losses(
    n_day,
    civil_time,
    latitude,
    longitude,
    surface_azimuth,
    surface_pitch,
    timestep,
    tmz_hrs_east,
    refraction_index=0.1,
):
    """Calculates the incident angle modifier (IAM) losses for solar panels based on the angle of incidence (AOI).
    Parameters:
    - n_day: Day of the year.
    - civil_time: Time of day in hours.
    - latitude, longitude: Geographic coordinates of the site.
    - surface_azimuth, surface_pitch: Orientation and tilt of the solar panel.
    - timestep: Time step in minutes.
    - tmz_hrs_east: Timezone offset from GMT in hours.
    - refraction_index: Refractive index of the panel's surface material.

    Returns:
    - iam_factor: IAM loss factor for each time step.
    """
    aoi = sr.calc_aoi(
        n_day,
        civil_time,
        latitude,
        longitude,
        surface_azimuth,
        surface_pitch,
        timestep,
        tmz_hrs_east,
    )

    iam_factor = 1 - refraction_index * ((1 / cos(radians(aoi))) - 1)
    iam_factor = np.where(aoi > 85, 0, iam_factor)

    return iam_factor


def calc_pv_derating(n_day, civil_time, pv_eol_derating, lifespan, year=1):
    """Calculates the derating factor for PV panels over their lifespan.

    Parameters:
    - n_day: Day of the year.
    - civil_time: Time of day in hours.
    - pv_eol_derating: End-of-life derating factor for the PV panel.
    - lifespan: Expected lifespan of the PV panel in years.
    - year: Current year of the panel's operation.

    Returns:
    - pv_derating: The derating factor for the PV panel at the given time.
    """
    yearly_derating = (1 - pv_eol_derating) / lifespan
    hourly_derating = ((1 - pv_eol_derating) / lifespan) / 8760

    pv_derating = 1 - (
        ((civil_time * hourly_derating) + ((n_day - 1) * (24 * hourly_derating)))
        + (yearly_derating * (year - 1))
    )

    return pv_derating


def calc_cell_temp(
    e_poa,
    ambient_temp,
    cell_temp_coeff=-0.0035,
    electrical_eff=0.21,
    cell_NOCT=42,
    ambient_NOCT=20,
    e_poa_NOCT=800,
    cell_temp_STC=25,
    transmittance_absorptance=0.9,
):
    """Calculates the cell temperature of a PV panel.

    Parameters:
    - e_poa: Plane of array irradiance in kW/m^2.
    - ambient_temp: Ambient temperature in degrees Celsius.
    - cell_temp_coeff: Temperature coefficient of the PV cell.
    - electrical_eff: Electrical efficiency of the PV panel.
    - cell_NOCT, ambient_NOCT: Nominal operating cell temperature and the corresponding ambient temperature.
    - e_poa_NOCT: Irradiance at NOCT conditions in W/m^2.
    - cell_temp_STC: Cell temperature at standard test conditions in degrees Celsius.
    - transmittance_absorptance: Transmittance and absorptance product of the PV panel.

    Returns:
    - Cell temperature of the PV panel.
    """
    temp_factor = (cell_NOCT - ambient_NOCT) * ((e_poa * 1000) / e_poa_NOCT)
    numerator = ambient_temp + temp_factor * (
        1
        - (electrical_eff * (1 - cell_temp_coeff * cell_temp_STC))
        / transmittance_absorptance
    )
    denominator = 1 + temp_factor * (
        cell_temp_coeff * electrical_eff / transmittance_absorptance
    )

    return numerator / denominator


def calc_cell_temp_coeff(
    e_poa, ambient_temp, cell_temp_coeff=-0.0035, cell_temp_STC=25
):
    """Adjusts the temperature coefficient based on current conditions.

    Parameters:
    - e_poa: Plane of array irradiance in kW/m^2.
    - ambient_temp: Ambient temperature in degrees Celsius.
    - cell_temp_coeff: Temperature coefficient of the PV cell.
    - cell_temp_STC: Cell temperature at standard test conditions in degrees Celsius.

    Returns:
    - Adjusted temperature coefficient.
    """
    cell_temp = calc_cell_temp(e_poa, ambient_temp, cell_temp_coeff)
    temp_coeff = np.minimum(1, (1 + cell_temp_coeff * (cell_temp - cell_temp_STC)))

    return temp_coeff


def calc_pv_power(
    pv_kwp,
    e_poa,
    ambient_temp,
    pv_derating=1,
    cell_temp_coeff=-0.0035,
    e_poa_STC=1,
    cell_temp_STC=25,
):
    """Calculates the power output of a PV panel.

    Parameters:
    - pv_kwp: Rated power of the PV panel in kWp.
    - e_poa: Plane of array irradiance in kW/m^2.
    - ambient_temp: Ambient temperature in degrees Celsius.
    - pv_derating: Derating factor for the PV panel.
    - cell_temp_coeff: Temperature coefficient of the PV cell.
    - e_poa_STC: Irradiance at standard test conditions in kW/m^2.
    - cell_temp_STC: Cell temperature at standard test conditions in degrees Celsius.

    Returns:
    - Tuple of (pv_power, temp_diff), where pv_power is the power output in W, and temp_diff is the temperature difference effect on power output.
    """
    temp_coeff = calc_cell_temp_coeff(
        e_poa, ambient_temp, cell_temp_coeff, cell_temp_STC
    )

    pv_power = (pv_kwp * pv_derating * (e_poa / e_poa_STC) * temp_coeff) * 1000
    temp_diff = (pv_power / temp_coeff) - pv_power

    return pv_power, temp_diff


def calc_solar_model(
    data,
    latitude,
    longitude,
    pv_kwp=1,
    surface_pitch=35,
    surface_azimuth=0,
    lifespan=25,
    pv_derating=0.88,
    electrical_eff=0.21,
    albedo=0.2,
    cell_temp_coeff=-0.0035,
    transmittance_absorptance=0.9,
    refraction_index=0.1,
    cell_NOCT=42,
    ambient_NOCT=20,
    e_poa_NOCT=800,
    e_poa_STC=1000,
    cell_temp_STC=25,
    timestep=60,
    tmz_hrs_east=0,
):
    """Processes TMY data to simulate solar PV system performance over a typical meteorological year.

    Args:
        data (DataFrame): TMY data including irradiance and temperature.
        latitude (float): Site latitude.
        longitude (float): Site longitude.
        pv_kwp (float): Rated power of the PV system in kWp.
        surface_pitch (float): Tilt angle of the PV panel from the horizontal plane.
        surface_azimuth (float): Orientation of the PV panel from true north.
        lifespan (int): Estimated lifespan of the PV system in years.
        pv_derating (float): Derating factor of the PV system to account for end-of-life performance.
        electrical_eff (float): Electrical efficiency of the PV panel.
        albedo (float): Ground reflectance factor.
        cell_temp_coeff (float): Temperature coefficient of the PV cell per degree Celsius.
        transmittance_absorptance (float): Product of transmittance and absorptance of the PV panel.
        refraction_index (float): Refractive index of the panel's cover.
        cell_NOCT (float): Nominal operating cell temperature under specified test conditions.
        ambient_NOCT (float): Ambient temperature at NOCT.
        e_poa_NOCT (float): Plane of array irradiance at NOCT.
        e_poa_STC (float): Plane of array irradiance under standard test conditions.
        cell_temp_STC (float): Cell temperature under standard test conditions.
        timestep (int): Time interval for calculations in minutes.
        tmz_hrs_east (float): Time zone offset from GMT in hours.

    Returns:
        DataFrame: Enhanced TMY data with added columns for solar radiation calculations and PV system performance metrics.
    """
    # Create copy of TMY_data
    tmy_data = data.copy()

    # Declination
    tmy_data["Declination_Angle"] = sr.calc_declination(tmy_data["Day_of_Year"])

    # Solar Time
    tmy_data["Solar_Time"] = sr.calc_solar_time(
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        longitude,
        timestep=60,
        tmz_hrs_east=0,
    )

    # Hour Angle
    tmy_data["Hour_Angle"] = sr.calc_hour_angle(
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        longitude,
        timestep=60,
        tmz_hrs_east=0,
    )

    # Angle of Incidence
    tmy_data["AOI"] = sr.calc_aoi(
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        latitude,
        longitude,
        surface_azimuth,
        surface_pitch,
        timestep=60,
        tmz_hrs_east=0,
    )

    # Zenith Angle
    tmy_data["Zenith_Angle"] = sr.calc_zenith(
        latitude,
        longitude,
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        timestep=60,
        tmz_hrs_east=0,
    )

    # Beam Radiation
    tmy_data["E_Beam_kWm2"] = sr.calc_beam_radiation(
        tmy_data["Gb(n)"],
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        latitude,
        longitude,
        surface_azimuth,
        surface_pitch,
        timestep,
        tmz_hrs_east,
    )

    # Diffuse Radiation
    tmy_data["E_Diffuse_kWm2"] = sr.calc_diffuse_radiation(
        tmy_data["Gd(h)"],
        tmy_data["G(h)"],
        surface_pitch,
        latitude,
        longitude,
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        timestep,
        tmz_hrs_east,
    )

    # Ground Radiation
    tmy_data["E_Ground_kWm2"] = sr.calc_ground_radiation(
        tmy_data["G(h)"], surface_pitch, albedo
    )

    # POA Irradiance
    tmy_data["E_POA_kWm2"] = (
        tmy_data["E_Beam_kWm2"] + tmy_data["E_Diffuse_kWm2"] + tmy_data["E_Ground_kWm2"]
    )

    # Adjust E_POA_kWm2 to include IAM losses
    tmy_data["Panel_POA_kWm2"] = (
        tmy_data["E_Beam_kWm2"]
        * (
            iam_losses(
                tmy_data["Day_of_Year"],
                tmy_data["Hour_of_Day"],
                latitude,
                longitude,
                surface_azimuth,
                surface_pitch,
                timestep,
                tmz_hrs_east,
                refraction_index,
            )
        )
        + tmy_data["E_Diffuse_kWm2"]
        + tmy_data["E_Ground_kWm2"]
    )

    # Record IAM losses
    tmy_data["IAM_Loss_kWm2"] = tmy_data["E_POA_kWm2"] - tmy_data["Panel_POA_kWm2"]

    # Extra Terrestrial Horizontal Irradiation
    tmy_data["ET_HRad_kWm2"] = sr.calc_et_horizontal_radiation(
        latitude,
        longitude,
        tmy_data["Day_of_Year"],
        tmy_data["Hour_of_Day"],
        timestep,
        tmz_hrs_east,
    )

    # PV Panel Derating
    tmy_data["PV_Derated_Eff"] = calc_pv_derating(
        tmy_data["Day_of_Year"], tmy_data["Hour_of_Day"], pv_derating, lifespan, year=1
    )

    # Cell Temperature
    tmy_data["Cell_Temp_C"] = calc_cell_temp(
        tmy_data["Panel_POA_kWm2"],
        tmy_data["T2m"],
        cell_temp_coeff,
        electrical_eff,
        cell_NOCT,
        ambient_NOCT,
        e_poa_NOCT,
        cell_temp_STC,
        transmittance_absorptance,
    )

    # PV Generation
    tmy_data["PV_Gen_kWh"], tmy_data["PV_Thermal_Loss_kWh"] = calc_pv_power(
        pv_kwp,
        tmy_data["Panel_POA_kWm2"],
        tmy_data["T2m"],
        tmy_data["PV_Derated_Eff"],
        cell_temp_coeff,
        e_poa_STC,
        cell_temp_STC,
    )

    return tmy_data


def combine_array_results(results):
    """Combines results from multiple PV array simulations into a single DataFrame.

    Parameters:
    - results: List of DataFrames, each representing the output from a single PV array simulation.

    Returns:
    - Combined DataFrame with metrics from all arrays.
    """
    columns_to_combine = [
        "E_Beam_kWm2",
        "E_Diffuse_kWm2",
        "E_Ground_kWm2",
        "E_POA_kWm2",
        "Panel_POA_kWm2",
        "ET_HRad_kWm2",
        "Cell_Temp_C",
        "PV_Gen_kWh",
        "PV_Thermal_Loss_kWh",
        "IAM_Loss_kWm2",
    ]

    columns_to_add = ["AOI", "Zenith_Angle"]
    time_columns = [
        "Declination_Angle",
        "Solar_Time",
        "Hour_Angle",
        "T2m",
        "Hour_of_Day",
        "Day_of_Year",
        "Week_of_Year",
        "Month_of_Year",
    ]
    array_data = []

    # Extract, label, and append data for each array
    for i, result in enumerate(results, start=1):
        # Select both types of columns to combine and to add
        df_to_combine = result["model_result"][columns_to_combine].copy()
        df_to_add = result["model_result"][columns_to_add].copy()

        # Rename columns for clarity
        df_to_combine.columns = [f"{col}_Array_{i}" for col in columns_to_combine]
        df_to_add.columns = [f"{col}_Array_{i}" for col in columns_to_add]

        # Concatenate horizontally to align the additional columns with the combined ones
        array_df = pd.concat([df_to_combine, df_to_add], axis=1)
        array_data.append(array_df)

    # Concatenate all arrays data side by side
    combined_df = pd.concat(array_data, axis=1)

    # For summing specific columns (excluding Cell_Temp_C), and averaging Cell_Temp_C
    for col in columns_to_combine:
        if col != "Cell_Temp_C":
            combined_df[f"{col}_Total"] = combined_df.filter(
                regex=f"^{col}_Array_"
            ).sum(axis=1)
        else:
            combined_df["Cell_Temp_C_Avg"] = combined_df.filter(
                regex=f"^{col}_Array_"
            ).mean(axis=1)

    # Append the time columns from the first result's DataFrame
    for time_col in time_columns:
        combined_df[time_col] = results[0]["model_result"][time_col]

    return combined_df


def total_array_results(results):
    """Aggregates results from multiple PV array simulations.

    Parameters:
    - results: List of DataFrames, each representing the output from a single PV array simulation.

    Returns:
    - DataFrame with aggregated metrics across all arrays.
    """
    columns_to_combine = [
        "E_Beam_kWm2",
        "E_Diffuse_kWm2",
        "E_Ground_kWm2",
        "E_POA_kWm2",
        "Panel_POA_kWm2",
        "ET_HRad_kWm2",
        "Cell_Temp_C",
        "PV_Gen_kWh",
        "PV_Thermal_Loss_kWh",
        "IAM_Loss_kWm2",
    ]

    time_columns = [
        "Declination_Angle",
        "Solar_Time",
        "Hour_Angle",
        "T2m",
        "Hour_of_Day",
        "Day_of_Year",
        "Week_of_Year",
        "Month_of_Year",
    ]

    combined_df = pd.DataFrame()

    # Combine specified columns across all arrays, summing or averaging as appropriate
    for col in columns_to_combine:
        if col != "Cell_Temp_C":
            combined_df[f"{col}_Total"] = sum(
                result["model_result"][col] for result in results
            )
        else:
            combined_df["Cell_Temp_C_Avg"] = sum(
                result["model_result"][col] for result in results
            ) / len(results)

    # Copy the time columns from the first result's DataFrame without modification
    for time_col in time_columns:
        combined_df[time_col] = results[0]["model_result"][time_col]

    return combined_df


def pv_stats(model_results, arrays):
    """Generates a summary of key PV performance metrics over the specified period.

    Parameters:
    - model_results: DataFrame containing model results.
    - arrays: List of PV array configurations used in the model.

    Returns:
    - Series with summarized PV performance metrics.
    """
    # Columns to sum
    columns_to_sum = [
        "E_Beam_kWm2_Total",
        "E_Diffuse_kWm2_Total",
        "E_Ground_kWm2_Total",
        "E_POA_kWm2_Total",
        "Panel_POA_kWm2_Total",
        "ET_HRad_kWm2_Total",
        "PV_Gen_kWh_Total",
        "PV_Thermal_Loss_kWh_Total",
        "IAM_Loss_kWm2_Total",
    ]

    # Columns to calculate the mean
    columns_to_mean = ["Cell_Temp_C_Avg", "T2m"]

    # Initialize a dictionary to hold the summary
    summary = {}

    # Sum the specified columns
    for col in columns_to_sum:
        summary[col] = model_results[col].sum()

    # Calculate the mean for the specified columns
    for col in columns_to_mean:
        summary[col] = model_results[col].mean()

    lifespan = arrays[0].lifespan
    eol_derating = arrays[0].pv_eol_derating
    yearly_derating = (1 - eol_derating) / lifespan
    total_gen = 0

    for i in range(lifespan):
        gen = (1 - (yearly_derating * (i + 1))) * summary["PV_Gen_kWh_Total"]
        total_gen += gen

    summary["PV_Gen_kWh_Lifetime"] = total_gen
    summary["PV_Gen_kWh_Annual"] = summary["PV_Gen_kWh_Total"]
    summary["PV_Thermal_Loss_kWh_Annual"] = summary["PV_Thermal_Loss_kWh_Total"]
    summary["IAM_Loss_kWm2_Annual"] = summary["IAM_Loss_kWm2_Total"]
    summary["Panel_POA_kWm2_Annual"] = summary["Panel_POA_kWm2_Total"]
    summary["E_POA_kWm2_Annual"] = summary["E_POA_kWm2_Total"]
    summary["E_Beam_kWm2_Annual"] = summary["E_Beam_kWm2_Total"]
    summary["E_Diffuse_kWm2_Annual"] = summary["E_Diffuse_kWm2_Total"]
    summary["E_Ground_kWm2_Annual"] = summary["E_Ground_kWm2_Total"]
    summary["ET_HRad_kWm2_Annual"] = summary["ET_HRad_kWm2_Total"]
    summary["T2m_Avg"] = summary["T2m"]

    # Define the desired order of keys
    desired_order = [
        "PV_Gen_kWh_Annual",
        "PV_Gen_kWh_Lifetime",
        "E_POA_kWm2_Annual",
        "Panel_POA_kWm2_Annual",
        "IAM_Loss_kWm2_Annual",
        "PV_Thermal_Loss_kWh_Annual",
        "E_Beam_kWm2_Annual",
        "E_Diffuse_kWm2_Annual",
        "E_Ground_kWm2_Annual",
        "ET_HRad_kWm2_Annual",
        "Cell_Temp_C_Avg",
        "T2m_Avg",
    ]

    # Create an OrderedDict with items in the desired order
    ordered_summary = OrderedDict((k, summary[k]) for k in desired_order)

    # Convert to pandas Series and round the values
    summary_series = pd.Series(ordered_summary).round(3)

    return summary_series


class SummaryGrouped:
    """Organizes grouped summary statistics of PV system performance.
    Parameters:
    - summaries (dict): Dictionary with time grouping as keys and summary statistics DataFrames as values.
    """

    def __init__(self, summaries):
        for key, df in summaries.items():
            setattr(self, key.lower(), df)


def pv_stats_grouped(model_results):
    """Generates grouped statistics of PV performance over different time frames.

    Parameters:
    - model_results: DataFrame containing model results.

    Returns:
    - SummaryGrouped object containing DataFrames of grouped statistics.
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
        "E_POA_kWm2_Total",
        "Panel_POA_kWm2_Total",
        "IAM_Loss_kWm2_Total",
        "PV_Thermal_Loss_kWh_Total",
        "E_Beam_kWm2_Total",
        "E_Diffuse_kWm2_Total",
        "E_Ground_kWm2_Total",
        "ET_HRad_kWm2_Total",
    ]
    columns_to_mean = ["Cell_Temp_C_Avg", "T2m"]

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
