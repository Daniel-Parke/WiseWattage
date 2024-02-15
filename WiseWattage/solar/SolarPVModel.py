from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd
import logging
import plotly.express as px

from meteo.Site import Site
from solar.SolarPVArray import SolarPVArray
from solar.solar_pv_model import calc_solar_model, combine_array_results
from solar.solar_pv_model import (
    total_array_results,
    pv_stats,
    pv_stats_grouped,
    SummaryGrouped,
)


@dataclass
class SolarPVModel:
    """
    A class representing a Solar PV Model.

    Attributes:
        site (Site): Information about the site.
        arrays (List[SolarPVArray]): List of SolarPVArray objects.
        models (List[Dict]): List of dictionaries containing model results for each array.
        all_models (pd.DataFrame): DataFrame containing all model results.
        combined_model (pd.DataFrame): DataFrame containing combined model results.
        summary (pd.DataFrame): DataFrame containing summary of model results.
        summary_grouped (SummaryGrouped): Object containing grouped summary of model results.
        albedo (float): Albedo coefficient.
        timestep (int): Time step for modeling.
    """

    site: Site
    arrays: List[SolarPVArray]
    models: List[Dict] = field(default_factory=list)
    all_models: pd.DataFrame = field(default=None, init=False)
    combined_model: pd.DataFrame = field(default=None, init=False)
    summary: pd.DataFrame = field(default=None, init=False)
    summary_grouped: SummaryGrouped = field(default=None, init=False)
    albedo: float = 0.2
    timestep: int = 60

    def __post_init__(self):
        """
        Post-initialization method.
        Runs model_solar_pv method to perform solar PV modeling and generate summaries.
        """
        self.model_solar_pv()  # Run solar PV modeling
        self.summary = pv_stats(
            self.all_models, self.arrays
        )  # Generate summary statistics
        logging.info("Model statistical analysis completed.")
        self.summary_grouped = pv_stats_grouped(
            self.all_models
        )  # Generate grouped summary statistics
        logging.info("Model statistical grouping completed.")
        logging.info("*******************")

    def model_solar_pv(self):
        """
        Method to perform solar PV modeling and generate model results.
        """
        logging.info("*******************")
        logging.info("Starting model calculations for SolarPVModel.")
        logging.info("*******************")
        models = []
        for array in self.arrays:
            log_message = (
                f"Simulating model for {array.pv_kwp}kWp, {array.surface_pitch} degrees pitch "
                f"& azimuth at {array.surface_azimuth} degrees WoS"
            )
            logging.info(log_message)
            result = calc_solar_model(
                self.site.tmy_data,
                self.site.latitude,
                self.site.longitude,
                array.pv_kwp,
                array.surface_pitch,
                array.surface_azimuth,
                array.lifespan,
                array.pv_eol_derating,
                array.electrical_eff,
                self.albedo,
                array.cell_temp_coeff,
                array.transmittance_absorptance,
                array.refraction_index,
                array.cell_NOCT,
                array.ambient_NOCT,
                array.e_poa_NOCT,
                array.e_poa_STC,
                array.cell_temp_STC,
                self.timestep,
                self.site.tmz_hrs_east,
            )
            models.append({"array_specs": array, "model_result": result})
        logging.info("*******************")
        logging.info("Model calculations completed.")
        self.models = models
        self.all_models = combine_array_results(models)
        logging.info("Model data aggregated.")
        self.combined_model = total_array_results(models)
        logging.info("Model data summary complete.")
        logging.info("*******************")

    def save_model_csv(self):
        """
        Method to save model results to a CSV file.
        """
        if self.all_models is not None:
            self.all_models.to_csv(f"Solar_Modelling_{self.site.name}_Combined.csv")
            logging.info("Model data saved to csv file completed.")
        logging.info("*******************")

    # Returns model for # array at # index location in results
    def array_model(self, n):
        """
        Returns model for a specific array at a given index location in results.

        Args:
            n (int): Index of the array.

        Returns:
            str or dict: Model result for the array.
        """
        try:
            return self.models[n]["model_result"]
        except IndexError:
            return f"There was no model at index {n}, check the number of arrays modelled and try again."

    def model_summary_html_export(self, freq=None, grouped=True):
        """
        Export model summary to HTML format.

        Args:
            freq (str): Frequency for grouped summary (e.g., 'daily', 'monthly').
            grouped (bool): Whether to export grouped summary or not.

        Returns:
            str or dict: HTML-formatted summary data or dictionary of summary data.
        """
        org_col_grouped = [
            "PV_Gen_kWh_Total",
            "Panel_POA_kWm2_Total",
            "IAM_Loss_kWm2_Total",
            "PV_Thermal_Loss_kWh_Total",
            "ET_HRad_kWm2_Total",
            "E_Beam_kWm2_Total",
            "E_Diffuse_kWm2_Total",
            "E_Ground_kWm2_Total",
            "Cell_Temp_C_Avg",
            "T2m",
        ]
        new_col_grouped = [
            "PV Generation (kWh)",
            "POA Radiation (kWh)",
            "AOI Losses (kWh)",
            "Thermal Losses (kWh)",
            "ET Horizontal Radiation",
            "Beam Radiation (kWh)",
            "Diffuse Radiation (kWh)",
            "Ground Radiation (kWh)",
            "Average Cell Temperature (C)",
            "Average Ambient Temp (C)",
        ]

        org_col_summary = [
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
        new_col_summary = [
            "PV Generation (kWh)",
            "Lifetime PV Generation (kWh)",
            "E POA (kWhm2)",
            "Panel POA (kWhm2)",
            "AOI Reflected Loss (kWhm2)",
            "PV Thermal Loss (kWh)",
            "Beam Radiation (kWhm2)",
            "Diffuse Radiation (kWhm2)",
            "Ground Radiation (kWhm2)",
            "Average Cell Temperature (C)",
            "Average Ambient Temp (C)",
        ]

        if grouped == True:
            data = getattr(self.summary_grouped, freq, None)
            data_new = data[org_col_grouped].copy()
            data_new.columns = new_col_grouped
            data_html = data_new.to_html()
            return data_html
        else:
            data = self.summary
            data_new = data[org_col_summary].copy()
            index_mapping = dict(zip(org_col_summary, new_col_summary))
            data_new = data_new.rename(index=index_mapping)
            data_dict = data_new.to_dict()
            return data_dict

    def plot_model(self, params, model_index=0, plot_type="line"):
        """
        Plot model data.

        Args:
            params (list): List of parameters to plot.
            model_index (int): Index of the model.
            plot_type (str): Type of plot (e.g., 'line', 'bar').

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        model_df = self.models[model_index]["model_result"]

        if params and all(param in model_df.columns for param in params):
            # Dynamically select the Plotly Express plotting function based on plot_type
            plot_func = getattr(px, plot_type.lower(), None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(
                    model_df,
                    x=model_df.index,
                    y=params[:],
                    title=f"Chart showing modelled value for {params} over a year.",
                )
                return fig
            else:
                print(
                    f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'."
                )
        else:
            print(
                "Error: Invalid parameters or parameters not found in DataFrame columns."
            )

    def plot_combined(self, params, plot_type="line"):
        """
        Plot combined model data.

        Args:
            params (list): List of parameters to plot.
            plot_type (str): Type of plot (e.g., 'line', 'bar').

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        # Ensure the DataFrame 'combined_model' has the columns specified in 'params'
        if params and all(param in self.combined_model.columns for param in params):
            # Dynamically select the Plotly Express plotting function based on plot_type
            plot_func = getattr(px, plot_type.lower(), None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(
                    self.combined_model,
                    x=self.combined_model.index,
                    y=params[:],
                    title=f"Chart showing aggregated modelled {params} over a year.",
                )
                return fig
            else:
                print(
                    f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'."
                )
        else:
            print(
                "Error: Invalid parameters or parameters not found in DataFrame columns."
            )

    def plot_sum(self, params, group="daily", plot_type="line"):
        """
        Plot grouped model data.

        Args:
            params (list): List of parameters to plot.
            group (str): Group for plotting (e.g., 'daily', 'monthly').
            plot_type (str): Type of plot (e.g., 'line', 'bar').

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        # Dynamically get the group DataFrame based on the 'group' string parameter
        group_df = getattr(self.summary_grouped, group, None)

        # Check if the group DataFrame exists and has the specified columns
        if (
            group_df is not None
            and params
            and all(param in group_df.columns for param in params)
        ):
            # Dynamically select the Plotly Express plotting function based on plot_type
            plot_func = getattr(px, plot_type, None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(
                    group_df,
                    x=group_df.index,
                    y=params[:],
                    title=f"Chart showing modelled {params} grouped by {group} values.",
                )
                return fig
            else:
                print(
                    f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'."
                )
        else:
            print(
                "Error: Invalid parameters, group not found, or parameters not found in DataFrame columns."
            )
