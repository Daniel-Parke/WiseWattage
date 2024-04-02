from dataclasses import dataclass, field
from typing import List, Dict, Union
import pandas as pd
import logging
from functools import cached_property

from meteo.Site import Site
from solar.SolarPVPanel import SolarPVPanel
from solar.SolarPVArray import SolarPVArray
from solar.solar_pv_model import (
    calc_solar_model,
    combine_array_results,
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
    """

    site: 'Site'
    arrays: Union['SolarPVArray', List['SolarPVArray']]
    models: List[Dict] = field(default_factory=list)
    all_models: pd.DataFrame = field(default=None, init=False)
    combined_model: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization method.
        Runs model_solar_pv method to perform solar PV modeling.
        """
        # Normalize self.arrays to be a list if not already
        if not isinstance(self.arrays, list):
            self.arrays = [self.arrays]
            
        # Run solar PV model
        self.model_solar_pv()


    def model_solar_pv(self):
        """
        Method to perform solar PV modeling and generate model results.
        """
        logging.info("*******************")
        logging.info(f"Starting Solar PV model simulations for {self.site.name}.")
        logging.info("*******************")
        models = []
        for array in self.arrays:
            log_message = (f"Simulating model - PV Size: {array.pv_kwp}kWp, Pitch: {array.surface_pitch} degrees, "
                           f"Azimuth {array.surface_azimuth} degrees WoS")
            logging.info(log_message)
            result = calc_solar_model(
                self.site.tmy_data,
                self.site.latitude,
                self.site.longitude,
                array.pv_kwp,
                array.surface_pitch,
                array.surface_azimuth,
                array.pv_panel.lifespan,
                array.pv_panel.pv_eol_derating,
                array.albedo,
                array.pv_panel.cell_temp_coeff,
                array.pv_panel.refraction_index,
                array.pv_panel.e_poa_STC,
                array.pv_panel.cell_temp_STC,
                self.site.timestep,
                self.site.tmz_hrs_east,
            )
            models.append({"array_specs": array, "model_result": result})
        
        # Arrange model results into class structure
        logging.info("*******************")
        logging.info(f"Solar PV model simulations for {self.site.name} completed.")
        self.models = models
        self.all_models = combine_array_results(models)
        logging.info(f"Solar PV model data aggregated.")
        self.combined_model = total_array_results(models)
        logging.info(f"Solar PV model data summary for {self.site.name} complete.")
        logging.info("      SUCCESS!      ") 


    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Generates a summary of model results the first time it is called and caches the result.

        Returns:
            pd.DataFrame: DataFrame containing summary of model results.
        """
        logging.info(f"Generating Solar PV model statistical analysis for {self.site.name}.")
        summary = pv_stats(self.all_models, self.arrays)
        logging.info(f"Solar PV model statistical analysis for {self.site.name} completed.")
        logging.info("*******************")
        return summary
    

    @cached_property
    def summary_grouped(self) -> 'SummaryGrouped':
        """
        Generates a grouped summary of model results the first time it is called and caches the result.

        Returns:
            SummaryGrouped: Object containing grouped summary of model results.
        """
        logging.info(f"Generating Solar PV model statistical grouping for {self.site.name}.")
        summary_grouped = pv_stats_grouped(self.all_models)
        logging.info(f"Solar PV model statistical grouping for {self.site.name} completed.")
        logging.info("*******************")
        return summary_grouped


    def save_model_csv(self):
        """
        Method to save model results to a CSV file.
        """
        if self.all_models is not None:
            self.all_models.to_csv(f"Solar_Modelling_{self.site.name}_All_Models.csv")
            self.combined_model.to_csv(f"Solar_Modelling_{self.site.name}_Combined.csv")
            logging.info(f"Model data for {self.site.name} saved to csv file completed.")
            logging.info("*******************")
        else:
            logging.info("Model data NOT saved, no model results found.")
            logging.info("*******************")

