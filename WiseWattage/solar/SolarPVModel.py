from dataclasses import dataclass, field
from typing import List, Dict, Union
import pandas as pd
import logging
from functools import cached_property

from meteo.Site import Site
from solar.SolarPVPanel import SolarPVPanel
from solar.SolarPVArray import SolarPVArray
from solar.solar_pv_model import (
    model_solar_pv,
    pv_stats,
    pv_grouped,
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

    cost: float = 0
    weight_kg: float = 0

    def __post_init__(self):
        """
        Post-initialization method.
        Runs model_solar_pv method to perform solar PV modeling.
        """
        # Normalize self.arrays to be a list if not already
        if not isinstance(self.arrays, list):
            self.arrays = [self.arrays]
            
        # Run solar PV model
        model_solar_pv(self) 


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
    def grouped(self) -> 'SummaryGrouped':
        """
        Generates a grouped summary of model results the first time it is called and caches the result.

        Returns:
            SummaryGrouped: Object containing grouped summary of model results.
        """
        logging.info(f"Generating Solar PV model statistical grouping for {self.site.name}.")
        grouped = pv_grouped(self.all_models)
        logging.info(f"Solar PV model statistical grouping for {self.site.name} completed.")
        logging.info("*******************")
        return grouped


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

