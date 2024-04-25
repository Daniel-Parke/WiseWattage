import logging
import pandas as pd

from dataclasses import dataclass, field
from typing import List, Union, Dict
from functools import cached_property

from solar.SolarPVModel import Site, SolarPVArray, SolarPVPanel, SolarPVModel
from demand.Load import Load
from utility.Grid import Grid
from utility.Inverter import Inverter
from storage.Battery import Battery
from finance.Finance import Finance

from ww_models.ww_model import (
    initialise_model, calc_solar_energy_flow, calc_battery_energy_flow, 
    calc_grid_energy_flow, sort_columns, model_stats, model_grouped, SummaryGrouped,
    calculate_capex, calculate_operation_costs, calculate_npc)

from misc.util import timer



@dataclass
class Model:
    site: 'Site'
    load: 'Load' = None
    grid: 'Grid' = None
    arrays: Union['SolarPVArray', List['SolarPVArray']] = None
    inverter: 'Inverter' = None
    battery: 'Battery' = None
    pv_model: 'SolarPVModel' = None
    finance: 'Finance' = None   
    name: str = ""
    project_lifespan: int = 25
    model: pd.DataFrame = field(default=None, init=False)

    capex: float = 0
    replacement_capex: float = 0
    opex_cost: float = 0
    export_value: float = 0
    npc: float = 0

    @timer
    def __post_init__(self):
        initialise_model(self)
        calc_solar_energy_flow(self)
        calc_battery_energy_flow(self)
        calc_grid_energy_flow(self)
        sort_columns(self)
        calculate_capex(self)
        calculate_operation_costs(self)
        calculate_npc(self)
            

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Generates a summary of model results the first time it is called and caches the result.

        Returns:
            pd.DataFrame: DataFrame containing summary of model results.
        """
        logging.info(f"Generating Model statistical analysis for {self.site.name}.")
        summary = model_stats(self.model, self.arrays)
        logging.info(f"Model statistical analysis for {self.site.name} completed.")
        logging.info("*******************")
        return summary
    

    @cached_property
    def grouped(self) -> 'SummaryGrouped':
        """
        Generates a grouped summary of model results the first time it is called and caches the result.

        Returns:
            SummaryGrouped: Object containing grouped summary of model results.
        """
        logging.info(f"Generating Model statistical grouping for {self.site.name}.")
        grouped = model_grouped(self.model)
        logging.info(f"Model statistical grouping for {self.site.name} completed.")
        logging.info("*******************")
        return grouped