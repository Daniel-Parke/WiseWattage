import pandas as pd

from dataclasses import dataclass, field
from typing import List, Union, Dict

from solar.SolarPVModel import Site, SolarPVArray, SolarPVPanel, SolarPVModel
from demand.Load import Load
from utility.Grid import Grid
from utility.Inverter import Inverter
from storage.Battery import Battery

from ww_models.ww_model import (initialise_model, calc_solar_energy_flow, 
                                calc_battery_energy_flow, calc_grid_energy_flow, sort_columns)

from misc.util import timer



@dataclass
class Model:
    site: 'Site'
    load: 'Load' = None
    grid: 'Grid' = None
    arrays: Union['SolarPVArray', List['SolarPVArray']] = None
    inverter: 'Inverter' = None
    battery: 'Battery' = None
    pv_model: SolarPVModel = None
    name: str = ""
    model: pd.DataFrame = field(default=None, init=False)

    @timer
    def __post_init__(self):
        initialise_model(self)
        calc_solar_energy_flow(self)
        calc_battery_energy_flow(self)
        calc_grid_energy_flow(self)
        sort_columns(self)
            
