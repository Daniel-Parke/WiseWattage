from dataclasses import dataclass
from typing import List, Union, Dict

from solar.SolarPVModel import Site, SolarPVArray, SolarPVPanel, SolarPVModel
from demand.Load import Load
from utility.Grid import Grid

from ww_models.ww_model import initialise_model



@dataclass
class Model:
    site: 'Site'
    arrays: Union['SolarPVArray', List['SolarPVArray']] = None
    load: 'Load' = None
    grid: 'Grid' = None
    pv_model: SolarPVModel = None
    name: str = ""


    def __post_init__(self):
        initialise_model(self)
            
