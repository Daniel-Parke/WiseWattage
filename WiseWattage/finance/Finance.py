import pandas as pd

from dataclasses import dataclass, field

from solar.SolarPVModel import SolarPVModel
from demand.Load import Load
from utility.Grid import Grid
from utility.Inverter import Inverter
from storage.Battery import Battery

from finance.financial import initialise_finance, calc_cashflow

@dataclass
class Finance:
    load: 'Load' = None
    grid: 'Grid' = None
    inverter: 'Inverter' = None
    battery: 'Battery' = None
    pv_model: SolarPVModel = None
    energy_model: pd.DataFrame = field(default=None, init=False)
    finance_model: pd.DataFrame = field(default=None, init=False)

    loan_amount: float = 10000
    fees: float = 1000
    upfront_fees: bool = True
    apr: float = 0.079
    payments_per_year: int = 12
    loan_years: int = 5

    project_lifespan: int = 25
    capex: float = 0
    replacement_capex: float = 0
    opex_cost: float = 0
    export_value: float = 0
    npc: float = 0

    def __post_init__(self):
        initialise_finance(self)
        calc_cashflow(self)