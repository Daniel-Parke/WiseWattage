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
    loan_amount: float = 0
    apr: float = 0.079
    payments_per_year: int = 12
    loan_years: int = 5
    fees: float = 0
    upfront_fees: bool = True

    finance_model: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        initialise_finance(self)
        calc_cashflow(self)