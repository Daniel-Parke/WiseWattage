
from solar.SolarPVModel import SolarPVModel
from demand.Load import Load
from utility.Grid import Grid


def initialise_model(self):
    if self.load is None:
        self.load = Load()

    if self.grid is None:
        self.grid = Grid()
        
    if self.arrays is not None:
        self.pv_model = SolarPVModel(self.site, self.arrays)