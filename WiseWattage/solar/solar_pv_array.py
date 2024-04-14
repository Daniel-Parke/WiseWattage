import logging

from solar.SolarPVPanel import SolarPVPanel

def initialise_array(self):
    if self.pv_panel is None:
        self.pv_panel = SolarPVPanel()

    self.pv_kwp = round(self.pv_panel.panel_kwp * self.num_panels, 3)
    self.area_m2 = round(self.pv_panel.size_m2 * self.num_panels, 3)

    logging.info(
            f"Solar PV array created: Size: {self.pv_kwp}kW, Size: {self.area_m2}m2, "
            f"Azimuth: {self.surface_azimuth}deg, Lifespan: {self.pv_panel.lifespan}yrs, "
            f"Pitch: {self.surface_pitch}deg, Efficiency: {self.pv_panel.eff*100}%"
        )