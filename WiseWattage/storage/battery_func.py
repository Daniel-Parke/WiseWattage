

def initialise_class(self):
    self.useable_capacity = round(self.max_capacity * (1-self.discharge_depth), 3)
    self.max_discharge_kW = self.useable_capacity * self.max_discharge_C
    self.max_charge_kW = self.useable_capacity * self.max_charge_C