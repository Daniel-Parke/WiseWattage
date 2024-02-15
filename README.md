# Wattsun and Homes - Solar PV Energy Modelling Application
### Video Demo:  <URL HERE> **WORK IN PROGRESS, APP WILL LAUNCH SOMETIME AFTER COMPLETION OF CS50 WEB**
### Description
This project is an energy modelling application designed to model solar PV potential across a range of sites and system parameters. It's accessed by creating Class objects which conduct the appropriate modelling when initialized. The models simulated can then be accessed using `.xxx` notation, with some basic time series data aggregation methods already built in. These models are returned as dataframes, with the library being built with a focus on using it offline for personal data analysis.

For this CS50 project, I will utilize the same codebase to build an online application where users can generate their own solar PV models. This will be completed using Flask, offering basic functionality for now. Users will be prompted to create an account to save the model results for access later, and they can generate an energy report summarizing the model created.

The library will be further expanded after CS50 to include the following features in the energy model:
<br>

**In development:**
- [x] Creation of `get_TMY` function to retrieve TMY meteorological data for a given location
- [x] Creation of solar radiation calculation codebase and integration with TMY data
- [x] Creation of fundamental solar PV power output calculations
- [x] Integration of sky model, clamping of beam/diffuse values & timestep adjustment
- [x] Adjustment of formulas to remove potential erroneous generation calc when AOI/Zenith close to 90 degrees
- [x] Inclusion of base physical processes affecting solar PV performance (Thermals, derating, etc.)
- [x] Inclusion of IAM losses to reduce beam insolation
- [x] Vectorised functionality added to radiation and modelling calculations to improve performance
- [x] Data aggregation functionality and time series grouping
- [x] Data logging implemented to assist with debugging and awareness of timescales for large models
- [x] Creation of `Site` class to allow for storing of TMY and project data
- [x] Function caching of TMY data enabled for faster reiteration
- [x] Creation of `SolarPVArray` class to allow for multiple scenarios to be modelled at the same location
- [x] Creation of `SolarPVModel` class to simulate solar PV performance at location
- [x] Data aggregation functions added to initial PVModel class that enable easier statistical analysis
- [x] Storing of individual full model results as well as aggregated data to be accessed separately
- [x] Enable saving of produced models as CSV file
- [x] Basic exception handling implemented
<br>

**Planned features:**
- [ ] Inclusion of low irradiance losses
- [ ] Creation of project class in which all other models and data generated will be stored and accessed
- [ ] Enable multiple iterations of models to be completed across sites for site analysis/comparison
- [ ] Inclusion of base project financials to estimate cost for solar PV project
- [ ] Further statistical analysis functions to be added that target desired columns
- [ ] Chart generation based on comparison of important characteristics
- [ ] Development of Flask web app to enable user to easily generate their own models
- [ ] Integration of generated statistical analysis and charts from Class object to Flask webpage
- [ ] Creation of user account and auth process to enable long term storage of models
- [ ] Creation and integration of SQL database to enable long term storage and easier integration elsewhere
<br>

**COMING SOON:**
- [ ] Ability to add load profile to compare demand to energy generation
- [ ] National Grid Connection with variable import/export tariffs for day-night tariffs
- [ ] Integration of energy storage technologies
- [ ] System controller for analysis of different energy priorities/scheduling
- [ ] Financial analysis and project creation for modelling multiple scenarios
- [ ] Alternative AC power sources such as off-grid generators
- [ ] Wind Turbine and other DC energy sources
- [ ] Integration of EV battery charge/discharge scheduling
<br>

**MEDIUM TERM GOALS:**
- [ ] Energy model optimisation procedure
- [ ] Built-in class methods for sensitivity analysis modelling
- [ ] Predictive and historical analysis based on user personal demand
<br>

**LONG TERM GOALS:**
- [ ] Real-time forecasting and integration into smart meter/IOT appliances
- [ ] Lead generation for equipment vendors and installers based on location, price, system, etc.
- [ ] Lead generation for sales/marketing teams to generate leads ahead of time or inform on the ground data
- [ ] Value generation for aggregated data generated through platform. Ensure anonymity of users!
- [ ] Wider app + database with documentation to allow access from external queries through API service without GUI
<br>


## System Schematic
**Future System Design Schematic**
![alt text](<static/Solar_Model_Technical_Diagram_2.png>)
<br>


## Model Classes

**Site**
```python
site = Site()
site.name
site.address
site.client
site.latitude
site.longitude
site.tmz_hrs_east
site.tmy_data
```

**SolarPVArray**
```python
array_1 = SolarPVArray()
array_1.pv_kwp
array_1.surface_pitch
array_1.surface_azimuth
array_1.electrical_eff
array_1.pv_eol_derating
array_1.lifespan
array_1.cell_temp_coeff
array_1.transmittance_absorptance
array_1.cell_NOCT
array_1.ambient_NOCT
array_1.e_poa_NOCT
array_1.e_poa_STC
array_1.cell_temp_STC
```

**SolarPVModel**
```python
pv_model = SolarPVModel()
pv_model.site
pv_model.arrays
pv_model.models
pv_model.all_models
pv_model.combined_model
pv_model.summary
pv_model.summary_grouped
pv_model.albedo
pv_model.timestep

pv_model.array_model(n)
pv_model.save_model_csv()
pv_model.model_summary_html_export(freq, grouped)
pv_model.plot_model(params, model_index, plot_type)
pv_model.plot_combined(params, plot_type)
pv_model.plot_sum(params, group, plot_type)

```
