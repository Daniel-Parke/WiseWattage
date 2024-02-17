# Wattsun and Homes - Energy Modelling Application
### Video Demo:  <URL HERE> **WORK IN PROGRESS, APP WILL LAUNCH SOMETIME AFTER COMPLETION OF CS50 WEB**
### Description
This project is an energy modelling application designed to model Solar PV potential across a range of sites and system parameters. It's accessed by creating Class objects which conduct the appropriate modelling when initialized. The models simulated can then be accessed using `.xxx` notation, with some basic time series data aggregation methods already built in. These models are returned as dataframes, with the library being built with a focus on using it offline for personal data analysis.

Once this project has been completed I will utilize the same codebase to build an online application where users can generate their own solar PV models. This will be completed using Flask, Django and/or FastAPI, offering basic API returns and a browser UI to run the model. Users will be prompted to create an account to save the model results for access later if they want, and they can generate an energy report summarizing the model created.

The library will be further expanded to include the following features in the energy model:
<br>

## System Schematic
**Future System Model Design Schematic**
![alt text](<static/Solar_Model_Technical_Diagram_2.png>)
<br>

**Planned features:**
- [ ] Inclusion of low irradiance losses
- [ ] Inclusion of spectral losses
- [ ] Inclusion of system/conversion losses
- [ ] Inclusion of base project financials to estimate cost for solar PV project
- [ ] Further statistical analysis functions to be added that target desired columns
- [ ] Full range of charts generated based on comparison of important characteristics
- [ ] Creation of project class in which all other models and data generated will be stored and accessed
- [ ] Batch processing and asynchronous functions for jrc_tmy when modelling multiple sides. Current
API response time is slowest part of modelling process (0.5s-1.5s API time VS 0.02s processing).
- [ ] Enable multiple iterations of models to be completed across sites for site analysis/comparison
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
- [ ] Development of Django web app to enable user to easily generate their own models
- [ ] Creation of user account and auth process to enable long term storage of models
<br>

**LONG TERM GOALS:**
- [ ] Predictive and historical analysis based on user personal demand
- [ ] Real-time forecasting and integration into smart meter/IOT appliances
- [ ] Lead generation for equipment vendors and installers based on location, price, system, etc.
- [ ] Lead generation for sales/marketing teams to generate leads ahead of time or inform on the ground data
- [ ] Value generation for aggregated data generated through platform. Ensure anonymity of users!
- [ ] Wider app + database with documentation to allow access from external queries through API service without GUI
- [ ] Increased market participation for platform users informed by shared data and models
<br>


## Model Classes

**Site**
```python
site = Site()
site.latitude
site.longitude
site.name
site.address
site.client
site.size
site.tmz_hrs_east
site.tmy_data
```

**SolarPVArray**
```python
array_1 = SolarPVArray()
array_1.pv_kwp
array_1.surface_pitch
array_1.surface_azimuth
array_1.lifespan
array_1.pv_eol_derating
array_1.cost_per_kwp
array_1.electrical_eff
array_1.cell_temp_coeff
array_1.transmittance_absorptance
array_1.refraction_index
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
```
