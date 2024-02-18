# WiseWattage - Energy Modelling Application
### Demo: <URL HERE> **WORK IN PROGRESS, ONLINE APPLICATION WILL LAUNCH SOMETIME IN Q4 2024**
### Description
This project is an energy modeling application designed to model on and off-grid energy generation potential across various sites and system parameters. The long-term goal is to create an all-in-one energy model that will quickly and easily generate accurate model results and statistics. There are many paid energy models on the market that do a similar job, but I have found they almost always add complexity that is not required for the vast majority of scenarios, which adds to the development and modeling timeframes.

Solutions offered by paid software, like PVSyst, Homer, and Plexos, typically target challenges inherent to particular and unique scenarios. These are fantastic modelling packages; however, they often require a particular level of industry knowledge and expertise to use effectively. This, in turn, makes it difficult for those curious about optimizing their energy systems but don't have the time to master all the concepts.

It is these issues, among others, that I aim to address for the wider data science and energy community by hopefully providing energy modeling tools that are 99% as powerful as existing libraries but with 99% less complexity when using them. When completed, WiseWattage will hopefully be a complete package that can be used offline to create Class objects representing a range of energy modeling scenarios.

My particular personal bias is towards promoting hybrid on and off-grid energy systems, with a focus on optimizing system autonomy and economics in a combined approach. However, the modeling package will be able to be arranged for any system configuration or modeling parameters, whether a system dependent wholly on fossil fuels or renewable technologies.

## System Schematic
**Example System Design Hybrid Energy System to be Modeled.**
![alt text](<static/Solar_Model_Technical_Diagram_2.png>)
<br>

The calculations behind the modeling are based on SAP methodologies, and documentation and referencing will be added once the project develops and methodologies have been finalized. Since I began experimenting with this project, I have changed the calculation methodology many times, and as such, I am reluctant to start formalizing the academic reporting until these areas are finalized.

Once this project has been completed, I will also hopefully utilize the same codebase to build an online application where users can generate their own solar PV models. This will be completed using Flask, Django, and/or FastAPI, offering basic API returns and a browser UI to run the model. Users will be prompted to create an account to save the model results for access later if they want, and they can generate an energy report summarizing the model created.

I am aiming to make the web app a free service indefinitely, with the long-term goal being to reduce the costs and time involved with energy modeling and assessments during the procurement phase of projects. I will initially be focusing on domestic scenarios first, but the model will be built with the ability to scale up or down indefinitely. WattSun & Homes will be the first implementation of an online application, which will focus on those in a domestic scenario interested in exploring the potential for rooftop Solar PV, and what other technologies could be incorporated. Hopefully this will offer the chance to gain invaluable feedback before the program is scaled up to include more complex scenarios.

## **Example of current implementation of model:**

The model is created by initializing Class objects which conduct the appropriate modeling when created. The models simulated can then be accessed using `.xxx` notation, with some basic time series data aggregation methods already built in. These models are returned as DataFrames, with the library being built with a focus on using it offline for personal data analysis.
<br>

```python
# Import required packages, assumes location is root of folder containing downloaded scripts.
from meteo.Site import Site
from solar.SolarPVArray import SolarPVArray
from solar.SolarPVModel import SolarPVModel

from misc.log_config import configure_logging

# Set up site and obtain TMY data
site = Site(latitude=54.60452, longitude=-5.92860)

# Set up PV arrays
pv_kwp = 1
surface_pitch = 35
surface_azimuth = -90
lifespan = 25
pv_eol_derating = 0.88

array_1 = SolarPVArray(pv_kwp, surface_pitch, surface_azimuth, lifespan, pv_eol_derating)
array_2 = SolarPVArray(1, 35, -45, 25, 0.88)
array_3 = SolarPVArray(1, 35, 0, 25, 0.88)
array_4 = SolarPVArray(1, 35, 45, 25, 0.88)
array_5 = SolarPVArray(1, 35, 90, 25, 0.88)
arrays = [array_1, array_2, array_3, array_4, array_5]

# Set up and run model
pv_model = SolarPVModel(site=site, arrays=arrays)
```
<br>

## Accessing Model Results

```python
# Example on how to access summary of model results
model_summary_df = pv_model.summary
model_summary_hourly_df = pv_model.summary_grouped.hourly
model_summary_daily_df = pv_model.summary_grouped.daily
model_summary_weekly_df = pv_model.summary_grouped.weekly
model_summary_monthly_df = pv_model.summary_grouped.monthly
model_summary_quarterly_df = pv_model.summary_grouped.quarterly

# Example on how to access specific model results
array_1_model_df = pv_model.models[0]["model_result"]
array_5_model_df = pv_model.models[4]["model_result"]

# Example on how to access combined model results
combined_model_df = pv_model.combined_model
```
This is an example implementation using basic inputs, however there are many more options for customising the model by adding additional inputs. These can be seen by looking into the specific functions called contained in the solar modelling .py files, and will be further detailed more as the project documentation improves over time.
<br>

## Current Model Classes & Methods

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

###### **Updated (18/02/2024 @ 00:17 GMT+00:00):** 
<br>

### **The library will be further expanded to include the following features:**

**COMING SOON™ (1 Month):**
- [ ] Ability to automatically build and simulate models from one single command loading CSV file in root folder. Template and guidance for this will be available online.
- [ ] Inclusion of low irradiance losses
- [ ] Inclusion of spectral losses
- [ ] Inclusion of system/conversion losses
- [ ] Inclusion of base project financials to estimate cost for solar PV project
- [ ] Further statistical analysis functions to be added that target desired columns
- [ ] Full range of charts generated based on comparison of important characteristics
- [ ] Creation of project Class in which all other models and data generated will be stored and accessed
- [ ] Batch processing and asynchronous functions for jrc_tmy when modelling multiple sides. Current
API response time is slowest part of modelling process (0.5s-1.5s API time VS 0.02s processing).
- [ ] Enable multiple iterations of models to be completed across sites for site analysis/comparison & sensitivity analysis.
- [ ] Creation and integration of SQL database to enable long term storage and easier integration further down the road.
###### **NOTE: We will not store accurate locations for areas being modelled after weather data has been returned, as this information is no longer required once TMY data is obtained. This is to improve user privacy and security in the event of any data breach. Ensuring and protecting user privacy will remain a core principle of this project as long as I am still involved in it's development.
<br>

**SHORT TERM GOALS (1-3 Months):**
- [ ] Financial analysis and project Class creation for modelling multiple scenarios.
- [ ] Estimated annual consumption and hourly load profile integrated into modelling process.
- [ ] Electricity Grid Connection with variable import/export tariffs for range of tariffs (day-night, economy 7, EV+, etc.).
- [ ] Integration of battery technologies optimised for range of services (load balancing, increased renewable %, reduced costs, etc.).
- [ ] Integration of alternative AC power sources such as off-grid generators.
- [ ] Integration of wind Turbine and other DC energy sources.
- [ ] System controller for analysis of different energy priorities/scheduling.
- [ ] Integration of EV battery charge/discharge scheduling.
- [ ] Integration of EV V2G and/or V2L scenarios (Vehicle to Grid, Vehicle to Load).
<br>

**MEDIUM TERM GOALS (3-6 Months):**
- [ ] Energy model optimisation procedure based initially on economics, emissions or renewable penetration %.
- [ ] Implementation of complex project financials required for even the most in depth procurement plans.
- [ ] Built-in Class methods for sensitivity analysis modelling that integrate with optimisation process.
- [ ] Development of Django/Flask web app to enable user to easily generate their own models
- [ ] Creation of user account and auth process to enable long term storage of models
- [ ] Integration of thermal demand and generation into profiling and modelling
- [ ] Creation of "mini" model API & apps to also assess other products available in the market (EV's, Insulation, Heat Pumps, Solar Thermal, etc.)
<br>

**LONG TERM GOALS (6-12 Months):**
- [ ] Predictive and historical analysis based on user personal demand.
- [ ] Real-time forecasting and integration into smart meter/IOT appliances.
- [ ] Lead generation for equipment vendors and installers based on location, price, system, etc.
- [ ] Lead generation for sales/marketing teams to generate leads ahead of time or inform on the ground data.
- [ ] Wider app + database with documentation to allow access from external queries through API service without GUI.
- [ ] Increased market participation for platform users informed by shared data and models.
- [ ] Exploration of integrating energy modelling/predictive API platform with suppliers to further integrate prosumers engagement in the market.
<br>

