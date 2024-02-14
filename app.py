from flask import Flask, redirect, render_template, request, session, jsonify
from flask_session import Session
from WiseWattage.wisewattage import Site, SolarPVArray, SolarPVModel

import pandas as pd
from plotly.offline import plot

app = Flask(__name__)

# Configure session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", name=session.get("name"))

@app.route("/solarmodel", methods=["GET", "POST"])
def solar_model():
    if request.method == "POST":
        name = request.form.get("Project Name")
        address = request.form.get("Address")
        latitude = float(request.form.get("Latitude"))
        longitude = float(request.form.get("Longitude"))
        pv_kwp = float(request.form.get("Solar PV kWp"))
        surface_pitch = float(request.form.get("Surface Pitch"))
        surface_azimuth = float(request.form.get("Solar Azimuth"))
        freq = request.form.get("freq")

        site = Site(name=name, address=address, latitude=latitude, longitude=longitude)
        array_1 = SolarPVArray(pv_kwp, surface_pitch, surface_azimuth)
        arrays = [array_1]
        pv_model = SolarPVModel(site=site, arrays=arrays)

        chart = pv_model.plot_sum(["T2m", "Cell_Temp_C_Avg"], group=freq, plot_type="bar")
        chart = plot(chart, output_type='div', include_plotlyjs=True)

        data_freq = ["hourly", "daily", "weekly", "monthly", "quarterly"]

        data_hourly = pv_model.model_summary_html_export(data_freq[0])
        data_daily = pv_model.model_summary_html_export(data_freq[1])
        data_weekly = pv_model.model_summary_html_export(data_freq[2])
        data_monthly = pv_model.model_summary_html_export(data_freq[3])
        data_quarterly = pv_model.model_summary_html_export(data_freq[4])

        data_summary = pv_model.model_summary_html_export(grouped=False)

        return render_template("modelresults.html", name=session.get("name"),chart=chart,
                               data_hourly=data_hourly,
                               data_daily=data_daily,
                               data_weekly=data_weekly,
                               data_monthly=data_monthly,
                               data_quarterly=data_quarterly,
                               data_summary=data_summary)

    return render_template("solarmodel.html", name=session.get("name"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["name"] = request.form.get("name")
        return redirect("/")
    if "name" not in session:
        return render_template("login.html")
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run()
