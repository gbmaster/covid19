#!/usr/bin/env python3

import datetime
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve
import sys
import urllib.request

# Logistic model equation
def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))


# Gauss model equation
def gauss_model(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


world_json_url = "https://pomber.github.io/covid19/timeseries.json"
ita_json_url = (
    "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-andamento-nazionale.json"
)

world_data = {}

# Import data from pomber
try:
    print("Downloading world data...")
    with urllib.request.urlopen(world_json_url) as url:
        world_json_data = json.loads(url.read().decode())
except urllib.error.HTTPError:
    print("Failed. Using local copy...")
    with open("timeseries.json", "r") as file:
        world_json_data = json.loads(file.read())

for country, days in world_json_data.items():
    world_data[country] = {}

    if country == "Italy":
        continue

    for day in days:
        date = datetime.datetime.strptime(day["date"], "%Y-%m-%d").date()
        world_data[country][date] = {
            "confirmed": day["confirmed"],
            "deaths": day["deaths"],
            "recovered": day["recovered"],
        }

# Integrate with updated data for Italy
try:
    print("Downloading data about Italy...")
    with urllib.request.urlopen(ita_json_url) as url:
        ita_json_data = json.loads(url.read().decode())
except urllib.error.HTTPError:
    print("Failed. Using local copy...")
    with open("dpc-covid19-ita-andamento-nazionale.json", "r") as file:
        ita_json_data = json.loads(file.read())

for day in ita_json_data:
    date = datetime.datetime.strptime(day["data"], "%Y-%m-%dT%H:%M:%S").date()

    world_data["Italy"][date] = {
        "confirmed": int(day["totale_casi"]),
        "deaths": int(day["deceduti"]),
        "recovered": int(day["dimessi_guariti"]),
    }


dataset = world_data["Italy"]
if len(dataset) == 0:
    print("No data available")
    sys.exit(0)

days_x = []
confirmed_y = []
deceases_y = []
recovered_y = []
remaining_y = []
first_day = None
for day in sorted(dataset):
    if first_day is None:
        first_day = day - datetime.timedelta(weeks=2)
    days_x += [(day - first_day).days]
    confirmed_y += [dataset[day]["confirmed"]]
    deceases_y += [dataset[day]["deaths"]]
    recovered_y += [dataset[day]["recovered"]]
    remaining_y += [dataset[day]["confirmed"] - dataset[day]["deaths"] - dataset[day]["recovered"]]

###########################
# Confirmed cases curve fit
###########################

# Compute hints for the confirmed cases curve fit
# Mean
confirmed_mean = sum(x * y for x, y in zip(days_x, confirmed_y)) / sum(confirmed_y)
# Total cases, so far...
confirmed_cases = confirmed_y[-1]

confirmed_fit, _ = curve_fit(logistic_model, days_x, confirmed_y, p0=[1, confirmed_mean, confirmed_cases])

end_infection = int(
    fsolve(
        lambda x: logistic_model(x, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2]) - int(confirmed_fit[2]),
        confirmed_fit[1],
    )
)

inflection_day = first_day + datetime.timedelta(days=int(confirmed_fit[1]))
print("Inflection on {0:s}".format(inflection_day.strftime("%d/%m/%Y")))
end_day = first_day + datetime.timedelta(days=end_infection)
print("End of infection on {0:s}".format(end_day.strftime("%d/%m/%Y")))
next_confirmed_1 = int(logistic_model(max(days_x) + 1, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2]))
next_confirmed_2 = int(logistic_model(max(days_x) + 2, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2]))
confirmed_percent_diff_1 = ((next_confirmed_1 - confirmed_y[-1]) * 100) / confirmed_y[-1]
confirmed_percent_diff_2 = ((next_confirmed_2 - next_confirmed_1) * 100) / next_confirmed_1
print(
    "Next two expected data on confirmed cases: {0:d} ({1:s}{2:.2f}% {1:s}{3:d}) / {4:d} ({5:s}{6:.2f}% {5:s}{7:d})".format(
        next_confirmed_1,
        "+" if confirmed_percent_diff_1 > 0 else "",
        confirmed_percent_diff_1,
        next_confirmed_1 - confirmed_y[-1],
        next_confirmed_2,
        "+" if confirmed_percent_diff_2 > 0 else "",
        confirmed_percent_diff_2,
        next_confirmed_2 - next_confirmed_1,
    )
)
print(
    "Expected total cases: {0:d}".format(
        int(logistic_model(end_infection, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2]))
    )
)

####################
# Deceases curve fit
####################

# Compute hints for the deceases curve fit
# Mean
deceases_mean = sum(x * y for x, y in zip(days_x, deceases_y)) / sum(deceases_y)
# Total deceases, so far...
total_deceases = deceases_y[-1]

deceases_fit, _ = curve_fit(logistic_model, days_x, deceases_y, p0=[1, deceases_mean, total_deceases])

#####################
# Recovered curve fit
#####################

# Compute hints for the recovered curve fit
# Mean
recovered_midpoint = (end_infection - days_x[0]) // 2
# Optimism: all recovered, at the end
total_recovered = logistic_model(end_infection, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2])

recovered_fit, _ = curve_fit(logistic_model, days_x, recovered_y, p0=[1, recovered_midpoint, total_recovered])

#####################
# Remaining cases fit
#####################

# Compute hints for the confirmed cases curve fit
# Mean
remaining_cases_mean = sum(x * y for x, y in zip(days_x, remaining_y)) / sum(remaining_y)
# Standard deviation
remaining_cases_sd = math.sqrt(sum(abs(i - remaining_cases_mean) ** 2 for i in days_x) / len(days_x))
# Peak height
remaining_cases_peak = remaining_y[int(remaining_cases_mean) - days_x[0]]

(remaining_cases_peak, remaining_cases_mean, remaining_cases_sd), _ = curve_fit(
    gauss_model, days_x, remaining_y, p0=[remaining_cases_peak, remaining_cases_mean, remaining_cases_sd],
)

peak_day = first_day + datetime.timedelta(days=int(remaining_cases_mean))
print("Peak on {0:s}".format(peak_day.strftime("%d/%m/%Y")))

pred_x = list(range(max(days_x), end_infection))
plt.rcParams["figure.figsize"] = [7, 7]
plt.rc("font", size=14)

plt.scatter(days_x, confirmed_y, label="Confirmed cases", color="red")
plt.scatter(days_x, deceases_y, label="Deceases", color="black")
plt.scatter(days_x, recovered_y, label="Recovered cases", color="green")
plt.scatter(days_x, remaining_y, label="Remaining cases", color="brown")

plt.plot(
    days_x + pred_x,
    [logistic_model(x, confirmed_fit[0], confirmed_fit[1], confirmed_fit[2]) for x in days_x + pred_x],
    label="Confirmed cases model",
    c="red",
)
plt.plot(
    days_x + pred_x,
    [logistic_model(x, deceases_fit[0], deceases_fit[1], deceases_fit[2]) for x in days_x + pred_x],
    label="Deceases model",
    c="black",
)
plt.plot(
    days_x + pred_x,
    [logistic_model(x, recovered_fit[0], recovered_fit[1], recovered_fit[2]) for x in days_x + pred_x],
    label="Recovered cases model",
    c="green",
)
plt.plot(
    days_x + pred_x,
    [gauss_model(x, remaining_cases_peak, remaining_cases_mean, remaining_cases_sd) for x in days_x + pred_x],
    label="Remaining cases model",
    c="brown",
)
plt.axvline(int(remaining_cases_mean), label="Peak", c="g")
plt.legend(prop={"size": 13})
plt.savefig("infections.png")
