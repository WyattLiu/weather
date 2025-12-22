#!/usr/bin/env python3
from ecmwf.opendata import Client

# Create client
client = Client()

# Get latest available data with multiple forecast steps
print("Fetching latest ECMWF forecast data with multiple time steps...")

# Download temperature data for multiple forecast steps (0-240 hours)
client.retrieve(
    type="fc",        # forecast
    step=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],  # Multiple forecast steps
    param=["2t"],     # 2m temperature
    target="forecast_data.grib2"
)

print("Data downloaded successfully to forecast_data.grib2")
