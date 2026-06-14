# 1. PVGIS-GERMANY-2020 DATASET
## About the datasets

This dataset contains hourly solar power production and weather data for Germany during 2022. The data was generated using PVGIS for a location at latitude 51.0° and longitude 10.0°, which is roughly in the central part of Germany.

The dataset includes measurements and estimates related to solar energy production, solar irradiation, temperature, wind speed, and sun position. Each row represents one hour.

### Columns

- `time` - date and time of the observation
- `P` - estimated PV power output (W)
- `G(i)` - solar irradiance on the panel surface (W/m²)
- `H_sun` - sun height above the horizon (degrees)
- `T2m` - air temperature at 2 meters (°C)
- `WS10m` - wind speed at 10 meters (m/s)
- 
### Dataset period

From January 1, 2022 to December 31, 2022, with hourly observations.

### Source

Data generated using the PVGIS (Photovoltaic Geographical Information System) platform.
