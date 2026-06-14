# 1. PVGIS-GERMANY-2022 DATASET
## About the dataset

This dataset contains hourly solar power production and weather data for Germany during 2022. The data was generated using PVGIS for a location at latitude 51.0° and longitude 10.0°, which is roughly in the central part of Germany.

The dataset includes measurements and estimates related to solar energy production, solar irradiation, temperature, wind speed, and sun position. Each row represents one hour.

### Columns

- `time` - date and time of the observation
- `P` - estimated PV power output (W)
- `G(i)` - solar irradiance on the panel surface (W/m²)
- `H_sun` - sun height above the horizon (degrees)
- `T2m` - air temperature at 2 meters (°C)
- `WS10m` - wind speed at 10 meters (m/s)


### Dataset period

From January 1, 2022 to December 31, 2022, with hourly observations.

### Source

Data generated using the PVGIS (Photovoltaic Geographical Information System) platform.


# 1. ERA-5-GERMANY-JULY-SEPTEMBER-2022 DATASET
## About the dataset

This dataset contains hourly weather data for Germany in 2022 obtained from the ERA5 reanalysis dataset. The data includes temperature, wind, cloud cover, precipitation, and solar radiation measurements.

Each row represents one hour of observations for a location at latitude 51.0° and longitude 10.0°.

### Features

- `valid_time` - date and time of the observation
- `latitude` - latitude of the location
- `longitude` - longitude of the location
- `u10` - east-west wind component at 10 meters (m/s)
- `v10` - north-south wind component at 10 meters (m/s)
- `t2m` - air temperature at 2 meters (K)
- `tcc` - total cloud cover (0–1)
- `tp` - total precipitation (m)
- `ssrd` - surface solar radiation downwards (J/m²)

### Dataset period

The dataset contains hourly observations collected from July to September 2022.

### Source

Data obtained from the ERA5 reanalysis dataset provided by the Climate Data Store.
