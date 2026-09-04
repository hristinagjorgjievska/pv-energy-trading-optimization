# AI-Driven Forecasting and Optimization of Photovoltaic Energy Storage and Market Selling Strategies in European Electricity Markets

**Course:** Introduction to Data Science  

**Authors:**  
- Hristina Gjorgjievska 
- Ognen Mladenovski

---

## Overview

This project explores the application of machine learning techniques for forecasting photovoltaic (PV) energy production and optimizing energy storage and market selling strategies in European electricity markets. By combining historical electricity prices, weather conditions, and solar irradiation data, the project evaluates predictive models and analyzes how accurate forecasts can support more efficient energy management and market participation.

## Experiments

The experimental evaluation was conducted to validate the proposed AI-based forecasting and optimization framework. 
The experiments are divided into four main research questions:

### RQ1: Electricity Price Forecasting

The goal of this experiment is to evaluate how accurately future electricity prices can be predicted using historical market data, meteorological variables, and photovoltaic-related features.

Experiments include:
- Training and evaluation of machine learning forecasting models.
- Comparison of different feature configurations.
- Evaluation using regression metrics:
  - MAE
  - RMSE
  - R² score

The best-performing model is used to generate future electricity price predictions required for the optimization stage.


### RQ2: Photovoltaic Production Forecasting

This experiment evaluates the ability to predict photovoltaic energy production based on weather conditions and solar radiation measurements.

Experiments include:
- Feature analysis of meteorological and solar radiation variables.
- Evaluation of forecasting approaches:
  - Machine learning models
  - Persistence baseline

The selected forecasting method provides estimated PV production values used as input for battery scheduling.


### RQ3: Battery Storage Optimization

The objective of this experiment is to determine the optimal battery charging and discharging strategy that maximizes electricity trading revenue.

The optimization considers:
- Predicted electricity prices
- Predicted PV production
- Battery constraints:
  - Maximum/minimum state of charge
  - Charging/discharging limits
  - Battery efficiency

The following strategies are compared:
- Rule-based strategy
- Model Predictive Control (MPC)
- Perfect foresight scenario (upper bound)


### RQ4: Uncertainty Analysis

This experiment investigates how prediction errors influence the final battery scheduling strategy and achieved revenue.

The analysis evaluates:
- Sensitivity of optimization results to forecasting errors.
- Impact of inaccurate price and PV production predictions.
- Difference between ideal and realistic forecasting scenarios.


### Experimental Setup

All experiments are performed using chronological train-validation-test splitting to avoid data leakage.

The evaluation pipeline consists of:

1. Data preprocessing and feature engineering.
2. Training forecasting models.
3. Generating future price and PV production predictions.
4. Applying battery optimization algorithms.
5. Comparing obtained revenues and performance metrics.
