# Predictive volatility of machine learning in micro-samples: a regularised assessment of regional poverty

[![Status: Under Review](https://img.shields.io/badge/Status-Under_Review-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the data and computational code necessary to fully replicate the analyses, simulations, and figures presented in the working manuscript: **"Predictive volatility of machine learning in micro-samples: a regularised assessment of regional poverty"** by Ahmad Hakiim Jamaluddin, Andrea Tri Rian Dani, Nor Idayu Mahat, Vita Ratnasari, and Shukor Sanim Mohd Fauzi.

*Note: A permanent DOI will be minted via Zenodo upon the formal acceptance and publication of the manuscript.*

## Overview

Small regional datasets pose a dual statistical challenge: correlated predictors inflate estimation variance, while highly flexible machine learning models can become unstable due to limited information per adaptive degree of freedom. This project investigates this phenomenon—termed **predictive volatility**—using both simulated environments and an empirical application predicting provincial poverty rates across 34 Indonesian provinces. 

The codebase evaluates a wide range of models, including:
* **Frequentist Shrinkage Models:** OLS, Ridge, LASSO, Elastic Net.
* **Bayesian Parametric Models:** Gaussian LM, Bayesian Ridge, Bayesian LASSO, Horseshoe priors, Beta regression, and Spike-and-Slab.
* **Spatial Diagnostics:** Dynamically connected k-nearest neighbors spatial graph (ICAR).
* **Machine Learning Ensembles:** Random Forest, XGBoost, BART, and Gaussian Processes.

## Repository Structure

* `scripts/`: Contains the master replication R scripts for data preparation, parametric and non-parametric cross-validation loops, prior sensitivity analysis, and the multi-scenario asymptotic simulation grid. Real poverty dataset is stored in the script.
* `0. Real dataset/`: Contains the poverty dataset used.
* `1. Findings/`: Automatically generated directory where the scripts output the tables, predictive distributions, and SHAP summary plots for real data analysis.
* `2. Simulation/`: Automatically generated directory where the scripts output the tables, predictive distributions, and SHAP summary plots for simulation study.

## System Requirements and Dependencies

All analyses, simulations, and visualizations were programmed and executed in the **R statistical computing environment**. Bayesian MCMC sampling chains were compiled and executed via Stan.
To run the master replication script, ensure you have the following R packages installed.

**Modeling & Inference:**
* `glmnet` (v4.1)
* `brms` (v2.20)
* `rstan`
* `loo`
* `posterior`
* `BoomSpikeSlab`
* `spdep`, `car`

**Machine Learning & Interpretability:**
* `BART`, `xgboost`, `randomForest`, `kernlab`
* `shapviz`

**Simulation & Data Wrangling:**
* `MASS`
* `dplyr`, `tidyr`, `sf`, `rnaturalearth`
* `ggplot2`, `ggcorrplot`, `bayesplot`

## Usage: Replicating the Analysis

To replicate the findings from the paper:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/HakiimJ/regional-poverty-volatility.git
cd regional-poverty-volatility
