# Physics-Informed Gaussian Process Regression for Wind Turbine Power Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![SALib](https://img.shields.io/badge/SALib-1.4.7-green.svg)](https://salib.readthedocs.io/)

## Overview

This repository contains the data, code, and results for the manuscript:

> **"Physics-Informed Gaussian Process Regression for Wind Turbine Power Prediction:
> Interpretable Decomposition, Uncertainty Quantification, and Residual Sensitivity Analysis"**
> *Under review — Renewable Energy, Elsevier, 2025*

The proposed **PI-GPR** framework decomposes wind turbine power prediction into:
1. A **Betz aerodynamic physics baseline** (with air-density temperature correction)
2. A **GPR residual model** that learns physics-unexplained variance from SCADA data

Sobol global sensitivity analysis is applied separately to the full GPR power model and the PI-GPR residual, revealing that the physics-unexplained variance is dominated by strongly coupled wind speed–rotor speed interactions (S_T − S₁ ≈ 0.32–0.33) corresponding to tip-speed ratio dynamics absent from the Betz model.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | Kelmarsh Wind Farm SCADA Data (2016–2021) |
| **Turbine** | Kelmarsh 1 — Senvion MM92 (rated 2050 kW) |
| **DOI** | https://doi.org/10.5281/zenodo.5946808 |
| **Raw records** | ~195,000 (10-minute averages) |
| **Sample used** | 5,000 (stratified by wind speed bins, seed = 42) |
| **Train / Test split** | 4,000 / 1,000 (80/20, seed = 42) |

### Features

| Feature | Unit | Role |
|---------|------|------|
| Wind Speed | m/s | Primary driver (Betz cubic) |
| Rotor Speed | RPM | TSR coupling |
| Ambient Temperature | °C | Air density correction |
| Wind Direction (sin) | — | Encoded direction component |
| Wind Direction (cos) | — | Encoded direction component |

**Target:** Active power output (kW)

---

## Models

| Model | Notes |
|-------|-------|
| Linear Regression | OLS baseline |
| Random Forest | 100 estimators, Gini importance |
| ANN (MLP) | 64-32 ReLU, Adam, early stopping |
| GPR | ARD-RBF + WhiteKernel, 500-pt kernel fitting subset |
| **PI-GPR** | **Proposed — Betz prior + GPR residual, full 4000-pt training** |

---

## Results

| Model | R² | RMSE (kW) | MAE (kW) | DM p-value |
|-------|----|-----------|----------|------------|
| Physics Baseline | 0.9826 | 81.78 | 58.01 | — |
| Linear Regression | 0.9334 | 159.91 | 129.32 | — |
| Random Forest | 0.9947 | 45.24 | 25.20 | — |
| ANN | 0.9807 | 86.15 | 58.26 | — |
| GPR | 0.9944 | 46.34 | 27.26 | 0.18 |
| **PI-GPR** | **0.9938** | **48.97** | **27.90** | — (reference) |

GPR vs PI-GPR RMSE difference (2.63 kW) is not statistically significant (Diebold-Mariano test, p = 0.18).

---

## Sobol Sensitivity Analysis

| Setting | Value |
|---------|-------|
| Library | SALib 1.4.7 |
| Estimator | Jansen |
| Sampling | Saltelli quasi-random |
| Base samples (N) | 1,024 |
| Total model evaluations | 12,288 |
| Applied to | GPR power model + PI-GPR residual model (separately) |

### Key Results — GPR Power Model

| Feature | S₁ | S_T | Interaction |
|---------|----|-----|-------------|
| Wind Speed | 0.535 | 0.618 | 0.083 |
| Rotor Speed | 0.382 | 0.465 | 0.083 |
| Temperature | 0.000 | 0.001 | 0.001 |
| Wind Dir (sin/cos) | <0.001 | <0.001 | 0.000 |

### Key Results — PI-GPR Residual Model

| Feature | S₁ | S_T | Interaction |
|---------|----|-----|-------------|
| Wind Speed | 0.412 | 0.734 | **0.322** |
| Rotor Speed | 0.385 | 0.718 | **0.333** |
| Temperature | 0.061 | 0.083 | 0.022 |
| Wind Dir (sin/cos) | <0.002 | <0.002 | 0.000 |

Interaction effects are ~4× larger in the residual model, confirming strongly coupled tip-speed ratio dynamics absent from the Betz prior.
---

## Reproducibility

All results are fully reproducible with `seed = 42`.

### Installation

```bash
git clone https://github.com/hadeedinsha/pi-gpr-wind-turbine-kelmarsh.git
cd pi-gpr-wind-turbine-kelmarsh
pip install -r requirements.txt
```

### requirements.txt

```
scikit-learn==1.3.0
SALib==1.4.7
numpy==1.24
matplotlib==3.7.0
pandas==2.0.0
scipy==1.11.0
```
---



## Data Availability

The Kelmarsh SCADA dataset used in this study is publicly available:

> Holleran D, Stuart P, Mander T. Kelmarsh Wind Farm SCADA Data (2016–2021).
> Zenodo; 2021. https://doi.org/10.5281/zenodo.5946808

---


