# Exeter-bike-sharing-analysis
BEMM457: Tfl bike-sharing analysis for Exeter city council
## Project Overview
This repository contains a business analytics coursework project for BEMM457 – Topics in Business Analytics.
The project analyses Transport for London (TfL) cycle hire data (2023–2024) to generate insights that can inform Exeter City Council’s sustainable mobility planning, particularly in relation to bike and e-scooter sharing systems.
The analysis applies descriptive analytics and regression modelling to identify key demand drivers and operational patterns relevant to public-sector decision-making.
## Business Objectives
- **Demand analysis:** Identify temporal and environmental factors influencing cycle hire demand  
- **Operational insight:** Support evidence-based decisions on fleet sizing and service planning  
- **Strategic learning:** Assess what Exeter can learn from London’s bike-sharing experience  
- **Ethical awareness:** Consider data governance and equity implications in mobility analytics  
## Repository Structure
├── data/
│   └── tfl_bike_data_2023_2024.csv        # Processed TfL cycle hire data
├── analysis/
│   ├── analyze_tfl_data.py                # Descriptive statistics and EDA
│   ├── regression_analysis.py             # Regression modelling
│   └── regression_results.csv             # Model outputs
├── visualizations/
│   ├── tfl_analysis_dashboard.png          # Descriptive analytics dashboard
│   └── tfl_regression_analysis.png         # Regression visualisations
└── README.md                              # Project overview and guidance
## Methodology
Framework: PPDAC cycle (Problem, Plan, Data, Analysis, Conclusion)
Techniques: Descriptive statistics and multiple linear regression (OLS)
Tools: Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
The analysis focuses on daily demand patterns and examines the influence of variables such as temperature, rainfall, weekdays, and public holidays.
## Key Findings (Summary)
Cycle hire demand exhibits strong seasonality and weekday effects.
Temperature is a significant positive predictor of demand.
Rainfall and public holidays are associated with reduced usage.
The regression model explains a high proportion of demand variation (R² ≈ 0.87).
Detailed interpretation of results and policy implications are provided in the coursework report submitted via ELE.
## Data Source
Transport for London (TfL) Open Data – Cycling
https://cycling.data.tfl.gov.uk/
The dataset is publicly available and used in accordance with TfL’s open data terms.
