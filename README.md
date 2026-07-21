# Who's Leaving? Customer Churn Prediction with Explainable AI

## Overview
Customer churn, which is defined as the discontinuation of a subscriber from a particular service, is a key driver of revenue loss in subscription-based businesses. This project builds a predictive model to identify customers at risk of churning, using a dataset of ~7000 telecommunications customers with demographic, account, and service-usage features. A multi-tool pipeline spanning Excel, Python, R, MATLAB, SQL, and Power BI is being implemented, with an emphasis on statistically rigorous exploratory analysis and model explainability.

## Research Questions
- **RQ1:** Which customer attributes are the strongest predictors of churn?
- **RQ2:** How does churn rates vary across tenure, monthly charges, contract type, and payment method?
- **RQ3:** Among the features most strongly associated with churn, do the key numeric predictors vary significantly across the groups defined by the key categorical predictors?
- **RQ4:** Can a machine learning model predict churn meaningfully better than a baseline?
- **RQ5:** Can the model's predictions be explained in a such a way that is actionable for a retention team? 

## Analytical Pipeline
| Stage | Tool | Purpose |
|---|---|---|
| 1 | Excel |  Data profiling and quality checks  |
| 2 | Python | Exploratory data analysis   |
| 3 | Python |  Feature engineering and preprocessing  |
| 4 | Python |  Model training and hyperparameter tuning  |
| 5 | MATLAB |  Model training (baseline)  |
| 6 | Python |  Model explainability  |
| 7 | R |  Statistical testing  |
| 8 | SQL |  Structured querying  |
| 9 | Power BI |  Interactive dashboard  |

## Project Status
🚧 **In progress.** EDA is complete; modeling (hyperparameter tuning via Bayesian Optimisation) and explainability (SHAP) are the current focus.

- [x] Data profiling and quality checks
- [x] Exploratory data analysis (distributions, redundancy analysis)
- [ ] Feature engineering and preprocessing
- [ ] Model training and hyperparameter tuning (Logistic Regression, Random Forest, XGBoost)
- [ ] Model explainability (SHAP)
- [ ] Final evaluation, reporting, and dashboard development

## Repository Structure
```
telco-churn-prediction/
├── data/          
│ └── raw/                      # Original, unmodified dataset 
├── notebooks/
│ ├── 01_eda.ipynb              # Exploratory data analysis
│ ├── 02_data_preparation.ipynb
│ ├── 03_modeling.ipynb
│ ├── 04_explainability.ipynb
├── r_analysis/
│ ├── statistical_tests.R       # Kruskal-Wallis tests
├── src/
│ └── eda_utils.py              # Reusable plotting/helper functions
├── reports/
│ └── figures/
├── environment.yml             # Conda environment specification
├── README.md
└── findings_summary.md
```

## Full Stack
| Category | Tools | 
| --- | --- |
| Data Profiling and Quality Checks | Excel | 
| Exploratory Analysis, Feature Engineering and Modeling | Python 3.12: pandas, NumPy, SciPy, scikit-learn, XGBoost, scikit-optimize |
| Statistical Testing | R: Kruskal-Wallis tests with epsilon-squared effect size |
| Baseline Modeling | MATLAB: logistic regression baseline |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn |
| Structured Querying | SQL |
| Reporting & Dashboarding | LaTeX, Power BI |
| Environment Management | conda (conda-forge channel) |

## Setup
```bash
conda env create -f environment.yml
conda activate churn_env
jupyter lab
```

## Data Sources
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Author
Joshua Pillay — [GitHub](https://github.com/joshua-pillay) · [LinkedIn](https://www.linkedin.com/in/joshua-m-pillay)