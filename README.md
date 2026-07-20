# Ethiopia Financial Inclusion Forecasting

Forecasting system for Ethiopia's digital financial transformation, built for Selam Analytics'
engagement with a consortium of development finance institutions, mobile money operators, and
the National Bank of Ethiopia.

## Project Goal

Forecast Ethiopia's progress on the two core Global Findex dimensions of financial inclusion for
2025-2027:

- **Access** - Account Ownership Rate (`ACC_OWNERSHIP`)
- **Usage** - Digital Payment Adoption Rate

The project models how policy changes, product launches, and infrastructure investments affect
these outcomes, and presents forecasts through an interactive dashboard.

## Repository Structure

ethiopia-fi-forecast/
├── data/
│ ├── raw/ # Starter dataset (as provided)
│ └── processed/ # Analysis-ready, cleaned/enriched data
├── notebooks/ # Exploration, EDA, modeling notebooks
├── src/ # Reusable Python modules
├── dashboard/
│ └── app.py # Streamlit dashboard
├── tests/ # Unit tests
├── models/ # Saved model artifacts
├── reports/
│ └── figures/ # Exported charts for the final report
├── requirements.txt
└── README.md


## Data

The starter dataset (`data/raw/ethiopia_fi_unified_data.csv`) uses a unified schema where every
row is one of four `record_type`s: `observation`, `event`, `impact_link`, or `target`. See
`data/raw/SCHEMA_README.md` for full field documentation and `data/raw/reference_codes.csv` for
valid categorical values.

**Note:** the starter file as distributed contains 30 observations, 10 events, and 3 targets, but
**0 impact_link records** (the brief describes 14). Task 1 of this project treats building the
event to indicator impact_link records as an enrichment deliverable rather than assuming they were
pre-populated. See `data/data_enrichment_log.md` for full documentation of this and all other
additions.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Tasks

| Task | Description | Branch |
|------|-------------|--------|
| 1 | Data exploration and enrichment | `task-1` |
| 2 | Exploratory data analysis | `task-2` |
| 3 | Event impact modeling | `task-3` |
| 4 | Forecasting Access and Usage | `task-4` |
| 5 | Dashboard development | `task-5` |

## Author

Meklit Workineh Daba - 10 Academy Data Science Bootcamp, Week 11
