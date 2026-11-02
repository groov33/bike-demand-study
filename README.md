# Bike sharing: buy more bikes or rebalance?

Round 2 case study using the UCI Bike Sharing dataset (Capital Bikeshare, Washington D.C., 2011-2012).

**Short answer:** the fleet looks big enough. Demand is squeezed into commute hours: on working days about 52% of
rides fall in 7-9am and 5-7pm, against 26% on weekends. Even the busiest hour in two years needs only about 450 bikes
(under stated assumptions), versus the 2,000 in the case. So the likely problem is bikes being in the wrong place at
the wrong time. Recommendation: forecast-based rebalancing and faster repairs, tested in a pilot before buying bikes.

Full write-up: `reports/Final_Report.pdf`

## What's in here
```
data/            hour.csv, day.csv (original dataset)
notebooks/
  01_data_inspection.ipynb      first look at the data and quality checks
  02_bike_demand_analysis.ipynb  EDA, hypothesis test, model training/testing, bikes needed
model/           demand_model.joblib (trained model), replay_predictions.csv, model_info.json
app.py           Streamlit app: ask questions in plain English + what-if planner
reports/         Final_Report.pdf
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain, run `notebooks/02_bike_demand_analysis.ipynb` top to bottom (about 5 min). It writes the files in `model/`.

## Approach
- Data checks: missing hours left as missing; 8 days with long gaps (e.g. Hurricane Sandy) kept out of training;
  `casual`/`registered` never used as features (they add up to the target).
- Hypothesis: working days have a bigger share of rides in commute hours, controlling for month and weather.
  Tested with a permutation test on daily values.
- Model: baseline vs Poisson GLM vs gradient boosting, each trained month by month on past data only.
  Picked on Jul-Sep 2012, tested once on Oct-Dec 2012. Gradient boosting (Poisson loss) won with
  about 17% error (WAPE), half the baseline's.
- Bikes needed: Little's law on the busiest hour, plus a reserve and a share of broken bikes (assumptions, adjustable in the app).
