# PRC Data Challenge 2026 — team zestful-jewel

Predict taxi-out time (`TAXITIME_SEC_mvt`) for departures at 10 major European airports.

## Final model

- Model: CatBoostRegressor (RMSE)
- Global model + airport-specific models (LIRF, EGLL, LTFM, EHAM, LFPG, EDDF) with 70/30 blending
- Features:
  - Categorical: ADEP, ADES, flight rule/type, aircraft type, operator, market segment, runway, stand
  - Time: hour/weekday/month sin-cos, IOBT/EOBT/ARVT deltas vs scheduled time
  - Congestion: dep/arr counts at airport and runway (hour + 2h window), queue share
  - Historical: month-blocked (leakage-safe) medians by airport-hour, airport-runway-hour, airport-stand
- No actual off-block/takeoff/taxi times used
- Final file: `zestful-jewel_final.parquet`

## Reproduce

1. Put official data in `data/raw/` (training_*.parquet, final_ranking.parquet, final_submitting.parquet)
2. Run the final script/notebook in `notebooks/` or `src/`
3. Submission is written to `results/zestful-jewel_final.parquet`

## Requirements

Python 3.10+, pandas, numpy, catboost, pyarrow

## License

GNU GPLv3
