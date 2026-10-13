import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

# ==========================================================
# FINAL V18 - for final_ranking + final_submitting
# Team: zestful-jewel
# ==========================================================
RAW = Path(
    "/kaggle/input/datasets/liveearthmap/"
    "prc-data-challenge-2026-local/"
    "PRC-Data-Challenge-2026-LOCAL/data/raw"
)
FINAL = Path(
    "/kaggle/input/datasets/liveearthmap/"
    "prc-data-challenge-2026-local"
)
OUT = Path("/kaggle/working/PRC-final/results")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "TAXITIME_SEC_mvt"
SPECIAL_AIRPORTS = ["LIRF", "EGLL", "LTFM", "EHAM", "LFPG", "EDDF"]
BLEND_WEIGHT = 0.70

CATEGORICAL_FEATURES = [
    "ADEP_mvt", "ADES_FILED_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt", "RUNWAY_mvt", "STAND_mvt",
]

NUMERIC_FEATURES_BASE = [
    "sched_hour", "sched_hour_sin", "sched_hour_cos",
    "sched_weekday_sin", "sched_weekday_cos",
    "sched_month_sin", "sched_month_cos",
    "iobt_minus_sched_min", "eobt1_minus_sched_min", "arvt1_minus_sched_min",
]

CONGESTION_FEATURES = [
    "dep_same_airport_hour", "dep_airport_2h_window",
    "dep_global_hour", "queue_share_of_hour",
    "arr_same_airport_hour", "arr_airport_2h_window",
    "dep_same_runway_hour", "dep_runway_2h_window",
    "arr_same_runway_hour", "arr_runway_2h_window",
]

HIST_FEATURES = [
    "taxi_median_adep_hour", "taxi_mean_adep_hour", "taxi_count_adep_hour",
    "taxi_median_adep_rwy_hour", "taxi_median_adep_stand",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES_BASE
    + CONGESTION_FEATURES
    + HIST_FEATURES
)

def cat_text(series):
    return series.astype("string").fillna("__MISSING__").astype(str)

def make_features(frame, hist_stats, full_movements=None):
    out = pd.DataFrame(index=frame.index)

    for column in CATEGORICAL_FEATURES:
        out[column] = cat_text(frame[column])

    sched = pd.to_datetime(frame["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    hour = sched.dt.hour + sched.dt.minute / 60.0 + sched.dt.second / 3600.0
    weekday = sched.dt.dayofweek
    month = sched.dt.month

    out["sched_hour"] = hour.astype(float)
    out["sched_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["sched_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["sched_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["sched_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["sched_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["sched_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    for source, name in [
        ("IOBT_flt", "iobt_minus_sched_min"),
        ("EOBT_1_flt", "eobt1_minus_sched_min"),
        ("ARVT_1_flt", "arvt1_minus_sched_min"),
    ]:
        event_time = pd.to_datetime(frame[source], utc=True, errors="coerce")
        minutes = (event_time - sched).dt.total_seconds() / 60.0
        out[name] = minutes.clip(-24 * 60, 24 * 60).astype(float)

    if full_movements is None:
        full_movements = frame

    full_sched = pd.to_datetime(full_movements["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    full_airport = full_movements["ADEP_mvt"].astype("string").fillna("__MISSING__").astype(str)
    full_runway = full_movements["RUNWAY_mvt"].astype("string").fillna("__MISSING__").astype(str)
    full_phase = full_movements["PHASE_mvt"].astype(str)
    full_bucket = full_sched.dt.floor("h")

    def get_counts(phase_filter, level="airport"):
        mask = full_phase == phase_filter
        if level == "airport":
            key = pd.DataFrame({
                "airport": full_airport[mask].to_numpy(),
                "bucket": full_bucket[mask],
            })
            return key.groupby(["airport", "bucket"], dropna=False).size()
        else:
            key = pd.DataFrame({
                "airport": full_airport[mask].to_numpy(),
                "runway": full_runway[mask].to_numpy(),
                "bucket": full_bucket[mask],
            })
            return key.groupby(["airport", "runway", "bucket"], dropna=False).size()

    dep_airport_counts = get_counts("DEP", "airport")
    arr_airport_counts = get_counts("ARR", "airport")
    dep_runway_counts = get_counts("DEP", "runway")
    arr_runway_counts = get_counts("ARR", "runway")

    airport = frame["ADEP_mvt"].astype("string").fillna("__MISSING__").astype(str)
    runway = frame["RUNWAY_mvt"].astype("string").fillna("__MISSING__").astype(str)
    bucket = sched.dt.floor("h")

    def lookup(counts, level="airport", shift=0):
        if level == "airport":
            idx = pd.MultiIndex.from_arrays([
                airport,
                bucket + pd.Timedelta(hours=shift),
            ])
        else:
            idx = pd.MultiIndex.from_arrays([
                airport,
                runway,
                bucket + pd.Timedelta(hours=shift),
            ])
        return counts.reindex(idx).fillna(0).to_numpy(dtype=float)

    same_dep = lookup(dep_airport_counts, "airport", 0)
    out["dep_same_airport_hour"] = same_dep
    out["dep_airport_2h_window"] = same_dep + lookup(dep_airport_counts, "airport", -1) + lookup(dep_airport_counts, "airport", 1)

    same_arr = lookup(arr_airport_counts, "airport", 0)
    out["arr_same_airport_hour"] = same_arr
    out["arr_airport_2h_window"] = same_arr + lookup(arr_airport_counts, "airport", -1) + lookup(arr_airport_counts, "airport", 1)

    global_counts = pd.Series(full_bucket).value_counts()
    out["dep_global_hour"] = bucket.map(global_counts).fillna(0).to_numpy(dtype=float)

    time_frame = pd.DataFrame({
        "airport": airport.to_numpy(),
        "bucket": bucket,
        "time": sched.to_numpy(),
    })
    position = time_frame.groupby(["airport", "bucket"], dropna=False)["time"].rank(method="first")
    out["queue_share_of_hour"] = ((position - 0.5) / np.maximum(same_dep, 1.0)).fillna(0.0).to_numpy(dtype=float)

    same_dep_rwy = lookup(dep_runway_counts, "runway", 0)
    out["dep_same_runway_hour"] = same_dep_rwy
    out["dep_runway_2h_window"] = same_dep_rwy + lookup(dep_runway_counts, "runway", -1) + lookup(dep_runway_counts, "runway", 1)

    same_arr_rwy = lookup(arr_runway_counts, "runway", 0)
    out["arr_same_runway_hour"] = same_arr_rwy
    out["arr_runway_2h_window"] = same_arr_rwy + lookup(arr_runway_counts, "runway", -1) + lookup(arr_runway_counts, "runway", 1)

    stats1, stats2, stats3 = hist_stats
    hour_int = hour.fillna(-1).astype(int)

    tmp = frame[["ADEP_mvt"]].copy()
    tmp["hour"] = hour_int
    m1 = tmp.merge(stats1, on=["ADEP_mvt", "hour"], how="left")
    out["taxi_median_adep_hour"] = m1["taxi_median_adep_hour"].to_numpy()
    out["taxi_mean_adep_hour"] = m1["taxi_mean_adep_hour"].to_numpy()
    out["taxi_count_adep_hour"] = m1["taxi_count_adep_hour"].fillna(0).to_numpy()

    tmp2 = frame[["ADEP_mvt", "RUNWAY_mvt"]].copy()
    tmp2["hour"] = hour_int
    m2 = tmp2.merge(stats2, on=["ADEP_mvt", "RUNWAY_mvt", "hour"], how="left")
    out["taxi_median_adep_rwy_hour"] = m2["taxi_median_adep_rwy_hour"].to_numpy()

    m3 = frame[["ADEP_mvt", "STAND_mvt"]].merge(stats3, on=["ADEP_mvt", "STAND_mvt"], how="left")
    out["taxi_median_adep_stand"] = m3["taxi_median_adep_stand"].to_numpy()

    global_median = stats1["taxi_median_adep_hour"].median()
    for col in HIST_FEATURES:
        out[col] = out[col].fillna(global_median)

    return out[FEATURES]

print("Loading training data...")
train_files = sorted(RAW.glob("training_*.parquet"))
all_frames, dep_frames = [], []

for path in train_files:
    df = pd.read_parquet(path)
    month_str = path.name.split("_")[1][:7]
    df["month_block"] = month_str
    all_frames.append(df)
    dep_frames.append(df.loc[df["PHASE_mvt"].eq("DEP")].copy())

full_train = pd.concat(all_frames, ignore_index=True)
train = pd.concat(dep_frames, ignore_index=True)
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
print("DEP rows:", len(train))

print("Computing historical statistics...")
months = sorted(train["month_block"].unique())
all_stats1, all_stats2, all_stats3 = [], [], []

for i, current_month in enumerate(months):
    past = train[train["month_block"] == current_month] if i == 0 else train[train["month_block"].isin(months[:i])]
    past = past.copy()
    past["hour"] = pd.to_datetime(past["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce").dt.hour

    s1 = (past.groupby(["ADEP_mvt", "hour"])[TARGET]
          .agg(taxi_median_adep_hour="median", taxi_mean_adep_hour="mean", taxi_count_adep_hour="count")
          .reset_index())
    s1["month_block"] = current_month

    s2 = (past.groupby(["ADEP_mvt", "RUNWAY_mvt", "hour"])[TARGET]
          .agg(taxi_median_adep_rwy_hour="median").reset_index())
    s2["month_block"] = current_month

    s3 = (past.groupby(["ADEP_mvt", "STAND_mvt"])[TARGET]
          .agg(taxi_median_adep_stand="median").reset_index())
    s3["month_block"] = current_month

    all_stats1.append(s1)
    all_stats2.append(s2)
    all_stats3.append(s3)

stats1_all = pd.concat(all_stats1, ignore_index=True)
stats2_all = pd.concat(all_stats2, ignore_index=True)
stats3_all = pd.concat(all_stats3, ignore_index=True)

train_temp = train.copy()
train_temp["hour"] = pd.to_datetime(train_temp["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce").dt.hour
stats1_full = (train_temp.groupby(["ADEP_mvt", "hour"])[TARGET]
               .agg(taxi_median_adep_hour="median", taxi_mean_adep_hour="mean", taxi_count_adep_hour="count")
               .reset_index())
stats2_full = (train_temp.groupby(["ADEP_mvt", "RUNWAY_mvt", "hour"])[TARGET]
               .agg(taxi_median_adep_rwy_hour="median").reset_index())
stats3_full = (train_temp.groupby(["ADEP_mvt", "STAND_mvt"])[TARGET]
               .agg(taxi_median_adep_stand="median").reset_index())
hist_stats_full = (stats1_full, stats2_full, stats3_full)

print("Building features...")
feature_parts = []
for current_month in months:
    part = train.loc[train["month_block"] == current_month].copy()
    full_part = full_train[full_train["month_block"] == current_month]
    s1 = stats1_all[stats1_all["month_block"] == current_month].drop(columns=["month_block"])
    s2 = stats2_all[stats2_all["month_block"] == current_month].drop(columns=["month_block"])
    s3 = stats3_all[stats3_all["month_block"] == current_month].drop(columns=["month_block"])
    feature_parts.append(make_features(part, (s1, s2, s3), full_part))

X = pd.concat(feature_parts, axis=0).sort_index()
y = train[TARGET]
airports = train["ADEP_mvt"].astype(str)
print("Features shape:", X.shape)

models = {}

print("\nTraining GLOBAL model...")
model_global = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE",
    iterations=900, learning_rate=0.08, depth=8, l2_leaf_reg=8.0,
    random_seed=4107, task_type="GPU", devices="0",
    allow_writing_files=False, verbose=100,
)
model_global.fit(X, y, cat_features=CATEGORICAL_FEATURES)
models["GLOBAL"] = model_global

for ap in SPECIAL_AIRPORTS:
    mask = airports == ap
    n = mask.sum()
    print(f"Training {ap} ({n} rows)...")
    if n < 8000:
        continue
    model_ap = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE",
        iterations=900, learning_rate=0.08, depth=8, l2_leaf_reg=8.0,
        random_seed=4107, task_type="GPU", devices="0",
        allow_writing_files=False, verbose=50,
    )
    model_ap.fit(X[mask], y[mask], cat_features=CATEGORICAL_FEATURES)
    models[ap] = model_ap

print("Models:", list(models.keys()))

print("Creating FINAL ranking predictions...")
ranking_full = pd.read_parquet(FINAL / "final_ranking.parquet")
ranking = ranking_full.loc[ranking_full["PHASE_mvt"].eq("DEP")].copy()
template = pd.read_parquet(FINAL / "final_submitting.parquet")

XR = make_features(ranking, hist_stats_full, ranking_full)
ranking_airports = ranking["ADEP_mvt"].astype(str).values

pred_global = models["GLOBAL"].predict(XR)
predictions = pred_global.copy()

for ap in models:
    if ap == "GLOBAL":
        continue
    mask = ranking_airports == ap
    if mask.sum() == 0:
        continue
    pred_ap = models[ap].predict(XR[mask])
    predictions[mask] = BLEND_WEIGHT * pred_ap + (1 - BLEND_WEIGHT) * pred_global[mask]

predictions = np.maximum(predictions, 0.0)

submission = template.copy()
submission[TARGET] = predictions.astype("float64")

out = OUT / "zestful-jewel_final.parquet"
submission.to_parquet(out, index=False)

print("\nFINAL COMPLETE")
print("FILE TO SUBMIT:", out)
print("Rows:", len(submission))
print("Unique IDs:", submission["MVT_ID_mvt"].nunique())
print("Missing:", submission[TARGET].isna().sum())
print("Median:", round(submission[TARGET].median(), 1))
print("Max:", round(submission[TARGET].max(), 1))
print("SHA256:", hashlib.sha256(out.read_bytes()).hexdigest())
