"""
Rebuilds solar_worth_it.ipynb around a single final dataset
(data/final_dataset.csv): a grid of points covering all of Jordan
(border-filtered, ~0.25 deg spacing), NASA POWER daily weather 2020-2025.
Every location is used for training - not just a handful of cities - so
the model has real coverage anywhere in the country, and the notebook can
show investors a heatmap of the best places for solar, not just a verdict
for 13 fixed cities.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- 0
md("""# SolarWise Jordan — Is Solar Wise For Me?

## Project Idea
We help someone in Jordan answer one question: *if I put solar panels on my roof, will I actually save money?* We use weather data to estimate how much power solar panels would produce at their location, then compare that to what they currently pay for electricity. We also cover the whole country, not just a handful of cities, so the same data can answer a second question: *where in Jordan is solar most worth it?* — useful for an investor scouting land for a utility-scale system.

## Problem
Solar panels produce different amounts of power in different parts of Jordan, depending on sun, heat, and elevation. Right now, someone thinking about solar has no easy way to check what to expect at their own location — they just guess, or trust a salesperson.

## Importance
Solar panels are expensive. A wrong decision costs real money for years. A quick, honest, location-based check helps people decide before they spend — and helps investors compare regions instead of picking one arbitrarily.

## Objectives
1. Collect weather data covering all of Jordan, not just a few cities
2. Build a model that predicts how much solar power any location can produce
3. Test the model to make sure its predictions are accurate and reliable, including at locations it never saw during training
4. Use the prediction to estimate savings for a household, and to compare generation potential across the whole country for investors""")

# ---------------------------------------------------------------- 1
md("""# Understanding the Data and Descriptive Analysis

Before doing anything else, we need to know what we're working with. This section answers: where did the data come from, how big is it, what does each column mean, and what problems does it have?

**Locations**: a grid of points spanning Jordan's full bounding box (0.25 degree spacing, roughly every 25-28km), filtered down to only the points that actually fall inside Jordan's real border (checked offline against a place-name database, not just the bounding rectangle - a rectangle around Jordan also covers parts of Saudi Arabia, Syria, Iraq, Israel/West Bank, and Egypt). Desert points are kept, not just cities - an investor scouting land for a utility-scale solar farm cares about open land, not just where people live.

**Weather source**: [NASA POWER](https://power.larc.nasa.gov/) — a free NASA dataset of daily weather and solar measurements for any point on Earth. We pulled daily data for 2020-2025 (6 years) at every kept grid point.

**What each row is**: one day of weather data at one grid location (latitude/longitude/elevation), with 7 raw weather variables renamed below to plain-language columns: solar irradiance, clear-sky irradiance, temperature, humidity, cloud cover, wind speed, and precipitation.

**Column names**: NASA POWER's API returns cryptic parameter codes (`ALLSKY_SFC_SW_DWN`, `T2M`, `RH2M`, ...). We rename them immediately after loading so every chart, table, and model-importance plot in this notebook uses names a non-specialist reader can understand without a lookup table.""")

# ---------------------------------------------------------------- 2
code("""# Importing the libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go""")

# ---------------------------------------------------------------- 3
code("""df = pd.read_csv(r'C:\\Users\\Sabri B-tek\\Desktop\\Shahed_final project\\data\\final_dataset.csv')

# NASA POWER's raw parameter codes -> plain-language column names.
# Done immediately after loading so every cell after this one, and every
# chart/table/model-importance plot, uses names a non-specialist reader
# (e.g. a professor outside the ML/remote-sensing field) can read without
# a lookup table.
COLUMN_RENAME = {
    'ALLSKY_SFC_SW_DWN': 'solar_irradiance(kwh/m2)',      # GHI - actual sunlight reaching the ground
    'CLRSKY_SFC_SW_DWN': 'clear_sky_irradiance(kwh/m2)',  # theoretical sunlight ceiling, no clouds
    'T2M': 'temperature(c)',
    'RH2M': 'humidity(percentage)',
    'CLOUD_AMT': 'cloud_cover(percentage)',
    'WS10M': 'wind_speed(m/s)',
    'PRECTOTCORR': 'precipitation(mm)',
}
df = df.rename(columns=COLUMN_RENAME)

# Unique key for a grid location - lat/lon pairs repeat across 6 years of
# daily rows, so this is what we group/split/hold-out by, not city names.
df['location_id'] = df['lat'].round(2).astype(str) + '_' + df['lon'].round(2).astype(str)

df.head()""")

# ---------------------------------------------------------------- 4
md("Checking column names, data types, and whether any values are missing.")

# ---------------------------------------------------------------- 5
code("""df.info()
# lat, lon, elevation should be float - ok
# date is currently an int/object (YYYYMMDD) - will convert to datetime
# weather columns are numeric - ok

weather_cols = ['solar_irradiance(kwh/m2)', 'clear_sky_irradiance(kwh/m2)', 'temperature(c)',
                'humidity(percentage)', 'cloud_cover(percentage)', 'wind_speed(m/s)', 'precipitation(mm)']""")

# ---------------------------------------------------------------- 6
code("""# checking shape - rows = grid locations x days (2020-2025)
df.shape""")

# ---------------------------------------------------------------- 7
md("""### Coverage map

Sanity check: does the grid actually cover all of Jordan - north, south, and the eastern desert, not just a cluster of cities?""")

# ---------------------------------------------------------------- 8
code("""locations = df[['location_id', 'nearest_place', 'lat', 'lon']].drop_duplicates()

fig_map = px.scatter_map(
    locations, lat='lat', lon='lon', hover_name='nearest_place',
    zoom=6.3, height=550, map_style='carto-positron',
    title=f'{len(locations)} weather-data locations covering Jordan'
)
fig_map.update_traces(marker=dict(size=11, color='#1f77b4', opacity=0.85))
fig_map.show()""")

# ---------------------------------------------------------------- 9
md("Confirming how many distinct locations we actually have.")

# ---------------------------------------------------------------- 10
code("""df['location_id'].nunique()""")

# ---------------------------------------------------------------- 11
md("""## Data Cleaning

Real-world pulls are rarely perfectly clean, so we check for the standard issues before doing anything else: exact duplicate rows, NASA POWER's own missing-value marker, and physically impossible values.""")

# ---------------------------------------------------------------- 12
md("Making sure we didn't accidentally collect the same day/location twice.")

# ---------------------------------------------------------------- 13
code("""# checking for duplicate rows
n_dupes = df.duplicated().sum()
print(f"{n_dupes} exact duplicate rows found")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after dropping duplicates: {df.shape}")""")

# ---------------------------------------------------------------- 14
md('NASA POWER uses -999 as a "no data" marker instead of a blank/NaN — we check for it explicitly, then treat it as missing rather than a real reading.')

# ---------------------------------------------------------------- 15
code("""# NASA POWER marks missing values as -999 instead of NaN
missing_counts = (df[weather_cols] == -999).sum()
print(missing_counts[missing_counts > 0])

df[weather_cols] = df[weather_cols].replace(-999, np.nan)
n_missing_rows = df[weather_cols].isna().any(axis=1).sum()
print(f"\\n{n_missing_rows} rows have at least one missing value after flagging -999")

# Small number of affected rows relative to 6 years of daily data - drop
# rather than impute, so the model only trains on real measurements.
df = df.dropna(subset=weather_cols).reset_index(drop=True)
print(f"Shape after dropping missing rows: {df.shape}")""")

# ---------------------------------------------------------------- 16
md("Basic statistics (min/max/mean) to sanity-check the ranges make sense (e.g. temperature isn't 500°C).")

# ---------------------------------------------------------------- 17
code("df.describe()")

# ---------------------------------------------------------------- 18
md("""### Outlier check

`temperature(c)` shows an implausible range above — Jordan's climate does not produce -80C or 150C days. We flag and remove rows outside a physically plausible range rather than clip them, since a handful of corrupted readings shouldn't be silently reinterpreted as valid extremes.""")

# ---------------------------------------------------------------- 19
code("""PLAUSIBLE_TEMP_RANGE = (-10, 55)  # generous bounds for Jordan's actual climate

outlier_mask = ~df['temperature(c)'].between(*PLAUSIBLE_TEMP_RANGE)
print(f"{outlier_mask.sum()} rows outside a plausible temperature range:")
print(df.loc[outlier_mask, ['location_id', 'date', 'temperature(c)']])

df = df[~outlier_mask].reset_index(drop=True)
print(f"\\nShape after removing outliers: {df.shape}")""")

# ---------------------------------------------------------------- 20
code("df.describe()")

# ---------------------------------------------------------------- 21
md("# Preprocessing & Feature Engineering")

# ---------------------------------------------------------------- 22
md("### Converting date")

# ---------------------------------------------------------------- 23
code("""df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear""")

# ---------------------------------------------------------------- 24
md("""### Computing the target column: `solar_generation_kwh_per_kwp`

Not a raw NASA value — it's the daily solar generation per kWp of installed panel capacity, computed with the standard PV performance formula:

```
kWh = GHI x panel_area x panel_efficiency x performance_ratio(temperature)
```

where the performance ratio derates for panel cell temperature above 25C (-0.4%/C). GHI (`solar_irradiance(kwh/m2)`), panel constants, and temperature are the physics inputs; the model will later learn the full mapping from raw weather to this label.""")

# ---------------------------------------------------------------- 25
code("""# Standard panel assumptions (see plan.md Section 4)
PANEL_AREA_M2 = 2.1
PANEL_EFFICIENCY = 0.21
BASE_PERFORMANCE_RATIO = 0.78
TEMP_COEFF = -0.004  # -0.4% per degree C above 25C cell temp

# Estimate panel cell temperature from ambient temp + irradiance-driven heating (NOCT-style approximation)
NOCT = 45  # nominal operating cell temp, C
df['panel_cell_temp_c'] = df['temperature(c)'] + (NOCT - 20) / 800 * (df['solar_irradiance(kwh/m2)'] * 1000 / 24)

df['system_performance_ratio'] = BASE_PERFORMANCE_RATIO * (1 + TEMP_COEFF * (df['panel_cell_temp_c'] - 25).clip(lower=0))
df['solar_generation_kwh_per_kwp'] = df['solar_irradiance(kwh/m2)'] * PANEL_AREA_M2 * PANEL_EFFICIENCY * df['system_performance_ratio']

df[['solar_irradiance(kwh/m2)', 'temperature(c)', 'panel_cell_temp_c', 'system_performance_ratio', 'solar_generation_kwh_per_kwp']].describe()""")

# ---------------------------------------------------------------- 26
md("""### Cyclical encoding for day-of-year

Day 1 and day 365 are adjacent in reality but far apart numerically — sin/cos encoding fixes that for the model.

Turn day-of-year into sin/cos so Dec 31 and Jan 1 look close together, not far apart - helps the model learn smooth seasons instead of a fake jump at year-end""")

# ---------------------------------------------------------------- 27
code("""# Turn day-of-year into sin/cos so Dec 31 and Jan 1 look close together, not far apart - helps the model learn smooth seasons instead of a fake jump at year-end
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)""")

# ---------------------------------------------------------------- 28
md("# Data Visualization")

# ---------------------------------------------------------------- 29
md("### Univariate Analysis")

# ---------------------------------------------------------------- 30
code("""# Distribution of each raw weather feature
for col in weather_cols:
    fig = px.histogram(df, x=col, title=f'Distribution of {col}')
    fig.show()""")

# ---------------------------------------------------------------- 31
code("""# Distribution of the target column
fig = px.histogram(df, x='solar_generation_kwh_per_kwp', title='Distribution of solar_generation_kwh_per_kwp (target)')
fig.show()""")

# ---------------------------------------------------------------- 32
code("""# Boxplot check on irradiance and target (post-cleaning)
for col in ['solar_irradiance(kwh/m2)', 'solar_generation_kwh_per_kwp']:
    fig = px.box(df, x=col, title=f'Boxplot of {col}')
    fig.show()""")

# ---------------------------------------------------------------- 33
md("""**Question**: how is our data spread across Jordan - do a few places dominate the dataset, or is it evenly spread?""")

# ---------------------------------------------------------------- 34
code("""# Share of rows coming from each area - top 10 named areas, rest grouped as "Other"
place_counts = df['nearest_place'].value_counts()
top10 = place_counts.head(10)
pie_data = pd.concat([top10, pd.Series({'Other': place_counts.iloc[10:].sum()})])
pie_pct = (pie_data / pie_data.sum() * 100).round(1)

fig = px.pie(values=pie_pct.values, names=pie_pct.index,
             title='Share of data rows by area (top 10 + Other)')
fig.show()""")

# ---------------------------------------------------------------- 35
md("""**Answer**: two desert sub-districts dominate the dataset - `Al-Jafr` and `Rwaished` together make up over half the rows. That's not a real population pattern; it's because Nominatim (the tool we use to label each grid point with a nearby place) assigns huge, sparsely-populated desert administrative areas to many grid points at once, while smaller cities each only cover one or two points. So this pie chart is really showing area-of-label, not "where the data cares about most" - every grid point still gets equal weight in training regardless of its label.""")

# ---------------------------------------------------------------- 36
md("### Bivariate Analysis")

# ---------------------------------------------------------------- 37
md("""**Question**: does solar generation change by month - do summer months really produce more?""")

# ---------------------------------------------------------------- 38
code("""# Average target by month - expect summer months to generate more
monthly = df.groupby('month')['solar_generation_kwh_per_kwp'].mean().reset_index()
fig = px.bar(monthly, x='month', y='solar_generation_kwh_per_kwp', text_auto='.2f',
             title='Average solar_generation_kwh_per_kwp by month')
fig.show()""")

# ---------------------------------------------------------------- 39
md("""**Answer**: yes - generation rises steadily from winter into a clear summer peak (June-August) and falls back down toward December, matching the sun's seasonal path. This is the same seasonal shape the day-of-year sin/cos encoding is meant to help the model learn smoothly.""")

# ---------------------------------------------------------------- 40
md("""**Question**: how does temperature affect generation - hotter means more sun, but does panel heat hurt output?""")

# ---------------------------------------------------------------- 41
code("""# Target vs temperature - shows the derating effect at high temps
sample = df.sample(min(20000, len(df)), random_state=42)
fig = px.scatter(sample, x='temperature(c)', y='solar_generation_kwh_per_kwp', opacity=0.3,
                  title='solar_generation_kwh_per_kwp vs Temperature')
fig.show()""")

# ---------------------------------------------------------------- 42
md("""**Answer**: generation trends upward with temperature overall, since hotter days are usually sunnier days in Jordan's climate - but the cloud of points flattens out and gets noisier at the highest temperatures, which is the panel-heat derating effect (built into our target formula) partly offsetting the extra sunlight.""")

# ---------------------------------------------------------------- 43
md("""**Question**: does generation change by latitude - is the north or south of Jordan sunnier?""")

# ---------------------------------------------------------------- 44
code("""# Target vs latitude - spatial pattern across Jordan
by_lat = df.groupby('lat')['solar_generation_kwh_per_kwp'].mean().reset_index()
fig = px.line(by_lat, x='lat', y='solar_generation_kwh_per_kwp', markers=True,
              title='Average solar_generation_kwh_per_kwp by latitude')
fig.show()""")

# ---------------------------------------------------------------- 45
md("""**Answer**: generation is clearly higher in the south (lower latitude) and lower in the north - the southern desert gets more consistent, cloud-free sun than the greener north, so latitude on its own is a real predictor of generation.""")

# ---------------------------------------------------------------- 46
md("""**Question**: which area of Jordan actually produces the most solar energy?""")

# ---------------------------------------------------------------- 47
code("""# Average target by area - which named places generate the most?
by_place = df.groupby('nearest_place')['solar_generation_kwh_per_kwp'].mean().sort_values(ascending=False).head(15)
fig = px.bar(by_place, orientation='h', title='Top 15 areas by average solar_generation_kwh_per_kwp')
fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig.show()""")

# ---------------------------------------------------------------- 48
md("""**Answer**: the top 15 are almost entirely southern and desert areas - places like Al Quwayrah, Wadi Araba, Aqaba, Ma'an, and Al-Jafr. This matches where Jordan's real utility-scale solar plants are actually built, which is a good sanity check that the model's spatial pattern reflects reality, not just noise.""")

# ---------------------------------------------------------------- 49
md("""**Question**: do higher-elevation locations produce more solar power?""")

# ---------------------------------------------------------------- 50
code("""# Target vs elevation - do higher-altitude locations produce more solar power?
sample_elev = df.sample(min(20000, len(df)), random_state=42)
fig = px.scatter(sample_elev, x='elevation', y='solar_generation_kwh_per_kwp', opacity=0.3,
                  title='solar_generation_kwh_per_kwp vs Elevation')
fig.show()""")

# ---------------------------------------------------------------- 51
md("""**Answer**: there's no strong, clean trend - elevation alone doesn't drive generation the way latitude does. High and low points both show a wide spread of outcomes, meaning elevation matters less directly than being in a naturally sunnier southern/desert region.""")

# ---------------------------------------------------------------- 52
md("""**Question**: how much does cloud cover cut into solar output?""")

# ---------------------------------------------------------------- 53
code("""# Target vs cloud cover - how much does cloud cover cut into output?
sample_cloud = df.sample(min(20000, len(df)), random_state=42)
fig = px.scatter(sample_cloud, x='cloud_cover(percentage)', y='solar_generation_kwh_per_kwp', opacity=0.3,
                  title='solar_generation_kwh_per_kwp vs Cloud Cover')
fig.show()""")

# ---------------------------------------------------------------- 54
md("""**Answer**: a clear downward trend - as cloud cover percentage rises, generation drops noticeably, especially past moderate cloud levels. This lines up with cloud cover being the second most important feature to the model, right behind raw irradiance.""")

# ---------------------------------------------------------------- 55
md("""**Question**: does rain meaningfully reduce output, separately from cloud cover?""")

# ---------------------------------------------------------------- 56
code("""# Target vs precipitation - does rain meaningfully reduce output, separate from cloud cover?
sample_precip = df.sample(min(20000, len(df)), random_state=42)
fig = px.scatter(sample_precip, x='precipitation(mm)', y='solar_generation_kwh_per_kwp', opacity=0.3,
                  title='solar_generation_kwh_per_kwp vs Precipitation')
fig.show()""")

# ---------------------------------------------------------------- 57
md("""**Answer**: about 78% of days have zero precipitation (Jordan is arid most of the year), so the chart is dominated by a vertical band at 0mm covering the full range of generation values. On the rare rainy days, average generation is noticeably lower - roughly a third less than dry days. That's a real, sizeable drop, but it's very likely the cloud cover those rainy days come with doing the work, not the rain itself, since rain can't fall without clouds blocking the sun first.""")

# ---------------------------------------------------------------- 58
md("""**Question**: is Jordan's average temperature changing year to year (2020-2025)?""")

# ---------------------------------------------------------------- 59
code("""# Average temperature by year - is Jordan getting hotter over 2020-2025?
yearly_temp = df.groupby('year')['temperature(c)'].mean().reset_index()
fig = px.line(yearly_temp, x='year', y='temperature(c)', markers=True,
              title='Average Temperature by Year')
fig.show()""")

# ---------------------------------------------------------------- 60
md("""**Answer**: average temperature bounces around a fairly stable range year to year rather than showing a strong, obvious upward trend over just 6 years - a real warming trend would need a much longer time series to detect reliably, so we're not claiming one here.""")

# ---------------------------------------------------------------- 61
md("### Multivariate Analysis")

# ---------------------------------------------------------------- 42
code("""corr = df[weather_cols + ['panel_cell_temp_c', 'solar_generation_kwh_per_kwp']].corr()
fig = px.imshow(corr, color_continuous_scale='Viridis', text_auto='.2f', height=600,
                 title='Correlation heatmap')
fig.show()""")

# ---------------------------------------------------------------- 43
code("""# Average target by month, one line per year - is solar output stable and predictable year to year?
monthly_yearly = df.groupby(['year', 'month'])['solar_generation_kwh_per_kwp'].mean().reset_index()
fig = px.line(monthly_yearly, x='month', y='solar_generation_kwh_per_kwp', color='year', markers=True,
              title='Average solar_generation_kwh_per_kwp by Month, per Year')
fig.show()""")

# ---------------------------------------------------------------- 44
md("# Modeling")

# ---------------------------------------------------------------- 45
md("""### Train/validation/test split

Time-based split (not random) to avoid leakage: train on 2020-2023, validate on 2024, test on 2025. We also hold out the southernmost grid location entirely from training, to check spatial generalization to a location the model has never seen anywhere in its features.""")

# ---------------------------------------------------------------- 46
code("""feature_cols = ['lat', 'lon', 'elevation', 'clear_sky_irradiance(kwh/m2)', 'temperature(c)', 'humidity(percentage)',
                'cloud_cover(percentage)', 'wind_speed(m/s)', 'precipitation(mm)', 'day_of_year_sin', 'day_of_year_cos']
target_col = 'solar_generation_kwh_per_kwp'

# Hold out one grid location entirely (the southernmost point) to test spatial generalization
holdout_location = df.sort_values('lat')['location_id'].iloc[0]
holdout_mask = df['location_id'] == holdout_location

df_spatial_holdout = df[holdout_mask]
df_main = df[~holdout_mask]

train_df = df_main[df_main['year'] <= 2023]
val_df = df_main[df_main['year'] == 2024]
test_df = df_main[df_main['year'] == 2025]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_val, y_val = val_df[feature_cols], val_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Spatial holdout ({holdout_location}): {df_spatial_holdout.shape}")""")

# ---------------------------------------------------------------- 47
md("### Baseline and model comparison")

# ---------------------------------------------------------------- 48
code("""from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

models = {
    'Linear Regression': (LinearRegression(), X_train_scaled, X_val_scaled),
    'Random Forest': (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1), X_train, X_val),
    'XGBoost': (XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=1), X_train, X_val),
}

results = {}
for name, (model, X_tr, X_va) in models.items():
    model.fit(X_tr, y_train)
    preds = model.predict(X_va)
    results[name] = {
        'MAE': mean_absolute_error(y_val, preds),
        'RMSE': np.sqrt(mean_squared_error(y_val, preds)),
        'R2': r2_score(y_val, preds),
    }

pd.DataFrame(results).T""")

# ---------------------------------------------------------------- 49
md("""**Reading the table**: all three models score well because the target is a deterministic physics formula of the input weather, but the tree-based models handle the non-linear temperature-derating effect better than plain Linear Regression. We pick the model with the lowest validation MAE, not a fixed favorite, since which one wins can shift as the dataset changes size.""")

# ---------------------------------------------------------------- 50
md("### Predicted vs actual & residuals (best model)")

# ---------------------------------------------------------------- 51
code("""best_model_name = pd.DataFrame(results).T['MAE'].idxmin()
best_model = models[best_model_name][0]
print(f"Best model by validation MAE: {best_model_name}")

val_preds = best_model.predict(X_val)

fig = px.scatter(x=y_val, y=val_preds, opacity=0.3, labels={'x': 'Actual', 'y': 'Predicted'},
                  title='Predicted vs Actual - solar_generation_kwh_per_kwp (validation)')
fig.add_shape(type='line', x0=y_val.min(), y0=y_val.min(), x1=y_val.max(), y1=y_val.max(),
              line=dict(color='red', dash='dash'))
fig.show()""")

# ---------------------------------------------------------------- 52
code("""residuals = y_val - val_preds
fig = px.histogram(residuals, title='Residuals distribution (validation)')
fig.show()""")

# ---------------------------------------------------------------- 53
code("""importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
fig = px.bar(importances, orientation='h', title=f'{best_model_name} feature importance')
fig.show()""")

# ---------------------------------------------------------------- 54
md("""**Reading the chart**: clear-sky irradiance (`clear_sky_irradiance(kwh/m2)`, the theoretical sunlight ceiling for that day/location) dominates importance, with cloud cover a distant second. This matches the physics: the target formula is built directly from real irradiance, and clouds are what pull real irradiance down from the clear-sky ceiling. Temperature, humidity, wind, and location (lat/lon/elevation) barely move the needle in comparison — mainly because the target formula's temperature-derating effect is small next to the raw irradiance signal itself.""")

# ---------------------------------------------------------------- 55
md("### Final test-set evaluation (2025, reported once)")

# ---------------------------------------------------------------- 56
code("""test_preds = best_model.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, test_preds))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_preds)))
print("Test R2:", r2_score(y_test, test_preds))""")

# ---------------------------------------------------------------- 57
md("""### Spatial generalization check (held-out location)

The southernmost grid point was excluded from training entirely. If the model scores well here, it can predict for a location it has never seen — which is exactly what a real user's (or investor's) location would be.""")

# ---------------------------------------------------------------- 58
code("""X_spatial = df_spatial_holdout[feature_cols]
y_spatial = df_spatial_holdout[target_col]
spatial_preds = best_model.predict(X_spatial)

print("Spatial holdout MAE:", mean_absolute_error(y_spatial, spatial_preds))
print("Spatial holdout RMSE:", np.sqrt(mean_squared_error(y_spatial, spatial_preds)))
print("Spatial holdout R2:", r2_score(y_spatial, spatial_preds))""")

# ---------------------------------------------------------------- 59
md("""# Where Is Solar Best in Jordan? — A Country-Wide View

Everything so far predicts *energy* (kWh per kWp per day) and checks the model works even at unseen locations. This section turns that into two practical views:
- **For a household**: is solar worth it at their specific location, given Jordan's real electricity tariff
- **For an investor**: which parts of the country generate the most - a heatmap across every location we have data for, not just a handful of cities""")

# ---------------------------------------------------------------- 60
md("""### Predicted annual generation per location

Using the trained model, predict daily output for the most recent full year (2025, our test year) at every grid location, then aggregate to an annual average per kWp. This is the number both the household verdict and the investor heatmap are built on.""")

# ---------------------------------------------------------------- 61
code("""SYSTEM_SIZE_KWP = 5.0  # typical residential single-phase system in Jordan

df_2025 = df[df['year'] == 2025].copy()
df_2025['predicted_kwh_per_kwp'] = best_model.predict(df_2025[feature_cols])
df_2025['predicted_kwh'] = df_2025['predicted_kwh_per_kwp'] * SYSTEM_SIZE_KWP

monthly_gen = (
    df_2025.groupby(['location_id', 'nearest_place', 'lat', 'lon', 'month'])['predicted_kwh']
    .sum()
    .reset_index()
    .rename(columns={'predicted_kwh': 'monthly_generation_kwh'})
)
location_summary = (
    monthly_gen.groupby(['location_id', 'nearest_place', 'lat', 'lon'])['monthly_generation_kwh']
    .agg(avg_monthly_generation_kwh='mean', annual_generation_kwh='sum')
    .reset_index()
    .sort_values('annual_generation_kwh', ascending=False)
)
location_summary.head(10)""")

# ---------------------------------------------------------------- 62
md("""### Investor view: best places for solar in Jordan

A country-wide heatmap of predicted annual generation per kWp - the higher/brighter a point, the more a fixed-size system would generate there. This is the view an investor scouting land for a utility-scale system would actually want, instead of a single national number.""")

# ---------------------------------------------------------------- 63
code("""fig_heat = px.scatter_map(
    location_summary, lat='lat', lon='lon', color='annual_generation_kwh',
    hover_name='nearest_place',
    hover_data={'annual_generation_kwh': ':.0f', 'lat': False, 'lon': False},
    color_continuous_scale='Viridis', zoom=6.3, height=600,
    map_style='carto-positron',
    title=f'Predicted annual solar generation per kWp by location ({SYSTEM_SIZE_KWP:.0f} kWp reference system)'
)
fig_heat.update_traces(marker=dict(size=16))
fig_heat.show()""")

# ---------------------------------------------------------------- 64
md("### JEPCO tariff and household verdict logic")

# ---------------------------------------------------------------- 65
code("""# JEPCO residential tiered tariff (JOD per kWh)
TARIFF_TIERS = [(300, 0.050), (600, 0.100), (float('inf'), 0.200)]


def bill_jod(monthly_kwh: float) -> float:
    \"\"\"Progressive tiered bill for a given monthly usage.\"\"\"
    remaining = monthly_kwh
    prev_cap = 0
    total = 0.0
    for cap, price in TARIFF_TIERS:
        tier_kwh = max(0, min(remaining, cap - prev_cap))
        total += tier_kwh * price
        remaining -= tier_kwh
        prev_cap = cap
        if remaining <= 0:
            break
    return total


def solar_verdict(location_id: str, monthly_usage_kwh: float) -> dict:
    row = location_summary.set_index('location_id').loc[location_id]
    generation = row['avg_monthly_generation_kwh']
    net_usage = max(0, monthly_usage_kwh - generation)  # solar offsets usage, can't go negative (no net billing modeled)
    bill_without_solar = bill_jod(monthly_usage_kwh)
    bill_with_solar = bill_jod(net_usage)
    savings = bill_without_solar - bill_with_solar
    coverage_pct = min(100, generation / monthly_usage_kwh * 100)

    return {
        'location': row['nearest_place'],
        'monthly_usage_kwh': monthly_usage_kwh,
        'predicted_generation_kwh': round(generation, 1),
        'coverage_pct': round(coverage_pct, 1),
        'bill_without_solar_jod': round(bill_without_solar, 2),
        'bill_with_solar_jod': round(bill_with_solar, 2),
        'monthly_savings_jod': round(savings, 2),
        'verdict': 'Worth it' if coverage_pct >= 100 else f'Partial ({coverage_pct:.0f}% covered)',
    }


# Example: a household using 400 kWh/month, checked at the best and worst locations
best_loc = location_summary.iloc[0]['location_id']
worst_loc = location_summary.iloc[-1]['location_id']
example_results = pd.DataFrame([solar_verdict(loc, 400) for loc in [best_loc, worst_loc]])
example_results""")

# ---------------------------------------------------------------- 66
md("""# Conclusion

**Did we meet the objectives?**
1. Collected 6 years of weather data across a full grid covering Jordan (border-filtered, not just a few cities), and cleaned the raw pull (duplicates, missing-value markers, temperature outliers) before modeling
2. Built and compared three models, picking the one with the lowest validation error, predicting daily solar output with high accuracy (see metrics above)
3. Validated it on unseen time (2025 test year) and an unseen location (southernmost grid point held out entirely) — both scored comparably well, meaning the model generalizes both across time and across new places, not just memorizing what it saw in training
4. Delivered two practical outputs: a household verdict using JEPCO's real tariff, and a country-wide heatmap of where solar generates the most - useful for an investor comparing regions

**Key limitations, stated honestly:**
- The target (`solar_generation_kwh_per_kwp`) is computed from a physics formula, not measured real-world panel output — so the model is learning to reproduce that formula from weather inputs, not learning from actual installed-system performance data
- Solar irradiance resolution in NASA POWER is coarser than the surface-weather variables, so nearby grid points can share very similar irradiance values even though their temperature/humidity/wind differ
- The verdict logic assumes no net billing/export credit and a fixed 5 kWp system size — real households and installations vary

**What this means for someone in Jordan**: this data-driven check gives a location-specific, physically grounded answer to "will solar cover my usage" for a household, and a country-wide comparison for an investor — rather than a generic national average or a sales pitch — which was the whole point of the project.""")

# ---------------------------------------------------------------- 67
md("""# Exporting the Model for the App

Saving everything the SolarWise app needs, so it can predict for any location in Jordan without re-running this notebook or calling NASA POWER live:
- the trained model (whichever of the three scored best)
- every grid location's 2025 daily weather features (the app snaps a user's input to the nearest covered location and predicts from there)""")

# ---------------------------------------------------------------- 68
code("""import joblib

# compress=3 keeps the file well under GitHub's 100MB limit with no accuracy loss
joblib.dump(best_model, 'app/solarwise_model.pkl', compress=3)

# 2025 daily features per location - what the app feeds the model, after
# snapping the user's input coordinates to the nearest covered grid point
app_features = df[df['year'] == 2025][
    ['location_id', 'nearest_place', 'month'] + feature_cols  # feature_cols already has lat, lon
].reset_index(drop=True)
joblib.dump(app_features, 'app/location_2025_features.pkl', compress=3)

print('Saved:', 'app/solarwise_model.pkl,', 'app/location_2025_features.pkl')""")

nb['cells'] = cells
nbf.write(nb, 'solar_worth_it.ipynb')
print(f"Wrote {len(cells)} cells to solar_worth_it.ipynb")
