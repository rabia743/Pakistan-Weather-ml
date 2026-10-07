import re
import pandas as pd

weather = pd.read_csv("weather.csv")
print(weather.head(10))
print(weather.columns)

# Data Cleaning
print(weather.isnull().sum())
print(weather.dtypes)
weather = weather.drop('visibility', axis=1)
weather = weather.drop('date', axis=1)
print(weather.isnull().sum())
weather = weather.rename(columns={
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "dayofweek": "Day_Of_Week",
    "is_weekend": "Is_Weekend",
    "season": "Season",
    "city": "City",
    "region": "Region",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "elevation": "Elevation",
    "tmin": "Tmin",
    "tmax": "Tmax",
    "tavg": "Tavg",
    "prcp": "Prcp",
    "wspd": "Wspd",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "dew_point": "Dew_Point",
    "cloud_cover": "Cloud_Cover",
    "temp_range": "Temp_Range",
    "is_hot_day": "Is_Hot_Day",
    "is_cold_day": "Is_Cold_Day",
    "rainfall_intensity": "Rainfall_Intensity",
    "wind_category": "Wind_Category"
})
print(weather.columns.tolist())
print(weather.duplicated().sum())
# add the new column using condition
weather["Football_Play"] = (
    (weather["Tavg"] >= 15) &
    (weather["Tavg"] <= 35) &
    (weather["Tmax"] <= 35) &
    (weather["Prcp"] <= 5) &
    (weather["Wspd"] <= 30) &
    (weather["Humidity"] <= 85) &
    (weather["Cloud_Cover"] <= 90)
).map({True: "Yes", False: "No"})
# just view the total no and yes in dataset
print(weather["Football_Play"].value_counts())

# text cleaning
text_cols = ["Season", "City", "Region", "Rainfall_Intensity", "Wind_Category"]
for col in text_cols:
    weather[col] = weather[col].str.strip()
    weather[col] = weather[col].str.lower()
    weather[col] = weather[col].str.replace(r'\s+', ' ', regex=True)
    # Remove special characters (sirf letters, numbers aur spaces rakhna hai)
    weather[col] = weather[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# weather.to_csv("cleaned weather data.csv", index=False)