import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import StandardScaler

weather = pd.read_csv("cleaned weather data.csv")
# print(weather.head(5))

# string converted
day_encoder = LabelEncoder()
season_encoder = LabelEncoder()
city_encoder = LabelEncoder()
region_encoder = LabelEncoder()
wind_encoder = LabelEncoder()
rainfall_encoder = LabelEncoder()
football_encoder = LabelEncoder()
weather["Day_Of_Week"] = day_encoder.fit_transform(weather["Day_Of_Week"])
weather["Season"] = season_encoder.fit_transform(weather["Season"])
weather["City"] = city_encoder.fit_transform(weather["City"])
weather["Region"] = region_encoder.fit_transform(weather["Region"])
weather["Wind_Category"] = wind_encoder.fit_transform(weather["Wind_Category"])
weather["Rainfall_Intensity"] = rainfall_encoder.fit_transform(weather["Rainfall_Intensity"])
weather["Football_Play"] = football_encoder.fit_transform(weather["Football_Play"])

# create the files for model trainig and webpage
encoders = {
    "day_encoder": day_encoder,
    "season_encoder": season_encoder,
    "city_encoder": city_encoder,
    "region_encoder": region_encoder,
    "wind_encoder": wind_encoder,
    "rainfall_encoder": rainfall_encoder,
    "football_encoder": football_encoder
}
# joblib.dump(encoders, "weather_encoders.pkl")


# # numeric convert
scaler = StandardScaler()
numeric_columns = ["Year","Month","Day","Is_Weekend","Latitude","Longitude","Elevation",
                   "Tmin","Tmax","Tavg","Prcp","Wspd","Humidity","Pressure","Dew_Point",
                   "Cloud_Cover","Temp_Range","Is_Hot_Day","Is_Cold_Day"]
# joblib.dump(scaler, "weather_scaler.pkl")

# weather.to_csv("weather_feature_engineered.csv", index=False)
# print("complete who!!")