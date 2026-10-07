import pandas as pd
weather = pd.read_csv(" after weather_cleaned.csv")
# check how many unique cities are there in the dataset
print("All Cities in Dataset:")
print(weather["City"].unique())
# Total kitni cities hain
print(f"\nTotal Cities: {weather["City"].nunique()}")

# Har city ka year range check karein
city_years = weather.groupby("City")["Year"].agg(["min", "max"])

print("="*60)
print("YEAR RANGE of EACH CITY:")
print("="*60)
print(city_years)

# Har city ke total years count
print("\n HAR CITY KA DATA COUNT:")
for city in weather["City"].unique():
    years = weather[weather["City"] == city]["Year"].nunique()
    min_year = weather[weather["City"] == city]["Year"].min()
    max_year = weather[weather["City"] == city]["Year"].max()
    print(f"  {city.upper()}: {min_year} to {max_year} ({years} years)")

    
# speicific city ke liye data filter karna mean lahore
# Lahore ka year-wise rainfall
lahore_data = weather[weather["City"] == "lahore"]
lahore_yearly = lahore_data.groupby("Year")["Prcp"].sum()

print("\n LAHORE YEAR-WISE RAINFALL:")
for year, rain in lahore_yearly.items():
    print(f"  {year}: {rain:.2f} mm")

print(f"\n Year with MOST Rain: {lahore_yearly.idxmax()} ({lahore_yearly.max():.2f} mm)")
print(f" Year with LEAST Rain: {lahore_yearly.idxmin()} ({lahore_yearly.min():.2f} mm)")
