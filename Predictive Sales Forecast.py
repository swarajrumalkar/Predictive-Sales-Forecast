import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stores_sales_forecasting.csv", encoding="latin1")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="mixed",
    errors="coerce"
)

df = df.dropna(subset=["Order Date"])

monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Sales"].sum()

forecast = monthly_sales.tail(3).mean()

print("Predicted Next Month Sales:", round(forecast, 2))

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")

plt.title("Predictive Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)
plt.tight_layout()

plt.savefig("sales_forecast.png")
plt.show()