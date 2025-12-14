import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =========================
# File paths (Mac + OneDrive)
# =========================
BASE_PATH = "/Users/CNN/Library/CloudStorage/OneDrive-UniversityofExeter"

DATA_PATH = f"{BASE_PATH}/tfl_bike_data_2023_2024.csv"
DASHBOARD_PATH = f"{BASE_PATH}/tfl_analysis_dashboard.png"
REGRESSION_FIG_PATH = f"{BASE_PATH}/tfl_regression_analysis.png"
RESULTS_CSV_PATH = f"{BASE_PATH}/regression_results.csv"

# =========================
# Plot settings (matplotlib only)
# =========================
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

# =========================
# Load data
# =========================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 70)
print("Regression Analysis: Predicting Daily Cycling Demand")
print("=" * 70)

# =========================
# Regression variables
# =========================
X = df[["temperature", "is_weekday", "is_holiday", "rainfall", "month"]]
y = df["daily_trips"]

# =========================
# Fit model
# =========================
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# =========================
# Model evaluation
# =========================
r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\nModel Coefficients")
print("-" * 70)
print(f"{'Variable':<20}{'Coefficient':>15}")
print("-" * 70)
print(f"{'Intercept':<20}{model.intercept_:>15,.0f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:<20}{coef:>15,.0f}")

print("\nModel Performance")
print("-" * 70)
print(f"R²:            {r2:.4f}")
print(f"Adjusted R²:   {adj_r2:.4f}")
print(f"RMSE:          {rmse:,.0f} trips/day")

# =========================
# F-test
# =========================
n = len(y)
k = X.shape[1]
f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
p_value = stats.f.sf(f_stat, k, n - k - 1)

print(f"\nF-statistic: {f_stat:.2f}")
print(f"P-value:     {p_value:.6f}")

# =========================
# Residuals
# =========================
residuals = y - y_pred

# =========================
# Dashboard figure
# =========================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Time series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df["date"], y, label="Actual", alpha=0.6)
ax1.plot(df["date"], y_pred, label="Predicted", linestyle="--")
ax1.set_title("TfL Cycle Hire Demand (2023–2024)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Daily Trips")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Actual vs predicted
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y, y_pred, alpha=0.4)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax2.set_title("Actual vs Predicted")
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.grid(True, alpha=0.3)

# Residual histogram
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
ax3.axvline(0, color="red", linestyle="--")
ax3.set_title("Residual Distribution")
ax3.set_xlabel("Residual")
ax3.set_ylabel("Frequency")

# Residual plot
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(y_pred, residuals, alpha=0.4)
ax4.axhline(0, color="red", linestyle="--")
ax4.set_title("Residuals vs Predicted")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Residual")
ax4.grid(True, alpha=0.3)

# Temperature effect
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(df["temperature"], y, alpha=0.3)
ax5.set_title("Temperature vs Demand")
ax5.set_xlabel("Temperature (°C)")
ax5.set_ylabel("Daily Trips")

# Weekday vs weekend
ax6 = fig.add_subplot(gs[2, 1])
weekday_mean = df.groupby("is_weekday")["daily_trips"].mean()
ax6.bar(["Weekend", "Weekday"], weekday_mean.values)
ax6.set_title("Weekday vs Weekend")
ax6.set_ylabel("Average Daily Trips")

# Monthly pattern
ax7 = fig.add_subplot(gs[2, 2])
monthly_avg = df.groupby("month")["daily_trips"].mean()
ax7.bar(monthly_avg.index, monthly_avg.values)
ax7.set_title("Monthly Demand Pattern")
ax7.set_xlabel("Month")
ax7.set_ylabel("Average Daily Trips")

plt.suptitle("TfL Cycle Hire Demand Analysis Dashboard", fontsize=16)
plt.savefig(DASHBOARD_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✓ Dashboard saved to: {DASHBOARD_PATH}")

# =========================
# Save regression results
# =========================
results_df = pd.DataFrame({
    "Variable": ["Intercept"] + list(X.columns),
    "Coefficient": [model.intercept_] + list(model.coef_),
    "Business interpretation": [
        "Baseline daily demand",
        "Effect per 1°C temperature change",
        "Weekday uplift vs weekend",
        "Holiday impact",
        "Effect per 1mm rainfall",
        "Monthly trend factor"
    ]
})

results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"✓ Regression results saved to: {RESULTS_CSV_PATH}")

print("\n" + "=" * 70)
print("✓ All analyses and visualisations completed!")
print("=" * 70)
