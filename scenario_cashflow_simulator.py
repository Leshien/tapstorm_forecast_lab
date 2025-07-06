
# Scenario-Based Cashflow Simulator for Tapstorm Studios
# Author: Tapstorm Financial Simulation Suite

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS (can be turned into user inputs in Streamlit)
start_month = "2025-01"
months_to_simulate = 12
starting_users = 120000

churn_rate = 0.05  # 5% monthly churn
cac = 2.5  # € per acquired user
arppu = 8.0  # € per paying user per month
monthly_acquisition_budget = 100000  # € per month

# Derived input: how many users we can acquire monthly
monthly_new_users = int(monthly_acquisition_budget / cac)

# Generate monthly simulation
dates = pd.date_range(start=start_month, periods=months_to_simulate, freq='MS')
users = [starting_users]
revenue = []
acquisition_costs = []

for _ in range(months_to_simulate):
    last_users = users[-1]
    churned = last_users * churn_rate
    gained = monthly_new_users
    new_users = last_users - churned + gained
    users.append(new_users)

    monthly_revenue = new_users * arppu
    monthly_cac_spend = monthly_new_users * cac

    revenue.append(monthly_revenue)
    acquisition_costs.append(monthly_cac_spend)

# Trim users list to match length
users = users[1:]

# Create DataFrame
df = pd.DataFrame({
    'Month': dates,
    'Active Users': users,
    'Cash Inflow (Revenue)': revenue,
    'Cash Outflow (CAC Spend)': acquisition_costs,
    'Net Cashflow': np.array(revenue) - np.array(acquisition_costs)
})

# Save to CSV
df.to_csv("tapstorm_scenario_cashflow_output.csv", index=False)

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['Month'], df['Cash Inflow (Revenue)'], label='Revenue')
plt.plot(df['Month'], df['Net Cashflow'], label='Net Cashflow', linestyle='--')
plt.title("Tapstorm Studios - Scenario-Based Cashflow Forecast")
plt.xlabel("Month")
plt.ylabel("€")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
