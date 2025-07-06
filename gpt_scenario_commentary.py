
import openai
import pandas as pd
import os

# Load Scenario forecast output
df = pd.read_csv("tapstorm_scenario_cashflow_output.csv")

# Format forecast preview
def build_prompt(df):
    preview = df[['Month', 'Active Users', 'Net Cashflow']].tail(12).to_string(index=False)
    prompt = f"""You are a financial strategy AI. Below is a 12-month scenario-based forecast for Tapstorm Studios.
The simulation uses:
- Starting users: 120,000
- Monthly churn: 5%
- CAC: €2.50
- ARPPU: €8.00
- Monthly acquisition budget: €100,000

Please generate a concise financial commentary:
- Interpret the revenue and user base trends
- Assess sustainability of cashflow trajectory
- Recommend any strategic adjustments (churn reduction, CAC control, ARPPU increase)

Forecast Preview:
{preview}
""" 
    return prompt

# GPT Call Function
def get_scenario_commentary(df):
    prompt = build_prompt(df)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful strategic finance analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response['choices'][0]['message']['content']

# Example usage:
# df = pd.read_csv("tapstorm_scenario_cashflow_output.csv")
# commentary = get_scenario_commentary(df)
# print(commentary)
