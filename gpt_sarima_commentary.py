
import openai
import pandas as pd
import os

# Load SARIMA forecast output
df = pd.read_csv("tapstorm_sarima_forecast_output.csv")

# Format forecast preview
def build_prompt(df):
    preview = df.tail(12)[['Month', 'yhat']].to_string(index=False)
    prompt = f"""You are a financial AI assistant. The following is a 12-month SARIMA-based forecast of Tapstorm Studios’ monthly cash collection.

Please provide a concise and strategic commentary:
- Highlight trend direction and confidence
- Note any volatility or growth flattening
- Make recommendations for finance leadership (CFO/controller)

Forecast:
{preview}
""" 
    return prompt

# GPT Call Function
def get_sarima_commentary(df):
    prompt = build_prompt(df)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response['choices'][0]['message']['content']

# Example usage:
# df = pd.read_csv("tapstorm_sarima_forecast_output.csv")
# commentary = get_sarima_commentary(df)
# print(commentary)
