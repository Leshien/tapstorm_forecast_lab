
import openai
import pandas as pd
import os

df = pd.read_csv("tapstorm_segmented_prophet_forecast_output.csv")

def build_prompt(df):
    sample = df.groupby('Segment').tail(1)[['ds', 'yhat', 'Segment']].to_string(index=False)
    prompt = f"""You're a financial data AI. Here's the final forecasted month for each customer segment from a Prophet model.

Please:
- Compare relative growth or stagnation across segments
- Suggest segment prioritization
- Highlight key risks or divergence

Data:
{sample}
""" 
    return prompt

def get_segmented_prophet_commentary(df):
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
