
# --- App Mode Selector ---
st.sidebar.title("đ§­ Start Here: Choose App Mode")
mode = st.sidebar.radio(
    "What do you want to explore?",
    ["đ Executive Summary", "đŹ Analyst Sandbox", "đ§Ş Demo All Models"],
    index=2
)

# --- Homepage Intro ---
st.title("đŽ Tapstorm Forecast Lab")
st.markdown("""
Welcome to the **Tapstorm Forecast Lab** â a capstone simulation platform for strategic forecasting and data storytelling.

This interactive Streamlit app lets you:
- Test multiple forecasting models (Prophet, SARIMA, LSTM)
- Upload your own data or use Tapstorm sample datasets
- Generate executive-style GPT commentary
- Simulate scenarios across churn, CAC, and ARPU

---

### đ Mode Selected: `""" + mode + """`

- **đ Executive Summary** â View simplified forecasts and strategic GPT insights.
- **đŹ Analyst Sandbox** â Upload, compare, and explore forecasting models in detail.
- **đ§Ş Demo All Models** â Full technical suite with all data views and download options.
""")

# Filter content based on selected mode
if mode == "đ Executive Summary":
    st.header("đ Coming Soon: Executive Forecast Dashboard")
    st.info("This mode will show simplified forecast comparisons and strategic commentary.")
elif mode == "đŹ Analyst Sandbox":
    st.header("đ Welcome to the Analyst Sandbox")
    st.markdown("All tools unlocked. Upload your data, test models, and extract insights.")
else:
    st.header("đ§Ş Demo Mode: Full Technical Suite Enabled")



import openai

# --- GPT Commentary Function ---
def generate_commentary(forecast_df, model_name):
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    recent_df = forecast_df.tail(12)
    text_preview = recent_df.to_csv(index=False)

    prompt = f"""
You are a financial forecasting analyst. Analyze the following {model_name} forecast (monthly cash inflows). Your job is to write a 150-word business-style commentary for a CFO.

Include:
- Observed trend and seasonality
- Any risk factors or anomalies
- Financial interpretation
- Strategic narrative tone

Forecast data:
{text_preview}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a financial analyst producing short internal memos."},
                {"role": "user", "content": prompt}
            ]
        )
        commentary = response["choices"][0]["message"]["content"]
        return commentary

    except Exception as e:
        return f"â ď¸ Error generating commentary: {str(e)}"




st.set_page_config(page_title="Tapstorm Forecast Lab", layout="wide")

st.title("đŽ Tapstorm Forecast Lab")
st.caption("Multi-model cashflow forecasting suite with GPT insights")

# Sidebar navigation
model = st.sidebar.selectbox(
    "Select Forecast Model",
    [
        "đ Prophet Forecast",
        "đ Segmented Prophet Forecast",
        "đ SARIMA Forecast",
        "đ§  LSTM Forecast",
        "đ§Ş Scenario Simulator"
    ]
)

st.markdown("---")

if model == "đ Prophet Forecast":
    st.header("Prophet Cashflow Forecast")
    st.write("This model uses Facebook Prophet to forecast future cash inflows based on historical data.")
    st.info("đ Forecast output: `tapstorm_prophet_forecast_output.csv`")

    try:
        df = pd.read_csv("tapstorm_prophet_forecast_output.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        st.subheader("đ Forecast Data Preview")
        st.dataframe(df.tail(12))

        st.subheader("đ Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df["ds"], df["yhat"], label="Forecast", color="tab:blue")
        ax.set_title("Prophet Forecast: Cash Collected (âŹ)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (âŹ)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§Ş Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "đ§Ş Scenario Simulator")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§  LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "đ§  LSTM Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "đ SARIMA Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "đ Segmented Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "đ Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("đĽ Download Forecast CSV", csv, "tapstorm_prophet_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("â ď¸ Forecast data not found. Please generate the Prophet forecast output first.")

elif model == "đ Segmented Prophet Forecast":
    st.header("Segmented Prophet Forecast")
    st.write("Prophet forecasts by customer segment (e.g. casual vs. high-spend users).")
    st.info("đ Forecast output: `tapstorm_segmented_prophet_forecast_output.csv`")

    try:
        df_seg = pd.read_csv("tapstorm_segmented_prophet_forecast_output.csv")
        df_seg["ds"] = pd.to_datetime(df_seg["ds"])
        segments = df_seg["segment"].unique()
        selected_segment = st.selectbox("Select Customer Segment", segments)

        df_filtered = df_seg[df_seg["segment"] == selected_segment]
        st.subheader("đ Forecast Data Preview")
        st.dataframe(df_filtered.tail(12))

        st.subheader(f"đ Forecast Chart: {selected_segment}")
        fig, ax = plt.subplots()
        ax.plot(df_filtered["ds"], df_filtered["yhat"], label="Forecast", color="tab:orange")
        ax.set_title(f"{selected_segment} Segment Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (âŹ)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§Ş Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "đ§Ş Scenario Simulator")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§  LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "đ§  LSTM Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "đ SARIMA Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "đ Segmented Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "đ Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    

        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("đĽ Download Segment Forecast CSV", csv, f"{selected_segment}_forecast.csv", "text/csv")

    except FileNotFoundError:
        st.error("â ď¸ Forecast data not found. Please generate the segmented forecast output first.")

elif model == "đ SARIMA Forecast":
    st.header("SARIMA Cashflow Forecast")
    st.write("Seasonal ARIMA model for capturing trend + seasonality with tuning.")
    st.info("đ Forecast output: `tapstorm_sarima_forecast_output.csv`")

    try:
        df_sarima = pd.read_csv("tapstorm_sarima_forecast_output.csv")
        df_sarima["ds"] = pd.to_datetime(df_sarima["ds"])
        st.subheader("đ Forecast Data Preview")
        st.dataframe(df_sarima.tail(12))

        st.subheader("đ Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_sarima["ds"], df_sarima["yhat"], label="Forecast", color="tab:green")
        ax.set_title("SARIMA Forecast: Cash Collected (âŹ)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (âŹ)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§Ş Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "đ§Ş Scenario Simulator")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§  LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "đ§  LSTM Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "đ SARIMA Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "đ Segmented Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "đ Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    

        csv = df_sarima.to_csv(index=False).encode('utf-8')
        st.download_button("đĽ Download Forecast CSV", csv, "tapstorm_sarima_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("â ď¸ Forecast data not found. Please generate the SARIMA forecast output first.")

elif model == "đ§  LSTM Forecast":
    st.header("LSTM Neural Network Forecast")
    st.write("Deep learning model that learns temporal dependencies in financial data.")
    st.info("đ Forecast output: `tapstorm_lstm_forecast_output.csv`")

    try:
        df_lstm = pd.read_csv("tapstorm_lstm_forecast_output.csv")
        df_lstm["ds"] = pd.to_datetime(df_lstm["ds"])
        st.subheader("đ Forecast Data Preview")
        st.dataframe(df_lstm.tail(12))

        st.subheader("đ Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_lstm["ds"], df_lstm["yhat"], label="Forecast", color="tab:red")
        ax.set_title("LSTM Forecast: Cash Collected (âŹ)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (âŹ)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§Ş Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "đ§Ş Scenario Simulator")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§  LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "đ§  LSTM Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "đ SARIMA Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "đ Segmented Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "đ Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    

        csv = df_lstm.to_csv(index=False).encode('utf-8')
        st.download_button("đĽ Download Forecast CSV", csv, "tapstorm_lstm_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("â ď¸ Forecast data not found. Please generate the LSTM forecast output first.")

elif model == "đ§Ş Scenario Simulator":
    st.header("Scenario-Based Cashflow Simulator")
    st.write("Play with churn, CAC, and ARPPU to simulate revenue and net cashflow.")
    st.info("đ Forecast output: `tapstorm_scenario_cashflow_output.csv`")

    try:
        df_scenario = pd.read_csv("tapstorm_scenario_cashflow_output.csv")
        df_scenario["ds"] = pd.to_datetime(df_scenario["ds"])
        st.subheader("đ Forecast Data Preview")
        st.dataframe(df_scenario.tail(12))

        st.subheader("đ Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_scenario["ds"], df_scenario["cashflow"], label="Net Cashflow", color="tab:purple")
        ax.set_title("Scenario Simulator: Cashflow Projection (âŹ)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cashflow (âŹ)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§Ş Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "đ§Ş Scenario Simulator")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ§  LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "đ§  LSTM Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "đ SARIMA Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "đ Segmented Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    


        st.subheader("đ§  GPT Commentary")
        if st.button("đ Generate CFO Commentary for đ Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "đ Prophet Forecast")
                st.markdown("### đŹ Strategic Insights")
                st.write(commentary)
    

        csv = df_scenario.to_csv(index=False).encode('utf-8')
        st.download_button("đĽ Download Forecast CSV", csv, "tapstorm_scenario_cashflow_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("â ď¸ Scenario simulation data not found. Please generate the output first.")

st.markdown("---")
st.caption("ÂŠ 2025 Tapstorm Studios | AI-enhanced Financial Forecasting Suite")



# --- Download Buttons ---
st.sidebar.header("đ¤ Download Forecasts")



st.sidebar.download_button("Download Prophet Forecast", convert_df(df), file_name="prophet_forecast.csv")
st.sidebar.download_button("Download Segmented Prophet Forecast", convert_df(segmented_forecast_df), file_name="segmented_prophet_forecast.csv")
st.sidebar.download_button("Download SARIMA Forecast", convert_df(sarima_df_out), file_name="sarima_forecast.csv")
st.sidebar.download_button("Download LSTM Forecast", convert_df(lstm_forecast_df), file_name="lstm_forecast.csv")
st.sidebar.download_button("Download Scenario Simulation", convert_df(simulator_forecast_df), file_name="scenario_forecast.csv")



# --- Forecast Comparison Dashboard ---
import streamlit as st

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")
import pandas as pd
import matplotlib.pyplot as plt

st.header("đ Forecast Comparison Dashboard")

# Load comparison dataset
comparison_df = pd.read_csv("tapstorm_forecast_comparison_dashboard.csv", parse_dates=["Date"])

# Optional filters
start_date = st.date_input("Start Date", value=comparison_df["Date"].min().date())
end_date = st.date_input("End Date", value=comparison_df["Date"].max().date())

filtered_df = comparison_df[
    (comparison_df["Date"] >= pd.to_datetime(start_date)) &
    (comparison_df["Date"] <= pd.to_datetime(end_date))
]

# Plot
st.subheader("đ Forecasts Over Time")
fig, ax = plt.subplots(figsize=(12, 6))