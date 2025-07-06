
# --- App Mode Selector ---
st.sidebar.title("🧭 Start Here: Choose App Mode")
mode = st.sidebar.radio(
    "What do you want to explore?",
    ["📈 Executive Summary", "🔬 Analyst Sandbox", "🧪 Demo All Models"],
    index=2
)

# --- Homepage Intro ---
st.title("🎮 Tapstorm Forecast Lab")
st.markdown("""
Welcome to the **Tapstorm Forecast Lab** — a capstone simulation platform for strategic forecasting and data storytelling.

This interactive Streamlit app lets you:
- Test multiple forecasting models (Prophet, SARIMA, LSTM)
- Upload your own data or use Tapstorm sample datasets
- Generate executive-style GPT commentary
- Simulate scenarios across churn, CAC, and ARPU

---

### 👉 Mode Selected: `""" + mode + """`

- **📈 Executive Summary** – View simplified forecasts and strategic GPT insights.
- **🔬 Analyst Sandbox** – Upload, compare, and explore forecasting models in detail.
- **🧪 Demo All Models** – Full technical suite with all data views and download options.
""")

# Filter content based on selected mode
if mode == "📈 Executive Summary":
    st.header("📊 Coming Soon: Executive Forecast Dashboard")
    st.info("This mode will show simplified forecast comparisons and strategic commentary.")
elif mode == "🔬 Analyst Sandbox":
    st.header("🔍 Welcome to the Analyst Sandbox")
    st.markdown("All tools unlocked. Upload your data, test models, and extract insights.")
else:
    st.header("🧪 Demo Mode: Full Technical Suite Enabled")



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
        return f"⚠️ Error generating commentary: {str(e)}"



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tapstorm Forecast Lab", layout="wide")

st.title("🎮 Tapstorm Forecast Lab")
st.caption("Multi-model cashflow forecasting suite with GPT insights")

# Sidebar navigation
model = st.sidebar.selectbox(
    "Select Forecast Model",
    [
        "📈 Prophet Forecast",
        "📊 Segmented Prophet Forecast",
        "📉 SARIMA Forecast",
        "🧠 LSTM Forecast",
        "🧪 Scenario Simulator"
    ]
)

st.markdown("---")

if model == "📈 Prophet Forecast":
    st.header("Prophet Cashflow Forecast")
    st.write("This model uses Facebook Prophet to forecast future cash inflows based on historical data.")
    st.info("📁 Forecast output: `tapstorm_prophet_forecast_output.csv`")

    try:
        df = pd.read_csv("tapstorm_prophet_forecast_output.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        st.subheader("📊 Forecast Data Preview")
        st.dataframe(df.tail(12))

        st.subheader("📈 Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df["ds"], df["yhat"], label="Forecast", color="tab:blue")
        ax.set_title("Prophet Forecast: Cash Collected (€)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (€)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧪 Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "🧪 Scenario Simulator")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧠 LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "🧠 LSTM Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📉 SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "📉 SARIMA Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📊 Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "📊 Segmented Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📈 Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "📈 Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "tapstorm_prophet_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("⚠️ Forecast data not found. Please generate the Prophet forecast output first.")

elif model == "📊 Segmented Prophet Forecast":
    st.header("Segmented Prophet Forecast")
    st.write("Prophet forecasts by customer segment (e.g. casual vs. high-spend users).")
    st.info("📁 Forecast output: `tapstorm_segmented_prophet_forecast_output.csv`")

    try:
        df_seg = pd.read_csv("tapstorm_segmented_prophet_forecast_output.csv")
        df_seg["ds"] = pd.to_datetime(df_seg["ds"])
        segments = df_seg["segment"].unique()
        selected_segment = st.selectbox("Select Customer Segment", segments)

        df_filtered = df_seg[df_seg["segment"] == selected_segment]
        st.subheader("📊 Forecast Data Preview")
        st.dataframe(df_filtered.tail(12))

        st.subheader(f"📈 Forecast Chart: {selected_segment}")
        fig, ax = plt.subplots()
        ax.plot(df_filtered["ds"], df_filtered["yhat"], label="Forecast", color="tab:orange")
        ax.set_title(f"{selected_segment} Segment Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (€)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧪 Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "🧪 Scenario Simulator")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧠 LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "🧠 LSTM Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📉 SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "📉 SARIMA Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📊 Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "📊 Segmented Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📈 Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "📈 Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    

        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Segment Forecast CSV", csv, f"{selected_segment}_forecast.csv", "text/csv")

    except FileNotFoundError:
        st.error("⚠️ Forecast data not found. Please generate the segmented forecast output first.")

elif model == "📉 SARIMA Forecast":
    st.header("SARIMA Cashflow Forecast")
    st.write("Seasonal ARIMA model for capturing trend + seasonality with tuning.")
    st.info("📁 Forecast output: `tapstorm_sarima_forecast_output.csv`")

    try:
        df_sarima = pd.read_csv("tapstorm_sarima_forecast_output.csv")
        df_sarima["ds"] = pd.to_datetime(df_sarima["ds"])
        st.subheader("📊 Forecast Data Preview")
        st.dataframe(df_sarima.tail(12))

        st.subheader("📈 Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_sarima["ds"], df_sarima["yhat"], label="Forecast", color="tab:green")
        ax.set_title("SARIMA Forecast: Cash Collected (€)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (€)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧪 Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "🧪 Scenario Simulator")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧠 LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "🧠 LSTM Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📉 SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "📉 SARIMA Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📊 Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "📊 Segmented Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📈 Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "📈 Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    

        csv = df_sarima.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "tapstorm_sarima_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("⚠️ Forecast data not found. Please generate the SARIMA forecast output first.")

elif model == "🧠 LSTM Forecast":
    st.header("LSTM Neural Network Forecast")
    st.write("Deep learning model that learns temporal dependencies in financial data.")
    st.info("📁 Forecast output: `tapstorm_lstm_forecast_output.csv`")

    try:
        df_lstm = pd.read_csv("tapstorm_lstm_forecast_output.csv")
        df_lstm["ds"] = pd.to_datetime(df_lstm["ds"])
        st.subheader("📊 Forecast Data Preview")
        st.dataframe(df_lstm.tail(12))

        st.subheader("📈 Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_lstm["ds"], df_lstm["yhat"], label="Forecast", color="tab:red")
        ax.set_title("LSTM Forecast: Cash Collected (€)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cash Collected (€)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧪 Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "🧪 Scenario Simulator")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧠 LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "🧠 LSTM Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📉 SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "📉 SARIMA Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📊 Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "📊 Segmented Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📈 Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "📈 Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    

        csv = df_lstm.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "tapstorm_lstm_forecast_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("⚠️ Forecast data not found. Please generate the LSTM forecast output first.")

elif model == "🧪 Scenario Simulator":
    st.header("Scenario-Based Cashflow Simulator")
    st.write("Play with churn, CAC, and ARPPU to simulate revenue and net cashflow.")
    st.info("📁 Forecast output: `tapstorm_scenario_cashflow_output.csv`")

    try:
        df_scenario = pd.read_csv("tapstorm_scenario_cashflow_output.csv")
        df_scenario["ds"] = pd.to_datetime(df_scenario["ds"])
        st.subheader("📊 Forecast Data Preview")
        st.dataframe(df_scenario.tail(12))

        st.subheader("📈 Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(df_scenario["ds"], df_scenario["cashflow"], label="Net Cashflow", color="tab:purple")
        ax.set_title("Scenario Simulator: Cashflow Projection (€)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cashflow (€)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧪 Scenario Simulator"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_scenario, "🧪 Scenario Simulator")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 🧠 LSTM Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_lstm, "🧠 LSTM Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📉 SARIMA Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_sarima, "📉 SARIMA Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📊 Segmented Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df_filtered, "📊 Segmented Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    


        st.subheader("🧠 GPT Commentary")
        if st.button("📝 Generate CFO Commentary for 📈 Prophet Forecast"):
            with st.spinner("Generating GPT insights..."):
                commentary = generate_commentary(df, "📈 Prophet Forecast")
                st.markdown("### 💬 Strategic Insights")
                st.write(commentary)
    

        csv = df_scenario.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "tapstorm_scenario_cashflow_output.csv", "text/csv")

    except FileNotFoundError:
        st.error("⚠️ Scenario simulation data not found. Please generate the output first.")

st.markdown("---")
st.caption("© 2025 Tapstorm Studios | AI-enhanced Financial Forecasting Suite")



# --- Download Buttons ---
st.sidebar.header("📤 Download Forecasts")

def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button("Download Prophet Forecast", convert_df(df), file_name="prophet_forecast.csv")
st.sidebar.download_button("Download Segmented Prophet Forecast", convert_df(segmented_forecast_df), file_name="segmented_prophet_forecast.csv")
st.sidebar.download_button("Download SARIMA Forecast", convert_df(sarima_df_out), file_name="sarima_forecast.csv")
st.sidebar.download_button("Download LSTM Forecast", convert_df(lstm_forecast_df), file_name="lstm_forecast.csv")
st.sidebar.download_button("Download Scenario Simulation", convert_df(simulator_forecast_df), file_name="scenario_forecast.csv")



# --- Forecast Comparison Dashboard ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("📊 Forecast Comparison Dashboard")

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
st.subheader("📈 Forecasts Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
for col in filtered_df.columns[1:]:
    ax.plot(filtered_df["Date"], filtered_df[col], label=col)

ax.set_title("Model Forecast Comparison")
ax.set_ylabel("Forecasted Revenue")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Optional download
st.download_button(
    label="📥 Download Forecast Comparison CSV",
    data=comparison_df.to_csv(index=False),
    file_name="tapstorm_forecast_comparison_dashboard.csv",
    mime="text/csv"
)



# --- Forecast Comparison Dashboard ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("📊 Forecast Comparison Dashboard")

# Load comparison dataset
comparison_df = pd.read_csv("tapstorm_forecast_comparison_dashboard.csv", parse_dates=["Date"])

# Optional filters
start_date = st.date_input("Start Date", value=comparison_df["Date"].min().date())
end_date = st.date_input("End Date", value=comparison_df["Date"].max().date())

filtered_df = comparison_df[
    (comparison_df["Date"] >= pd.to_datetime(start_date)) &
    (comparison_df["Date"] <= pd.to_datetime(end_date))
]

# Model toggles
st.sidebar.markdown("### Toggle Models")
model_columns = list(comparison_df.columns[1:])
selected_models = [col for col in model_columns if st.sidebar.checkbox(col, True)]

# Simulated standard deviations (for error bands)
std_map = {
    "Prophet": 1500,
    "Segmented Prophet": 2000,
    "SARIMA": 1800,
    "LSTM": 2200,
    "Scenario Simulator": 2500
}

# Plot
st.subheader("📈 Forecasts Over Time")
fig, ax = plt.subplots(figsize=(12, 6))

for col in selected_models:
    ax.plot(filtered_df["Date"], filtered_df[col], label=col)
    if col in std_map:
        std_dev = std_map[col]
        upper = filtered_df[col] + std_dev
        lower = filtered_df[col] - std_dev
        ax.fill_between(filtered_df["Date"], lower, upper, alpha=0.2)

ax.set_title("Model Forecast Comparison")
ax.set_ylabel("Forecasted Revenue")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# GPT commentary block (simulated)
st.subheader("💬 GPT Forecast Commentary")
st.markdown("""
- **Prophet** predicts moderate growth with seasonal undulations, reflecting historical patterns.
- **Segmented Prophet** shows optimistic lift, likely biased by high outlier performance in a key segment.
- **SARIMA** forecasts more conservative values with tighter variance, suitable for stable planning.
- **LSTM** anticipates subtle non-linear upticks, best viewed over the long term.
- **Scenario Simulator** spikes sharply — appropriate for aggressive growth or promotional assumptions.

📌 Use toggles to isolate or combine models based on business appetite for risk, optimism, and realism.
""")

# Optional download
st.download_button(
    label="📥 Download Forecast Comparison CSV",
    data=comparison_df.to_csv(index=False),
    file_name="tapstorm_forecast_comparison_dashboard.csv",
    mime="text/csv"
)
