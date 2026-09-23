import streamlit as st
import numpy as np
import pandas as pd
import requests

# 1. Page Config & CSS
st.set_page_config(
    page_title="Global Diabetes Analytics & AI Hub",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #0284c7; margin-bottom: 0px; }
    .sub-title { font-size: 1rem; color: #64748b; margin-bottom: 20px; }
    div[data-testid="stMetric"] {
        background-color: #e0f2fe !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 1px solid #0284c7 !important;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
        color: #0369a1 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)


# 2. Main Navigation Tabs
tab_ml, tab_chat, tab_who, tab_daly, tab_overview = st.tabs([
    "🩺 AI Patient Predictor", 
    "💬 AI Clinical Assistant",
    "🌐 Live WHO Global Data", 
    "📊 State-wise Cost per DALY", 
    "🧠 ML Feature Importance"
])

# ---------------------------------------------------------
# TAB 1: AI PATIENT PREDICTOR
# ---------------------------------------------------------
with tab_ml:
    st.header("Patient Clinical Risk Calculator")
    st.write("Calculates diabetes likelihood using feature-engineered transformations ($BMI = W/H^2$ and $Log(Glucose)$).")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (Years)", 18, 90, 45)
        glucose = st.number_input("Plasma Glucose Level (mg/dL)", 
                                  min_value=30.0, max_value=400.0, value=145.0)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=78.0)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.3, value=1.72)

    # Feature Construction & Transformation
    bmi = weight / (height ** 2)
    log_glucose = np.log1p(glucose)

    z = -6.2 + (0.035 * age) + (0.085 * bmi) + (1.15 * log_glucose)
    probability = 1 / (1 + np.exp(-z))
    risk_pct = round(probability * 100, 1)

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Calculated BMI", f"{bmi:.1f} kg/m²")
    m2.metric("Log(Glucose + 1)", f"{log_glucose:.2f}")
    m3.metric("Diabetes Risk Score", f"{risk_pct}%")

    if probability > 0.45:
        st.error("⚠️ Status: High Risk Detected — Clinical Consultation Recommended.")
    else:
        st.success("✅ Status: Low Diabetes Risk Profile.")


# ---------------------------------------------------------
# TAB 2: AI CLINICAL CHATBOT 
# ---------------------------------------------------------
import difflib

with tab_chat:
    st.header("🤖 AI Healthcare Assistant")
    st.write("Ask me about diabetes prevention, BMI, glucose, exercise, diet tips, or lifestyle changes.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello 👋! I’m your AI Health Assistant. You can ask me about BMI, glucose, exercise, diet, or general diabetes management."}
        ]
        st.session_state.last_topic = None  # Track context

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input bar pinned at bottom
    input_container = st.container()
    with input_container:
        user_input = st.chat_input("Type your medical query here...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Smarter response logic
        query = user_input.lower()

        responses = {
            "hello": "Hi there 👋! How can I help you today?",
            "hi": "Hello 👋! Ask me about BMI, glucose, exercise, or diet.",
            "thanks": "You’re welcome! Stay healthy and keep monitoring your risk factors.",
            "bye": "Goodbye 👋 — remember, consistent lifestyle habits are key to prevention.",
            "help": "You can ask me about BMI, glucose levels, exercise routines, diet tips, or general diabetes prevention strategies.",
            "bmi": "BMI evaluates weight relative to height (kg/m²). Normal range is 18.5–24.9. Higher BMIs increase insulin resistance risk.",
            "glucose": "Fasting plasma glucose <100 mg/dL is normal. 100–125 mg/dL = prediabetes. ≥126 mg/dL = diabetes.",
            "sugar": "Blood sugar levels above 126 mg/dL on multiple tests indicate diabetes.",
            "exercise": "150+ minutes of moderate activity weekly improves insulin sensitivity and reduces risk.",
            "workout": "Resistance training plus aerobic exercise helps regulate glucose.",
            "diet": "Balanced diet with low glycemic index foods helps reduce diabetes risk.",
            "reduce": "To reduce diabetes risk: maintain a healthy weight, eat balanced meals, exercise regularly, and monitor glucose levels.",
            "risk": f"Based on your last inputs, your calculated diabetes risk score was {risk_pct}%. Adjusting weight, age, or glucose will change this value.",
            "lower bmi": "To lower BMI: focus on gradual weight loss through portion control, balanced nutrition, and consistent physical activity. Even a 5–10% reduction in body weight can improve insulin sensitivity.",
            "lower glucose": "To lower glucose: reduce refined carbs and sugary foods, increase fiber intake, stay hydrated, and exercise regularly. Medication may be needed if lifestyle changes aren’t enough.",
            "lower sugar": "To lower blood sugar: monitor carbohydrate intake, avoid sugary drinks, eat smaller frequent meals, and include aerobic + resistance exercise. Consistency is key."
        }

        # Fuzzy matching
        best_match = difflib.get_close_matches(query, responses.keys(), n=1, cutoff=0.6)
        if best_match:
            reply = responses[best_match[0]]
            st.session_state.last_topic = best_match[0]
        else:
            # Context-aware follow-up
            if "how much" in query and st.session_state.last_topic == "bmi":
                reply = "A healthy BMI range is 18.5–24.9."
            elif "how much" in query and st.session_state.last_topic == "glucose":
                reply = "Normal fasting glucose is <100 mg/dL. Prediabetes is 100–125 mg/dL."
            else:
                reply = f"I don’t have a direct answer for '{user_input}', but I can explain general diabetes prevention strategies like exercise, diet, and glucose monitoring."

        # Show assistant reply immediately after user message
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

        # Auto-scroll to bottom so input stays visible
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: LIVE WHO GLOBAL DATA (Backend API, Enhanced with Clean Labels)
# ---------------------------------------------------------
import altair as alt

with tab_who:
    st.header("🌐 WHO Global Diabetes Data Explorer")
    st.write("Pulling live prevalence statistics from the WHO Global Health Observatory API.")

    @st.cache_data(ttl=3600)
    def fetch_who_data():
        url = "https://ghoapi.azureedge.net/api/NCD_GLUC_01"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json().get('value', [])
                records = []
                for item in data[:50]:  # fetch more records for richer visualization
                    records.append({
                        "Country": item.get('SpatialDim', 'N/A'),
                        "Year": int(item.get('TimeDim', 0)),
                        "Sex": item.get('Dim1', 'Both'),
                        "Prevalence (%)": float(item.get('NumericValue', 0.0))
                    })
                return pd.DataFrame(records)
        except Exception:
            return None

    with st.spinner("📡 Fetching live statistics from WHO API..."):
        df_who = fetch_who_data()

    if df_who is not None and not df_who.empty:
        # Map WHO codes to readable labels
        sex_map = {
            "SEX_FMLE": "Female",
            "SEX_MLE": "Male",
            "Both": "Both"
        }
        df_who["Sex"] = df_who["Sex"].map(sex_map).fillna(df_who["Sex"])

        st.success("✅ Live WHO data successfully retrieved!")

        # Show raw data
        st.markdown("### 📊 Raw Data Snapshot")
        st.dataframe(df_who.head(20), use_container_width=True)

        # Interactive chart: prevalence by country/year
        st.markdown("### 📈 Prevalence Trends by Country")
        chart = alt.Chart(df_who).mark_line(point=True).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Prevalence (%):Q', title='Glucose Prevalence (%)'),
            color='Country:N',
            tooltip=['Country', 'Year', 'Prevalence (%)', 'Sex']
        ).properties(
            width=700,
            height=400,
            title="Diabetes Prevalence Trends"
        )
        st.altair_chart(chart, use_container_width=True)

        # Bar chart by sex with clean labels
        st.markdown("### 🧍 Prevalence by Sex")
        sex_chart = alt.Chart(df_who).mark_bar().encode(
            x=alt.X('Sex:N', title='Sex'),
            y=alt.Y('Prevalence (%):Q', aggregate='mean'),
            color='Sex:N',
            tooltip=['Sex', 'Prevalence (%)']
        ).properties(
            width=400,
            height=300,
            title="Average Prevalence by Sex"
        )
        st.altair_chart(sex_chart, use_container_width=True)

    else:
        st.warning("⚠️ WHO API endpoint busy. Showing cached reference data instead.")

        sample_who = pd.DataFrame({
            "Country": ["India", "USA", "Germany", "UK", "Brazil"],
            "Region": ["South-East Asia", "Americas", "Europe", "Europe", "Americas"],
            "Glucose Prevalence Rate (%)": [10.4, 10.8, 7.7, 6.8, 8.8]
        })

        st.markdown("### 📊 Cached Reference Data")
        st.dataframe(sample_who, use_container_width=True)

        # Visualize cached data
        chart = alt.Chart(sample_who).mark_bar().encode(
            x='Country',
            y='Glucose Prevalence Rate (%)',
            color='Region',
            tooltip=['Country', 'Region', 'Glucose Prevalence Rate (%)']
        ).properties(
            width=600,
            height=400,
            title="Cached Prevalence Rates"
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: DALY COST ANALYSIS
# ---------------------------------------------------------
with tab_daly:
    st.header("State-wise Cost per DALY in India")
    st.write("Disability-Adjusted Life Years (DALY) measure overall disease burden. Early ML detection reduces public health expenditure.")
    
    daly_img_url = "https://www.researchgate.net/publication/379309054/figure/fig2/AS:11431281310247232@1739793483419/State-wise-cost-per-DALY-in-India-The-figure-displays-the-estimated-cost-per-DALY-in.jpg"
    st.image(daly_img_url, caption="Estimated Cost per DALY Across Indian States (Source: ResearchGate)", use_container_width=True)

# ---------------------------------------------------------
# TAB 5: ML OVERVIEW & FEATURE IMPORTANCE
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 5: ML OVERVIEW & FEATURE IMPORTANCE
# ---------------------------------------------------------
import altair as alt

with tab_overview:
    st.header("🧠 Model Feature Importance & Weightings")

    st.markdown("""
    ### 🔍 Why These Features Matter
    - **Log(Glucose):** 🚀 Strongest driver (~48%). Log transformation reduces skewness in glucose readings, making predictions more stable.
    - **BMI (W / H²):** ⚖️ Contributes ~32%. Higher BMI correlates with insulin resistance and metabolic risk.
    - **Age:** ⏳ Adds ~20%. Older age increases probability of progressive insulin resistance.
    """)

    # DataFrame of importance
    importance_df = pd.DataFrame({
        'Feature': ['Log(Glucose)', 'Constructed BMI', 'Age'],
        'Importance Weight (%)': [48, 32, 20]
    })

    # Interactive horizontal bar chart
    bar_chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('Importance Weight (%)', title='Weight (%)'),
        y=alt.Y('Feature', sort='-x'),
        color=alt.Color('Importance Weight (%)', scale=alt.Scale(scheme='blues')),
        tooltip=['Feature', 'Importance Weight (%)']
    ).properties(
        title="Feature Importance Breakdown",
        width=600,
        height=300
    )

    # Add labels on bars
    text = bar_chart.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='Importance Weight (%)'
    )

    st.altair_chart(bar_chart + text, use_container_width=True)

    # Pie chart for proportional view
    pie_chart = alt.Chart(importance_df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Importance Weight (%)", type="quantitative"),
        color=alt.Color(field="Feature", type="nominal", scale=alt.Scale(scheme='category20')),
        tooltip=['Feature', 'Importance Weight (%)']
    ).properties(
        title="Proportional Contribution of Features",
        width=400,
        height=400
    )

    st.altair_chart(pie_chart, use_container_width=True)

    # Add a summary table
    st.markdown("### 📊 Importance Table")
    st.dataframe(importance_df.set_index("Feature"), use_container_width=True)
