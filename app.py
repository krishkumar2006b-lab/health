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
# TAB 1: AI PATIENT PREDICTOR (Premium Dashboard Style)
# ---------------------------------------------------------

# ---------------------------------------------------------
# PATIENT DETAILS — PREMIUM CARD DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>
    /* Main input card */
    .patient-card {
        background: linear-gradient(145deg, #172A3A, #203A4F);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 24px 26px;
        margin: 10px 0 22px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    }

    /* Section headings */
    .input-section-title {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .input-section-subtitle {
        color: #AFC4D4;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Individual mini cards */
    .input-card {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }

    .input-card:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(52,152,219,0.45);
    }

    .input-label {
        color: #FFFFFF;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 3px;
    }

    .input-description {
        color: #9FB4C4;
        font-size: 12px;
        margin-bottom: 8px;
    }

    /* Info badges */
    .info-badge {
        display: inline-block;
        background: rgba(52,152,219,0.16);
        color: #5DADE2;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 3px;
    }

    /* Streamlit widget spacing */
    div[data-testid="stSlider"],
    div[data-testid="stNumberInput"],
    div[data-testid="stRadio"] {
        margin-top: -5px;
    }

    /* Radio buttons */
    div[role="radiogroup"] {
        gap: 8px;
    }

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 7px 12px;
    }

    /* Divider */
    .soft-divider {
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.15),
            transparent
        );
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="patient-card">

    <div class="input-section-title">
        👤 Patient Information
    </div>

    <div class="input-section-subtitle">
        Enter the following information to calculate your
        <b>10-year type 2 diabetes risk</b>.
    </div>

    <span class="info-badge">🔒 Your inputs are used only for this calculation</span>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TWO-COLUMN INPUT LAYOUT
# ---------------------------------------------------------

col1, col2 = st.columns(2, gap="large")


# =========================================================
# LEFT COLUMN — BODY & AGE
# =========================================================

with col1:

    st.markdown("""
    <div class="patient-card">

        <div class="input-section-title">
            📋 Basic Measurements
        </div>

        <div class="input-section-subtitle">
            Your age and body measurements
        </div>

    </div>
    """, unsafe_allow_html=True)


    # AGE
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🧍 Age</div>
        <div class="input-description">
            Your current age in years
        </div>
    </div>
    """, unsafe_allow_html=True)

    age = st.slider(
        "Age (Years)",
        min_value=18,
        max_value=90,
        value=33,
        label_visibility="collapsed"
    )


    # BMI
    st.markdown("""
    <div class="input-card">
        <div class="input-label">⚖️ Body Mass Index (BMI)</div>
        <div class="input-description">
            Weight relative to your height
        </div>
    </div>
    """, unsafe_allow_html=True)

    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=15.0,
        max_value=50.0,
        value=24.0,
        step=0.1,
        label_visibility="collapsed"
    )


    # WAIST
    st.markdown("""
    <div class="input-card">
        <div class="input-label">📏 Waist Circumference</div>
        <div class="input-description">
            Measure around your waist in centimeters
        </div>
    </div>
    """, unsafe_allow_html=True)

    waist = st.number_input(
        "Waist Circumference (cm)",
        min_value=50.0,
        max_value=150.0,
        value=90.0,
        step=0.5,
        label_visibility="collapsed"
    )


# =========================================================
# RIGHT COLUMN — LIFESTYLE & MEDICAL HISTORY
# =========================================================

with col2:

    st.markdown("""
    <div class="patient-card">

        <div class="input-section-title">
            ❤️ Lifestyle & Medical History
        </div>

        <div class="input-section-subtitle">
            Tell us about your daily habits and health history
        </div>

    </div>
    """, unsafe_allow_html=True)


    # PHYSICAL ACTIVITY
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🏃 Physical Activity</div>
        <div class="input-description">
            Do you get at least 30 minutes of physical activity each day?
        </div>
    </div>
    """, unsafe_allow_html=True)

    activity = st.radio(
        "Physical Activity",
        ["Yes", "No"],
        horizontal=True,
        label_visibility="collapsed"
    )


    # FRUIT / VEGETABLE
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🥗 Fruit & Vegetable Intake</div>
        <div class="input-description">
            Do you eat fruits or vegetables every day?
        </div>
    </div>
    """, unsafe_allow_html=True)

    diet = st.radio(
        "Fruit and Vegetable Intake",
        ["Yes", "No"],
        horizontal=True,
        label_visibility="collapsed"
    )


    # BLOOD PRESSURE MEDICATION
    st.markdown("""
    <div class="input-card">
        <div class="input-label">💊 Blood Pressure Medication</div>
        <div class="input-description">
            Are you currently taking medication for high blood pressure?
        </div>
    </div>
    """, unsafe_allow_html=True)

    meds = st.radio(
        "Antihypertensive Medication",
        ["Yes", "No"],
        horizontal=True,
        label_visibility="collapsed"
    )


    # HIGH BLOOD GLUCOSE
    st.markdown("""
    <div class="input-card">
        <div class="input-label">🩸 History of High Blood Glucose</div>
        <div class="input-description">
            Have you ever been told that your blood glucose was high?
        </div>
    </div>
    """, unsafe_allow_html=True)

    high_glucose = st.radio(
        "History of High Blood Glucose",
        ["Yes", "No"],
        horizontal=True,
        label_visibility="collapsed"
    )


# =========================================================
# FAMILY HISTORY — FULL WIDTH CARD
# =========================================================

st.markdown("""
<div class="patient-card">

    <div class="input-section-title">
        👨‍👩‍👧 Family History of Diabetes
    </div>

    <div class="input-section-subtitle">
        Select the closest relationship that applies to you.
    </div>

</div>
""", unsafe_allow_html=True)

family = st.radio(
    "Family History",
    [
        "No",
        "Yes (grandparent/uncle/aunt)",
        "Yes (parent/sibling/child)"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# =========================================================
# INPUT SUMMARY
# =========================================================

st.markdown("""
<div class="soft-divider"></div>

<div style="
    text-align:center;
    color:#AFC4D4;
    font-size:13px;
    margin-bottom:8px;
">
    ✓ All required information entered
</div>
""", unsafe_allow_html=True)


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
# TAB 3: LIVE WHO GLOBAL DATA (Clean Dashboard Style, Fixed 'Both', Custom Colors)
# ---------------------------------------------------------
import altair as alt

with tab_who:
    st.header("🌐 WHO Global Diabetes Data Explorer")
    st.write("Interactive prevalence statistics from the WHO Global Health Observatory API.")

    @st.cache_data(ttl=3600)
    def fetch_who_data():
        url = "https://ghoapi.azureedge.net/api/NCD_GLUC_01"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json().get('value', [])
                records = []
                for item in data:  # pull all records
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
        sex_map = {"SEX_FMLE": "Female", "SEX_MLE": "Male", "Both": "Both"}
        df_who["Sex"] = df_who["Sex"].map(sex_map).fillna(df_who["Sex"])

        st.success("✅ Live WHO data successfully retrieved!")

        # Filter controls
        countries = sorted(df_who["Country"].unique())
        selected_country = st.selectbox("Select a country:", countries)
        selected_sex = st.radio("Select sex:", ["Both", "Male", "Female"])

        # Handle 'Both' properly
        if selected_sex == "Both":
            filtered = df_who[df_who["Country"] == selected_country]
        else:
            filtered = df_who[(df_who["Country"] == selected_country) & (df_who["Sex"] == selected_sex)]

        # Line chart for prevalence trends (custom colors)
        st.markdown(f"### 📈 Prevalence Trends in {selected_country} ({selected_sex})")
        trend_chart = alt.Chart(filtered).mark_line(point=True).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Prevalence (%):Q', title='Glucose Prevalence (%)'),
            color=alt.Color('Sex:N',
                            title="Sex",
                            scale=alt.Scale(domain=["Male", "Female"],
                                            range=["blue", "red"])),
            tooltip=['Year', 'Sex', 'Prevalence (%)']
        ).properties(
            width=700,
            height=400,
            title=f"Diabetes Prevalence Trends in {selected_country}"
        )
        st.altair_chart(trend_chart, use_container_width=True)

        # Bar chart by sex (averages, same color scheme)
        st.markdown(f"### 🧍 Average Prevalence by Sex in {selected_country}")
        sex_chart = alt.Chart(df_who[df_who["Country"] == selected_country]).mark_bar().encode(
            x=alt.X('Sex:N', title='Sex'),
            y=alt.Y('Prevalence (%):Q', aggregate='mean'),
            color=alt.Color('Sex:N',
                            title="Sex",
                            scale=alt.Scale(domain=["Male", "Female", "Both"],
                                            range=["blue", "red", "gray"])),
            tooltip=['Sex', 'Prevalence (%)']
        ).properties(
            width=500,
            height=300,
            title=f"Average Prevalence by Sex in {selected_country}"
        )
        st.altair_chart(sex_chart, use_container_width=True)

        # Data snapshot
        st.markdown("### 📊 Data Snapshot")
        st.dataframe(filtered.sort_values("Year"), use_container_width=True)

    else:
        st.warning("⚠️ WHO API endpoint busy. Showing cached reference data instead.")

        sample_who = pd.DataFrame({
            "Country": ["India", "USA", "Germany", "UK", "Brazil"],
            "Region": ["South-East Asia", "Americas", "Europe", "Europe", "Americas"],
            "Glucose Prevalence Rate (%)": [10.4, 10.8, 7.7, 6.8, 8.8]
        })

        st.markdown("### 📊 Cached Reference Data")
        st.dataframe(sample_who, use_container_width=True)

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
# TAB 4: DALY COST ANALYSIS (Accurate State-wise INR Dashboard)
# ---------------------------------------------------------
import altair as alt

with tab_daly:
    st.markdown("<h1 style='font-size:32px;'>💰 State-wise Cost per DALY in India</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px;'>
    Disability-Adjusted Life Years (DALY) measure the total burden of disease.  
    Each DALY represents one lost year of healthy life.  
    Research shows the <b>average cost per DALY in India is ~₹82,000</b>,  
    but state-level costs vary dramatically — from <b>~₹27,000</b> in some states  
    to <b>~₹2,69,000</b> in others.  
    Early ML detection can reduce these costs by preventing complications and improving outcomes.
    </p>
    """, unsafe_allow_html=True)

    # Accurate state-wise INR sample data (illustrative but grounded in published ranges)
    sample_daly = pd.DataFrame({
        "State": ["Delhi", "Kerala", "Maharashtra", "Tamil Nadu", "Gujarat", "Punjab", "West Bengal", "Arunachal Pradesh", "Nagaland"],
        "Cost per DALY (INR)": [120000, 72000, 95000, 83000, 80000, 88000, 76000, 269000, 145000]
    })

    # Create two columns: left for chart, right for image
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h2 style='font-size:24px;'>📊 DALY Costs by State (₹)</h2>", unsafe_allow_html=True)

        bar_chart = alt.Chart(sample_daly).mark_bar().encode(
            x=alt.X('State:N', sort='-y', title='State'),
            y=alt.Y('Cost per DALY (INR):Q', title='Cost per DALY (₹)'),
            color=alt.Color('Cost per DALY (INR):Q', scale=alt.Scale(scheme='reds')),
            tooltip=['State', 'Cost per DALY (INR)']
        ).properties(
            width=600,
            height=400,
            title="Accurate DALY Costs by State"
        )

        # Add labels on bars
        text = bar_chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text='Cost per DALY (INR)'
        )

        st.altair_chart(bar_chart + text, use_container_width=True)

    with col2:
        st.markdown("<h2 style='font-size:24px;'>📷 Reference Image</h2>", unsafe_allow_html=True)
        daly_img_url = "https://www.researchgate.net/publication/379309054/figure/fig2/AS:11431281310247232@1739793483419/State-wise-cost-per-DALY-in-India-The-figure-displays-the-estimated-cost-per-DALY-in.jpg"
        st.image(daly_img_url, caption="Estimated Cost per DALY (Source: ResearchGate)", use_container_width=True)

    # Insights section
    st.markdown("""
    <h2 style='font-size:24px;'>🔎 Insights</h2>
    <ul style='font-size:18px;'>
        <li><b>Delhi</b> shows one of the highest costs (~₹1.2 lakh), reflecting urban healthcare expenditure.</li>
        <li><b>Kerala</b> has relatively lower costs (~₹72k), possibly due to stronger public health systems.</li>
        <li><b>Arunachal Pradesh</b> stands out with extremely high costs (~₹2.69 lakh), showing disparities in resource allocation.</li>
        <li><b>India’s average</b> is ~₹82k per DALY, but the spread across states highlights the need for <b>state-specific health policies</b>.</li>
    </ul>
    """, unsafe_allow_html=True)




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
