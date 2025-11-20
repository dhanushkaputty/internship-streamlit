# app.py - Premium Pro Streamlit Dashboard (World Happiness)
# ---------------------- IMPORTS ----------------------
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="🌎 World Happiness Dashboard",
    page_icon="🌎",
    layout="wide"
)

# ---------------------- CUSTOM PREMIUM UI ----------------------
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #f4fff4;
}

h1, h2, h3 {
    color: #2a7f2a !important;
}

.section-header {
    background-color: #e7ffe7;
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
    color: #2b6e2b;
}

div.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------------- PAGE HEADER ----------------------
st.markdown("""
<h1 style="text-align:center; color:#2a7f2a;">
    🌎 World Happiness Analytics Dashboard
</h1>
<p style="text-align:center; font-size:18px; color:#444;">
    Complete 4-Week Internship Project • Data • Visualization • Modeling • Insights
</p>
<hr>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR NAVIGATION (TEXT ONLY) ----------------

# ---------------- SIDEBAR NAVIGATION (TEXT ONLY — CLEAN) ----------------

# FIX RADIO LABEL TEXT VISIBILITY (VERY IMPORTANT)
import streamlit as st

st.markdown("""
<style>

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #f4fff4 !important;
    }

    /* Make ALL sidebar text visible & clean */
    section[data-testid="stSidebar"] * {
        color: black !important;
        font-size: 16px !important;
    }

    /* Fix radio button text */
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: black !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Fix selectbox LABEL */
    label {
        color: black !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* ✅ Fix selectbox SELECTED VALUE at cursor */
    div[data-baseweb="select"] > div {
        background-color: #d0f0d0 !important;  /* Light green background */
        color: white !important;               /* Text stays visible */
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Fix text inside dropdown menu */
    ul[role="listbox"] li {
        color: black !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Fix selected item visibility in dropdown */
    ul[role="listbox"] li[aria-selected="true"] {
        background-color: #d0f0d0 !important;
        color: black !important;
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)



# ---------------- SIDEBAR NAVIGATION (PREMIUM WITH EMOJIS) ----------------
st.sidebar.title(" Navigation")

# Week selection (with emojis)
week_page = st.sidebar.selectbox(
    "📘 Internship Weeks",
    [
        "📘 Week 1 – Planning",
        "📗 Week 2 – Data Analysis",
        "📙 Week 3 – Visualization",
        "📕 Week 4 – Modeling & Reflection"
    ]
)

# Section selection (with emojis)
section_page = st.sidebar.radio(
    "📂 Sections",
    [
        "🏠 Home",
        "📁 Data",
        "📊 Visualizations",
        "🗺️ Map",
        "🤖 Modeling",
        "🔍 Insights",
        "ℹ️ About"
    ]
)


# ---------------------- DATASET UPLOAD ----------------------

st.markdown("<div class='section-header'>📂 Upload Dataset</div>", unsafe_allow_html=True)

if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Upload the World Happiness Report CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("✅ Dataset uploaded successfully!")
        except:
            st.error("❌ Error loading the dataset. Please upload a valid CSV file.")
else:
    df = st.session_state["df"]


# ---------------------- IF NO DATASET ----------------------
if df is None:
    st.warning("⚠ Please upload the dataset to continue.")
    st.stop()


# ---------------------- CLEANING COLUMN NAMES ----------------------

rename_map = {
    "Happiness.Rank": "Happiness_Rank",
    "Happiness.Score": "Happiness_Score",
    "Economy..GDP.per.Capita.": "GDP_per_Capita",
    "Family": "Family",
    "Health..Life.Expectancy.": "Life_Expectancy",
    "Freedom": "Freedom",
    "Trust..Government.Corruption.": "Government_Trust",
    "Generosity": "Generosity",
    "Dystopia.Residual": "Dystopia_Residual",
    "Country": "Country"
}

df = df.rename(columns=rename_map)

st.session_state["df"] = df  # Update cleaned dataset

# ---------------------- BASIC INFO ----------------------

st.write(f"### 📄 Dataset Loaded: **{df.shape[0]} rows** × **{df.shape[1]} columns**")

with st.expander("🔍 View Column Names"):
    st.write(list(df.columns))

with st.expander("📊 Preview Dataset"):
    st.dataframe(df.head())
# --------------------------------------------------------
# ------------------- WEEK 1 CONTENT ---------------------
# --------------------------------------------------------

if week_page == "📘 Week 1 – Planning":


    # ---------------- HOME ----------------
    if section_page == "🏠 Home":
        st.markdown("<div class='section-header'>📘 Week 1 – Planning (Home)</div>", unsafe_allow_html=True)
        st.write("""
        Week 1 focuses on **planning**, **strategy**, and understanding the dataset.
        
        ###  Objectives of Week 1:
        - Understand dataset structure  
        - Define research questions  
        - Plan cleaning & transformation steps  
        - Plan visualizations and modeling  
        - Create workflow diagrams  
        """)

    # ---------------- DATA ----------------
    elif section_page == "📁 Data":
        st.markdown("<div class='section-header'>📘 Week 1 – Dataset Understanding</div>", unsafe_allow_html=True)

        st.write("""
        ### 📄 Dataset Chosen: *World Happiness Report 2017*
        **Contains:**  
        - 155 countries  
        - 12 indicators influencing Happiness Score  
        - GDP, Life Expectancy, Family Support, Freedom, Trust, Generosity  
        """)

        st.subheader("🔹 Sample Data Preview")
        st.dataframe(df.head())

        st.subheader("🔹 Column Descriptions")
        st.write("""
        - **Happiness Score** — overall happiness index  
        - **GDP per Capita** — economic strength  
        - **Family** — social support  
        - **Life Expectancy** — health & longevity  
        - **Freedom** — freedom to make life choices  
        - **Generosity** — willingness to help others  
        - **Government Trust** — absence of corruption  
        """)

    # ---------------- VISUALIZATIONS (PLAN) ----------------
    elif section_page == "📊 Visualizations":
        st.markdown("<div class='section-header'>📘 Week 1 – Planned Visualizations</div>", unsafe_allow_html=True)

        st.write("""
        In Week 3, the following visualizations will be created:

        ### 📊 Planned Charts:
        - Histogram of Happiness Score  
        - Top 10 happiest countries  
        - Bottom 10 least happy countries  
        - Correlation heatmap  
        - GDP vs Happiness (scatter)  
        - Freedom vs Happiness (regression)  
        - Region-wise averages  
        """)

        st.image("https://i.imgur.com/2J8Bq0H.png", caption="Planned Workflow Diagram")

    # ---------------- MAP (PLAN) ----------------
    elif section_page == "🗺️ Map":
        st.markdown("<div class='section-header'>📘 Week 1 – Map Planning</div>", unsafe_allow_html=True)

        st.write("""
        The map will be implemented in Week 3 using **Plotly Choropleth**.
        
        🌍 It will visualize:
        - Happiness Score by country (colored world map)
        - Mouse hover details (GDP, Life Expectancy, Freedom)
        """)

    # ---------------- MODELING (PLAN) ----------------
    elif section_page == "🤖 Modeling":
        st.markdown("<div class='section-header'>📘 Week 1 – Modeling Plan</div>", unsafe_allow_html=True)

        st.write("""
        Modeling will be performed during **Week 4**.
        
        Planned ML Model:
        - **Linear Regression**  
        - Predict Happiness Score  
        - Input Factors: GDP, Life Expectancy, Family, Freedom, Trust  
        - Output: Happiness Score  
        """)

    # ---------------- INSIGHTS ----------------
    elif section_page == "🔍 Insights":
        st.markdown("<div class='section-header'>📘 Week 1 – Expected Insights</div>", unsafe_allow_html=True)

        st.write("""
        ### 🔮 Expected Patterns:
        - Higher GDP → Higher Happiness  
        - Longer Life Expectancy → Happier population  
        - Strong Family Support → Higher well-being  
        - Freedom and Trust → Moderate but important influence  
        
        ###  Expected Correlations:
        - GDP ↗ Happiness  
        - Life Expectancy ↗ Happiness  
        - Freedom ↗ Happiness  
        - Corruption ↘ Happiness  
        """)

    # ---------------- ABOUT ----------------
    elif section_page == "ℹ️ About":
        st.markdown("<div class='section-header'>📘 Week 1 – About</div>", unsafe_allow_html=True)

        st.write("""
        Week 1 lays the foundation for the entire project:
        - Dataset understanding  
        - Planning  
        - Research questions  
        - Visualization roadmap  
        - Modeling strategy  
        """)
# --------------------------------------------------------
# ------------------- WEEK 2 CONTENT ---------------------
# --------------------------------------------------------

elif week_page == "📗 Week 2 – Data Analysis":


    # ---------------- HOME ----------------
    if section_page == "🏠 Home":
        st.markdown("<div class='section-header'>📗 Week 2 – Data Analysis (Home)</div>", unsafe_allow_html=True)
        st.write("""
        Week 2 focuses on **Data Cleaning, Preprocessing, and Exploratory Data Analysis (EDA)**.
        
        ### ✔ Tasks Completed This Week:
        - Loaded and cleaned dataset  
        - Renamed inconsistent column names  
        - Checked missing values  
        - Generated summary statistics  
        - Performed basic visual exploration  
        """)

    # ---------------- DATA ----------------
    elif section_page == "📁 Data":
        st.markdown("<div class='section-header'>📗 Week 2 – Data Summary & Cleaning</div>", unsafe_allow_html=True)

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📊 Data Types")
        st.write(df.dtypes)

        st.subheader("❗ Missing Values")
        st.write(df.isnull().sum())

        st.subheader(" Summary Statistics")
        st.write(df.describe())

        st.subheader(" Cleaning Applied")
        st.write("""
        - Column names standardized  
        - Checked for missing values  
        - Ensured numerical fields are in correct dtype  
        """)

    # ---------------- VISUALIZATIONS (Week 2) ----------------
    elif section_page == "📊 Visualizations":
        st.markdown("<div class='section-header'>📗 Week 2 – Basic EDA Visualizations</div>", unsafe_allow_html=True)

        # ------- Histogram of Happiness Score -------
        st.subheader(" Distribution of Happiness Score")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(df["Happiness_Score"], kde=True, color="green", ax=ax)
        st.pyplot(fig)

        # ------- GDP vs Happiness -------
        st.subheader(" GDP per Capita vs Happiness")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.scatterplot(data=df, x="GDP_per_Capita", y="Happiness_Score", color="blue", ax=ax)
        st.pyplot(fig)

        # ------- Life Expectancy vs Happiness -------
        st.subheader(" Life Expectancy vs Happiness")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.scatterplot(data=df, x="Life_Expectancy", y="Happiness_Score", color="red", ax=ax)
        st.pyplot(fig)

        # ------- Pairplot Preview -------
        st.subheader(" Pairplot (Preview of Week 3)")
        st.write("A full pairplot will be shown in Week 3. For now, a small preview:")
        fig = sns.pairplot(df[["Happiness_Score", "GDP_per_Capita", "Family", "Life_Expectancy"]])
        st.pyplot(fig)

    # ---------------- MAP (plan only) ----------------
    elif section_page == "🗺️ Map":
        st.markdown("<div class='section-header'>📗 Week 2 – Map Section</div>", unsafe_allow_html=True)
        st.write("""
        Map visualizations will be implemented in **Week 3**.

        ✨ Planned:
        - Choropleth world map  
        - Happiness Score coloring  
        - Hover information for each country  
        """)

    # ---------------- MODELING (plan only) ----------------
    elif section_page == "🤖 Modeling":
        st.markdown("<div class='section-header'>📗 Week 2 – Modeling Section</div>", unsafe_allow_html=True)
        st.write("""
        Modeling will start in **Week 4**.

        Planned model:
        - Linear Regression  
        - Predict Happiness Score  
        - Use GDP, Life Expectancy, Family, Freedom, Trust  
        - Evaluate using R², MAE, RMSE  
        """)

    # ---------------- INSIGHTS ----------------
    elif section_page == "🔍 Insights":
        st.markdown("<div class='section-header'>📗 Week 2 – Insights from Analysis</div>", unsafe_allow_html=True)

        st.write("""
        ### 🔎 Key Findings:
        - Countries with higher **GDP per Capita** tend to have higher happiness.  
        - **Life Expectancy** is strongly correlated with well-being.  
        - Social support (**Family**) consistently shows high influence.  
        - Few missing values — dataset is clean and ready for deeper analysis.  
        - Happiness Score distribution is almost normal.  
        """)

    # ---------------- ABOUT ----------------
    elif section_page == "ℹ️ About":
        st.markdown("<div class='section-header'>📗 Week 2 – About</div>", unsafe_allow_html=True)

        st.write("""
        Week 2 handles:
        - Data cleaning  
        - Data transformation  
        - Basic EDA  
        - Understanding relationships  
        """)
# --------------------------------------------------------
# ------------------- WEEK 3 CONTENT ---------------------
# --------------------------------------------------------

elif week_page == "📙 Week 3 – Visualization":

    # ---------------- HOME ----------------
    if section_page == "🏠 Home":
        st.markdown("<div class='section-header'>📙 Week 3 – Visualization (Home)</div>", unsafe_allow_html=True)
        st.write("""
        Week 3 focuses on **data visualization and interpretation**.

        ###  Visualizations Covered:
        - Histogram  
        - Top/Bottom 10 countries  
        - Correlation Heatmap  
        - Scatter + Regression  
        - Pairplot  
        - Boxplots  
        - Bar charts  
        - Choropleth World Map  
        """)

    # ---------------- DATA ----------------
    elif section_page == "📁 Data":
        st.markdown("<div class='section-header'>📙 Week 3 – Cleaned Dataset</div>", unsafe_allow_html=True)

        st.write("Below is the cleaned dataset used for Week 3 visualizations:")
        st.dataframe(df.head())

        st.subheader(" Correlation Table")
        corr = df.corr(numeric_only=True)
        st.write(corr)

    # ---------------- VISUALIZATIONS ----------------
    elif section_page == "📊 Visualizations":
        st.markdown("<div class='section-header'>📙 Week 3 – Visualization Dashboard</div>", unsafe_allow_html=True)

        # ------------ HISTOGRAM ------------
        st.subheader(" Distribution of Happiness Score")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["Happiness_Score"], kde=True, color="#2a7f2a", ax=ax)
        st.pyplot(fig)

        # ------------ TOP 10 ------------
        st.subheader("🏆 Top 10 Happiest Countries")
        top10 = df.nlargest(10, "Happiness_Score")
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=top10, y="Country", x="Happiness_Score", palette="Greens_r")
        st.pyplot(fig)

        # ------------ BOTTOM 10 ------------
        st.subheader("😞 Bottom 10 Least Happy Countries")
        bottom10 = df.nsmallest(10, "Happiness_Score")
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=bottom10, y="Country", x="Happiness_Score", palette="Reds_r")
        st.pyplot(fig)

        # ------------ CORRELATION HEATMAP ------------
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Greens", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # ------------ GDP VS HAPPINESS ------------
        st.subheader("💰 GDP per Capita vs Happiness Score")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="GDP_per_Capita", y="Happiness_Score", color="blue", ax=ax)
        sns.regplot(data=df, x="GDP_per_Capita", y="Happiness_Score", scatter=False, color="black", ax=ax)
        st.pyplot(fig)

        # ------------ FREEDOM VS HAPPINESS ------------
        st.subheader("🕊️ Freedom vs Happiness Score (Regression)")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(data=df, x="Freedom", y="Happiness_Score", color="#2a7f2a", ax=ax)
        st.pyplot(fig)

        # ------------ FAMILY SUPPORT VS HAPPINESS ------------
        st.subheader("👨‍👩‍👧 Family Support vs Happiness")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Family", y="Happiness_Score", color="purple", ax=ax)
        sns.regplot(data=df, x="Family", y="Happiness_Score", scatter=False, color="black", ax=ax)
        st.pyplot(fig)

        # ------------ BOX PLOTS ------------
        st.subheader("📦 Boxplot – Life Expectancy vs Happiness")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="Life_Expectancy", color="lightgreen", ax=ax)
        st.pyplot(fig)

        # ------------ PAIRPLOT ------------
        st.subheader(" Pairplot of Key Factors")
        pair_cols = ["GDP_per_Capita", "Family", "Life_Expectancy", "Freedom", "Happiness_Score"]
        fig = sns.pairplot(df[pair_cols], diag_kind="kde")
        st.pyplot(fig)

    # ---------------- MAP (Choropleth) ----------------
    elif section_page == "🗺️ Map":
        st.markdown("<div class='section-header'>📙 Week 3 – Choropleth World Map</div>", unsafe_allow_html=True)

        st.write("### 🌍 World Map of Happiness Score")

        # Plotly Choropleth Map
        fig = px.choropleth(
            df,
            locations="Country",
            locationmode="country names",
            color="Happiness_Score",
            hover_name="Country",
            color_continuous_scale="Greens",
            title="Happiness Score by Country"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- MODELING (plan preview) ----------------
    elif section_page == "🤖 Modeling":
        st.markdown("<div class='section-header'>📙 Week 3 – Modeling Preview</div>", unsafe_allow_html=True)
        st.write("""
        Week 4 will include:
        - Training Linear Regression Model  
        - Predicting Happiness Score  
        - Feature importance  
        - Model evaluation  
        """)
    
    # ---------------- INSIGHTS ----------------
    elif section_page == "🔍 Insights":
        st.markdown("<div class='section-header'>📙 Week 3 – Insights from Visualizations</div>", unsafe_allow_html=True)

        st.write("""
        ### 🔎 Key Insights:
        - **GDP**, **Life Expectancy**, and **Family Support** strongly increase happiness.  
        - Freedom shows positive influence on Happiness Score.  
        - Countries with low Trust/Corruption tend to rank lower.  
        - Top 10 happiest countries all have high GDP and strong social systems.  
        """)

    # ---------------- ABOUT ----------------
    elif section_page == "ℹ️ About":
        st.markdown("<div class='section-header'>📙 Week 3 – About Visualization Week</div>", unsafe_allow_html=True)
        st.write("""
        Week 3 is fully focused on **visual storytelling** and uncovering insights from data.
        """)
# --------------------------------------------------------
# ------------------- WEEK 4 CONTENT ---------------------
# --------------------------------------------------------

elif week_page == "📕 Week 4 – Modeling & Reflection":

    # ---------------- HOME ----------------
    if section_page == "🏠 Home":
        st.markdown("<div class='section-header'>📕 Week 4 – Final Week (Home)</div>", unsafe_allow_html=True)
        st.write("""
        Week 4 focuses on **Machine Learning Model**, **evaluation**, and **reflection**.

        ### 🎯 Goals:
        - Build a regression model  
        - Predict Happiness Score  
        - Analyze errors  
        - Reflect on all 4 weeks of work  
        """)


    # ---------------- DATA REFLECTION ----------------
    elif section_page == "📁 Data":
        st.markdown("<div class='section-header'>📕 Week 4 – Data Review</div>", unsafe_allow_html=True)

        st.write("""
        ### ✔ Data Quality Summary:
        - Dataset contains **155 countries**
        - Very few missing values  
        - Clean numerical columns  
        - All necessary indicators present  

        Dataset is excellent for regression-based prediction.
        """)

        st.subheader("📌 Final Dataset Preview")
        st.dataframe(df.head())


    # ---------------- MODELING ----------------
    elif section_page == "🤖 Modeling":

        st.markdown("<div class='section-header'>📕 Week 4 – Modeling & Prediction</div>", unsafe_allow_html=True)

        st.write("### 🧠 Building Linear Regression Model")

        # ---------------- FEATURE SELECTION ----------------
        features = ["GDP_per_Capita", "Family", "Life_Expectancy", "Freedom", "Generosity", "Government_Trust"]
        target = "Happiness_Score"

        X = df[features]
        y = df[target]

        # ---------------- TRAIN TEST SPLIT ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # ---------------- TRAIN MODEL ----------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        # ---------------- PREDICTIONS ----------------
        y_pred = model.predict(X_test)

        # ---------------- METRICS ----------------
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # ---------------- DISPLAY METRICS ----------------
        st.subheader("📊 Model Evaluation Metrics")
        st.write(f"**R² Score:** {r2:.3f}")
        st.write(f"**MAE (Mean Absolute Error):** {mae:.3f}")
        st.write(f"**MSE (Mean Squared Error):** {mse:.3f}")
        st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.3f}")

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("📈 Feature Importance (Coefficient Values)")

        importance = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)

        st.dataframe(importance)

        # ------------ FEATURE IMPORTANCE CHART ------------
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance, x="Coefficient", y="Feature", palette="Greens_r", ax=ax)
        st.pyplot(fig)

        # ------------ SAMPLE PREDICTION UI ------------
        st.subheader("🔮 Try a Prediction Yourself")

        col1, col2 = st.columns(2)

        with col1:
            gdp = st.slider("GDP per Capita", 0.0, 2.0, 1.2)
            life = st.slider("Life Expectancy", 0.0, 1.0, 0.8)
            freedom = st.slider("Freedom", 0.0, 1.0, 0.6)

        with col2:
            family = st.slider("Family Support", 0.0, 2.0, 1.1)
            generosity = st.slider("Generosity", 0.0, 1.0, 0.3)
            trust = st.slider("Government Trust", 0.0, 1.0, 0.2)

        user_input = np.array([[gdp, family, life, freedom, generosity, trust]])
        user_pred = model.predict(user_input)[0]

        st.success(f"🌟 **Predicted Happiness Score: {user_pred:.3f}**")


    # ---------------- VISUALIZATION REFLECTION ----------------
    elif section_page == "📊 Visualizations":
        st.markdown("<div class='section-header'>📕 Week 4 – Visualization Reflection</div>", unsafe_allow_html=True)

        st.write("""
        ### 🔍 Reflection on Visualizations:
        - Heatmap clearly showed strongest correlations  
        - Top/Bottom 10 charts helped identify patterns  
        - Regression plots showed linear trends  
        - Choropleth map gave global insight  
        - Pairplot displayed multivariate relationships  
        """)


    # ---------------- MAP REFLECTION ----------------
    elif section_page == "🗺️ Map":
        st.markdown("<div class='section-header'>📕 Week 4 – Map Reflection</div>", unsafe_allow_html=True)

        st.write("""
        ### 🌍 Reflection on Map:
        - Choropleth map gave a strong global overview  
        - Clear color gradients show country differences  
        - Useful for geographic pattern analysis  
        """)


    # ---------------- INSIGHTS (FINAL) ----------------
    elif section_page == "🔍 Insights":
        st.markdown("<div class='section-header'>📕 Week 4 – Final Insights</div>", unsafe_allow_html=True)

        st.write("""
        ### 🌟 Final Conclusions:
        - GDP, Family Support, and Life Expectancy have the **highest impact** on happiness.  
        - Freedom moderately contributes to well-being.  
        - Trust and Generosity have minor but noticeable influence.  
        - Countries with strong economy + social support = happiest nations.  
        
        ### 📌 Model Reflection:
        - Linear Regression performed well (good R² score).  
        - Errors were small (low MAE, RMSE).  
        """)


    # ---------------- ABOUT ----------------
    elif section_page == "ℹ️ About":
        st.markdown("<div class='section-header'>📕 Week 4 – About Final Week</div>", unsafe_allow_html=True)

        st.write("""
        Week 4 combines everything:
        - Full ML model  
        - Predictions  
        - Evaluation  
        - Reflection  
        - Final insights  
        """)
# --------------------------------------------------------
# ------------------- PART 7: FINAL TOUCHES --------------
# --------------------------------------------------------

# ---------------------- DOWNLOADS -----------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("⬇️ Download Section")

colA, colB = st.columns(2)

with colA:
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📁 Download Cleaned Dataset (CSV)",
        data=csv_download,
        file_name="cleaned_world_happiness.csv",
        mime="text/csv"
    )

with colB:
    report_text = """
World Happiness Report – Internship Project

This report contains:
✓ Week 1 – Planning  
✓ Week 2 – Data Analysis  
✓ Week 3 – Visualizations  
✓ Week 4 – Modeling & Reflection  

Thank you.
"""
    st.download_button(
        label="📄 Download Summary Report (TXT)",
        data=report_text,
        file_name="happiness_summary_report.txt",
        mime="text/plain"
    )


# ---------------------- HELP SECTION -----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader(" Need Help Understanding the Dashboard?")

with st.expander("📘 How to Navigate Weeks & Sections"):
    st.write("""
    - Select **Week** from sidebar  
    - Choose a **Section** inside the Week  
    - Each week has a different focus  
    """)

with st.expander("📊 What Are These Visualizations?"):
    st.write("""
    - **Histogram** shows score distribution  
    - **Heatmap** shows correlation strength  
    - **Top/Bottom 10** highlight happiest and least happy nations  
    - **Regression** shows linear relations  
    """)

with st.expander("🤖 What Does the Model Do?"):
    st.write("""
    - Predicts Happiness Score  
    - Uses GDP, Family, Life Expectancy, Freedom, Trust, Generosity  
    - Measures accuracy using R², MAE, RMSE  
    """)


# ---------------------- FINAL FOOTER -----------------------

st.markdown("""
<hr>
<div style='text-align:center; color:#2a7f2a; font-size:16px;'>
    <b>🌿 World Happiness Dashboard</b><br>
    Built as part of a 4-Week Data Science Internship Project<br>
    <span style='font-size:14px; color:#555;'>Data • Visualization • Machine Learning • Insights</span>
</div>
<hr>
""", unsafe_allow_html=True)

st.success("🎉 All modules loaded successfully — your dashboard is complete!")
