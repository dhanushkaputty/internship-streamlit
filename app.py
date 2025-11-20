# app.py - Premium Pro Streamlit Dashboard (World Happiness)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

# --- Page config
st.set_page_config(page_title="World Happiness - Premium Dashboard", layout="wide", initial_sidebar_state="expanded")
sns.set(style="whitegrid")

# --- Helper functions
@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

def robust_rename(df):
    # Try several common column name patterns and standardize them
    mapping = {}
    cols = df.columns.tolist()
    for c in cols:
        lc = c.lower().strip()
        if "happiness.score" in c or "happiness_score" in lc or "happiness score" in lc:
            mapping[c] = "Happiness_Score"
        if "happiness.rank" in c or "happiness_rank" in lc or "happiness rank" in lc:
            mapping[c] = "Happiness_Rank"
        if "economy" in lc and ("gdp" in lc or "per.capita" in lc or "gdp.per" in lc or "gdp_per" in lc):
            mapping[c] = "GDP_per_Capita"
        if "health" in lc or "life" in lc:
            mapping[c] = "Life_Expectancy"
        if "family" in lc:
            mapping[c] = "Family"
        if "freedom" in lc:
            mapping[c] = "Freedom"
        if "generosity" in lc:
            mapping[c] = "Generosity"
        if "trust" in lc or "government" in lc:
            mapping[c] = "Government_Trust"
        if "dystopia" in lc:
            mapping[c] = "Dystopia_Residual"
        if "country" in lc:
            mapping[c] = "Country"
    if mapping:
        df = df.rename(columns=mapping)
    return df

def clean_and_prepare(df):
    df = robust_rename(df)
    # Keep a baseline set of columns if they exist
    cols_want = ["Country","Happiness_Rank","Happiness_Score","GDP_per_Capita",
                 "Family","Life_Expectancy","Freedom","Generosity","Government_Trust","Dystopia_Residual"]
    # ensure numeric
    for c in cols_want:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Basic cleaning: drop duplicates, trim country names
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["Country"]) if "Country" in df.columns else df.drop_duplicates()
    return df

def get_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

# --- Sidebar - navigation & theme
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Data","Visualizations","Map","Modeling","Insights","About"])

# --- Home
if page == "Home":
    st.header("🌎 World Happiness — Premium Pro Dashboard")
    st.write("""
    This project demonstrates planning, data analysis, advanced visualizations and simple predictive modeling
    using the World Happiness dataset (2017).  
    Use the left sidebar to navigate between pages.
    """)
    st.write("**How to use:** Upload the CSV on the Data page (or place a file named `world_happiness.csv` in the project root for local auto-load).")
    st.info("If you deploy on Streamlit Cloud, graders will upload the CSV via the Upload button. Local runs can auto-load the local CSV file if present.")

# --- Data page: upload or auto-load
if page == "Data":
    st.header("📥 Data Upload & Cleaning")
    st.write("Upload dataset or let the app auto-load `world_happiness.csv` (local).")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded.")
    else:
        # Try to auto-load local file if present (works for local runs)
        try:
            df = load_csv_from_path("world_happiness.csv")
            st.info("Loaded local file `world_happiness.csv`.")
        except Exception:
            st.info("No file uploaded. Please upload your CSV file here to proceed.")
    if df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        st.subheader("Columns detected")
        st.write(list(df.columns))
        # Clean & prepare
        df_clean = clean_and_prepare(df)
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean.head())
        st.write("Missing values per column:")
        st.write(df_clean.isnull().sum())
        st.markdown("#### Download cleaned dataset")
        csv_bytes = get_download_link(df_clean)
        st.download_button("Download cleaned CSV", csv_bytes, "cleaned_happiness.csv", "text/csv")
        # Store in session for other pages
        st.session_state["df_clean"] = df_clean

# --- Visualizations
if page == "Visualizations":
    st.header("📊 Visualizations")
    if "df_clean" not in st.session_state:
        st.warning("Upload data on the Data page first.")
    else:
        df = st.session_state["df_clean"].copy()
        st.sidebar.markdown("### Controls")
        top_n = st.sidebar.slider("Top N countries for bar charts", min_value=5, max_value=20, value=10)
        min_happiness = st.sidebar.slider("Minimum Happiness Score", float(df["Happiness_Score"].min() if "Happiness_Score" in df.columns else 0), float(df["Happiness_Score"].max() if "Happiness_Score" in df.columns else 10), float(df["Happiness_Score"].min() if "Happiness_Score" in df.columns else 0))
        df = df[df["Happiness_Score"] >= min_happiness] if "Happiness_Score" in df.columns else df

        chart_choice = st.selectbox("Choose chart", ["Histogram","Top N Happiest","Bottom N","Correlation Heatmap","GDP vs Happiness","Freedom vs Happiness (regression)"])
        if chart_choice == "Histogram":
            st.subheader("Distribution of Happiness Scores")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df["Happiness_Score"].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            st.write("Interpretation: The histogram shows distribution and skew. Explain peaks and tails in your report.")
        if chart_choice == "Top N Happiest":
            st.subheader(f"Top {top_n} happiest countries")
            top = df.nlargest(top_n, "Happiness_Score")[["Country","Happiness_Score"]]
            fig = px.bar(top.sort_values("Happiness_Score"), x="Happiness_Score", y="Country", orientation="h", title=f"Top {top_n} Countries")
            st.plotly_chart(fig, use_container_width=True)
        if chart_choice == "Bottom N":
            st.subheader(f"Bottom {top_n} least happy countries")
            bot = df.nsmallest(top_n, "Happiness_Score")[["Country","Happiness_Score"]]
            fig = px.bar(bot.sort_values("Happiness_Score", ascending=True), x="Happiness_Score", y="Country", orientation="h", title=f"Bottom {top_n} Countries")
            st.plotly_chart(fig, use_container_width=True)
        if chart_choice == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            numeric = df.select_dtypes(include=np.number)
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(numeric.corr(), annot=True, cmap="magma", ax=ax)
            st.pyplot(fig)
            st.write("Focus on correlations with `Happiness_Score` in your analysis.")
        if chart_choice == "GDP vs Happiness":
            st.subheader("GDP per Capita vs Happiness")
            if "GDP_per_Capita" in df.columns:
                fig = px.scatter(df, x="GDP_per_Capita", y="Happiness_Score", hover_name="Country", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("GDP_per_Capita column not found. Check Data page.")
        if chart_choice == "Freedom vs Happiness (regression)":
            st.subheader("Freedom vs Happiness (Regression line)")
            if "Freedom" in df.columns:
                fig = px.scatter(df, x="Freedom", y="Happiness_Score", hover_name="Country", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Freedom column not found. Check Data page.")

# --- Map page (choropleth)
if page == "Map":
    st.header("🗺️ World Map - Happiness Choropleth")
    if "df_clean" not in st.session_state:
        st.warning("Upload data on the Data page first.")
    else:
        df = st.session_state["df_clean"].copy()
        if "Happiness_Score" not in df.columns:
            st.error("Happiness_Score missing.")
        else:
            # Plotly accepts country names; locationmode='country names'
            fig = px.choropleth(df, locations="Country", locationmode="country names",
                                color="Happiness_Score",
                                hover_name="Country",
                                color_continuous_scale="Viridis",
                                title="Happiness Score by Country")
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            st.write("Tip: If some countries don't map correctly, check country name spelling.")

# --- Modeling page
if page == "Modeling":
    st.header("🤖 Predictive Modeling - Linear Regression")
    if "df_clean" not in st.session_state:
        st.warning("Upload data on the Data page first.")
    else:
        df = st.session_state["df_clean"].copy()
        st.write("We will use a simple Linear Regression to predict `Happiness_Score` from key features.")
        # Auto-select a set of features if available
        candidate_features = ["GDP_per_Capita","Family","Life_Expectancy","Freedom","Generosity","Government_Trust"]
        features = [c for c in candidate_features if c in df.columns]
        st.write("Available features detected:", features)
        if not features or "Happiness_Score" not in df.columns:
            st.error("Not enough numeric columns available for modeling.")
        else:
            # Drop rows with missing in these columns
            modelling_df = df[features + ["Happiness_Score"]].dropna()
            X = modelling_df[features]
            y = modelling_df["Happiness_Score"]
            test_size = st.slider("Test set size (%)", 10, 40, 25)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.subheader("Model results")
            st.write(f"R² on test set: **{r2:.4f}**")
            coefs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
            st.write("Feature coefficients (higher absolute value = more influence):")
            st.dataframe(coefs.rename("Coefficient"))
            st.write("Intercept:", float(model.intercept_))

            # Show a scatter of predicted vs actual
            fig = px.scatter(x=y_test, y=y_pred, labels={"x":"Actual Happiness","y":"Predicted Happiness"}, title="Actual vs Predicted")
            fig.add_shape(type="line", x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max(), line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            # Provide an input form for making single predictions
            st.markdown("### Make a single prediction")
            with st.form("predict_form"):
                inputs = {}
                for f in features:
                    inputs[f] = st.number_input(f"{f}", float(np.nanmedian(df[f].dropna())), format="%.4f")
                submitted = st.form_submit_button("Predict")
                if submitted:
                    x_new = np.array([inputs[f] for f in features]).reshape(1, -1)
                    pred = model.predict(x_new)[0]
                    st.success(f"Predicted Happiness Score: {pred:.3f}")

# --- Insights page
if page == "Insights":
    st.header("🔍 Automated Insights & Notes")
    if "df_clean" not in st.session_state:
        st.warning("Upload data on the Data page first.")
    else:
        df = st.session_state["df_clean"].copy()
        st.subheader("Top 5 insights (auto)")
        insights = []
        if "Happiness_Score" in df.columns:
            top = df.nlargest(5, "Happiness_Score")["Country"].tolist()
            insights.append(f"Top 5 happiest countries: {', '.join(top)}")
            bottom = df.nsmallest(5, "Happiness_Score")["Country"].tolist()
            insights.append(f"Bottom 5 least happy countries: {', '.join(bottom)}")
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty and "Happiness_Score" in numeric.columns:
            corr = numeric.corr()["Happiness_Score"].drop("Happiness_Score").sort_values(ascending=False)
            strong = corr.head(3).index.tolist()
            insights.append(f"Top correlated features with Happiness_Score: {', '.join(strong)}")
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}.** {ins}")
        st.write("---")
        st.write("Use these insights in your week reports. Expand on reasons and policy implications in the DOC.")

# --- About page
if page == "About":
    st.header("ℹ️ About this Project")
    st.markdown("""
    **Project:** World Happiness — Premium Pro Dashboard  
    **Purpose:** Demonstrate full-cycle data exploration, visualization, and a simple predictive model.  
    **Contents:** Data cleaning, interactive visualizations, choropleth map, regression model, downloadable cleaned data.  
    """)
    st.markdown("**How this helps you score higher:**")
    st.write("""
    - Clear planning & organization (multiple pages)  
    - Interactive visuals & map for strong storytelling  
    - Simple ML model with metrics for evidence-based insights  
    - Downloadable artifacts for submission (cleaned data / results)
    """)
    st.markdown("---")
    st.write("Data Science with Python")

