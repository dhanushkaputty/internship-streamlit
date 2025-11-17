import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Internship Data Analytics – Streamlit Dashboard", layout="wide")
sns.set(style="whitegrid")

st.title(" Internship Data Analytics Dashboard – World Happiness Report")
st.write("#### Developed by: Sweety ")

page = st.sidebar.selectbox(
    "Navigate",
    ("Home", "Week 1 – Planning", "Week 2 – Data Analysis", "Week 3 – Visualization", "Week 4 – Reflection")
)

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.header(" Welcome to Your Internship Analytics Project")
    st.write("""
    This Streamlit Web App demonstrates the entire 4-week internship project:
    
    **✔ Week 1 – Strategy Planning**  
    **✔ Week 2 – Data Cleaning & Analysis**  
    **✔ Week 3 – Visualization & Interpretation**  
    **✔ Week 4 – Evaluation & Reflection**
    
    Please upload your dataset below to continue.
    """)

    uploaded_file = st.file_uploader("Upload the World Happiness CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Rename columns immediately so all pages use correct column names
        df = df.rename(columns={
            'Happiness.Rank': 'Happiness_Rank',
            'Happiness.Score': 'Happiness_Score',
            'Economy..GDP.per.Capita.': 'GDP_per_Capita',
            'Health..Life.Expectancy.': 'Life_Expectancy',
            'Trust..Government.Corruption.': 'Government_Trust'
        })

        st.session_state["df"] = df
        st.success("Dataset uploaded successfully! Go to Week 2 or Week 3 to view analysis.")

# ---------------- WEEK 1 ----------------
if page == "Week 1 – Planning":
    st.header(" Week 1 – Strategic Data Exploration Planning")
    
    st.subheader("Dataset Chosen")
    st.write("""
    **World Happiness Report 2017**  
    Source: Kaggle / Public Dataset  
    Contains **155 countries** with **12 happiness indicators**.
    """)

    st.subheader("Research Questions")
    st.write("""
    1️ What factors influence the Happiness Score of a country?  
    2️ How strongly do GDP, Life Expectancy, Freedom, and Family affect happiness?  
    3️ Which countries rank highest and lowest?  
    4️ Are social factors or economic factors more important?  
    """)

    st.subheader("Planned Methodology")
    st.write("""
    ✔ Data Cleaning – handle missing values, rename columns  
    ✔ Exploratory Data Analysis – summary statistics  
    ✔ Visualization – heatmaps, bar plots, scatterplots  
    ✔ Insight Extraction – correlation, top/bottom ranking  
    """)

    st.subheader("Flowchart for Week 1 Plan")
    st.image("https://i.imgur.com/2J8Bq0H.png")

# ---------------- WEEK 2 ----------------
if page == "Week 2 – Data Analysis":
    st.header(" Week 2 – Execution of Data Analysis")
    
    if "df" not in st.session_state:
        st.warning("Please upload dataset from Home page first.")
    else:
        df = st.session_state["df"]

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.success("Column names cleaned and dataset ready for analysis.")

# ---------------- WEEK 3 ----------------
if page == "Week 3 – Visualization":
    st.header(" Week 3 – Data Visualization & Interpretation")
    
    if "df" not in st.session_state:
        st.warning("Upload dataset first from Home page.")
    else:
        df = st.session_state["df"]

        vis = st.selectbox("Choose visualization:", [
            "Histogram – Happiness Score",
            "Top 10 Happiest Countries",
            "Bottom 10 Countries",
            "Correlation Heatmap",
            "GDP vs Happiness",
            "Freedom vs Happiness Regression"
        ])

        # ------- HISTOGRAM -------
        if vis == "Histogram – Happiness Score":
            st.subheader("🔹 Distribution of Happiness Scores")
            fig = plt.figure(figsize=(8,5))
            sns.histplot(df["Happiness_Score"], kde=True, color="skyblue")
            st.pyplot(fig)
            st.write("**Interpretation:** Most countries have moderate happiness scores between 4 and 6.")

        # ------- TOP 10 -------
        if vis == "Top 10 Happiest Countries":
            st.subheader(" Top 10 Happiest Countries")
            top10 = df.nlargest(10, "Happiness_Score")
            fig = plt.figure(figsize=(10,6))
            sns.barplot(data=top10, x="Happiness_Score", y="Country", palette="crest")
            st.pyplot(fig)
            st.write("**These countries have strong GDP, healthcare, and freedom indicators.**")

        # ------- BOTTOM 10 -------
        if vis == "Bottom 10 Countries":
            st.subheader(" Bottom 10 Least Happy Countries")
            bottom10 = df.nsmallest(10, "Happiness_Score")
            fig = plt.figure(figsize=(10,6))
            sns.barplot(data=bottom10, x="Happiness_Score", y="Country", palette="coolwarm")
            st.pyplot(fig)
            st.write("**These countries often face conflict, poverty, or lack of social support.**")

        # ------- HEATMAP -------
        if vis == "Correlation Heatmap":
            st.subheader(" Correlation Heatmap of All Factors")
            fig = plt.figure(figsize=(12,7))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="magma")
            st.pyplot(fig)
            st.write("**GDP, Life Expectancy, and Family have the strongest positive correlation with Happiness Score.**")

        # ------- GDP vs Happiness -------
        if vis == "GDP vs Happiness":
            st.subheader(" GDP per Capita vs Happiness Score")
            fig = plt.figure(figsize=(8,5))
            sns.scatterplot(data=df, x="GDP_per_Capita", y="Happiness_Score")
            st.pyplot(fig)
            st.write("**Countries with high GDP generally report higher happiness.**")

        # ------- Freedom vs Happiness -------
        if vis == "Freedom vs Happiness Regression":
            st.subheader("🕊 Freedom vs Happiness Score")
            fig = plt.figure(figsize=(8,5))
            sns.regplot(data=df, x="Freedom", y="Happiness_Score", color="green")
            st.pyplot(fig)
            st.write("**Freedom is an important social factor contributing to national happiness.**")

# ---------------- WEEK 4 ----------------
if page == "Week 4 – Reflection":
    st.header(" Week 4 – Comprehensive Evaluation & Reflection")
    
    st.write("""
    ✔ This dashboard summarizes all 4 weeks of the internship  
    ✔ Demonstrates planning → analysis → visualization → insights  
    ✔ Shows clear learning, improvement and understanding  
    """)

    st.subheader("Reflection Summary")
    st.write("""
    - Improved understanding of Python, Pandas, Seaborn and Streamlit  
    - Learned how economic and social factors impact national happiness  
    - Week 3 visualizations helped identify strong correlations  
    - Streamlit made the project interactive and professional  
    - Challenges included choosing correct features and fixing column names  
    - Successfully completed full end-to-end data analysis workflow  
    """)

    st.success("Your internship project is now fully demonstrated in Streamlit!")

# -------- FOOTER --------
st.markdown("---")
st.write("Developed by **Dhanushka** | Internship 2025 | Data Science with Python ")
