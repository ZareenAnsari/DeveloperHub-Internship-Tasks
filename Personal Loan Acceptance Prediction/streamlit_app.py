import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")

st.title(" Bank Loan Marketing Analysis")
st.markdown("Analyze customer groups and identify who is likely to accept a personal loan.")

# 1. Load Data
@st.cache_data
def load_data():
    # Matches the file in your VS Code explorer
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    return df

try:
    df = load_data()
    
    # Sidebar: Navigation
    page = st.sidebar.selectbox("Choose a Page", ["Data Overview", "Customer Analysis", "Outlier Handling"])

    if page == "Data Overview":
        st.header("Dataset Overview")
        st.write(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
        with col2:
            acceptance_rate = (df['y'] == 'yes').mean() * 100
            st.metric("Loan Acceptance Rate", f"{acceptance_rate:.2f}%")

    elif page == "Customer Analysis":
        st.header(" Which groups accept the offer?")
        
        # Select category to analyze
        category = st.selectbox("Select Category", ["job", "education", "marital", "contact"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=category, hue='y', data=df, ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.info(f"Insight: This chart shows how '{category}' impacts the likelihood of a 'yes' (loan acceptance).")

    elif page == "Outlier Handling":
        st.header(" Outlier Detection & Handling")
        
        # Selecting numeric column
        num_col = st.selectbox("Select Numeric Feature", ["age", "duration", "campaign"])
        
        # Visualizing before cleaning
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=df[num_col], ax=ax[0], color='salmon')
        ax[0].set_title(f"Original {num_col}")
        
        # Handling Outliers (IQR Method)
        Q1 = df[num_col].quantile(0.25)
        Q3 = df[num_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Clipping (Capping)
        df_cleaned = df.copy()
        df_cleaned[num_col] = df_cleaned[num_col].clip(lower=lower, upper=upper)
        
        sns.boxplot(x=df_cleaned[num_col], ax=ax[1], color='lightblue')
        ax[1].set_title(f"Cleaned {num_col}")
        
        st.pyplot(fig)
        st.success(f"Applied IQR Capping. Values outside [{lower:.1f}, {upper:.1f}] were adjusted.")

except FileNotFoundError:
    st.error("File 'bank-additional-full.csv' not found. Please ensure it is in the same directory as this script.")