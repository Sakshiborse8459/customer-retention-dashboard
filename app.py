import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Bank Retention Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("../data/customer_churn.csv")
    return df

df = load_data()

# -----------------------------
# DATA CLEANING
# -----------------------------

df.drop_duplicates(inplace=True)

binary_cols = ["HasCrCard", "IsActiveMember", "Exited"]

for col in binary_cols:
    df[col] = df[col].astype(int)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

def engagement_status(row):

    if row["IsActiveMember"] == 1 and row["NumOfProducts"] >= 2:
        return "Active Engaged Customer"

    elif row["IsActiveMember"] == 0 and row["Balance"] > 100000:
        return "Inactive High Balance Customer"

    elif row["IsActiveMember"] == 1 and row["NumOfProducts"] < 2:
        return "Active Low Product Customer"

    else:
        return "Inactive Disengaged Customer"


df["Engagement_Status"] = df.apply(engagement_status, axis=1)

# Product Category

def product_category(x):

    if x == 1:
        return "Single Product"

    elif x == 2:
        return "Multi Product"

    else:
        return "High Product Usage"


df["Product_Depth_Category"] = df["NumOfProducts"].apply(product_category)

# Relationship Strength

df["Relationship_Strength_Index"] = (
    df["IsActiveMember"] * 30 +
    df["NumOfProducts"] * 20 +
    df["HasCrCard"] * 20 +
    df["Tenure"] * 3
)

scaler = MinMaxScaler(feature_range=(0,100))

df["Relationship_Strength_Index"] = scaler.fit_transform(
    df[["Relationship_Strength_Index"]]
)

# Premium At Risk

df["Premium_At_Risk_Flag"] = np.where(
    (df["Balance"] > 120000) &
    (df["EstimatedSalary"] > 100000) &
    (df["IsActiveMember"] == 0) &
    (df["Exited"] == 1),
    1,
    0
)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.title("Filters")

geo = st.sidebar.multiselect(
    "Select Geography",
    df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[
    (df["Geography"].isin(geo)) &
    (df["Gender"].isin(gender))
]

# -----------------------------
# KPI SECTION
# -----------------------------

total_customers = filtered_df.shape[0]

churned_customers = filtered_df["Exited"].sum()

churn_rate = round(
    (churned_customers / total_customers) * 100,
    2
)

active_ratio = round(
    (filtered_df["IsActiveMember"].mean()) * 100,
    2
)

avg_relationship = round(
    filtered_df["Relationship_Strength_Index"].mean(),
    2
)

high_balance_disengaged = round(
    (
        filtered_df[
            (filtered_df["Balance"] > 100000) &
            (filtered_df["IsActiveMember"] == 0)
        ].shape[0]
        /
        total_customers
    ) * 100,
    2
)

# -----------------------------
# TITLE
# -----------------------------

st.title("Customer Engagement & Product Utilization Analytics")

# -----------------------------
# KPI CARDS
# -----------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Customers", total_customers)
c2.metric("Churned Customers", churned_customers)
c3.metric("Churn Rate %", f"{churn_rate}%")
c4.metric("Active Member Ratio", f"{active_ratio}%")
c5.metric("Avg Relationship Score", avg_relationship)

# -----------------------------
# PAGE 1
# -----------------------------

st.header("Executive Overview")

col1, col2 = st.columns(2)

geo_churn = (
    filtered_df.groupby("Geography")["Exited"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    geo_churn,
    x="Geography",
    y="Exited",
    title="Geography Wise Churn"
)

col1.plotly_chart(fig1, use_container_width=True)

gender_churn = (
    filtered_df.groupby("Gender")["Exited"]
    .sum()
    .reset_index()
)

fig2 = px.pie(
    gender_churn,
    names="Gender",
    values="Exited",
    hole=0.5,
    title="Gender Wise Churn"
)

col2.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# PAGE 2
# -----------------------------

st.header("Engagement Analytics")

engagement = (
    filtered_df.groupby("Engagement_Status")["CustomerId"]
    .count()
    .reset_index()
)

fig3 = px.bar(
    engagement,
    x="Engagement_Status",
    y="CustomerId",
    color="Engagement_Status",
    title="Engagement Profile Distribution"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# PAGE 3
# -----------------------------

st.header("Product Utilization Analysis")

product_churn = (
    filtered_df.groupby("Product_Depth_Category")["Exited"]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    product_churn,
    x="Product_Depth_Category",
    y="Exited",
    color="Product_Depth_Category",
    title="Product Usage vs Churn"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# PAGE 4
# -----------------------------

st.header("Financial Commitment Analysis")

fig5 = px.scatter(
    filtered_df,
    x="Balance",
    y="EstimatedSalary",
    color="Exited",
    title="Balance vs Salary vs Churn"
)

st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# PAGE 5
# -----------------------------

st.header("High Risk Customer Detector")

risk_df = filtered_df[
    filtered_df["Premium_At_Risk_Flag"] == 1
]

st.dataframe(risk_df)

csv = risk_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download High Risk Customers",
    csv,
    "high_risk_customers.csv",
    "text/csv"
)

# -----------------------------
# PAGE 6
# -----------------------------

st.header("Retention Strategy Recommendations")

st.markdown("""
### Key Insights

- Customers with low engagement show higher churn.
- Inactive high-balance customers are risky.
- Multi-product users retain better.
- Germany region shows elevated churn.

### Recommendations

- Launch loyalty programs for inactive customers.
- Encourage cross-selling of banking products.
- Offer premium support to high balance customers.
- Increase engagement campaigns.
""")

# -----------------------------
# MACHINE LEARNING
# -----------------------------

st.header("Churn Prediction Model")

features = [
    "CreditScore",
    "Age",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "EstimatedSalary"
]

X = filtered_df[features]
y = filtered_df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

st.subheader(f"Model Accuracy: {round(acc*100,2)}%")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

fig6 = px.bar(
    importance,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

st.plotly_chart(fig6, use_container_width=True)