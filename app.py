import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Engagement & Product Utilization Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
/* Remove top black header */
header {
    visibility: hidden;
}

/* Remove Streamlit menu */
#MainMenu {
    visibility: hidden;
}

/* Remove footer */
footer {
    visibility: hidden;
}

/* Remove deploy button */
[data-testid="stToolbar"] {
    display: none;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #1e293b 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
    font-weight: 600;
}
/* Main background */
.stApp {
    background-color: #f5f7fb;
}

/* Tabs Styling */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #1f2937 !important;
    background-color: transparent;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ef4444 !important;
    border-bottom: 3px solid #ef4444;
}

/* Hover effect */
button[data-baseweb="tab"]:hover {
    color: #2563eb !important;
}
/* Plotly chart text fix */

</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f6f8fb 0%, #eef3f9 100%);
            color: #102a43;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0b1f33;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            color: #486581;
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }
        .kpi-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            min-height: 120px;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #486581;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #102a43;
            margin-bottom: 0.1rem;
        }
        .kpi-delta {
            font-size: 0.85rem;
            color: #627d98;
        }
        .section-note {
            background: rgba(255,255,255,0.75);
            border-left: 4px solid #2f80ed;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            color: #334e68;
            margin-top: 0.4rem;
            margin-bottom: 0.7rem;
        }
        .footer {
            text-align: center;
            color: #486581;
            padding: 1.2rem 0 0.3rem 0;
            font-size: 0.95rem;
        }
        /* Fix KPI metric values */
[data-testid="stMetricValue"] {
    color: #0b1f33 !important;
    font-weight: 700 !important;
}

/* Fix KPI labels */
[data-testid="stMetricLabel"] {
    color: #486581 !important;
    font-weight: 600 !important;
}
        /* Fix download button */
.stDownloadButton button {
    background-color: white !important;
    color: #102a43 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
}

/* Hover effect */
.stDownloadButton button:hover {
    background-color: #f3f4f6 !important;
    color: #0b1f33 !important;
}
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants and helpers
# -----------------------------
DATA_PATH = "enhanced_customer_analysis.csv"

REQUIRED_COLUMNS = [
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
    "EngagementSegment",
    "ProductCategory",
    "HighValueCustomer",
    "SalaryBalanceRatio",
    "StickyCustomer",
    "RelationshipScore",
    "RelationshipCategory",
    "RiskScore",
    "RiskCategory",
]

NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "SalaryBalanceRatio",
    "RelationshipScore",
    "RiskScore",
]

BINARY_COLUMNS = [
    "HasCrCard",
    "IsActiveMember",
    "Exited",
    "HighValueCustomer",
    "StickyCustomer",
]

COLOR_SEQ = ["#0F4C81", "#2F80ED", "#56CCF2", "#27AE60", "#F2994A", "#EB5757"]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and validate source data."""
    df = pd.read_csv("C:\Banking_Retention_project\data\enhanced_customer_analysis.csv")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    for col in NUMERIC_COLUMNS + BINARY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic type cleanup for categorical fields.
    categorical_cols = [
        "Geography",
        "Gender",
        "EngagementSegment",
        "ProductCategory",
        "RelationshipCategory",
        "RiskCategory",
        "Surname",
    ]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Normalize binary values for stable KPI calculations.
    for col in BINARY_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    # Create analysis labels used in charts.
    df["CustomerStatus"] = np.where(df["Exited"] == 1, "Churned", "Retained")
    df["MemberActivity"] = np.where(df["IsActiveMember"] == 1, "Active", "Inactive")
    df["CardStatus"] = np.where(df["HasCrCard"] == 1, "Card Holder", "No Card")
    df["ValueTier"] = np.where(df["HighValueCustomer"] == 1, "High Value", "Standard Value")
    df["StickinessFlag"] = np.where(df["StickyCustomer"] == 1, "Sticky", "Non-Sticky")
    df["ProductBand"] = np.where(df["NumOfProducts"] > 1, "Multi-Product", "Single Product")

    # Handle any remaining missing numeric values conservatively.
    numeric_fill_cols = NUMERIC_COLUMNS + ["Balance", "EstimatedSalary"]
    for col in set(numeric_fill_cols):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV for download."""
    return df.to_csv(index=False).encode("utf-8")


def safe_rate(series: pd.Series) -> float:
    """Return mean percentage safely."""
    if len(series) == 0:
        return 0.0
    return float(series.mean() * 100)


@st.cache_data(show_spinner=False)
def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create correlation matrix for numerical dashboard features."""
    cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "SalaryBalanceRatio",
        "RelationshipScore",
        "RiskScore",
        "HasCrCard",
        "IsActiveMember",
        "HighValueCustomer",
        "StickyCustomer",
        "Exited",
    ]
    available_cols = [col for col in cols if col in df.columns]
    return df[available_cols].corr(numeric_only=True).round(2)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all sidebar filters and search state."""
    filtered = df.copy()

    if st.session_state.geo_filter:
        filtered = filtered[filtered["Geography"].isin(st.session_state.geo_filter)]

    if st.session_state.gender_filter:
        filtered = filtered[filtered["Gender"].isin(st.session_state.gender_filter)]

    filtered = filtered[
        filtered["NumOfProducts"].between(
            st.session_state.product_range[0], st.session_state.product_range[1]
        )
    ]
    filtered = filtered[
        filtered["Balance"].between(
            st.session_state.balance_range[0], st.session_state.balance_range[1]
        )
    ]
    filtered = filtered[
        filtered["EstimatedSalary"].between(
            st.session_state.salary_range[0], st.session_state.salary_range[1]
        )
    ]

    if st.session_state.relationship_filter:
        filtered = filtered[
            filtered["RelationshipCategory"].isin(st.session_state.relationship_filter)
        ]

    if st.session_state.risk_filter:
        filtered = filtered[filtered["RiskCategory"].isin(st.session_state.risk_filter)]

    search_term = st.session_state.search_term.strip().lower()
    if search_term:
        search_mask = (
            filtered["CustomerId"].astype(str).str.lower().str.contains(search_term, na=False)
            | filtered["Surname"].astype(str).str.lower().str.contains(search_term, na=False)
            | filtered["Geography"].astype(str).str.lower().str.contains(search_term, na=False)
            | filtered["RelationshipCategory"].astype(str).str.lower().str.contains(search_term, na=False)
            | filtered["RiskCategory"].astype(str).str.lower().str.contains(search_term, na=False)
        )
        filtered = filtered[search_mask]

    return filtered


def render_kpi_card(title: str, value: str, delta_text: str) -> None:
    """Render a styled KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_no_data_message() -> None:
    """Standard no-data warning."""
    st.warning("No records match the current filters. Adjust the sidebar filters to continue.")


def build_sidebar(df: pd.DataFrame) -> None:
    """Create sidebar filters and search controls."""
    st.sidebar.markdown("## Dashboard Filters")
    st.sidebar.caption("Refine the portfolio view for customer retention and utilization analysis.")

    # Session state initialization for filter persistence.
    defaults = {
        "geo_filter": sorted(df["Geography"].dropna().unique().tolist()),
        "gender_filter": sorted(df["Gender"].dropna().unique().tolist()),
        "product_range": (
            int(df["NumOfProducts"].min()),
            int(df["NumOfProducts"].max()),
        ),
        "balance_range": (
            float(df["Balance"].min()),
            float(df["Balance"].max()),
        ),
        "salary_range": (
            float(df["EstimatedSalary"].min()),
            float(df["EstimatedSalary"].max()),
        ),
        "relationship_filter": sorted(df["RelationshipCategory"].dropna().unique().tolist()),
        "risk_filter": sorted(df["RiskCategory"].dropna().unique().tolist()),
        "search_term": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.sidebar.button("Reset Filters", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

    st.sidebar.multiselect(
        "Geography",
        options=sorted(df["Geography"].dropna().unique().tolist()),
        key="geo_filter",
    )
    st.sidebar.multiselect(
        "Gender",
        options=sorted(df["Gender"].dropna().unique().tolist()),
        key="gender_filter",
    )
    st.sidebar.slider(
        "Product Count",
        min_value=int(df["NumOfProducts"].min()),
        max_value=int(df["NumOfProducts"].max()),
        key="product_range",
    )
    st.sidebar.slider(
        "Balance Range",
        min_value=float(df["Balance"].min()),
        max_value=float(df["Balance"].max()),
        key="balance_range",
        format="%.0f",
    )
    st.sidebar.slider(
        "Salary Range",
        min_value=float(df["EstimatedSalary"].min()),
        max_value=float(df["EstimatedSalary"].max()),
        key="salary_range",
        format="%.0f",
    )
    st.sidebar.multiselect(
        "Relationship Category",
        options=sorted(df["RelationshipCategory"].dropna().unique().tolist()),
        key="relationship_filter",
    )
    st.sidebar.multiselect(
        "Risk Category",
        options=sorted(df["RiskCategory"].dropna().unique().tolist()),
        key="risk_filter",
    )

    st.sidebar.markdown("---")
    st.sidebar.text_input(
        "Search Customer",
        placeholder="Customer ID, surname, geography...",
        key="search_term",
    )


def chart_layout(fig, height: int = 420):
    """Apply a consistent professional visual theme to charts."""

    fig.update_layout(
        height=height,
        template="plotly_white",
        colorway=COLOR_SEQ,

        margin=dict(l=10, r=10, t=50, b=10),

        legend_title_text="",

        paper_bgcolor="white",
        plot_bgcolor="white",

        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=14,
            color="#102a43"
        ),

        title_font=dict(
            size=20,
            color="#0b1f33"
        ),

        xaxis=dict(
            title_font=dict(size=14, color="#102a43"),
            tickfont=dict(size=13, color="#102a43"),
            showgrid=False
        ),

        yaxis=dict(
            title_font=dict(size=14, color="#102a43"),
            tickfont=dict(size=13, color="#102a43"),
            gridcolor="rgba(15, 23, 42, 0.08)"
        ),

        legend=dict(
            font=dict(size=13, color="#102a43")
        )
    )

    return fig


def render_top_kpis(filtered_df: pd.DataFrame) -> None:
    """Render the top summary KPI band."""
    total_customers = len(filtered_df)
    churn_rate = safe_rate(filtered_df["Exited"]) if total_customers else 0.0
    active_rate = safe_rate(filtered_df["IsActiveMember"]) if total_customers else 0.0
    avg_balance = filtered_df["Balance"].mean() if total_customers else 0.0
    high_value_count = int(filtered_df["HighValueCustomer"].sum()) if total_customers else 0
    high_risk_count = len(filtered_df[filtered_df["RiskCategory"].str.contains("High", case=False, na=False)])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card("Total Customers", f"{total_customers:,}", "Filtered customer base")
    with c2:
        render_kpi_card("Churn Rate", f"{churn_rate:.1f}%", "Current portfolio churn")
    with c3:
        render_kpi_card("Active Customer %", f"{active_rate:.1f}%", "Member activity penetration")
    with c4:
        render_kpi_card("Average Balance", f"${avg_balance:,.0f}", "Average account balance")
    with c5:
        render_kpi_card("High Value Customers", f"{high_value_count:,}", "Premium relationship count")
    with c6:
        render_kpi_card("High Risk Customers", f"{high_risk_count:,}", "Priority retention watchlist")


def executive_overview_tab(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    """Executive overview section."""
    if filtered_df.empty:
        render_no_data_message()
        return

    st.markdown("### Executive Overview")
    st.markdown(
        '<div class="section-note">A concise portfolio view of churn exposure, regional performance, and core relationship drivers.</div>',
        unsafe_allow_html=True,
    )

    executive_container = st.container()
    with executive_container:
        left, right = st.columns([1, 1.3])

        with left:
            churn_counts = (
                filtered_df["CustomerStatus"].value_counts().rename_axis("CustomerStatus").reset_index(name="Count")
            )
            pie_fig = px.pie(
                churn_counts,
                names="CustomerStatus",
                values="Count",
                title="Churn Distribution",
                hole=0.45,
                color="CustomerStatus",
                color_discrete_map={"Retained": "#27AE60", "Churned": "#EB5757"},
            )
            st.plotly_chart(chart_layout(pie_fig), use_container_width=True)
            churn_rate = safe_rate(filtered_df["Exited"])
            st.caption(
                f"Business insight: {churn_rate:.1f}% of customers in the filtered portfolio have churned, which sets the immediate retention baseline for leadership review."
            )

        with right:
            geo_churn = (
                filtered_df.groupby("Geography", as_index=False)
                .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"))
            )
            geo_churn["ChurnRate"] = (geo_churn["ChurnRate"] * 100).round(2)
            geo_fig = px.bar(
                geo_churn.sort_values("ChurnRate", ascending=False),
                x="Geography",
                y="ChurnRate",
                color="Geography",
                text="ChurnRate",
                title="Geography Churn Analysis",
                labels={"ChurnRate": "Churn Rate (%)"},
            )
            geo_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(chart_layout(geo_fig), use_container_width=True)
            if not geo_churn.empty:
                top_geo = geo_churn.sort_values("ChurnRate", ascending=False).iloc[0]
                st.caption(
                    f"Business insight: {top_geo['Geography']} shows the highest churn rate at {top_geo['ChurnRate']:.1f}%, making it the first geography to review for targeted intervention."
                )

    corr = get_correlation_matrix(filtered_df)
    heatmap_fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Correlation Heatmap",
    )
    st.plotly_chart(chart_layout(heatmap_fig, height=520), use_container_width=True)
    st.caption(
        "Business insight: The heatmap highlights the strongest statistical relationships across churn, engagement, product depth, and customer value measures for quick executive validation."
    )

    with st.expander("Executive KPI Summary", expanded=True):
        retained = int((filtered_df["Exited"] == 0).sum())
        churned = int((filtered_df["Exited"] == 1).sum())
        avg_relationship = filtered_df["RelationshipScore"].mean()
        avg_risk = filtered_df["RiskScore"].mean()
        st.markdown(
            f"""
            - Retained customers: **{retained:,}**
            - Churned customers: **{churned:,}**
            - Average relationship score: **{avg_relationship:.2f}**
            - Average risk score: **{avg_risk:.2f}**
            """
        )


def engagement_analytics_tab(filtered_df: pd.DataFrame) -> None:
    """Engagement analytics section."""
    if filtered_df.empty:
        render_no_data_message()
        return

    st.markdown("### Engagement Analytics")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        engagement_churn = (
            filtered_df.groupby(["EngagementSegment", "CustomerStatus"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig = px.bar(
            engagement_churn,
            x="EngagementSegment",
            y="Count",
            color="CustomerStatus",
            barmode="group",
            title="Engagement vs Churn",
            labels={"Count": "Customers"},
            color_discrete_map={"Retained": "#27AE60", "Churned": "#EB5757"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Engagement bands with larger churn concentration should receive the earliest retention campaigns and service outreach."
        )

    with row1_col2:
        active_analysis = (
            filtered_df.groupby("MemberActivity", as_index=False)
            .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"))
        )
        active_analysis["ChurnRate"] = (active_analysis["ChurnRate"] * 100).round(2)
        fig = px.bar(
            active_analysis,
            x="MemberActivity",
            y="ChurnRate",
            color="MemberActivity",
            text="ChurnRate",
            title="Active vs Inactive Analysis",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Activity status offers a direct operational signal for retention because inactive customers typically carry a higher exit likelihood."
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        relationship_churn = (
            filtered_df.groupby("RelationshipCategory", as_index=False)
            .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"))
        )
        relationship_churn["ChurnRate"] = (relationship_churn["ChurnRate"] * 100).round(2)
        fig = px.bar(
            relationship_churn.sort_values("ChurnRate", ascending=False),
            x="RelationshipCategory",
            y="ChurnRate",
            color="RelationshipCategory",
            title="Relationship Category Analysis",
            text="ChurnRate",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Relationship tiers with elevated churn rates indicate where relationship-building actions can deliver the fastest retention gains."
        )

    with row2_col2:
        sticky_analysis = (
            filtered_df.groupby("StickinessFlag", as_index=False)
            .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"))
        )
        sticky_analysis["ChurnRate"] = (sticky_analysis["ChurnRate"] * 100).round(2)
        fig = px.bar(
            sticky_analysis,
            x="StickinessFlag",
            y="ChurnRate",
            color="StickinessFlag",
            text="ChurnRate",
            title="Sticky Customer Analysis",
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Stickiness segments show whether deeper relationship anchors are successfully protecting customers from churn."
        )


def product_utilization_tab(filtered_df: pd.DataFrame) -> None:
    """Product utilization section."""
    if filtered_df.empty:
        render_no_data_message()
        return

    st.markdown("### Product Utilization")
    col1, col2 = st.columns(2)

    with col1:
        product_churn = (
            filtered_df.groupby("NumOfProducts", as_index=False)
            .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"))
        )
        product_churn["ChurnRate"] = (product_churn["ChurnRate"] * 100).round(2)
        fig = px.bar(
            product_churn,
            x="NumOfProducts",
            y="ChurnRate",
            color="NumOfProducts",
            text="ChurnRate",
            title="Product Count vs Churn",
            labels={"ChurnRate": "Churn Rate (%)", "NumOfProducts": "Product Count"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Product depth trends help quantify whether broader product adoption is reducing or accelerating churn exposure."
        )

    with col2:
        product_engagement = (
            filtered_df.groupby("ProductCategory", as_index=False)
            .agg(ActiveRate=("IsActiveMember", "mean"), Customers=("CustomerId", "count"))
        )
        product_engagement["ActiveRate"] = (product_engagement["ActiveRate"] * 100).round(2)
        fig = px.bar(
            product_engagement.sort_values("ActiveRate", ascending=False),
            x="ProductCategory",
            y="ActiveRate",
            color="ProductCategory",
            text="ActiveRate",
            title="Product Engagement Analysis",
            labels={"ActiveRate": "Active Customer (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Product groups with weaker activity rates can be prioritized for onboarding, cross-sell, or service activation plays."
        )

    multi_product = (
        filtered_df.groupby("ProductBand", as_index=False)
        .agg(RetentionRate=("Exited", lambda x: (1 - x.mean()) * 100), Customers=("CustomerId", "count"))
    )
    multi_product["RetentionRate"] = multi_product["RetentionRate"].round(2)
    fig = px.bar(
        multi_product,
        x="ProductBand",
        y="RetentionRate",
        color="ProductBand",
        text="RetentionRate",
        title="Multi-Product Retention Analysis",
        labels={"RetentionRate": "Retention Rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.caption(
        "Business insight: Comparing single-product and multi-product customers reveals how product penetration contributes to long-term retention resilience."
    )


def premium_customer_risk_tab(filtered_df: pd.DataFrame) -> None:
    """Premium and risk-focused section."""
    if filtered_df.empty:
        render_no_data_message()
        return

    st.markdown("### Premium Customer Risk")

    high_balance_threshold = filtered_df["Balance"].quantile(0.75)
    high_balance_inactive = filtered_df[
        (filtered_df["Balance"] >= high_balance_threshold)
        & (filtered_df["IsActiveMember"] == 0)
    ]

    col1, col2 = st.columns(2)

    with col1:
        if high_balance_inactive.empty:
            st.info("No high-balance inactive customers are present in the current filtered view.")
        else:
            premium_geo = (
                high_balance_inactive.groupby("Geography", as_index=False)
                .size()
                .rename(columns={"size": "Customers"})
            )
            fig = px.bar(
                premium_geo.sort_values("Customers", ascending=False),
                x="Geography",
                y="Customers",
                color="Geography",
                text="Customers",
                title="High Balance Inactive Customers",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
            st.caption(
                "Business insight: High-balance inactive customers represent a premium attrition risk segment and should be proactively re-engaged."
            )

    with col2:
        fig = px.scatter(
            filtered_df,
            x="EstimatedSalary",
            y="Balance",
            color="RiskCategory",
            size="RelationshipScore",
            hover_data=["CustomerId", "Surname", "Geography", "NumOfProducts"],
            title="Balance vs Salary Scatter Plot",
            labels={"EstimatedSalary": "Estimated Salary", "Balance": "Balance"},
        )
        st.plotly_chart(chart_layout(fig, height=460), use_container_width=True)
        st.caption(
            "Business insight: The salary-balance view helps isolate premium clients whose financial profile and risk category merit high-touch retention strategies."
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        risk_dist = filtered_df.groupby("RiskCategory", as_index=False).size().rename(columns={"size": "Customers"})
        fig = px.bar(
            risk_dist.sort_values("Customers", ascending=False),
            x="RiskCategory",
            y="Customers",
            color="RiskCategory",
            text="Customers",
            title="Risk Category Distribution",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Risk segmentation distribution shows the volume of customers requiring preventive retention monitoring."
        )

    with row2_col2:
        premium_risk = filtered_df[filtered_df["HighValueCustomer"] == 1].copy()
        if premium_risk.empty:
            st.info("No high-value customers are present in the current filtered view.")
        else:
            premium_risk_analysis = (
                premium_risk.groupby("RiskCategory", as_index=False)
                .agg(ChurnRate=("Exited", "mean"), Customers=("CustomerId", "count"))
            )
            premium_risk_analysis["ChurnRate"] = (premium_risk_analysis["ChurnRate"] * 100).round(2)
            fig = px.bar(
                premium_risk_analysis.sort_values("ChurnRate", ascending=False),
                x="RiskCategory",
                y="ChurnRate",
                color="RiskCategory",
                text="ChurnRate",
                title="Premium Churn Risk Analysis",
                labels={"ChurnRate": "Churn Rate (%)"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
            st.caption(
                "Business insight: Among premium customers, higher-risk bands require immediate personalized retention treatment to protect revenue concentration."
            )


def relationship_strength_tab(filtered_df: pd.DataFrame) -> None:
    """Relationship strength section."""
    if filtered_df.empty:
        render_no_data_message()
        return

    st.markdown("### Relationship Strength")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_df,
            x="RelationshipCategory",
            color="CustomerStatus",
            barmode="group",
            title="Relationship Category Histogram",
            labels={"count": "Customers"},
            color_discrete_map={"Retained": "#27AE60", "Churned": "#EB5757"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Relationship category distribution shows where customer volume is concentrated across stronger and weaker relationship tiers."
        )

    with col2:
        fig = px.histogram(
            filtered_df,
            x="RelationshipScore",
            color="CustomerStatus",
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            title="Relationship Score Analysis",
            color_discrete_map={"Retained": "#2F80ED", "Churned": "#EB5757"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(
            "Business insight: Relationship score distributions reveal whether weaker scores are overrepresented within churned customers."
        )

    loyalty_seg = (
        filtered_df.groupby("RelationshipCategory", as_index=False)
        .agg(RetentionRate=("Exited", lambda x: (1 - x.mean()) * 100), Customers=("CustomerId", "count"))
    )
    loyalty_seg["RetentionRate"] = loyalty_seg["RetentionRate"].round(2)
    fig = px.bar(
        loyalty_seg.sort_values("RetentionRate", ascending=False),
        x="RelationshipCategory",
        y="RetentionRate",
        color="RelationshipCategory",
        text="RetentionRate",
        title="Loyalty Segmentation Analysis",
        labels={"RetentionRate": "Retention Rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.caption(
        "Business insight: Retention performance by relationship tier clarifies where relationship strengthening can produce the highest portfolio lift."
    )


def customer_explorer_tab(filtered_df: pd.DataFrame) -> None:
    """Interactive customer explorer section."""
    st.markdown("### Customer Explorer")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with summary_col2:
        st.metric("Unique Geographies", f"{filtered_df['Geography'].nunique() if not filtered_df.empty else 0}")
    with summary_col3:
        st.metric("Visible Churn Rate", f"{safe_rate(filtered_df['Exited']):.1f}%" if not filtered_df.empty else "0.0%")

    csv_data = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Download Filtered Dataset",
        data=csv_data,
        file_name="\"C:\Banking_Retention_project\data\enhanced_customer_analysis.csv",
        mime="text/csv",
        use_container_width=False,
    )

    if filtered_df.empty:
        render_no_data_message()
        return

    display_cols = [
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "EngagementSegment",
        "ProductCategory",
        "RelationshipScore",
        "RelationshipCategory",
        "RiskScore",
        "RiskCategory",
        "HighValueCustomer",
        "StickyCustomer",
        "Exited",
    ]

    st.dataframe(
        filtered_df[display_cols].sort_values(["RiskScore", "RelationshipScore"], ascending=[False, False]),
        use_container_width=True,
        height=500,
    )

    with st.expander("Explorer Notes", expanded=False):
        st.write(
            "Use the sidebar filters and search box to isolate customer cohorts, review churn signals, and export the resulting portfolio slice for further action."
        )


def main() -> None:
    """Main dashboard application."""
    st.markdown('<div class="main-title">Customer Engagement & Product Utilization Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Retention Strategy Dashboard for engagement, product depth, relationship strength, and premium churn risk monitoring.</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Dataset file '{DATA_PATH}' was not found. Place the CSV file in the same folder as app.py and rerun the dashboard."
        )
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty. Please provide a valid CSV file.")
        st.stop()
    except ValueError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected data loading error: {exc}")
        st.stop()

    build_sidebar(df)
    filtered_df = filter_dataframe(df)
    st.session_state["filtered_customer_count"] = len(filtered_df)

    render_top_kpis(filtered_df)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Engagement Analytics",
            "Product Utilization",
            "Premium Customer Risk",
            "Relationship Strength",
            "Customer Explorer",
        ]
    )

    with tabs[0]:
        executive_overview_tab(filtered_df, df)
    with tabs[1]:
        engagement_analytics_tab(filtered_df)
    with tabs[2]:
        product_utilization_tab(filtered_df)
    with tabs[3]:
        premium_customer_risk_tab(filtered_df)
    with tabs[4]:
        relationship_strength_tab(filtered_df)
    with tabs[5]:
        customer_explorer_tab(filtered_df)

    st.markdown(
        '<div class="footer">Developed for Customer Engagement & Product Utilization Analytics for Retention Strategy</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
