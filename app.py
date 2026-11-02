
import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px

from spark_engine import analyze_transactions


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Banking Transaction Analytics",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Transaction Analytics")
st.caption(
    "Big Data Analysis and Suspicious Transaction Monitoring "
    "using Python and Apache Spark"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(100, 120, 150, 0.10);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(120, 120, 120, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------

@st.cache_data(show_spinner="Processing transactions with PySpark...")
def load_data(csv_bytes):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:
            temp_file.write(csv_bytes)
            temp_path = temp_file.name

        result = analyze_transactions(temp_path)
        return result

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Dashboard Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Banking CSV",
    type=["csv"]
)

if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
    st.sidebar.success("Uploaded dataset selected")
else:
    default_path = "data/banking_transactions.csv"

    if not os.path.exists(default_path):
        st.error("Sample dataset not found. Run generate_data.py first.")
        st.stop()

    with open(default_path, "rb") as file:
        csv_bytes = file.read()

    st.sidebar.info("Using sample banking dataset")


# ---------------- PROCESS DATA ----------------

try:
    transactions, type_summary, daily_summary, customer_summary = (
        load_data(csv_bytes)
    )
except Exception as error:
    st.error(f"Unable to process the dataset: {error}")
    st.stop()


if transactions.empty:
    st.warning("No valid transaction records found.")
    st.stop()


transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

# ---------------- FILTERS ----------------

st.sidebar.subheader("🔎 Filters")

transaction_types = sorted(
    transactions["transaction_type"].dropna().unique().tolist()
)

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    transaction_types,
    default=transaction_types
)

statuses = sorted(
    transactions["status"].dropna().unique().tolist()
)

selected_statuses = st.sidebar.multiselect(
    "Transaction Status",
    statuses,
    default=statuses
)

min_date = transactions["transaction_date"].min().date()
max_date = transactions["transaction_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Transaction Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

suspicious_only = st.sidebar.checkbox(
    "Show suspicious transactions only",
    value=False
)

filtered = transactions[
    transactions["transaction_type"].isin(selected_types)
    & transactions["status"].isin(selected_statuses)
].copy()

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered = filtered[
        (filtered["transaction_date"].dt.date >= start_date)
        & (filtered["transaction_date"].dt.date <= end_date)
    ]

if suspicious_only:
    filtered = filtered[filtered["is_suspicious"] == 1]


# ---------------- KPI METRICS ----------------

st.subheader("📊 Transaction Overview")

total_transactions = len(filtered)
total_amount = filtered["amount"].sum()
average_amount = filtered["amount"].mean() if total_transactions else 0
suspicious_count = int(filtered["is_suspicious"].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Amount", f"₹{total_amount:,.2f}")
col3.metric("Average Amount", f"₹{average_amount:,.2f}")
col4.metric("Flagged Transactions", f"{suspicious_count:,}")


# ---------------- CHARTS ----------------

st.divider()
st.subheader("📈 Transaction Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    type_data = (
        filtered.groupby("transaction_type")
        .size()
        .reset_index(name="count")
    )

    fig_type = px.bar(
        type_data,
        x="transaction_type",
        y="count",
        color="transaction_type",
        title="Transactions by Type",
        labels={
            "transaction_type": "Transaction Type",
            "count": "Number of Transactions"
        }
    )

    st.plotly_chart(fig_type, use_container_width=True)

with chart_col2:
    fig_status = px.pie(
        filtered,
        names="status",
        title="Transaction Status Distribution",
        hole=0.4
    )

    st.plotly_chart(fig_status, use_container_width=True)


# ---------------- DAILY TREND ----------------

st.subheader("📅 Daily Transaction Trends")

daily_filtered = (
    filtered.assign(
        transaction_day=filtered["transaction_date"].dt.date
    )
    .groupby("transaction_day")
    .agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum")
    )
    .reset_index()
)

fig_daily = px.line(
    daily_filtered,
    x="transaction_day",
    y="total_amount",
    markers=True,
    title="Daily Transaction Amount",
    labels={
        "transaction_day": "Date",
        "total_amount": "Total Amount (₹)"
    }
)

st.plotly_chart(fig_daily, use_container_width=True)


# ---------------- SUSPICIOUS TRANSACTIONS ----------------

st.divider()
st.subheader("🚨 Suspicious Transaction Analysis")

suspicious_data = filtered[
    filtered["is_suspicious"] == 1
].copy()

if suspicious_data.empty:
    st.success("No transactions were flagged with the selected filters.")
else:
    st.warning(
        f"{len(suspicious_data)} transaction(s) flagged for review."
    )

    st.dataframe(
        suspicious_data[
            [
                "transaction_id",
                "customer_id",
                "transaction_date",
                "amount",
                "transaction_type",
                "location",
                "suspicious_reason"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ---------------- CUSTOMER ANALYSIS ----------------

st.divider()
st.subheader("👤 Customer Transaction Analysis")

customer_data = (
    filtered.groupby("customer_id")
    .agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum")
    )
    .reset_index()
    .sort_values("total_amount", ascending=False)
    .head(10)
)

fig_customer = px.bar(
    customer_data,
    x="customer_id",
    y="total_amount",
    title="Top 10 Customers by Transaction Amount",
    labels={
        "customer_id": "Customer ID",
        "total_amount": "Total Amount (₹)"
    }
)

st.plotly_chart(fig_customer, use_container_width=True)


# ---------------- DOWNLOAD REPORT ----------------

st.divider()
st.subheader("📥 Download Reports")

csv_output = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Transactions CSV",
    data=csv_output,
    file_name="banking_transaction_report.csv",
    mime="text/csv"
)

st.caption(
    "Academic demonstration project. Flagged transactions are "
    "rule-based alerts and must be reviewed; they do not establish fraud."
)
