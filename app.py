from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.data_utils import generate_synthetic_experiment, prepare_experiment_dataframe
from analytics.stats import build_interpretation, compute_experiment_stats

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_FILE = BASE_DIR / "data" / "kaggle_style_ab_sample.csv"
CSS_FILE = BASE_DIR / "assets" / "style.css"


def inject_css() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def guess_column(columns: list[str], keywords: list[str]) -> Optional[str]:
    for column in columns:
        lowered = column.lower()
        if any(keyword in lowered for keyword in keywords):
            return column
    return None


@st.cache_data(show_spinner=False)
def load_bundled_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_FILE)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


st.set_page_config(
    page_title="A/B Testing Analytics Dashboard",
    page_icon="🧪",
    layout="wide",
)
inject_css()

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">A/B Testing Analytics Dashboard</div>
        <div class="hero-subtitle">Upload experiment data or generate synthetic traffic to measure p-values, confidence intervals, and conversion lift.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data Source")
    source = st.radio(
        "Choose input mode",
        ("Upload CSV", "Bundled Kaggle-style sample", "Synthetic generator"),
        index=0,
    )
    alpha = st.slider("Significance level (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

raw_df: Optional[pd.DataFrame] = None

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload experiment CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to start analyzing your experiment.")
        st.stop()
    raw_df = pd.read_csv(uploaded_file)
elif source == "Bundled Kaggle-style sample":
    raw_df = load_bundled_data()
    st.caption(
        "Bundled dataset format is compatible with public Kaggle A/B testing datasets, e.g. "
        "https://www.kaggle.com/datasets/rohankulakarni/ab-test-marketing-campaign-dataset"
    )
else:
    with st.sidebar:
        st.markdown("### Synthetic Generator")
        control_size = st.number_input("Control sample size", min_value=500, max_value=500000, value=5000, step=500)
        treatment_size = st.number_input("Treatment sample size", min_value=500, max_value=500000, value=5000, step=500)
        control_rate = st.slider("Control conversion rate", min_value=0.01, max_value=0.50, value=0.12, step=0.01)
        expected_lift = st.slider(
            "Expected treatment lift",
            min_value=-0.20,
            max_value=0.50,
            value=0.08,
            step=0.01,
            help="Relative lift applied to control conversion rate.",
        )
        seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    raw_df = generate_synthetic_experiment(
        control_size=control_size,
        treatment_size=treatment_size,
        control_rate=control_rate,
        treatment_lift=expected_lift,
        seed=int(seed),
    )

if raw_df is None or raw_df.empty:
    st.error("The selected data source returned no rows.")
    st.stop()

with st.expander("Preview dataset", expanded=False):
    st.dataframe(raw_df.head(50), use_container_width=True)

st.markdown("### Configure experiment columns")
columns = list(raw_df.columns)

variant_guess = guess_column(columns, ["group", "variant", "arm", "bucket", "test"])
conversion_guess = guess_column(columns, ["converted", "conversion", "is_conversion", "outcome", "visitpageflag"])
date_guess = guess_column(columns, ["date", "time", "timestamp", "event"])
segment_guess = guess_column(columns, ["country", "segment", "device", "channel", "region"])

left, right = st.columns(2)
with left:
    variant_col = st.selectbox(
        "Variant/group column",
        options=columns,
        index=columns.index(variant_guess) if variant_guess in columns else 0,
    )
    conversion_col = st.selectbox(
        "Conversion column (0/1, true/false, yes/no)",
        options=columns,
        index=columns.index(conversion_guess) if conversion_guess in columns else min(1, len(columns) - 1),
    )
with right:
    optional_columns = ["None", *columns]
    date_col_opt = st.selectbox(
        "Time column (optional)",
        options=optional_columns,
        index=optional_columns.index(date_guess) if date_guess in optional_columns else 0,
    )
    segment_col_opt = st.selectbox(
        "Segment column (optional)",
        options=optional_columns,
        index=optional_columns.index(segment_guess) if segment_guess in optional_columns else 0,
    )

date_col = None if date_col_opt == "None" else date_col_opt
segment_col = None if segment_col_opt == "None" else segment_col_opt

try:
    prepared_df = prepare_experiment_dataframe(
        raw_df,
        variant_col=variant_col,
        conversion_col=conversion_col,
        date_col=date_col,
        segment_col=segment_col,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

variants = sorted(prepared_df["variant"].dropna().unique().tolist())
if len(variants) < 2:
    st.error("At least two groups are required for A/B testing.")
    st.stop()

default_control = next((v for v in variants if "control" in v.lower()), variants[0])
default_treatment = next((v for v in variants if any(x in v.lower() for x in ["treat", "variant", "test", "new"])), None)
if default_treatment is None or default_treatment == default_control:
    default_treatment = variants[1]

selector_left, selector_right = st.columns(2)
with selector_left:
    control_label = st.selectbox(
        "Control group label",
        options=variants,
        index=variants.index(default_control),
    )
with selector_right:
    treatment_options = [v for v in variants if v != control_label]
    treatment_label = st.selectbox(
        "Treatment group label",
        options=treatment_options,
        index=treatment_options.index(default_treatment) if default_treatment in treatment_options else 0,
    )

results = compute_experiment_stats(
    prepared_df,
    control_label=control_label,
    treatment_label=treatment_label,
    alpha=alpha,
)

group_stats = results["group_stats"]

metric_cols = st.columns(5)
metric_payload = [
    ("Control CVR", format_pct(results["control_rate"]), f"n={results['control_n']:,}"),
    ("Treatment CVR", format_pct(results["treatment_rate"]), f"n={results['treatment_n']:,}"),
    ("Relative lift", format_pct(results["relative_lift"]), f"Absolute: {format_pct(results['absolute_lift'])}"),
    ("p-value", f"{results['p_value']:.4g}", f"alpha={alpha:.2f}"),
    (
        "95% CI (absolute lift)",
        f"{format_pct(results['ci_low'])} → {format_pct(results['ci_high'])}",
        "Treatment - Control",
    ),
]

for column, (title, value, subtitle) in zip(metric_cols, metric_payload):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

chart_left, chart_right = st.columns((1.15, 1))

with chart_left:
    chart_df = group_stats.copy()
    chart_df["rate_pct"] = chart_df["rate"] * 100
    chart_df["err_plus"] = ((chart_df["ci_high"] - chart_df["rate"]).clip(lower=0)) * 100
    chart_df["err_minus"] = ((chart_df["rate"] - chart_df["ci_low"]).clip(lower=0)) * 100

    fig_rates = go.Figure()
    colors = ["#5B8DEF", "#34D399"]
    for i, (_, row) in enumerate(chart_df.iterrows()):
        fig_rates.add_bar(
            x=[row["variant"]],
            y=[row["rate_pct"]],
            name=row["variant"],
            marker_color=colors[i % len(colors)],
            text=[f"{row['rate_pct']:.2f}%"],
            textposition="outside",
            error_y=dict(
                type="data",
                array=[row["err_plus"]],
                arrayminus=[row["err_minus"]],
                visible=True,
            ),
        )
    fig_rates.update_layout(
        title="Conversion Rates with Confidence Intervals",
        template="plotly_dark",
        showlegend=False,
        yaxis_title="Conversion rate (%)",
        xaxis_title="Variant",
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
    )
    st.plotly_chart(fig_rates, use_container_width=True)

with chart_right:
    significance_flag = "Significant" if results["is_significant"] else "Not significant"
    relative_lift_value = results["relative_lift"]
    if pd.isna(relative_lift_value):
        gauge_value = 0
    else:
        gauge_value = max(min(relative_lift_value * 100, 100), -100)
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": f"Relative Lift ({significance_flag})"},
            gauge={
                "axis": {"range": [-100, 100]},
                "bar": {"color": "#5B8DEF"},
                "steps": [
                    {"range": [-100, 0], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [0, 100], "color": "rgba(16, 185, 129, 0.25)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20), height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

if "event_time" in prepared_df.columns:
    trend_df = prepared_df[prepared_df["variant"].isin([control_label, treatment_label])].copy()
    trend_df["date"] = trend_df["event_time"].dt.date
    trend_summary = (
        trend_df.groupby(["date", "variant"])["converted"]
        .agg(conversions="sum", users="count")
        .reset_index()
        .sort_values(["variant", "date"])
    )
    trend_summary["cum_conversions"] = trend_summary.groupby("variant")["conversions"].cumsum()
    trend_summary["cum_users"] = trend_summary.groupby("variant")["users"].cumsum()
    trend_summary["cum_rate_pct"] = (trend_summary["cum_conversions"] / trend_summary["cum_users"]) * 100

    fig_trend = px.line(
        trend_summary,
        x="date",
        y="cum_rate_pct",
        color="variant",
        markers=True,
        title="Cumulative Conversion Trend",
        labels={"cum_rate_pct": "Cumulative conversion rate (%)", "date": "Date", "variant": "Group"},
        color_discrete_sequence=["#5B8DEF", "#34D399"],
    )
    fig_trend.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10), height=360)
    st.plotly_chart(fig_trend, use_container_width=True)

if "segment" in prepared_df.columns:
    segment_df = (
        prepared_df[prepared_df["variant"].isin([control_label, treatment_label])]
        .groupby(["segment", "variant"])["converted"]
        .mean()
        .reset_index()
    )
    segment_df["conversion_rate_pct"] = segment_df["converted"] * 100
    fig_segments = px.bar(
        segment_df,
        x="segment",
        y="conversion_rate_pct",
        color="variant",
        barmode="group",
        title="Segment-wise Conversion Rate",
        labels={"segment": "Segment", "conversion_rate_pct": "Conversion rate (%)", "variant": "Group"},
        color_discrete_sequence=["#5B8DEF", "#34D399"],
    )
    fig_segments.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10), height=360)
    st.plotly_chart(fig_segments, use_container_width=True)

st.markdown("### Interpretation")
st.markdown(build_interpretation(results, alpha=alpha))

summary_df = pd.DataFrame(
    [
        {
            "control_group": control_label,
            "treatment_group": treatment_label,
            "control_n": results["control_n"],
            "treatment_n": results["treatment_n"],
            "control_rate": results["control_rate"],
            "treatment_rate": results["treatment_rate"],
            "absolute_lift": results["absolute_lift"],
            "relative_lift": results["relative_lift"],
            "p_value": results["p_value"],
            "ci_low": results["ci_low"],
            "ci_high": results["ci_high"],
            "is_significant": results["is_significant"],
            "recommended_n_per_group_80_power": results["recommended_n_per_group"],
        }
    ]
)
st.download_button(
    "Download experiment summary as CSV",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="ab_test_summary.csv",
    mime="text/csv",
)
