from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.crowd_model import (  # noqa: E402
    classify_crowd,
    predict_monument_with_model,
    summarize_training,
    train_crowd_models,
)
from scripts.recommendations import (  # noqa: E402
    get_quick_insights,
    list_monument_names,
    load_tourism_data,
    recommend_hidden_gems,
)


DATA_PATH = BASE_DIR / "data" / "processed_tourism.csv"
CROWD_COLORS = {"High": "#d94841", "Medium": "#f6b73c", "Low": "#5dbb63"}
PAGE_OPTIONS = [
    "🏠 Home",
    "🧳 Plan My Heritage Trip",
    "🔮 Crowd Forecast",
    "💎 Hidden Gems",
    "🗺 India Map",
]
CIRCLE_COORDS = {
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Amaravati": {"lat": 16.5062, "lon": 80.6480},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433},
    "Banglore": {"lat": 12.9716, "lon": 77.5946},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Dharwad": {"lat": 15.4589, "lon": 75.0078},
    "Goa": {"lat": 15.2993, "lon": 74.1240},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Hampi": {"lat": 15.3350, "lon": 76.4600},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Jabalpur": {"lat": 23.1815, "lon": 79.9864},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Jhansi": {"lat": 25.4484, "lon": 78.5685},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Leh": {"lat": 34.1526, "lon": 77.5771},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Raiganj": {"lat": 25.6185, "lon": 88.1256},
    "Raipur": {"lat": 21.2514, "lon": 81.6296},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022},
    "Sarnath": {"lat": 25.3811, "lon": 83.0218},
    "Shimla": {"lat": 31.1048, "lon": 77.1734},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973},
    "Thrissur": {"lat": 10.5276, "lon": 76.2144},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812},
}
TOLERANCE_WEIGHTS = {
    "Very low crowds": {"Low": 1.0, "Medium": 0.45, "High": 0.05},
    "Balanced": {"Low": 0.9, "Medium": 1.0, "High": 0.35},
    "I can handle busy icons": {"Low": 0.7, "Medium": 0.9, "High": 1.0},
}


def format_number(value: float | int) -> str:
    return f"{int(round(value)):,}"


def render_card(title: str, value: str, subtitle: str = "", accent: str = "#f59e0b") -> None:
    st.markdown(
        f"""
        <div class="premium-card" style="border-top: 4px solid {accent};">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, body: str) -> None:
    body_html = body.replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="premium-card text-card">
            <div class="text-card-title">{title}</div>
            <div class="text-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    return load_tourism_data(DATA_PATH)


@st.cache_resource
def load_model_bundle(data_signature: float):
    df = load_processed_data()
    return train_crowd_models(df)


def infer_interest_tags(monument_name: str) -> set[str]:
    lower_name = str(monument_name).strip().lower()
    keyword_map = {
        "Forts & Palaces": ["fort", "palace", "mahal", "garh"],
        "Temples & Spiritual": ["temple", "stupa", "mosque", "tomb", "monastery", "buddhist"],
        "Archaeology & Ruins": ["cave", "caves", "excavated", "remains", "ruined", "edict", "rock"],
        "Museums & Culture": ["museum", "observatory", "hall", "gate"],
        "UNESCO Icons": ["taj mahal", "fatehpur", "mamallapuram"],
    }
    tags = {
        category
        for category, keywords in keyword_map.items()
        if any(keyword in lower_name for keyword in keywords)
    }
    return tags or {"Heritage Discovery"}


def build_map_dataframe(prepared_df: pd.DataFrame, model_bundle) -> pd.DataFrame:
    map_df = prepared_df.copy().sort_values(["circle", "monument"]).reset_index(drop=True)
    forecasts = [
        predict_monument_with_model(model_bundle, monument)
        for monument in map_df["monument"].tolist()
    ]
    forecast_df = pd.DataFrame(forecasts)
    map_df["predicted_visitors"] = forecast_df["predicted_visitors"]
    map_df["predicted_crowd"] = forecast_df["predicted_crowd"]
    map_df["best_time_to_visit"] = forecast_df["best_time_to_visit"]

    group_index = map_df.groupby("circle").cumcount()
    for idx, row in map_df.iterrows():
        base = CIRCLE_COORDS.get(row["circle"])
        if not base:
            continue
        angle = group_index.iloc[idx] * 2.3999632297
        radius = 0.12 + 0.015 * math.sqrt(group_index.iloc[idx] + 1)
        lon_adjust = radius * math.sin(angle) / max(math.cos(math.radians(base["lat"])), 0.35)
        lat_adjust = radius * math.cos(angle)
        map_df.loc[idx, "lat"] = base["lat"] + lat_adjust
        map_df.loc[idx, "lon"] = base["lon"] + lon_adjust

    return map_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def build_trip_plan(
    prepared_df: pd.DataFrame,
    model_bundle,
    city: str,
    days: int,
    crowd_tolerance: str,
) -> list[list[dict[str, object]]]:
    city_df = prepared_df[prepared_df["circle"] == city].copy()
    if city_df.empty:
        return []

    rows = []
    for _, row in city_df.iterrows():
        forecast = predict_monument_with_model(model_bundle, row["monument"])
        monument_tags = infer_interest_tags(row["monument"])
        tolerance_score = TOLERANCE_WEIGHTS[crowd_tolerance][forecast["predicted_crowd"]]
        itinerary_score = (
            0.45 * float(row["heritage_score"])
            + 0.35 * float(row["hidden_gem_score"])
            + 0.20 * tolerance_score
        )
        rows.append(
            {
                "monument": row["monument"],
                "circle": row["circle"],
                "tags": ", ".join(sorted(monument_tags)),
                "heritage_score": float(row["heritage_score"]),
                "hidden_gem_score": float(row["hidden_gem_score"]),
                "predicted_visitors": int(forecast["predicted_visitors"]),
                "predicted_crowd": forecast["predicted_crowd"],
                "best_time_to_visit": forecast["best_time_to_visit"],
                "itinerary_score": itinerary_score,
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        ["itinerary_score", "heritage_score", "hidden_gem_score"],
        ascending=[False, False, False],
    )
    selected = ranked.head(max(days * 2, days)).to_dict(orient="records")
    per_day = max(1, min(2, math.ceil(len(selected) / max(days, 1))))
    return [selected[i : i + per_day] for i in range(0, len(selected), per_day)]


st.set_page_config(page_title="SmartHeritage", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

:root {
    --bg-a: #fff7ed;
    --bg-b: #fde68a;
    --bg-c: #f6ad55;
    --bg-d: #8b5e34;
    --text-main: #4b2e16;
    --text-soft: #7c5a3d;
    --card: rgba(255, 250, 240, 0.82);
}

.stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(circle at top left, rgba(255, 214, 165, 0.95), transparent 26%),
        radial-gradient(circle at bottom right, rgba(252, 211, 77, 0.45), transparent 20%),
        linear-gradient(135deg, #fff8ef 0%, #fff1d8 28%, #f6d59a 65%, #d7a15f 100%);
    color: var(--text-main);
}

[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

.top-shell {
    background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,244,227,0.58));
    border: 1px solid rgba(139, 94, 52, 0.12);
    border-radius: 28px;
    padding: 1.25rem 1.35rem 1.1rem;
    box-shadow: 0 18px 44px rgba(133, 90, 46, 0.12);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #6b3f1d;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #8b6848;
    font-size: 1.05rem;
    line-height: 1.7;
}

.chip {
    display: inline-block;
    margin-right: 0.5rem;
    margin-top: 0.85rem;
    padding: 0.38rem 0.8rem;
    border-radius: 999px;
    background: rgba(255, 237, 213, 0.95);
    color: #8c4d1c;
    border: 1px solid rgba(213, 119, 21, 0.12);
    font-size: 0.88rem;
}

.premium-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,248,238,0.65));
    border: 1px solid rgba(139, 94, 52, 0.10);
    border-radius: 24px;
    padding: 1rem 1.05rem 1.1rem;
    box-shadow: 0 18px 36px rgba(164, 104, 41, 0.10);
    min-height: 138px;
}

.text-card {
    min-height: 120px;
}

.card-title {
    color: #9a6b37;
    font-size: 0.92rem;
    margin-bottom: 0.55rem;
}

.card-value {
    color: #5f3414;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}

.card-subtitle {
    color: #8b6848;
    font-size: 0.9rem;
    line-height: 1.55;
}

.text-card-title {
    color: #6f411f;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.text-card-body {
    color: #8a6b52;
    line-height: 1.65;
    font-size: 0.95rem;
}

.day-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,244,231,0.70));
    border-radius: 26px;
    padding: 1rem 1.05rem;
    border: 1px solid rgba(139, 94, 52, 0.10);
    box-shadow: 0 16px 34px rgba(164, 104, 41, 0.10);
    margin-bottom: 0.9rem;
}

.day-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #7b4b25;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stop-card {
    background: rgba(255, 251, 246, 0.95);
    border-radius: 18px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.65rem;
    border: 1px solid rgba(139, 94, 52, 0.08);
}

h1, h2, h3, h4, p, label, .stMarkdown, .stCaption {
    color: var(--text-main) !important;
}

[data-baseweb="radio"] > div {
    gap: 0.5rem;
}

[role="radiogroup"] label {
    background: rgba(255,255,255,0.72);
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(139, 94, 52, 0.10);
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.72);
    border-radius: 22px;
    border: 1px solid rgba(139,94,52,0.08);
}

[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div,
.stNumberInput input,
.stTextInput input {
    background: rgba(255,255,255,0.72);
    border-radius: 18px;
}

[data-baseweb="select"] *,
[data-baseweb="base-input"] *,
.stNumberInput input,
.stTextInput input {
    color: #5f3414 !important;
}

[data-baseweb="select"] svg {
    fill: #8b5e34 !important;
}

.stSlider [data-baseweb="slider"] * {
    color: #8b5e34 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

data_signature = DATA_PATH.stat().st_mtime
source_df = load_processed_data()
model_bundle = load_model_bundle(data_signature)
prepared_df = model_bundle.prepared_df.copy()
monument_names = list_monument_names(source_df)
city_names = sorted(prepared_df["circle"].dropna().astype(str).unique().tolist())
hidden_gems_df = recommend_hidden_gems(source_df, top_n=12)
insights = get_quick_insights(source_df)
training_summary = summarize_training(model_bundle)

top_row_left, top_row_right = st.columns([0.78, 0.22])
with top_row_left:
    st.markdown(
        """
        <div class="top-shell">
            <div class="hero-title">SmartHeritage</div>
            <div class="hero-subtitle">
                An AI-first tourism planner for heritage travellers. Build calm itineraries,
                forecast monument crowds, uncover hidden gems, and explore India through
                soft storytelling instead of raw dashboards.
            </div>
            <span class="chip">AI itinerary planning</span>
            <span class="chip">Visitor forecasting</span>
            <span class="chip">Hidden gem scoring</span>
            <span class="chip">Interactive India map</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_row_right:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

page = st.radio("Navigation", PAGE_OPTIONS, horizontal=True, label_visibility="collapsed")
st.write("")

if page == "🏠 Home":
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("Monuments Modelled", format_number(insights["total_sites"]), "Cleaned heritage records ready for planning", "#d97706")
    with c2:
        render_card("Annual Visitors", format_number(insights["average_visitors"] * insights["total_sites"]), "Tourism footprint from the processed dataset", "#f59e0b")
    with c3:
        render_card("Visitor R²", f"{model_bundle.metrics['visitor_r2']:.2f}", "RandomForestRegressor quality on the modelled training set", "#92400e")

    a, b = st.columns(2)
    with a:
        render_text_card(
            "How the planner thinks",
            "It blends heritage score, hidden gem score, predicted crowd level, and your own tolerance for busy places to recommend the best order of stops.",
        )
    with b:
        metrics_text = " | ".join(
            f"{row.metric}: {row.value}" for row in training_summary.itertuples(index=False)
        )
        render_text_card(
            "Model snapshot",
            f"{metrics_text}. Forecasts are generated directly from the processed tourism baseline, so travellers only choose the place and crowd comfort level.",
        )

    st.subheader("Featured Hidden Gems")
    gem_cols = st.columns(3)
    for idx, gem in enumerate(hidden_gems_df.head(3).itertuples(index=False)):
        with gem_cols[idx % 3]:
            render_text_card(
                gem.monument_name,
                (
                    f"{gem.circle} | Hidden Gem Score {gem.hidden_gem_score:.2f}\n\n"
                    f"Heritage Score {gem.heritage_score:.2f}. Best visit window: {gem.best_time_to_visit}."
                ),
            )

elif page == "🧳 Plan My Heritage Trip":
    st.subheader("Plan My Heritage Trip")
    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        selected_city = st.selectbox("Area / heritage circle", city_names, index=0, key="trip_city")
        trip_days = st.number_input("Number of days", min_value=1, max_value=7, value=2)
    with input_col2:
        crowd_tolerance = st.selectbox(
            "Crowd tolerance",
            list(TOLERANCE_WEIGHTS.keys()),
            index=1,
            key="trip_tolerance",
        )
    with input_col3:
        render_text_card(
            "AI planning context",
            (
                f"Using `{selected_city}` as the trip area.\n\n"
                "Forecast baseline: processed tourism dataset.\n\n"
                "The planner balances heritage value, hidden-gem potential, and your crowd comfort level."
            ),
        )

    itinerary = build_trip_plan(
        prepared_df,
        model_bundle,
        selected_city,
        int(trip_days),
        crowd_tolerance,
    )

    if not itinerary:
        st.warning("No monuments were found for this city.")
    else:
        st.markdown(
            f"### Your {trip_days}-day plan for `{selected_city}`"
        )
        st.caption(
            "The itinerary is optimized using your crowd tolerance and the current tourism baseline learned from the dataset."
        )
        for day_index, day_stops in enumerate(itinerary[: int(trip_days)], start=1):
            st.markdown(f"<div class='day-card'><div class='day-title'>Day {day_index}</div>", unsafe_allow_html=True)
            for stop in day_stops:
                st.markdown(
                    f"""
                    <div class="stop-card">
                        <strong>{stop['monument']}</strong><br>
                        <span style="color:#8a6b52;">{stop['tags']}</span><br><br>
                        Predicted crowd: <strong>{stop['predicted_crowd']}</strong> |
                        Predicted visitors: <strong>{format_number(stop['predicted_visitors'])}</strong><br>
                        Best time: <strong>{stop['best_time_to_visit']}</strong> |
                        Heritage score: <strong>{stop['heritage_score']:.2f}</strong> |
                        Hidden gem score: <strong>{stop['hidden_gem_score']:.2f}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔮 Crowd Forecast":
    st.subheader("Crowd Forecast")
    selected_monument = st.selectbox("Monument", monument_names, index=0, key="forecast_monument")
    forecast = predict_monument_with_model(model_bundle, selected_monument)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("Predicted Visitors", format_number(forecast["predicted_visitors"]), "Baseline forecast estimate", "#ea580c")
    with c2:
        render_card("Predicted Crowd", forecast["predicted_crowd"], "Converted from predicted visitor volume", "#d97706")
    with c3:
        render_card("Best Time", forecast["best_time_to_visit"], "Recommended visit window", "#f59e0b")
    with c4:
        render_card("Top Model Driver", forecast["top_driver"], "Most influential feature globally", "#92400e")

    compare_df = pd.DataFrame(
        {
            "Series": ["Historical monthly baseline", "Predicted monthly visitors"],
            "Visitors": [forecast["actual_visitors"], forecast["predicted_visitors"]],
        }
    )
    fig = px.bar(
        compare_df,
        x="Series",
        y="Visitors",
        color="Series",
        color_discrete_sequence=["#d6b36f", "#d97706"],
        template="plotly_white",
        title=f"Forecast Story for {selected_monument}",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Visitors")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "This forecast uses the processed tourism baseline directly, so the traveller only picks the monument and sees a clear predicted crowd outcome."
    )

elif page == "💎 Hidden Gems":
    st.subheader("Hidden Gems")
    gem_city = st.selectbox("Focus on an area or show all", ["All India"] + city_names, index=0, key="gem_city")
    gem_count = st.slider("How many gems to surface?", min_value=3, max_value=12, value=6)
    gems = (
        recommend_hidden_gems(source_df, top_n=gem_count, circle_name=gem_city)
        if gem_city != "All India"
        else recommend_hidden_gems(source_df, top_n=gem_count)
    )

    if gems.empty:
        st.warning("No hidden gems match the selected filters.")
    else:
        if gem_city == "All India":
            st.caption("These are low-crowd monuments across the full dataset.")
        else:
            st.caption(
                f"These are the low-crowd monuments within `{gem_city}`. In the current app rule, every `Low` crowd monument is treated as a hidden gem."
            )
        cols = st.columns(3)
        for idx, gem in enumerate(gems.itertuples(index=False)):
            with cols[idx % 3]:
                render_text_card(
                    gem.monument_name,
                    (
                        f"{gem.circle} | {gem.hidden_gem_category}\n\n"
                        f"Crowd Level: {gem.crowd_level}\n\n"
                        f"Visitors: {format_number(gem.total_visitors)}\n\n"
                        f"Best time: {gem.best_time_to_visit}"
                    ),
                )

elif page == "🗺 India Map":
    st.subheader("India Map")
    st.caption(
        "Map colors reflect the predicted crowd baseline for each monument based on the processed tourism dataset."
    )
    map_df = build_map_dataframe(prepared_df, model_bundle)
    crowd_filters = st.multiselect("Show crowd levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    map_df = map_df[map_df["predicted_crowd"].isin(crowd_filters)]

    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        color="predicted_crowd",
        size="predicted_visitors",
        hover_name="monument",
        hover_data={
            "circle": True,
            "predicted_visitors": ":,.0f",
            "best_time_to_visit": True,
            "lat": False,
            "lon": False,
        },
        color_discrete_map=CROWD_COLORS,
        title="Predicted heritage crowd map for current conditions",
        projection="natural earth",
    )
    fig.update_traces(marker=dict(line=dict(width=0.8, color="white"), opacity=0.90))
    fig.update_geos(
        lataxis_range=[6, 38],
        lonaxis_range=[67, 98],
        showcountries=True,
        countrycolor="rgba(107,63,29,0.30)",
        showland=True,
        landcolor="rgba(255,245,230,0.55)",
        showocean=True,
        oceancolor="rgba(254,243,199,0.35)",
        showlakes=True,
        lakecolor="rgba(254,243,199,0.35)",
        showframe=False,
        coastlinecolor="rgba(107,63,29,0.25)",
        bgcolor="rgba(0,0,0,0)",
        center={"lat": 22.8, "lon": 79.2},
        projection_scale=4.2,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="Crowd level",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Markers are colored by predicted crowd level: green for low, yellow for medium, and red for high. "
        "All markers refresh whenever the processed dataset changes."
    )
