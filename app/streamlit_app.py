"""Urban Mobility & Traffic Intelligence System
A high-performance interactive dashboard for NYC traffic congestion, forecasting, and incident hotspots.
"""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.db import (
    get_connection,
    get_kpis,
    query_top_hotspots,
    query_peak_hours,
    query_rolling_7day_avg,
    query_crash_density,
    query_zones_getting_worse,
)


DB_PATH = Path("data/urban.db")
GEOJSON_PATH = Path("data/taxi_zones.geojson")

st.set_page_config(
    page_title="NYC Urban Mobility Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM THEME & CSS INJECTION
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, .metric-value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Card container */
.metric-card {
    background-color: #12151C;
    border: 1px solid #232733;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.metric-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #9BA3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #E6E8EE;
}

.metric-subtitle {
    font-size: 0.80rem;
    color: #6C7BFF;
    margin-top: 4px;
}

.caveat-banner {
    background-color: rgba(108, 123, 255, 0.08);
    border-left: 4px solid #6C7BFF;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 18px;
    font-size: 0.88rem;
    color: #E6E8EE;
}

/* Hide default streamlit clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# DATABASE & DATA LOADERS (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def get_db():
    """Maintain a singleton read-only SQLite connection."""
    return get_connection(DB_PATH, read_only=True)


@st.cache_data
def load_kpis():
    conn = get_db()
    return get_kpis(conn)


@st.cache_data
def load_geojson():
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_hotspots_data(day_type: str, start_hour: int, end_hour: int):
    conn = get_db()
    # SQL query filtered by day_type if needed
    day_filter = ""
    if day_type == "Weekday":
        day_filter = "AND CAST(strftime('%w', zh.hour_ts) AS INTEGER) BETWEEN 1 AND 5"
    elif day_type == "Weekend":
        day_filter = "AND CAST(strftime('%w', zh.hour_ts) AS INTEGER) IN (0, 6)"

    sql = f"""
    WITH zone_metrics AS (
        SELECT 
            zh.zone_id,
            z.zone_name,
            z.borough,
            COUNT(zh.hour_ts) AS active_hours,
            SUM(zh.trips) AS total_trips,
            ROUND(AVG(zh.avg_speed), 2) AS mean_speed_mph,
            ROUND(AVG(zh.congestion_index), 4) AS mean_congestion
        FROM zone_hourly zh
        JOIN zones z ON zh.zone_id = z.zone_id
        WHERE CAST(strftime('%H', zh.hour_ts) AS INTEGER) BETWEEN :start_hour AND :end_hour
          {day_filter}
          AND zh.congestion_index IS NOT NULL
        GROUP BY zh.zone_id, z.zone_name, z.borough
        HAVING SUM(zh.trips) >= 100
    )
    SELECT 
        RANK() OVER (ORDER BY mean_congestion DESC) AS congestion_rank,
        zone_id,
        zone_name,
        borough,
        active_hours,
        total_trips,
        mean_speed_mph,
        mean_congestion
    FROM zone_metrics
    ORDER BY congestion_rank;
    """
    return pd.read_sql_query(sql, conn, params={"start_hour": start_hour, "end_hour": end_hour})


@st.cache_data
def load_peak_hours_data(zone_id: int):
    conn = get_db()
    return query_peak_hours(conn, zone_id=zone_id)


@st.cache_data
def load_top_zones():
    conn = get_db()
    sql = """
    SELECT zh.zone_id, z.zone_name, z.borough, SUM(zh.trips) as total_trips
    FROM zone_hourly zh
    JOIN zones z ON zh.zone_id = z.zone_id
    GROUP BY zh.zone_id, z.zone_name, z.borough
    ORDER BY total_trips DESC
    LIMIT 20
    """
    return pd.read_sql_query(sql, conn)


@st.cache_data
def load_forecasts(zone_id: int):
    conn = get_db()
    f_df = pd.read_sql_query(
        """
        SELECT hour_ts, actual_congestion, forecast_congestion, lower_ci, upper_ci, is_anomaly, is_future
        FROM forecasts
        WHERE zone_id = :zone_id
        ORDER BY hour_ts
        """,
        conn,
        params={"zone_id": zone_id},
    )
    e_df = pd.read_sql_query(
        "SELECT * FROM forecast_eval WHERE zone_id = :zone_id",
        conn,
        params={"zone_id": zone_id},
    )
    return f_df, e_df


@st.cache_data
def load_crashes_summary():
    conn = get_db()
    density_df = query_crash_density(conn, min_crashes=10)
    hourly_df = pd.read_sql_query(
        """
        SELECT CAST(strftime('%H', crash_ts) AS INTEGER) as crash_hour,
               COUNT(*) as total_crashes,
               SUM(injured) as total_injured,
               SUM(killed) as total_killed
        FROM crashes
        GROUP BY crash_hour
        ORDER BY crash_hour
        """,
        conn,
    )
    factors_df = pd.read_sql_query(
        """
        SELECT factor, COUNT(*) as count
        FROM crashes
        WHERE factor != 'Unspecified' AND factor IS NOT NULL
        GROUP BY factor
        ORDER BY count DESC
        LIMIT 10
        """,
        conn,
    )
    crashes_coords = pd.read_sql_query(
        "SELECT lat, lon, injured, killed FROM crashes WHERE lat IS NOT NULL AND lon IS NOT NULL LIMIT 4000",
        conn,
    )
    return density_df, hourly_df, factors_df, crashes_coords


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
with st.sidebar:
    st.title("🚕 NYC Mobility")
    st.markdown("**Traffic Intelligence & Forecasting**")
    st.markdown("---")

    st.subheader("Global Timeframe")
    st.caption("Active data period: **August 1 – September 30, 2026**")

    st.markdown("---")
    st.markdown(
        """
        **System Architecture**
        - **Data Pipeline**: TLC Yellow Taxi (4.91M trips) + TLC Zones + NYC Crashes (13.4k incidents)
        - **Storage**: SQLite 3 (`data/urban.db`, 19.6 MB)
        - **Analytics**: SQL window functions & CTEs
        - **Modeling**: statsmodels SARIMAX (48h ahead)
        """
    )
    st.markdown("---")
    st.caption("Google Antigravity Mobility Platform v2.0")


# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
kpis = load_kpis()

st.title("Urban Mobility & Traffic Intelligence System")
st.markdown(
    """
    <div class="caveat-banner">
        <strong>⚠️ Data Proxy Notice:</strong> Congestion indices displayed in this system are a 
        <strong>proxy metric computed from NYC TLC Yellow Taxi speeds</strong> relative to observed off-peak free-flow baselines, 
        not direct physical traffic sensor measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_map, tab_peak, tab_forecast, tab_crashes, tab_method = st.tabs([
    "📊 Overview",
    "🗺️ Hotspots Map",
    "⏱️ Peak-Hour Analysis",
    "📈 48h Forecast",
    "💥 Crashes & Safety",
    "📖 Data & Method",
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_overview:
    st.header("Executive Overview & KPIs")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Trips Analyzed</div>
                <div class="metric-value">{kpis['total_trips']:,}</div>
                <div class="metric-subtitle">Cleaned 2-month sample</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">City Mean Congestion</div>
                <div class="metric-value">{kpis['city_mean_congestion']:.1%}</div>
                <div class="metric-subtitle">Speed deficit vs free-flow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Worst Zone Overall</div>
                <div class="metric-value" style="font-size: 1.25rem;">{kpis['worst_zone'].split(' - ')[0]}</div>
                <div class="metric-subtitle">{kpis['worst_zone'].split(' - ')[-1]} mean congestion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Traffic Crashes</div>
                <div class="metric-value">{kpis['total_crashes']:,}</div>
                <div class="metric-subtitle">{kpis['total_injured']:,} injured / {kpis['total_killed']} killed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Key Findings (Computed from Empirical Data)")
    
    col_findings_1, col_findings_2 = st.columns(2)
    with col_findings_1:
        st.info(
            f"""
            - **Worst Hour City-Wide**: Traffic congestion peaks consistently at **{kpis['worst_hour']}**, with city-wide average traffic speeds slowing by over 55% compared to off-peak night hours.
            - **Core Hotspots**: Manhattan's commercial core (**Midtown Center, UN/Turtle Bay South, and Battery Park City**) experiences average speeds below 7.5 mph during weekday business hours.
            - **Weekly Rhythms**: Thursday and Friday late afternoons (16:00 – 19:00) represent the most severe city-wide congestion bottlenecks.
            """
        )
    with col_findings_2:
        st.warning(
            f"""
            - **Crash Geography**: High-collision corridors are concentrated along arterial routes outside Midtown—led by **Crown Heights North** (267 collisions) and **East New York** (237 collisions).
            - **Casualty Impact**: Across the 61-day window, {kpis['total_crashes']:,} reported collisions resulted in {kpis['total_injured']:,} injuries and {kpis['total_killed']} fatalities.
            - **Causal Distinction**: High-congestion zones in Manhattan show dense minor fender-benders, whereas high-injury collisions concentrate on higher-speed outer-borough avenues.
            """
        )

    # 7-day rolling trend chart
    st.markdown("### City-Wide 7-Day Rolling Congestion Trend")
    conn = get_db()
    rolling_df = query_rolling_7day_avg(conn)
    fig_rolling = go.Figure()
    fig_rolling.add_trace(
        go.Scatter(
            x=rolling_df["hour_ts"],
            y=rolling_df["congestion_index"],
            mode="lines",
            name="Hourly Congestion",
            line=dict(color="#232733", width=1),
            opacity=0.4,
        )
    )
    fig_rolling.add_trace(
        go.Scatter(
            x=rolling_df["hour_ts"],
            y=rolling_df["rolling_7day_congestion"],
            mode="lines",
            name="7-Day Rolling Average",
            line=dict(color="#6C7BFF", width=3),
        )
    )
    fig_rolling.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        xaxis_title="Timestamp",
        yaxis_title="Congestion Index (0 = Free-Flow, 1 = Gridlock)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_rolling, use_container_width=True)


# =========================================================
# TAB 2: HOTSPOTS MAP
# =========================================================
with tab_map:
    st.header("NYC Congestion Hotspots & Spatial Distribution")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 3, 2])
    with col_ctrl1:
        map_day_type = st.selectbox("Day Type Filter", ["All Days", "Weekday", "Weekend"], index=0)
    with col_ctrl2:
        hour_range = st.slider("Hour of Day Window", 0, 23, (7, 19))
    with col_ctrl3:
        show_crash_heat = st.checkbox("Overlay Crash HeatMap", value=False)

    hotspots_df = load_hotspots_data(map_day_type, hour_range[0], hour_range[1])
    geojson_data = load_geojson()

    map_col, table_col = st.columns([3, 2])

    with map_col:
        st.subheader(f"Choropleth ({hour_range[0]:02d}:00 – {hour_range[1]:02d}:00)")
        if geojson_data and not hotspots_df.empty:
            # Map congestion map lookup
            cong_lookup = hotspots_df.set_index("zone_id")["mean_congestion"].to_dict()
            speed_lookup = hotspots_df.set_index("zone_id")["mean_speed_mph"].to_dict()
            trips_lookup = hotspots_df.set_index("zone_id")["total_trips"].to_dict()

            # Center map on NYC
            m = folium.Map(
                location=[40.75, -73.98],
                zoom_start=11,
                tiles="CartoDB dark_matter",
                control_scale=True,
            )
            # Add basemap tile layer switcher
            folium.TileLayer("CartoDB positron", name="Light Theme").add_to(m)

            # Color scale function
            def get_color(cong_val):
                if cong_val is None or np.isnan(cong_val):
                    return "#232733"
                if cong_val < 0.25:
                    return "#1b2a6b"
                elif cong_val < 0.45:
                    return "#6C7BFF"
                elif cong_val < 0.55:
                    return "#ffb86c"
                else:
                    return "#ff5d5d"

            # Style function for GeoJSON
            def style_fn(feature):
                zid = feature["properties"].get("LocationID")
                c_val = cong_lookup.get(zid)
                return {
                    "fillColor": get_color(c_val),
                    "color": "#12151C",
                    "weight": 0.8,
                    "fillOpacity": 0.75 if c_val is not None else 0.2,
                }

            # Add fields to properties for tooltip
            for feat in geojson_data["features"]:
                zid = feat["properties"].get("LocationID")
                feat["properties"]["congestion"] = f"{cong_lookup.get(zid, 0.0):.1%}" if zid in cong_lookup else "N/A"
                feat["properties"]["speed"] = f"{speed_lookup.get(zid, 0.0):.1f} mph" if zid in speed_lookup else "N/A"
                feat["properties"]["trips"] = f"{trips_lookup.get(zid, 0):,}" if zid in trips_lookup else "0"

            folium.GeoJson(
                geojson_data,
                style_function=style_fn,
                tooltip=folium.GeoJsonTooltip(
                    fields=["zone", "borough", "congestion", "speed", "trips"],
                    aliases=["Zone:", "Borough:", "Mean Congestion:", "Avg Speed:", "Trips:"],
                    localize=True,
                ),
            ).add_to(m)

            # Optional Crash Heatmap
            if show_crash_heat:
                _, _, _, crashes_coords = load_crashes_summary()
                heat_data = [[row["lat"], row["lon"]] for _, row in crashes_coords.iterrows()]
                HeatMap(heat_data, radius=9, blur=12, max_zoom=13).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=560)
        else:
            st.warning("Map geometry or data currently unavailable.")

    with table_col:
        st.subheader("Ranked Top 10 Congestion Hotspots")
        if not hotspots_df.empty:
            display_top10 = hotspots_df.head(10)[[
                "congestion_rank", "zone_name", "borough", "mean_congestion", "mean_speed_mph", "total_trips"
            ]].copy()
            display_top10["mean_congestion"] = display_top10["mean_congestion"].apply(lambda v: f"{v:.1%}")
            display_top10["mean_speed_mph"] = display_top10["mean_speed_mph"].apply(lambda v: f"{v:.1f} mph")
            display_top10["total_trips"] = display_top10["total_trips"].apply(lambda v: f"{v:,}")
            display_top10.columns = ["Rank", "Zone Name", "Borough", "Congestion", "Avg Speed", "Trips"]
            st.dataframe(display_top10, hide_index=True, use_container_width=True)
        else:
            st.info("No zones met the minimum trips filter for the selected window.")


# =========================================================
# TAB 3: PEAK-HOUR ANALYSIS
# =========================================================
with tab_peak:
    st.header("Temporal Congestion Dynamics & Peak Hours")

    top_zones_df = load_top_zones()
    zone_options = {"City-Wide (All Zones)": -1}
    for _, r in top_zones_df.iterrows():
        zone_options[f"{r['zone_name']} ({r['borough']})"] = int(r["zone_id"])

    selected_zone_name = st.selectbox("Select Target Geographic Unit", list(zone_options.keys()), index=0)
    selected_zone_id = zone_options[selected_zone_name]

    peak_df = load_peak_hours_data(selected_zone_id)

    if not peak_df.empty:
        # Create Pivot table for heatmap: Day of Week (0-6) x Hour of Day (0-23)
        day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        pivot_cong = peak_df.pivot(index="day_of_week", columns="hour_of_day", values="mean_congestion")
        pivot_cong.index = [day_names[i] for i in pivot_cong.index]

        fig_heat = px.imshow(
            pivot_cong,
            labels=dict(x="Hour of Day", y="Day of Week", color="Congestion"),
            x=[f"{h:02d}:00" for h in range(24)],
            y=pivot_cong.index.tolist(),
            color_continuous_scale=["#1b2a6b", "#6C7BFF", "#ffb86c", "#ff5d5d"],
            aspect="auto",
        )
        fig_heat.update_layout(
            title=f"Hourly Congestion Profile Heatmap: {selected_zone_name}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=340,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        col_peak_1, col_peak_2 = st.columns(2)
        with col_peak_1:
            st.subheader("Average 24-Hour Congestion Profile")
            daily_profile = peak_df.groupby("hour_of_day")["mean_congestion"].mean().reset_index()
            fig_line = px.line(
                daily_profile,
                x="hour_of_day",
                y="mean_congestion",
                markers=True,
                labels={"hour_of_day": "Hour of Day (00:00 - 23:00)", "mean_congestion": "Mean Congestion Index"},
            )
            fig_line.update_traces(line_color="#6C7BFF", line_width=3)
            fig_line.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col_peak_2:
            st.subheader("Worst Hour per Borough")
            conn = get_db()
            worst_borough_df = pd.read_sql_query(
                """
                WITH borough_hourly AS (
                    SELECT z.borough,
                           CAST(strftime('%H', zh.hour_ts) AS INTEGER) as hr,
                           AVG(zh.congestion_index) as cong
                    FROM zone_hourly zh
                    JOIN zones z ON zh.zone_id = z.zone_id
                    WHERE zh.congestion_index IS NOT NULL AND z.borough != 'Unknown'
                    GROUP BY z.borough, hr
                ),
                ranked AS (
                    SELECT borough, hr, cong,
                           ROW_NUMBER() OVER (PARTITION BY borough ORDER BY cong DESC) as rn
                    FROM borough_hourly
                )
                SELECT borough, hr, cong FROM ranked WHERE rn = 1 ORDER BY cong DESC
                """,
                conn,
            )
            worst_borough_df["hour_label"] = worst_borough_df["hr"].apply(lambda h: f"{h:02d}:00")
            fig_bar = px.bar(
                worst_borough_df,
                x="borough",
                y="cong",
                color="cong",
                text="hour_label",
                color_continuous_scale=["#6C7BFF", "#ff5d5d"],
                labels={"borough": "Borough", "cong": "Peak Congestion"},
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hourly data available for this selection.")


# =========================================================
# TAB 4: 48H FORECAST
# =========================================================
with tab_forecast:
    st.header("48-Hour Congestion Forecasting & Evaluation")

    top_zones_df = load_top_zones()
    forecast_options = {"City-Wide (All Zones)": 0}
    for _, r in top_zones_df.iterrows():
        forecast_options[f"{r['zone_name']} ({r['borough']})"] = int(r["zone_id"])

    sel_forecast_name = st.selectbox("Select Zone for Forecast", list(forecast_options.keys()), index=0)
    sel_forecast_id = forecast_options[sel_forecast_name]

    f_data, e_data = load_forecasts(sel_forecast_id)

    if not e_data.empty:
        eval_row = e_data.iloc[0]
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">SARIMAX MAE</div>
                    <div class="metric-value">{eval_row['model_mae']:.4f}</div>
                    <div class="metric-subtitle">sMAPE: {eval_row['model_smape']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Seasonal Baseline MAE</div>
                    <div class="metric-value">{eval_row['baseline_mae']:.4f}</div>
                    <div class="metric-subtitle">sMAPE: {eval_row['baseline_smape']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Holdout Winner</div>
                    <div class="metric-value" style="color: {'#6C7BFF' if eval_row['winner'] == 'SARIMAX' else '#ffb86c'};">{eval_row['winner']}</div>
                    <div class="metric-subtitle">Honest time-based test</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mc4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Holdout Window</div>
                    <div class="metric-value" style="font-size: 1.35rem;">Last 7 Days</div>
                    <div class="metric-subtitle">168 unseen hourly steps</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("ℹ️ **Honest Evaluation Note**: In recurrent traffic patterns with strict weekly periodicity, seasonal naive baselines (repeating the exact hour of the previous week) provide a strong benchmark that low-order parametric SARIMA models often find challenging to outperform over multi-step horizons.")

    if not f_data.empty:
        fig_fc = go.Figure()

        # Historical actual
        hist_data = f_data[f_data["is_future"] == 0]
        fut_data = f_data[f_data["is_future"] == 1]

        fig_fc.add_trace(
            go.Scatter(
                x=hist_data["hour_ts"],
                y=hist_data["actual_congestion"],
                mode="lines",
                name="Actual Congestion (History)",
                line=dict(color="#9BA3B8", width=1.5),
            )
        )

        # In-sample fitted
        fig_fc.add_trace(
            go.Scatter(
                x=hist_data["hour_ts"],
                y=hist_data["forecast_congestion"],
                mode="lines",
                name="Fitted In-Sample",
                line=dict(color="#6C7BFF", width=1, dash="dot"),
                opacity=0.6,
            )
        )

        # Confidence intervals for future
        if not fut_data.empty:
            fig_fc.add_trace(
                go.Scatter(
                    x=fut_data["hour_ts"].tolist() + fut_data["hour_ts"].tolist()[::-1],
                    y=fut_data["upper_ci"].tolist() + fut_data["lower_ci"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(108, 123, 255, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence Interval",
                    showlegend=True,
                )
            )

            # Future forecast line
            fig_fc.add_trace(
                go.Scatter(
                    x=fut_data["hour_ts"],
                    y=fut_data["forecast_congestion"],
                    mode="lines+markers",
                    name="48h Ahead Forecast",
                    line=dict(color="#FFB86C", width=3, dash="dash"),
                    marker=dict(size=4),
                )
            )

        # Anomalies
        anom_data = hist_data[hist_data["is_anomaly"] == 1]
        if not anom_data.empty:
            fig_fc.add_trace(
                go.Scatter(
                    x=anom_data["hour_ts"],
                    y=anom_data["actual_congestion"],
                    mode="markers",
                    name="Anomaly (>2 Residual σ)",
                    marker=dict(color="#FF5D5D", size=8, symbol="circle-open", line=dict(width=2)),
                )
            )

        fig_fc.update_layout(
            title=f"SARIMAX 48-Hour Congestion Projection: {sel_forecast_name}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=420,
            xaxis_title="Timeline",
            yaxis_title="Congestion Index (0 to 1)",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)


# =========================================================
# TAB 5: CRASHES & SAFETY
# =========================================================
with tab_crashes:
    st.header("Collision Density & Contributing Factors")

    c_density, c_hourly, c_factors, _ = load_crashes_summary()

    col_crash_1, col_crash_2 = st.columns([3, 2])

    with col_crash_1:
        st.subheader("Top Collision Zones in NYC")
        if not c_density.empty:
            disp_c = c_density.head(10)[["crash_rank", "zone_name", "borough", "total_crashes", "total_injured", "total_killed"]].copy()
            disp_c.columns = ["Rank", "Zone Name", "Borough", "Total Crashes", "Injured", "Killed"]
            st.dataframe(disp_c, hide_index=True, use_container_width=True)
        else:
            st.info("No crash density data available.")

    with col_crash_2:
        st.subheader("Top Contributing Factors")
        if not c_factors.empty:
            fig_fac = px.bar(
                c_factors,
                y="factor",
                x="count",
                orientation="h",
                color="count",
                color_continuous_scale=["#6C7BFF", "#ff5d5d"],
                labels={"count": "Incidents", "factor": "Factor"},
            )
            fig_fac.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_fac, use_container_width=True)

    st.subheader("Collisions by Hour of Day")
    if not c_hourly.empty:
        fig_chr = px.bar(
            c_hourly,
            x="crash_hour",
            y=["total_injured", "total_crashes"],
            barmode="group",
            labels={"crash_hour": "Hour of Day (0-23)", "value": "Count", "variable": "Metric"},
            color_discrete_map={"total_crashes": "#6C7BFF", "total_injured": "#FFB86C"},
        )
        fig_chr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig_chr, use_container_width=True)

    st.info(
        "🧠 **Statistical Caution**: Correlation does not equal causation. "
        "Areas with the highest crash counts are frequently broad arterial roadways with high vehicle throughput "
        "rather than the slowest gridlocked core streets."
    )


# =========================================================
# TAB 6: DATA & METHOD
# =========================================================
with tab_method:
    st.header("Methodology, Limitations & Reproducibility")

    st.markdown(
        r"""
        ### 1. Data Sources (Free Public Data)
        1. **NYC TLC Yellow Taxi Trip Records**:
           - Source: Official NYC Taxi & Limousine Commission Parquet data on AWS CloudFront.
           - Schema columns utilized: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, `trip_distance`, `passenger_count`.
        2. **NYC Taxi Zones Boundary Shapefile**:
           - Source: Official TLC taxi zones shapefile reprojected to EPSG:4326.
        3. **NYC Motor Vehicle Collisions (Crashes)**:
           - Source: NYC Open Data (Socrata API, dataset `h9gi-nx95`), filtered to the identical 2-month window.

        ### 2. Cleaning & Filtration Protocol
        Raw trip records were processed in bounded row-group batches (300k rows/batch) applying strict validation:
        - Trips with duration $< 1$ minute or $> 180$ minutes dropped.
        - Distance $\le 0.1$ miles or $> 100$ miles dropped.
        - Implied trip average speed $> 80$ mph dropped as physical GPS or recording errors.
        - Nulls in location, timestamp, or distance dropped.

        ### 3. Congestion Index Formulation
        For each TLC Taxi Zone $z$ and hour timestamp $t$:
        $$\\text{speed}_{z,t} = \\frac{1}{N_{z,t}} \\sum_{i=1}^{N_{z,t}} \\frac{\\text{distance}_i}{\\text{duration}_i}$$
        
        The **Free-Flow Speed** $\\text{FFS}_z$ is established as the **90th percentile of hourly speeds during off-peak hours** (22:00 to 05:00):
        $$\\text{FFS}_z = P_{90}\\left(\\left\\{ \\text{speed}_{z,t} \\mid t \\in \\text{OffPeak} \\right\\}\\right)$$

        The normalized **Congestion Index** $C_{z,t} \\in [0, 1]$ measures relative speed deficit:
        $$C_{z,t} = \\operatorname{clip}\\left(1 - \\frac{\\text{speed}_{z,t}}{\\text{FFS}_z}, 0, 1\\right), \\quad \\text{for } N_{z,t} \\ge 10$$

        ### 4. Known Limitations
        - **Taxi Sample Proxy**: TLC trip records reflect yellow medallion taxi patterns, which are overrepresented in Manhattan and at international airports (JFK, LGA).
        - **Exogenous Variables**: The baseline SARIMAX model does not currently ingest real-time precipitation, weather events, or transit disruptions.
        - **Time Scope**: Aggregated over a 2-month operational window (August – September 2026).

        ### 5. How to Rebuild
        ```bash
        # 1. Activate venv
        source .venv/bin/activate  # or .\\.venv\\Scripts\\activate on Windows

        # 2. Re-run pipeline (extracts, cleans, and builds data/urban.db)
        python -m src.pipeline.run

        # 3. Precompute SARIMAX forecasts and evaluate holdout
        python -m src.forecast.train

        # 4. Launch Streamlit UI
        streamlit run app/streamlit_app.py
        ```
        """
    )
