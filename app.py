"""
CS 230 Final Project — NASA Fireballs (1988–2026) Explorer
Author: Naban Ahmed
Section: 7
Dataset: NASA Fireballs (1988–2026) — nasa_fireballs.xlsx (sheet: 'in')
Streamlit URL: [Add your Streamlit Community Cloud URL after deploying]
Description: Interactive Streamlit app for a high school astronomy teacher to explore
NASA fireball events (1988–2026), atmospheric entry energy vs. calculated impact
energy (kt), yearly counts, and largest events across customizable year ranges,
impact energy thresholds, and hemispheres.
LLM(s) Used: Gemini
References: NASA/JPL CNEOS Fireball Data (nasa_fireballs.xlsx); Streamlit, Pandas,
OpenPyXL, Matplotlib, and PyDeck documentation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="NASA Fireballs Explorer (1988–2026)",
    page_icon="☄️",
    layout="wide",
)

DATA_PATHS = [
    Path("nasa_fireballs.xlsx"),
    Path("data/nasa_fireballs.xlsx"),
]

# #[DICT_DRIVEN] AUGMENT – Added display labels, descriptions, and hemisphere
# sign rules for 'Both', 'Northern', and 'Southern' from nasa_fireballs.xlsx.
HEMISPHERE_CONFIG = {
    "Both": {
        "label": "Both Hemispheres (N & S)",
        "description": "All recorded fireball events across northern and southern latitudes.",
    },
    "Northern": {
        "label": "Northern Hemisphere (lat-dir == 'N')",
        "description": "Fireballs north of the equator (signed latitude >= 0°).",
    },
    "Southern": {
        "label": "Southern Hemisphere (lat-dir == 'S')",
        "description": "Fireballs south of the equator (signed latitude < 0°).",
    },
}


# #[STR_CLEAN] REVISE / REWRITE – Combines separate numeric 'lat'/'lon' and
# directional 'lat-dir'/'lon-dir' columns from nasa_fireballs.xlsx into signed floats.
def combine_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw lat/lat-dir and lon/lon-dir columns into signed latitude/longitude."""
    out = df.copy()
    lat_num = pd.to_numeric(out["lat"], errors="coerce")
    lon_num = pd.to_numeric(out["lon"], errors="coerce")

    lat_dir = out["lat-dir"].astype(str).str.strip().str.upper()
    lon_dir = out["lon-dir"].astype(str).str.strip().str.upper()

    lat_sign = np.where(lat_dir == "S", -1.0, 1.0)
    lon_sign = np.where(lon_dir == "W", -1.0, 1.0)

    out["latitude"] = lat_num * lat_sign
    out["longitude"] = lon_num * lon_sign
    out["coord_label"] = (
        lat_num.round(1).astype(str)
        + lat_dir
        + ", "
        + lon_num.round(1).astype(str)
        + lon_dir
    )
    return out


@st.cache_data
def load_fireballs_data() -> pd.DataFrame:
    """Load and clean nasa_fireballs.xlsx using a relative path."""
    excel_path = next((p for p in DATA_PATHS if p.exists()), None)
    if excel_path is None:
        raise FileNotFoundError(
            "Could not find nasa_fireballs.xlsx in the project root or data/ folder."
        )

    xls = pd.ExcelFile(excel_path)
    sheet_name = "in" if "in" in xls.sheet_names else xls.sheet_names[0]
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)

    df = combine_coordinates(raw)
    df["peak_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["peak_dt", "latitude", "longitude", "impact-e"]).copy()
    df["year"] = df["peak_dt"].dt.year.astype(int)
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df["impact-e"] = pd.to_numeric(df["impact-e"], errors="coerce")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
    df["vel"] = pd.to_numeric(df["vel"], errors="coerce")
    return df


# #[LOOP_VS_PANDAS] AUGMENT – Counts fireballs per year via a Python dictionary loop
# and verifies exact agreement with a Pandas value_counts/reindex calculation.
def yearly_counts_loop_vs_pandas(
    filtered_df: pd.DataFrame, start_year: int, end_year: int
) -> pd.DataFrame:
    """Return yearly fireball counts across [start_year, end_year] verified with Pandas."""
    counts_by_year = {yr: 0 for yr in range(start_year, end_year + 1)}
    for yr in filtered_df["year"]:
        if yr in counts_by_year:
            counts_by_year[yr] += 1

    pd_counts = (
        filtered_df["year"]
        .value_counts()
        .reindex(range(start_year, end_year + 1), fill_value=0)
    )
    assert all(counts_by_year[yr] == int(pd_counts.loc[yr]) for yr in counts_by_year), (
        "Discrepancy between loop counts and Pandas counts"
    )

    return pd.DataFrame(
        {
            "year": list(counts_by_year.keys()),
            "count": [counts_by_year[yr] for yr in counts_by_year],
        }
    )


# #[AI_REWRITE] REVISE / REWRITE – Replaced dense chained filter expression with
# explicit step-by-step boolean masks for year range, impact energy, and hemisphere.
def filter_fireballs(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    min_impact_e: float,
    hemisphere: str,
) -> pd.DataFrame:
    """Filter nasa_fireballs.xlsx rows by year range, minimum impact-e (kt), and hemisphere."""
    # Filter 1: Keep events within the selected start and end years
    mask = (df["year"] >= start_year) & (df["year"] <= end_year)

    # Filter 2: Keep events at or above the selected calculated impact energy (kt)
    mask = mask & (df["impact-e"] >= min_impact_e)

    # Filter 3: Apply hemisphere filter using signed latitude
    if hemisphere == "Northern":
        mask = mask & (df["latitude"] >= 0.0)
    elif hemisphere == "Southern":
        mask = mask & (df["latitude"] < 0.0)

    return df.loc[mask].copy()


def main() -> None:
    st.title("☄️ NASA Fireballs (1988–2026) Classroom Explorer")
    st.caption(
        "Built for a high school astronomy teacher exploring NASA CNEOS fireball records "
        "from nasa_fireballs.xlsx (1988–2026)."
    )

    try:
        df = load_fireballs_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Filter Controls")

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        # Control 1: Year range slider
        start_year, end_year = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min(2010, max_year), min(2020, max_year)),
        )

        # Control 2: Minimum calculated impact energy slider (kt)
        min_impact_e = st.slider(
            "Minimum Calculated Impact Energy (kt)",
            min_value=0.0,
            max_value=25.0,
            value=1.0,
            step=0.1,
        )

        # Control 3: Hemisphere radio buttons
        hemisphere = st.radio(
            "Hemisphere",
            options=list(HEMISPHERE_CONFIG.keys()),
            index=0,
        )

        # Control 4: Top N number input
        top_n = st.number_input(
            "Top N Largest Fireballs",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )

    st.info(
        f"**Current View:** {HEMISPHERE_CONFIG[hemisphere]['label']} | "
        f"**Years:** {start_year}–{end_year} | **Min Impact Energy:** {min_impact_e:.1f} kt. "
        f"*Data Caveat:* `alt` has 286 missing entries and `vel` has 525 missing entries out of "
        f"{len(df)} total rows in `nasa_fireballs.xlsx`, and detection depends on sensor coverage."
    )

    filtered = filter_fireballs(df, start_year, end_year, min_impact_e, hemisphere)

    if filtered.empty:
        st.warning(
            "No fireballs matched these filters. Try lowering Minimum Calculated Impact Energy (kt), "
            "expanding the Year Range, or selecting 'Both'."
        )
        st.stop()

    # #[LIST_COMP] IDENTIFY & EXPLAIN – Builds formatted event labels using a list comprehension
    # over matching rows so teachers have a quick summary string for each fireball.
    # Traces a 1-row input (2026-10-04, impact-e=0.18) -> ["2026-10-04 (0.18 kt)"].
    event_summaries = [
        f"{row.peak_dt.strftime('%Y-%m-%d')} ({row._4:.2f} kt)"
        if hasattr(row, "_4")
        else f"{row.peak_dt.strftime('%Y-%m-%d')} ({getattr(row, 'impact_e', 0.0):.2f} kt)"
        for row in filtered.itertuples()
    ]

    k1, k2, k3 = st.columns(3)
    k1.metric("Matching Fireballs", f"{len(filtered):,}")
    k2.metric("Peak Impact Energy (kt)", f"{filtered['impact-e'].max():.2f}")
    k3.metric("Median Impact Energy (kt)", f"{filtered['impact-e'].median():.2f}")

    # Q1: Map of fireball locations
    st.subheader(
        f"1. Where did fireballs hit with at least {min_impact_e:.1f} kt impact energy "
        f"between {start_year} and {end_year}?"
    )
    map_data = filtered.copy()
    map_data["date_str"] = map_data["peak_dt"].dt.strftime("%Y-%m-%d")
    map_data["radius_m"] = np.clip(np.sqrt(map_data["impact-e"]) * 220000, 120000, 950000)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_radius="radius_m",
        get_fill_color="[220, 76, 45, 180]",
        get_line_color="[255, 230, 170, 220]",
        line_width_min_pixels=1,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=10.0, longitude=0.0, zoom=0.75, pitch=0),
        tooltip={
            "html": "<b>Date:</b> {date_str}<br/><b>Impact Energy:</b> {impact-e} kt<br/>"
            "<b>Coords:</b> {coord_label}",
            "style": {"backgroundColor": "#141821", "color": "#FFFFFF"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)

    # Q2: Bar chart of fireball count per year
    st.subheader(
        f"2. How many fireballs were recorded each year between {start_year} and {end_year}?"
    )
    yearly_df = yearly_counts_loop_vs_pandas(filtered, start_year, end_year)

    fig1, ax1 = plt.subplots(figsize=(9, 3.8))
    ax1.bar(
        yearly_df["year"],
        yearly_df["count"],
        color="#2B6CB0",
        edgecolor="#1A365D",
        width=0.65,
    )
    ax1.set_title(
        f"Fireball Count per Year ({start_year}–{end_year}, >= {min_impact_e:.1f} kt, {hemisphere})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Fireballs")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig1)

    # Q3: Top N largest fireballs chart + table
    st.subheader(
        f"3. What are the top {int(top_n)} biggest fireballs in the {hemisphere.lower()} hemisphere?"
    )
    top_df = filtered.sort_values(by="impact-e", ascending=False).head(int(top_n)).copy()
    top_df["label"] = top_df["peak_dt"].dt.strftime("%Y-%m-%d") + " (" + top_df["coord_label"] + ")"

    fig2, ax2 = plt.subplots(figsize=(9, max(3.0, 0.35 * len(top_df) + 1.2)))
    ax2.barh(
        top_df["label"][::-1],
        top_df["impact-e"][::-1],
        color="#C05621",
        edgecolor="#7B341E",
    )
    ax2.set_title(
        f"Top {len(top_df)} Fireballs by Calculated Impact Energy (kt)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Calculated Impact Energy (kilotons)")
    ax2.set_ylabel("Date & Coordinates")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)

    display_cols = ["date", "coord_label", "impact-e", "energy", "alt", "vel"]
    st.dataframe(top_df[display_cols].reset_index(drop=True), use_container_width=True)

    with st.expander("Formatted Event Summaries"):
        st.write(event_summaries[:25])


if __name__ == "__main__":
    main()