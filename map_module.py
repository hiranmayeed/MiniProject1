"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Module: map_module.py — Telangana Choropleth Heatmap
=============================================================================
Purpose : Generates an interactive Folium choropleth map of Telangana
          districts coloured by Departure from LPA (Kharif) or any
          numeric metric column. Highlights the user-selected district
          with a thick border and auto-zooms to its bounds.

Usage in app.py:
    from map_module import create_state_map, load_telangana_geojson

Integration:
    pip install folium streamlit-folium requests
=============================================================================
"""

import folium
import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import FloatImage
import branca.colormap as cm


# =============================================================================
# SECTION 1 — CONSTANTS
# =============================================================================

# Geographic center of Telangana state (latitude, longitude)
# Used to initialise the map view before any district is selected.
TELANGANA_CENTER = [17.8496, 79.1151]
TELANGANA_DEFAULT_ZOOM = 7

# GeoJSON sources tried in order — first success wins.
# Multiple sources guard against any single URL going down.
GEOJSON_URLS = [
    # India GeoJSON repo — district level, filtered to Telangana
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    # Datameet India districts
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/Telangana.geojson",
    # rforindia alternative
    "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States",
]

# Departure from LPA colour thresholds (IMD classification)
# Used to build the legend and colour bins.
DEPARTURE_BINS   = [-100, -20, -5, 5, 20, 100]
DEPARTURE_COLORS = ["#d73027", "#fc8d59", "#ffffbf", "#91cf60", "#1a9850"]
DEPARTURE_LABELS = ["Large Deficit (<-20%)", "Below Normal (-20 to -5%)",
                    "Normal (-5 to +5%)",    "Above Normal (+5 to +20%)",
                    "Large Excess (>+20%)"]


# =============================================================================
# SECTION 2 — GEOJSON LOADER
# =============================================================================

@st.cache_data(ttl=86400, show_spinner="Loading Telangana district boundaries...")
def load_telangana_geojson() -> dict:
    """
    Loads the Telangana district boundary GeoJSON from a public URL.
    Tries multiple sources in order. Falls back to embedded centroid
    data (point-based map) if all URLs fail.

    Returns:
        GeoJSON dict with Telangana district polygons, or
        None if all sources failed (caller falls back gracefully).

    The result is cached for 24 hours (ttl=86400 seconds) so the
    network call only happens once per day per user session.
    """

    for url in GEOJSON_URLS:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            geojson = response.json()

            # The geohacker/india repo has all Indian districts —
            # filter to Telangana only by state name property
            if "india_district" in url:
                geojson = _filter_telangana_features(geojson)

            # Validate we actually got district polygons
            if geojson and len(geojson.get("features", [])) > 0:
                print(f"✅ GeoJSON loaded from: {url} "
                      f"({len(geojson['features'])} features)")
                return geojson

        except Exception as e:
            print(f"⚠️  GeoJSON source failed ({url}): {e}")
            continue

    # All URLs failed — return None, caller will use circle markers fallback
    print("❌ All GeoJSON sources failed. Map will use marker fallback.")
    return None


def _filter_telangana_features(geojson: dict) -> dict:
    """
    Filters a national district GeoJSON to Telangana features only.
    Checks common property keys used for state name across different
    GeoJSON sources (ST_NM, state, STATE, statename).
    """
    telangana_variants = {"telangana", "telegana"}  # handle common misspellings

    filtered_features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})

        # Check multiple possible state name keys
        for key in ["ST_NM", "state", "STATE", "statename", "State"]:
            state_val = props.get(key, "")
            if state_val and state_val.lower() in telangana_variants:
                filtered_features.append(feature)
                break

    return {"type": "FeatureCollection", "features": filtered_features}


def _get_district_name_from_feature(feature: dict) -> str:
    """
    Extracts the district name from a GeoJSON feature's properties.
    Tries multiple common property key names used across different
    GeoJSON sources.
    """
    props = feature.get("properties", {})
    for key in ["dtname", "DISTRICT", "district", "District",
                "NAME_2", "name", "dt_name", "DT_NAME"]:
        val = props.get(key)
        if val:
            return str(val).strip().title()
    return "Unknown"


# =============================================================================
# SECTION 3 — NAME MATCHING UTILITY
# =============================================================================

def _match_district_name(geojson_name: str,
                          df_districts: list) -> str:
    """
    Fuzzy-matches a GeoJSON district name to the closest name in the
    DataFrame's district list. Handles minor spelling differences between
    the GeoJSON source and your IMD CSV data.

    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Substring match (one contains the other)
      3. Word overlap score (highest overlap wins)

    Returns the best matching DataFrame district name, or None.
    """
    geojson_clean = geojson_name.lower().strip()
    df_lower      = {d.lower().strip(): d for d in df_districts}

    # 1. Exact match
    if geojson_clean in df_lower:
        return df_lower[geojson_clean]

    # 2. Substring match
    for df_name_lower, df_name_orig in df_lower.items():
        if geojson_clean in df_name_lower or df_name_lower in geojson_clean:
            return df_name_orig

    # 3. Word overlap score
    geojson_words = set(geojson_clean.split())
    best_match    = None
    best_score    = 0

    for df_name_lower, df_name_orig in df_lower.items():
        df_words = set(df_name_lower.split())
        overlap  = len(geojson_words & df_words)
        if overlap > best_score:
            best_score = overlap
            best_match = df_name_orig

    return best_match if best_score > 0 else None


# =============================================================================
# SECTION 4 — COLOUR MAPPING
# =============================================================================

def _get_departure_color(value: float) -> str:
    """
    Maps a departure % value to a hex colour following IMD's classification.
    Used to colour each district polygon on the choropleth.

    Colour scale (RdYlGn diverging):
        Large Deficit  (<-20%) → Deep Red    #d73027
        Below Normal   (-20 to -5%) → Orange #fc8d59
        Normal         (-5 to +5%) → Yellow  #ffffbf
        Above Normal   (+5 to +20%) → Lt Green #91cf60
        Large Excess   (>+20%) → Deep Green  #1a9850
        No data        → Gray               #cccccc
    """
    if value is None or pd.isna(value):
        return "#cccccc"

    for i, (low, high) in enumerate(zip(DEPARTURE_BINS[:-1], DEPARTURE_BINS[1:])):
        if low <= value < high:
            return DEPARTURE_COLORS[i]

    # Handle value exactly at +100
    return DEPARTURE_COLORS[-1]


def _build_colormap() -> cm.StepColormap:
    """
    Builds a Branca StepColormap that matches the IMD departure bins.
    This colormap is added to the Folium map as a visible legend/colorbar.
    """
    colormap = cm.StepColormap(
        colors  = DEPARTURE_COLORS,
        vmin    = -100,
        vmax    = 100,
        index   = DEPARTURE_BINS,
        caption = "Departure from LPA (%)"
    )
    return colormap


# =============================================================================
# SECTION 5 — CORE MAP FUNCTION
# =============================================================================

def create_state_map(df: pd.DataFrame,
                     selected_district: str,
                     metric_col: str = "departure_pct",
                     season: str = "Kharif",
                     year: int = None) -> folium.Map:
    """
    Creates an interactive Folium choropleth map of Telangana districts.

    Each district polygon is:
      - Coloured by its metric value (departure from LPA by default).
      - Given a tooltip showing district name + key metrics on hover.
      - Highlighted with a thick yellow border if it's the selected district.

    The map auto-zooms to the selected district's bounds.

    Args:
        df               : DataFrame from 03_seasonal_with_departure.csv,
                           filtered to the desired season and year.
                           Must contain 'district_name' and metric_col columns.
        selected_district: The district currently selected in the sidebar.
        metric_col       : Column name to colour districts by.
                           Default: 'departure_pct' (Departure from LPA %).
        season           : Season label for display in tooltips ("Kharif" etc).
        year             : Year for display in tooltips. Uses latest if None.

    Returns:
        folium.Map object ready to be rendered by st_folium().

    Example (in app.py):
        seasonal_filtered = seasonal_df[
            (seasonal_df["season"] == "Kharif") &
            (seasonal_df["year"] == 2023)
        ]
        m = create_state_map(seasonal_filtered, selected_district)
        st_folium(m, use_container_width=True, height=480)
    """

    # ── Prepare the metric data ────────────────────────────────────────────
    # Build a lookup dict: district_name → metric value
    # This is used to colour each polygon as we iterate over GeoJSON features.
    if year is None and "year" in df.columns:
        year = int(df["year"].max())

    # Filter to the right season+year rows
    plot_df = df.copy()
    if "season" in plot_df.columns:
        plot_df = plot_df[plot_df["season"] == season]
    if year and "year" in plot_df.columns:
        plot_df = plot_df[plot_df["year"] == year]

    # Build lookup: district_name (lowercased) → row data
    metric_lookup = {}
    for _, row in plot_df.iterrows():
        key = row["district_name"].lower().strip()
        metric_lookup[key] = row

    # ── Load GeoJSON ───────────────────────────────────────────────────────
    geojson = load_telangana_geojson()

    # ── Initialise base map ────────────────────────────────────────────────
    # CartoDB Positron is a clean light basemap — minimal visual noise so
    # the choropleth colours read clearly against the background.
    m = folium.Map(
        location = TELANGANA_CENTER,
        zoom_start= TELANGANA_DEFAULT_ZOOM,
        tiles    = "CartoDB positron",
        prefer_canvas = True    # canvas renderer is faster for many polygons
    )

    # ── Render map: choropleth or fallback markers ─────────────────────────
    selected_bounds = None    # will be set when we find the selected district

    if geojson:
        selected_bounds = _add_choropleth_layer(
            m, geojson, metric_lookup, metric_col,
            selected_district, season, year
        )
    else:
        # GeoJSON unavailable — fall back to circle markers at centroids
        selected_bounds = _add_marker_fallback(
            m, plot_df, metric_col, selected_district, season
        )

    # ── Auto-zoom to selected district ────────────────────────────────────
    if selected_bounds:
        # fit_bounds takes [[south, west], [north, east]]
        m.fit_bounds(selected_bounds, padding=[20, 20])
    # If no bounds found, map stays at full Telangana view

    # ── Add colormap legend ────────────────────────────────────────────────
    colormap = _build_colormap()
    colormap.add_to(m)

    # ── Add title overlay ─────────────────────────────────────────────────
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background: white; padding: 6px 16px; border-radius: 6px;
                border: 1px solid #ddd; font-family: sans-serif;
                font-size: 13px; font-weight: 600; color: #333; z-index: 1000;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
        Telangana — {season} {year} Departure from LPA
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def _add_choropleth_layer(m: folium.Map,
                           geojson: dict,
                           metric_lookup: dict,
                           metric_col: str,
                           selected_district: str,
                           season: str,
                           year: int) -> list:
    """
    Adds GeoJSON polygon layers to the Folium map — one per district.
    Returns the bounding box [[S,W],[N,E]] of the selected district
    for auto-zoom, or None if the selected district wasn't found.

    We use individual GeoJSON layers (not folium.Choropleth) because
    individual layers give us per-feature style control — critical for
    the highlighted selected-district border.
    """
    selected_bounds = None
    districts_matched = 0

    for feature in geojson.get("features", []):

        # Get district name from GeoJSON feature properties
        geojson_name = _get_district_name_from_feature(feature)

        # Match GeoJSON name to DataFrame district name
        df_districts = list(metric_lookup.keys())
        matched_name = _match_district_name(geojson_name, df_districts)

        # Get metric value and row data for this district
        row_data  = metric_lookup.get(matched_name) if matched_name else None
        value     = float(row_data[metric_col]) \
                    if row_data is not None and not pd.isna(row_data[metric_col]) \
                    else None
        fill_color = _get_departure_color(value)

        # Determine if this is the selected district
        is_selected = (
            geojson_name.lower().strip() ==
            selected_district.lower().strip()
        ) or (
            matched_name and
            matched_name.lower().strip() ==
            selected_district.lower().strip()
        )

        # Style: selected district gets thick yellow border + full opacity
        if is_selected:
            style = {
                "fillColor"  : fill_color,
                "color"      : "#FFD700",   # gold/yellow highlight border
                "weight"     : 4,           # thick border
                "fillOpacity": 0.85,
                "opacity"    : 1.0,
            }
        else:
            style = {
                "fillColor"  : fill_color,
                "color"      : "#555555",   # standard thin grey border
                "weight"     : 0.8,
                "fillOpacity": 0.65,
                "opacity"    : 0.8,
            }

        # Build tooltip HTML — shown on hover
        if row_data is not None:
            departure = row_data.get("departure_pct", "N/A")
            total_mm  = row_data.get("total_rainfall_mm", "N/A")
            lpa_mm    = row_data.get("lpa_mm", "N/A")
            category  = row_data.get("anomaly_category", "N/A")

            dep_str   = f"{departure:+.1f}%" if isinstance(departure, float) else departure
            tot_str   = f"{total_mm:.0f} mm"  if isinstance(total_mm, float)  else total_mm
            lpa_str   = f"{lpa_mm:.0f} mm"    if isinstance(lpa_mm, float)    else lpa_mm

            tooltip_html = f"""
            <div style="font-family:sans-serif; font-size:13px; min-width:180px">
                <b style="font-size:14px">{geojson_name}</b><br>
                <hr style="margin:4px 0; border-color:#eee">
                <b>Season:</b> {season} {year}<br>
                <b>Departure from LPA:</b>
                    <span style="color:{'#2d7a4f' if isinstance(departure, float) and departure > 5
                                        else '#b03030' if isinstance(departure, float) and departure < -5
                                        else '#c07a00'}">{dep_str}</span><br>
                <b>Actual Rainfall:</b> {tot_str}<br>
                <b>LPA:</b> {lpa_str}<br>
                <b>Category:</b> {category}
                {'<br><b style="color:#FFD700">★ Selected District</b>' if is_selected else ''}
            </div>
            """
        else:
            tooltip_html = f"""
            <div style="font-family:sans-serif; font-size:13px">
                <b>{geojson_name}</b><br>
                <i style="color:#999">No data available</i>
            </div>
            """

        # Add the district as a GeoJSON layer with style + tooltip
        folium.GeoJson(
            feature,
            style_function    = lambda x, s=style: s,
            highlight_function= lambda x: {
                "weight"     : 3,
                "color"      : "#333333",
                "fillOpacity": 0.9,
            },
            tooltip = folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(m)

        if row_data is not None:
            districts_matched += 1

        # Capture the bounding box of the selected district for auto-zoom
        if is_selected:
            selected_bounds = _get_feature_bounds(feature)

    print(f"Choropleth: matched {districts_matched} districts to metric data.")
    return selected_bounds


def _add_marker_fallback(m: folium.Map,
                          df: pd.DataFrame,
                          metric_col: str,
                          selected_district: str,
                          season: str) -> list:
    """
    Fallback when GeoJSON is unavailable. Renders circle markers at
    district centroid coordinates. Less visually rich than choropleth
    but fully functional.
    """

    selected_bounds = None

    for _, row in df.iterrows():
        district = row.get("district_name", "Unknown")
        value    = row.get(metric_col)
        color    = _get_departure_color(value)
        is_sel   = district.lower().strip() == selected_district.lower().strip()

        # Look up centroid for this district
        centroid = TELANGANA_CENTROIDS.get(district)
        if not centroid:
            continue

        lat, lon = centroid

        folium.CircleMarker(
            location    = [lat, lon],
            radius      = 14 if is_sel else 10,
            color       = "#FFD700" if is_sel else "#555",
            weight      = 4 if is_sel else 1,
            fill        = True,
            fill_color  = color,
            fill_opacity= 0.85,
            tooltip     = folium.Tooltip(
                f"<b>{district}</b><br>"
                f"Departure: {value:+.1f}%<br>"
                f"Season: {season}"
            )
        ).add_to(m)

        if is_sel:
            # Approximate bounding box from centroid (±0.5°)
            selected_bounds = [[lat - 0.5, lon - 0.5], [lat + 0.5, lon + 0.5]]

    return selected_bounds


def _get_feature_bounds(feature: dict) -> list:
    """
    Computes the bounding box [[south, west], [north, east]] of a
    GeoJSON feature by scanning all coordinate pairs in its geometry.
    Works for both Polygon and MultiPolygon geometries.
    """
    try:
        geometry = feature.get("geometry", {})
        geo_type = geometry.get("type", "")
        coords   = geometry.get("coordinates", [])

        all_lons, all_lats = [], []

        def extract_coords(coord_list):
            """Recursively flatten nested coordinate arrays."""
            for item in coord_list:
                if isinstance(item[0], (int, float)):
                    all_lons.append(item[0])
                    all_lats.append(item[1])
                else:
                    extract_coords(item)

        extract_coords(coords)

        if all_lats and all_lons:
            return [
                [min(all_lats), min(all_lons)],   # [south, west]
                [max(all_lats), max(all_lons)]    # [north, east]
            ]
    except Exception:
        pass
    return None


# =============================================================================
# SECTION 6 — TELANGANA DISTRICT CENTROIDS (fallback coordinates)
# =============================================================================
# Used when GeoJSON polygon data is unavailable.
# Coordinates: (latitude, longitude) of approximate district centers.

TELANGANA_CENTROIDS = {
    "Adilabad"               : (19.664, 78.532),
    "Bhadradri Kothagudem"   : (17.555, 80.620),
    "Hyderabad"              : (17.385, 78.486),
    "Jagitial"               : (18.795, 78.917),
    "Jangaon"                : (17.723, 79.152),
    "Jayashankar Bhupalapally": (18.476, 79.888),
    "Jogulamba Gadwal"        : (16.234, 77.803),
    "Kamareddy"              : (18.322, 78.340),
    "Karimnagar"             : (18.438, 79.128),
    "Khammam"                : (17.247, 80.150),
    "Kumuram Bheem Asifabad" : (19.366, 79.282),
    "Mahabubabad"            : (17.603, 80.014),
    "Mahabubnagar"           : (16.737, 77.983),
    "Mancherial"             : (18.875, 79.458),
    "Medak"                  : (18.048, 78.263),
    "Medchal Malkajgiri"     : (17.632, 78.560),
    "Mulugu"                 : (18.193, 80.236),
    "Nagarkurnool"           : (16.481, 78.325),
    "Nalgonda"               : (17.058, 79.267),
    "Narayanpet"             : (16.741, 77.496),
    "Nirmal"                 : (19.096, 78.344),
    "Nizamabad"              : (18.672, 78.094),
    "Peddapalli"             : (18.614, 79.384),
    "Rajanna Sircilla"       : (18.386, 78.833),
    "Ranga Reddy"            : (17.235, 78.068),
    "Sangareddy"             : (17.619, 78.086),
    "Siddipet"               : (18.100, 78.852),
    "Suryapet"               : (17.141, 79.622),
    "Vikarabad"              : (17.335, 77.904),
    "Wanaparthy"             : (16.362, 78.065),
    "Warangal (Rural)"       : (18.001, 79.588),
    "Warangal (Urban)"       : (17.977, 79.600),
    "Yadadri Bhuvanagiri"    : (17.582, 79.143),
}


# =============================================================================
# SECTION 7 — STREAMLIT RENDER HELPER
# =============================================================================

def render_map_section(seasonal_df: pd.DataFrame,
                        selected_district: str,
                        selected_season: str = "Kharif") -> None:
    """
    Complete self-contained function to render the map section inside
    the Streamlit app. Call this from main() in app.py.

    Handles:
      - Season and year selector controls
      - Reset Zoom button
      - Map rendering via st_folium
      - Responsive height

    Args:
        seasonal_df       : Full seasonal DataFrame from the pipeline.
        selected_district : District currently selected in the sidebar.
        selected_season   : Season to display by default ("Kharif").
    """

    st.markdown(
        '<div class="section-header">State-wide Rainfall Heatmap</div>',
        unsafe_allow_html=True
    )

    # ── Map controls row ──────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

    with ctrl_col1:
        available_seasons = sorted(seasonal_df["season"].unique().tolist())
        map_season = st.selectbox(
            "Season",
            available_seasons,
            index = available_seasons.index(selected_season)
                    if selected_season in available_seasons else 0,
            key   = "map_season_selector"
        )

    with ctrl_col2:
        season_years = sorted(
            seasonal_df[seasonal_df["season"] == map_season]["year"]
            .unique().tolist(),
            reverse=True
        )
        map_year = st.selectbox(
            "Year",
            season_years,
            index = 0,    # default to most recent year
            key   = "map_year_selector"
        )

    with ctrl_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        reset_zoom = st.button("↺ Reset Zoom", key="map_reset_zoom")

    # ── Filter data for selected season + year ────────────────────────────
    map_df = seasonal_df[
        (seasonal_df["season"] == map_season) &
        (seasonal_df["year"]   == map_year)
    ].copy()

    if map_df.empty:
        st.warning(f"No data available for {map_season} {map_year}.")
        return

    # ── Build the map ─────────────────────────────────────────────────────
    # When reset_zoom is clicked, pass None as selected_district so the
    # map initialises at full Telangana view instead of zooming in.
    district_for_map = None if reset_zoom else selected_district

    try:
        folium_map = create_state_map(
            df                = map_df,
            selected_district = district_for_map or "",
            metric_col        = "departure_pct",
            season            = map_season,
            year              = int(map_year)
        )

        # st_folium renders the Folium map inside Streamlit.
        # returned_objects captures click events for future interactivity.
        # use_container_width=True makes it fill the column width.
        map_output = st_folium(
            folium_map,
            use_container_width = True,
            height              = 480,
            returned_objects    = ["last_object_clicked_tooltip"],
            key                 = f"telangana_map_{map_season}_{map_year}"
        )

    except Exception as e:
        st.error(
            f"Map rendering failed: {e}\n\n"
            "Ensure `folium` and `streamlit-folium` are installed:\n"
            "`pip install folium streamlit-folium`"
        )
        return

    # ── Data summary below the map ────────────────────────────────────────
    with st.expander(f"📋 {map_season} {map_year} — All Districts Summary"):
        summary = map_df[[
            "district_name", "total_rainfall_mm",
            "lpa_mm", "departure_pct", "anomaly_category"
        ]].sort_values("departure_pct", ascending=False).reset_index(drop=True)

        summary.columns = [
            "District", "Rainfall (mm)", "LPA (mm)",
            "Departure (%)", "Category"
        ]

        def colour_category(val):
            colours = {
                "Large Excess"  : "background-color:#c8f7c5",
                "Above Normal"  : "background-color:#d4edda",
                "Normal"        : "background-color:#fff9c4",
                "Below Normal"  : "background-color:#fde8d8",
                "Large Deficit" : "background-color:#f9c0c0",
            }
            return colours.get(val, "")

        styled = (
            summary.style
            .applymap(colour_category, subset=["Category"])
            .format({
                "Rainfall (mm)": "{:.0f}",
                "LPA (mm)"     : "{:.0f}",
                "Departure (%)": "{:+.1f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, height=280)