"""
Map-based analyses for planecrashinfo_clean.csv.

Generates interactive HTML maps into the `maps/` directory:
  - accidents_by_country.html
  - accidents_by_country_decade.html (animated by decade)
  - fatality_ratio_by_country.html
  - accidents_scatter_geo.html (optional, if latitude/longitude exist)
"""

import os

import pandas as pd
import plotly.express as px

DATA_PATH = "planecrashinfo_clean.csv"
MAPS_DIR = "maps"


COUNTRY_FIX = {
    # United States variants
    "USA": "United States",
    "U.S.A.": "United States",
    "U.S.": "United States",
    "United States of America": "United States",
    "US": "United States",
    # UK variants
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "UK": "United Kingdom",
    # Russia / USSR
    "Russia": "Russian Federation",
    "Soviet Union": "Russia",
    # other common partials can be added as you spot them
}


def ensure_output_dir(path=MAPS_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} in current directory.")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Date → year / decade ---
    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    elif "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT

    df["year"] = df["date_parsed"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    # --- Fatalities / aboard ---
    if "fatalities_total" in df.columns:
        df["fatalities_total"] = pd.to_numeric(df["fatalities_total"], errors="coerce")
    else:
        df["fatalities_total"] = pd.NA

    # total aboard (if not already there)
    if "aboard_total" in df.columns:
        df["aboard_total"] = pd.to_numeric(df["aboard_total"], errors="coerce")
    elif "aboard" in df.columns:
        df["aboard_total"] = (
            df["aboard"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        )
    else:
        df["aboard_total"] = pd.NA

    # fatality ratio (per accident)
    df["fatality_ratio"] = df["fatalities_total"] / df["aboard_total"]

    # location_country should exist from your cleaning pipeline
    if "location_country" not in df.columns:
        # fallback: if "country" exists, rename it
        if "country" in df.columns:
            df = df.rename(columns={"country": "location_country"})
        else:
            df["location_country"] = None

    # normalize country strings and apply COUNTRY_FIX
    df["location_country"] = (
        df["location_country"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    df["location_country"] = df["location_country"].replace(COUNTRY_FIX)

    return df


# --------------------- MAP BUILDERS --------------------- #


def map_accidents_by_country(df: pd.DataFrame, outdir: str):
    """
    Choropleth of total accidents per country.
    """
    subset = df.dropna(subset=["location_country"])
    if subset.empty:
        print("Skipping accidents_by_country (no location_country data).")
        return

    agg = subset.groupby("location_country").size().reset_index(name="accidents")

    if agg.empty:
        print("Skipping accidents_by_country (no grouped data).")
        return

    fig = px.choropleth(
        agg,
        locations="location_country",
        locationmode="country names",
        color="accidents",
        color_continuous_scale="Viridis",
        title="Total recorded accidents by country",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar_title="Accidents",
    )

    fname = os.path.join(outdir, "accidents_by_country.html")
    fig.write_html(fname)
    print(f"Saved {fname}")


def map_accidents_by_country_decade(df: pd.DataFrame, outdir: str):
    """
    Animated choropleth: accidents per country per decade.
    """
    subset = df.dropna(subset=["location_country", "decade"])
    if subset.empty:
        print("Skipping accidents_by_country_decade (no country/decade data).")
        return

    agg = (
        subset.groupby(["location_country", "decade"])
        .size()
        .reset_index(name="accidents")
    )

    if agg.empty:
        print("Skipping accidents_by_country_decade (no grouped data).")
        return

    fig = px.choropleth(
        agg,
        locations="location_country",
        locationmode="country names",
        color="accidents",
        hover_name="location_country",
        animation_frame="decade",
        color_continuous_scale="Plasma",
        title="Accidents by country over time (per decade)",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar_title="Accidents",
    )

    fname = os.path.join(outdir, "accidents_by_country_decade.html")
    fig.write_html(fname)
    print(f"Saved {fname}")


def map_fatality_ratio_by_country(df: pd.DataFrame, outdir: str):
    """
    Choropleth of country-level severity:
    (sum of fatalities) / (sum of aboard) per country.
    """
    subset = df.dropna(subset=["location_country", "fatalities_total", "aboard_total"])
    subset = subset[subset["aboard_total"] > 0]

    if subset.empty:
        print("Skipping fatality_ratio_by_country (no valid rows).")
        return

    agg = (
        subset.groupby("location_country")
        .agg(
            total_fatalities=("fatalities_total", "sum"),
            total_aboard=("aboard_total", "sum"),
            accidents=("fatalities_total", "size"),
        )
        .reset_index()
    )

    agg["country_fatality_ratio"] = agg["total_fatalities"] / agg["total_aboard"]

    if agg.empty:
        print("Skipping fatality_ratio_by_country (no grouped data).")
        return

    fig = px.choropleth(
        agg,
        locations="location_country",
        locationmode="country names",
        color="country_fatality_ratio",
        hover_name="location_country",
        hover_data={
            "country_fatality_ratio": ":.2f",
            "accidents": True,
            "total_fatalities": True,
            "total_aboard": True,
        },
        color_continuous_scale="Reds",
        range_color=(0, min(1.0, float(agg["country_fatality_ratio"].max()))),
        title="Country-level fatality ratio (Σ fatalities / Σ aboard)",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar_title="Fatalities / aboard",
    )

    fname = os.path.join(outdir, "fatality_ratio_by_country.html")
    fig.write_html(fname)
    print(f"Saved {fname}")


def map_scatter_if_latlon(df: pd.DataFrame, outdir: str):
    """
    Optional: scatter_geo of individual accidents if latitude/longitude columns exist.
    Looks for columns named 'lat'/'latitude' and 'lon'/'lng'/'longitude'.
    """
    lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon", "lng", "longitude")]

    if not lat_cols or not lon_cols:
        print("Skipping accidents_scatter_geo (no latitude/longitude columns found).")
        return

    lat_col = lat_cols[0]
    lon_col = lon_cols[0]

    subset = df.dropna(subset=[lat_col, lon_col, "fatalities_total"])
    if subset.empty:
        print("Skipping accidents_scatter_geo (no valid lat/lon rows).")
        return

    # make sure numeric
    subset[lat_col] = pd.to_numeric(subset[lat_col], errors="coerce")
    subset[lon_col] = pd.to_numeric(subset[lon_col], errors="coerce")
    subset = subset.dropna(subset=[lat_col, lon_col])

    if subset.empty:
        print("Skipping accidents_scatter_geo (no numeric lat/lon rows).")
        return

    fig = px.scatter_geo(
        subset,
        lat=lat_col,
        lon=lon_col,
        color="decade",
        size="fatalities_total",
        hover_name="location" if "location" in subset.columns else None,
        hover_data={
            "date_parsed": True if "date_parsed" in subset.columns else False,
            "operator": True if "operator" in subset.columns else False,
            "fatalities_total": True,
            "aboard_total": True,
        },
        title="Individual accidents (bubble size = fatalities, color = decade)",
        opacity=0.7,
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))

    fname = os.path.join(outdir, "accidents_scatter_geo.html")
    fig.write_html(fname)
    print(f"Saved {fname}")


# --------------------- MAIN --------------------- #


def main():
    outdir = ensure_output_dir()
    df = load_data()
    df = preprocess(df)

    print(f"Generating map visualizations into {outdir}/")

    map_accidents_by_country(df, outdir)
    map_accidents_by_country_decade(df, outdir)
    map_fatality_ratio_by_country(df, outdir)
    map_scatter_if_latlon(df, outdir)

    print("Done.")


if __name__ == "__main__":
    main()
