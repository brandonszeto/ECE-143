import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


DATA_PATH = "planecrashinfo_clean.csv"
PLOTS_DIR = "plots"


def ensure_output_dir(path=PLOTS_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} in current directory.")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Dates ---
    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    elif "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT

    df["year"] = df["date_parsed"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    # --- Fatalities ---
    for col in [
        "fatalities_total",
        "fatalities_passengers",
        "fatalities_crew",
        "ground_fatalities",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    df["is_fatal"] = df["fatalities_total"] > 0

    # --- Aboard total ---
    if "aboard_total" in df.columns:
        df["aboard_total"] = pd.to_numeric(df["aboard_total"], errors="coerce")
    else:
        if "aboard" in df.columns:
            df["aboard_total"] = (
                df["aboard"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
        else:
            df["aboard_total"] = pd.NA

    # --- Fatality ratio ---
    df["fatality_ratio"] = df["fatalities_total"] / df["aboard_total"]

    # --- Time of day (hour) ---
    if "time_hhmm" in df.columns:
        # assume HH:MM or similar
        df["hour"] = (
            df["time_hhmm"]
            .astype(str)
            .str.extract(r"^(\d{1,2})", expand=False)
            .astype(float)
        )
    else:
        df["hour"] = pd.NA

    return df


def plot_yearly_trends(df, outdir):
    if "year" not in df.columns or "fatalities_total" not in df.columns:
        print("Skipping yearly trends plot (missing 'year' or 'fatalities_total').")
        return

    yearly = (
        df.groupby("year", dropna=True)
        .agg(crashes=("year", "count"), fatalities=("fatalities_total", "sum"))
        .reset_index()
    )

    if yearly.empty:
        print("Skipping yearly trends plot (no yearly data after grouping).")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(yearly["year"], yearly["crashes"], label="Crashes")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of crashes")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["fatalities"], color="tab:red", label="Fatalities")
    ax2.set_ylabel("Total fatalities")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Crashes and fatalities per year")
    plt.tight_layout()
    fname = os.path.join(outdir, "yearly_crashes_fatalities.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_top_countries(df, outdir, top_n=20):
    if "location_country" not in df.columns:
        print("Skipping top countries plot (missing 'location_country').")
        return

    counts = df["location_country"].dropna().value_counts().head(top_n).sort_values()

    if counts.empty:
        print("Skipping top countries plot (no country data).")
        return

    plt.figure(figsize=(8, 6))
    counts.plot(kind="barh")
    plt.xlabel("Number of accidents")
    plt.ylabel("Country")
    plt.title(f"Top {top_n} countries by number of accidents")
    plt.tight_layout()
    fname = os.path.join(outdir, "top_countries_accidents.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_top_operators(df, outdir, top_n=15):
    if "operator" not in df.columns:
        print("Skipping operator plot (missing 'operator').")
        return

    counts = df["operator"].dropna().value_counts().head(top_n).sort_values()

    if counts.empty:
        print("Skipping operator plot (no operator data).")
        return

    plt.figure(figsize=(8, 6))
    counts.plot(kind="barh")
    plt.xlabel("Number of accidents")
    plt.ylabel("Operator")
    plt.title(f"Top {top_n} operators by number of accidents")
    plt.tight_layout()
    fname = os.path.join(outdir, "top_operators_accidents.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_aircraft_severity(df, outdir, top_n=15):
    if "aircraft_type" not in df.columns or "fatality_ratio" not in df.columns:
        print(
            "Skipping aircraft severity plot (missing 'aircraft_type' or 'fatality_ratio')."
        )
        return

    subset = df.dropna(subset=["aircraft_type", "fatality_ratio"])
    if subset.empty:
        print("Skipping aircraft severity plot (no valid rows).")
        return

    stats = (
        subset.groupby("aircraft_type")["fatality_ratio"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )

    if stats.empty:
        print("Skipping aircraft severity plot (no grouped stats).")
        return

    plt.figure(figsize=(8, 6))
    stats.plot(kind="barh")
    plt.xlabel("Median fatality ratio")
    plt.ylabel("Aircraft type")
    plt.title(f"Aircraft types by median fatality ratio (top {top_n})")
    plt.tight_layout()
    fname = os.path.join(outdir, "aircraft_type_median_fatality_ratio.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_aboard_vs_fatalities(df, outdir):
    if "aboard_total" not in df.columns or "fatalities_total" not in df.columns:
        print(
            "Skipping aboard vs fatalities plot (missing 'aboard_total' or 'fatalities_total')."
        )
        return

    subset = df.dropna(subset=["aboard_total", "fatalities_total"])
    if subset.empty:
        print("Skipping aboard vs fatalities plot (no valid rows).")
        return

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=subset,
        x="aboard_total",
        y="fatalities_total",
        hue="decade",
        alpha=0.6,
        palette="viridis",
    )
    max_aboard = subset["aboard_total"].max()
    plt.plot([0, max_aboard], [0, max_aboard], linestyle="--", color="gray")
    plt.xlabel("Number aboard")
    plt.ylabel("Fatalities")
    plt.title("Fatalities vs number aboard (color = decade)")
    plt.tight_layout()
    fname = os.path.join(outdir, "aboard_vs_fatalities_by_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_fatality_ratio_by_decade(df, outdir):
    if "fatality_ratio" not in df.columns or "decade" not in df.columns:
        print("Skipping fatality ratio density (missing 'fatality_ratio' or 'decade').")
        return

    subset = df[
        (df["fatality_ratio"].notna())
        & (df["fatality_ratio"] >= 0)
        & (df["fatality_ratio"] <= 1)
        & df["decade"].notna()
    ].copy()

    if subset.empty:
        print("Skipping fatality ratio density (no valid rows).")
        return

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=subset, x="fatality_ratio", hue="decade", common_norm=False)
    plt.xlabel("Fatalities / aboard")
    plt.title("Distribution of fatality ratios by decade")
    plt.tight_layout()
    fname = os.path.join(outdir, "fatality_ratio_density_by_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_hour_hist(df, outdir):
    if "hour" not in df.columns:
        print("Skipping hour-of-day histogram (missing 'hour').")
        return

    subset = df["hour"].dropna()
    if subset.empty:
        print("Skipping hour-of-day histogram (no valid hour data).")
        return

    plt.figure(figsize=(8, 4))
    sns.histplot(subset, bins=24, discrete=True)
    plt.xlabel("Hour of day")
    plt.ylabel("Number of accidents")
    plt.title("Accidents by time of day")
    plt.tight_layout()
    fname = os.path.join(outdir, "accidents_by_hour_of_day.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_fatalities_by_group_decade(df, outdir):
    needed = ["decade", "fatalities_passengers", "fatalities_crew", "ground_fatalities"]
    if any(col not in df.columns for col in needed):
        print("Skipping fatalities by group/decade (missing columns).")
        return

    agg = (
        df.groupby("decade", dropna=True)
        .agg(
            pax=("fatalities_passengers", "sum"),
            crew=("fatalities_crew", "sum"),
            ground=("ground_fatalities", "sum"),
        )
        .reset_index()
    )

    if agg.empty:
        print("Skipping fatalities by group/decade (no grouped data).")
        return

    melted = agg.melt(
        id_vars="decade",
        value_vars=["pax", "crew", "ground"],
        var_name="group",
        value_name="fatalities",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="decade", y="fatalities", hue="group")
    plt.title("Passenger, crew, and ground fatalities by decade")
    plt.xlabel("Decade")
    plt.ylabel("Fatalities")
    plt.tight_layout()
    fname = os.path.join(outdir, "fatalities_by_group_decade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def main():
    outdir = ensure_output_dir()
    df = load_data()
    df = preprocess(df)

    print("Generating plots into", outdir)

    plot_yearly_trends(df, outdir)
    plot_top_countries(df, outdir)
    plot_top_operators(df, outdir)
    plot_aircraft_severity(df, outdir)
    plot_aboard_vs_fatalities(df, outdir)
    plot_fatality_ratio_by_decade(df, outdir)
    plot_hour_hist(df, outdir)
    plot_fatalities_by_group_decade(df, outdir)

    print("Done.")


if __name__ == "__main__":
    main()
