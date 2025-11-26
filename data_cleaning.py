import re
import pandas as pd

RAW_CSV = "./planecrashinfo_accidents.csv"
CLEAN_CSV = "./planecrashinfo_clean.csv"


def read_raw_data(path: str) -> pd.DataFrame:
    """
    Read the CSV produced by the scraper.
    That file is a standard comma-separated CSV with a header row.
    """
    df = pd.read_csv(
        path,
        dtype=str,
        on_bad_lines="skip",  # if any weird rows, just skip
    )
    print("Read data with shape:", df.shape)
    print("Original columns:", list(df.columns))
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case and map known ones
    to a consistent schema.
    """
    # Strip whitespace first
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    col_map = {}
    for col in df.columns:
        col_clean = col.strip().lower()

        if col_clean.startswith("aboard"):
            col_map[col] = "aboard"
        elif "type" in col_clean:
            col_map[col] = "aircraft_type"
        elif col_clean.startswith("cn"):
            col_map[col] = "cn_ln"
        elif col_clean == "date":
            col_map[col] = "date"
        elif col_clean == "detail_url":
            col_map[col] = "detail_url"
        elif "fatalit" in col_clean:
            col_map[col] = "fatalities"
        elif "flight" in col_clean:
            col_map[col] = "flight_no"
        elif col_clean in ("ground", "ground_fatalities"):
            col_map[col] = "ground_fatalities"
        elif col_clean == "location":
            col_map[col] = "location"
        elif "operator" in col_clean:
            col_map[col] = "operator"
        elif "registr" in col_clean:
            col_map[col] = "registration"
        elif col_clean == "route":
            col_map[col] = "route"
        elif col_clean == "summary":
            col_map[col] = "summary"
        elif col_clean == "time":
            col_map[col] = "time"
        elif "year_page_url" in col_clean:
            col_map[col] = "year_page_url"
        else:
            # fallback: generic snake_case
            tmp = re.sub(r"[^0-9a-zA-Z]+", "_", col_clean).strip("_")
            col_map[col] = tmp or col_clean

    df = df.rename(columns=col_map)
    print("Normalized columns:", list(df.columns))
    return df


def parse_fatalities(text: str):
    """
    Examples:
        "22   (passengers:?  crew:?)"
        "1    (passengers:1  crew:0)"
        "20   (passengers:?  crew:?)"
    Returns: total, pax, crew  (ints or None)
    """
    if pd.isna(text):
        return None, None, None

    s = str(text)

    # Extract leading integer (total fatalities)
    m_total = re.search(r"(\d+)", s)
    total = int(m_total.group(1)) if m_total else None

    # passengers
    m_pax = re.search(r"passengers:\s*([0-9?]+)", s, re.IGNORECASE)
    pax = None
    if m_pax and m_pax.group(1) != "?":
        pax = int(m_pax.group(1))

    # crew
    m_crew = re.search(r"crew:\s*([0-9?]+)", s, re.IGNORECASE)
    crew = None
    if m_crew and m_crew.group(1) != "?":
        crew = int(m_crew.group(1))

    return total, pax, crew


def split_location(loc: str):
    """
    Heuristic split of location → (city, state, country).
    """
    if pd.isna(loc):
        return None, None, None

    s = str(loc).strip()

    # If there is no comma at all, treat as region or country
    if "," not in s:
        if any(ctry in s for ctry in ["Germany", "France", "Belgium", "Italy", "England"]):
            return None, None, s
        else:
            return s, None, None

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0], None, None
    elif len(parts) == 2:
        city_region = parts[0]
        last = parts[1]
        known_countries = {"England", "Germany", "France", "Belgium", "Italy"}
        if last in known_countries:
            return city_region, None, last
        else:
            return city_region, last, None
    else:
        city_region = parts[0]
        state = parts[1]
        country = parts[-1]
        return city_region, state, country


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Times look like "1718", "2345", "700", "?" etc.
    Convert to "HH:MM" where possible; keep original as 'time_raw'.
    """
    if "time" not in df.columns:
        return df

    df["time_raw"] = df["time"]

    def _parse_time(t):
        if pd.isna(t):
            return None
        s = str(t).strip()
        if s == "?" or s == "":
            return None

        # Remove non-digits
        s = re.sub(r"\D", "", s)
        if not s:
            return None

        # pad to 4 digits if needed (e.g. "700" → "0700")
        if len(s) <= 2:
            return None
        if len(s) == 3:
            s = "0" + s
        elif len(s) > 4:
            s = s[-4:]

        hh = int(s[:2])
        mm = int(s[2:4])
        if hh > 23 or mm > 59:
            return None
        return f"{hh:02d}:{mm:02d}"

    df["time_hhmm"] = df["time"].apply(_parse_time)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # Parse dates
    df = parse_date_col(df)

    # Parse time
    df = parse_time_col(df)

    # Fatalities split
    if "fatalities" in df.columns:
        totals = df["fatalities"].apply(lambda x: parse_fatalities(x))
        df["fatalities_total"] = totals.apply(lambda x: x[0])
        df["fatalities_passengers"] = totals.apply(lambda x: x[1])
        df["fatalities_crew"] = totals.apply(lambda x: x[2])

        df["fatalities_total"] = pd.to_numeric(df["fatalities_total"], errors="coerce")
        df["fatalities_passengers"] = pd.to_numeric(df["fatalities_passengers"], errors="coerce")
        df["fatalities_crew"] = pd.to_numeric(df["fatalities_crew"], errors="coerce")

    # Location split
    if "location" in df.columns:
        loc_split = df["location"].apply(lambda x: split_location(x))
        df["location_city"] = loc_split.apply(lambda x: x[0])
        df["location_state"] = loc_split.apply(lambda x: x[1])
        df["location_country"] = loc_split.apply(lambda x: x[2])

    # Ground fatalities numeric
    if "ground_fatalities" in df.columns:
        df["ground_fatalities"] = pd.to_numeric(df["ground_fatalities"], errors="coerce")

    return df


def main():
    df = read_raw_data(RAW_CSV)
    df_clean = clean_dataset(df)

    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"Saved cleaned data to {CLEAN_CSV}")
    print(df_clean.head())


if __name__ == "__main__":
    main()
