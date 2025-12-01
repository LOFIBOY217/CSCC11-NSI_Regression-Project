import pandas as pd

def fill_occ_fields_from_date(df, date_col="OCC_DATE"):
    """
    Fill or generate OCC_* fields based on OCC_DATE (EXCEPT OCC_HOUR).
    Ensures consistent data types:
        OCC_YEAR, OCC_DAY, OCC_DOY -> Int64
        OCC_MONTH, OCC_DOW -> str
    """
    # Parse OCC_DATE into datetime format
    df[date_col] = pd.to_datetime(
        df[date_col],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Derive date-based features (no OCC_HOUR here)
    derived = pd.DataFrame(index=df.index)
    derived["OCC_YEAR"]  = df[date_col].dt.year.astype("Int64")
    derived["OCC_MONTH"] = df[date_col].dt.month_name().astype(str)
    derived["OCC_DAY"]   = df[date_col].dt.day.astype("Int64")
    derived["OCC_DOY"]   = df[date_col].dt.dayofyear.astype("Int64")
    derived["OCC_DOW"]   = df[date_col].dt.day_name().astype(str)

    # Expected dtypes (no OCC_HOUR)
    type_map = {
        "OCC_YEAR": "Int64",
        "OCC_MONTH": "str",
        "OCC_DAY": "Int64",
        "OCC_DOY": "Int64",
        "OCC_DOW": "str",
    }

    # Fill missing values and enforce dtype consistency
    for col, dtype in type_map.items():
        fill_vals = derived[col]
        if col not in df.columns:
            df[col] = fill_vals
        else:
            df[col] = df[col].where(df[col].notna(), fill_vals)

        # Force conversion after filling
        if dtype == "str":
            df[col] = df[col].astype(str)
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df