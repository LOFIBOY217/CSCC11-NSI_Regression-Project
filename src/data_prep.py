import pandas as pd
import src.preprocess as pp
import src.build_feature_matrix as build_feature_matrix

def prepare_monthly_dataset(csv_path):
    """
    Load raw crime csv and transform into monthly NSI dataset.
    Applies all preprocessing + feature engineering steps.
    Returns a cleaned monthly dataframe ready for modeling.
    """

    df = pd.read_csv(csv_path)

    # ---- basic preprocessing ----
    df = pp.fill_occ_fields_from_date(df)
    df = build_feature_matrix.add_severity_weights(df)

    # ---- aggregate + compute NSI ----
    monthly = build_feature_matrix.aggregate_monthly_scores(df)
    monthly = build_feature_matrix.compute_nsi(monthly)
    monthly = build_feature_matrix.encode_basic_features(monthly)
    monthly = build_feature_matrix.add_nsi_lag(monthly)
    monthly = build_feature_matrix.add_nsi_rolling_avg(monthly)

    # ---- drop unusable first-rows ----
    monthly = monthly.dropna(subset=['Prev_Month_NSI', 'NSI_3M_Avg']).copy()

    return monthly