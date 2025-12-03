from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ------------------------
# Severity score mapping for different crime categories
# ------------------------
severity_map = {
    'Robbery': 5,
    'Assault': 4,
    'Break and Enter': 3,
    'Auto Theft': 2,
    'Theft Over': 1
}

# ------------------------
# Convert month names to numeric form for ML models
# ------------------------
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def add_severity_weights(df):
    """
    Add numeric crime severity scores to the dataset.
    """
    df['Severity_Weight'] = df['MCI_CATEGORY'].map(severity_map)
    return df


def aggregate_monthly_scores(df):
    """
    Aggregate crime data by neighborhood and month.

    Operation:
        Group by:
            - NEIGHBOURHOOD_158
            - REPORT_YEAR
            - REPORT_MONTH
    """
    monthly = (
        df.groupby(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])
          .agg(
              TotalCrimeScore=('Severity_Weight', 'sum'),  # weighted severity sum
              Crime_Count=('Severity_Weight', 'count'),    # incident count
              x=('x', 'mean'),                             # avg longitude
              y=('y', 'mean')                              # avg latitude
          )
          .reset_index()
    )
    return monthly


def compute_nsi(df):
    """
    Compute the Neighborhood Safety Index (NSI).
    """
    gmax = df['TotalCrimeScore'].max()
    gmin = df['TotalCrimeScore'].min()

    # Normalize crime score into [0,1] safety index (inverted scale)
    df['NSI'] = 1 - (df['TotalCrimeScore'] - gmin) / (gmax - gmin)
    return df


def add_prev_month_nsi(df):
    """
    Add previous month's NSI as a predictive feature.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])
    df['Prev_Month_NSI'] = df.groupby('NEIGHBOURHOOD_158')['NSI'].shift(1)
    return df


def add_nsi_3m_avg(df):
    """
    Add 3-month rolling average NSI (excluding current month).
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])
    df['NSI_3M_Avg'] = (
        df.groupby('NEIGHBOURHOOD_158')['NSI']
          .transform(lambda s: s.shift(1).rolling(3).mean())
    )
    return df

def encode_basic_features(df):
    """
    Encode categorical fields into numeric form for ML models.
    """
    le = LabelEncoder()

    # Convert neighborhood name → numeric ID
    df['NEIGHBOURHOOD_158'] = le.fit_transform(df['NEIGHBOURHOOD_158'].astype(str))

    # Convert month string → numeric month
    df['REPORT_MONTH'] = df['REPORT_MONTH'].map(month_map).astype(int)

    return df


def encode_report_month_numeric(df, column='REPORT_MONTH'):
    """
    Convert month string → numeric month without touching NEIGHBOURHOOD_158.
    Used by pipelines that need to keep the categorical neighbourhood label.
    """
    df = df.copy()
    df[column] = df[column].map(month_map).astype(int)
    return df


def build_neighbourhood_one_hot_features(
    df,
    numeric_feature_cols,
    neighbourhood_col='NEIGHBOURHOOD_158',
    prefix='NEIGHBOURHOOD'
):
    """
    Append a one-hot encoding of the neighbourhood column and return the expanded
    dataframe along with the full feature column list.
    """
    df = df.copy()
    one_hot = pd.get_dummies(df[neighbourhood_col], prefix=prefix)
    df = df.drop(columns=[neighbourhood_col])
    df = pd.concat([df, one_hot], axis=1)
    feature_cols = numeric_feature_cols + one_hot.columns.tolist()
    return df, feature_cols


def add_prev_crime_count(df):
    """
    Add previous month's crime count as a predictive feature.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])
    df['Prev_Crime_Count'] = df.groupby('NEIGHBOURHOOD_158')['Crime_Count'].shift(1)
    return df

def add_crime_6m_avg(df, window=6):
    """
    Add rolling average crime count of previous N months for each neighbourhood.
    Creates Prev_6M_Crime_Avg column.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])

    df['Prev_6M_Crime_Avg'] = (
        df.groupby('NEIGHBOURHOOD_158')['Crime_Count']
          .transform(lambda s: s.shift(1).rolling(window).mean())
    )

    return df

def add_prev_year_nsi(df):
    """
    Add previous year's NSI for the same month as a predictive feature.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])

    df['Prev_Year_NSI'] = (
        df.groupby(['NEIGHBOURHOOD_158', 'REPORT_MONTH'])['NSI']
          .shift(1)  # previous year's NSI for the same neighbourhood & month
    )

    return df
