from sklearn.preprocessing import LabelEncoder

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


def add_nsi_lag(df):
    """
    Add previous month's NSI as a predictive feature.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])
    df['Prev_Month_NSI'] = df.groupby('NEIGHBOURHOOD_158')['NSI'].shift(1)
    return df

def add_nsi_rolling_avg(df, window=3):
    """
    Add rolling mean NSI of previous N months for each neighbourhood.
    Creates NSI_3M_Avg column.
    """
    df = df.sort_values(['NEIGHBOURHOOD_158', 'REPORT_YEAR', 'REPORT_MONTH'])

    df['NSI_3M_Avg'] = (
        df.groupby('NEIGHBOURHOOD_158')['NSI']
          .transform(lambda s: s.shift(1).rolling(window).mean())
    )

    return df

def encode_basic_features(df):
    """
    Encode categorical fields into numeric form for ML models.

    Operations:
        * NEIGHBOURHOOD_158 → Label encoded integers
        * REPORT_MONTH → converted from month name to 1-12
    """
    le = LabelEncoder()

    # Convert neighborhood name → numeric ID
    df['NEIGHBOURHOOD_158'] = le.fit_transform(df['NEIGHBOURHOOD_158'].astype(str))

    # Convert month string → numeric month
    df['REPORT_MONTH'] = df['REPORT_MONTH'].map(month_map).astype(int)

    return df