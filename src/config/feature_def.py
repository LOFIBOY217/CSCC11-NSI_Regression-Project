# Base temporal model
FEATURE_BASE = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'x',
    'y'
]

# Add short-term safety trend (3-month average)
FEATURE_NSI_3M = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'NSI_3M_Avg',
    'x',
    'y'
]

# For one-hot neighbourhood pipelines (neighbourhood handled separately)
FEATURE_NSI_3M_NUMERIC = [
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'NSI_3M_Avg',
    'x',
    'y'
]

# Add yearly seasonal memory
FEATURE_NSI_YEAR = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'Prev_Year_NSI',
    'x',
    'y'
]

# Add crime-based temporal dynamics (short + long term)
FEATURE_CRIME_6M = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'NSI_3M_Avg',
    'Prev_Crime_Count',
    'Prev_6M_Crime_Avg',
    'x',
    'y'
]


TARGET_COL = 'NSI'
