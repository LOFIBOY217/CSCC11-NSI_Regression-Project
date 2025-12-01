FEATURE_COLS = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'x',
    'y'
]

FEATURE_WITH_3M_COLS = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'NSI_3M_Avg',
    'x',
    'y'
]

FEATURE_NN_COLS = [
    'NEIGHBOURHOOD_158',
    'REPORT_YEAR',
    'REPORT_MONTH',
    'Prev_Month_NSI',
    'NSI_3M_Avg',
    'x',
    'y',
    'Crime_Count',
    'TotalCrimeScore'
]

TARGET_COL = 'NSI'