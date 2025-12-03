import src.data_prep as data_prep
import src.build_feature_matrix as build_feature_matrix
import src.models.linear_reg as linear


def run_onehot_linear_pipeline(
    csv_path,
    numeric_feature_cols,
    target_col,
    scale=True,
    neighbourhood_col='NEIGHBOURHOOD_158',
    prefix='NEIGHBOURHOOD'
):
    """
    Full linear-regression pipeline that keeps neighbourhoods as categorical
    variables by one-hot encoding them before training.
    Steps:
        1) build the monthly dataset without label-encoding neighbourhoods
        2) expand the dataframe with one-hot columns
        3) train/evaluate the linear model using the shared utilities
    Returns trained artifacts plus the expanded feature list for inspection.
    """
    monthly = data_prep.prepare_monthly_dataset_onehot(csv_path)

    monthly_onehot, feature_cols = build_feature_matrix.build_neighbourhood_one_hot_features(
        monthly,
        numeric_feature_cols,
        neighbourhood_col=neighbourhood_col,
        prefix=prefix
    )

    model, y_test, y_pred, metrics = linear.train_linear_model(
        monthly_onehot,
        feature_cols,
        target_col,
        scale=scale
    )

    return model, y_test, y_pred, metrics, feature_cols, monthly_onehot
