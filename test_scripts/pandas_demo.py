import pandas as pd


def calculate_summary_statistics(dataframe):
    """Calculates basic summary statistics on age, height, and weight.

    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame containing columns 'age', 'height', and 'weight'.

    Returns
    -------
    age_sum : float
        Sum of all ages in the dataframe's 'age' column.
    height_min : float
        Minimum value in the dataframe's 'height' column.
    weight_mean : float
        Mean of the dataframe's 'weight' column.

    None
        If 'age', 'height', or 'weight' columns do not exist in the dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'age': [23, 45, 19],
                            'height': [178, 163, 183],
                            'weight': [71, 59, 88]})
    >>> calculate_summary_statistics(df)
    (87, 163, 72.66667)
    """
    if (
        "age" in dataframe.columns
        and "height" in dataframe.columns
        and "weight" in dataframe.columns
    ):
        age_sum = dataframe["age"].sum()

        height_min = dataframe["height"].min()

        weight_mean = dataframe["weight"].mean()

        return age_sum, height_min, weight_mean
    else:
        return None
