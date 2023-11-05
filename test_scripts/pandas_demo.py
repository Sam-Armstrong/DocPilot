import pandas as pd


def calculate_summary_statistics(dataframe):
    if 'age' in dataframe.columns and 'height' in dataframe.columns and 'weight' in dataframe.columns:
        age_sum = dataframe['age'].sum()
        
        height_min = dataframe['height'].min()
        
        weight_mean = dataframe['weight'].mean()
        
        return age_sum, height_min, weight_mean
    else:
        return None
