import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtw_kmedoids
import random
import statistics as stat
import time


def time_series_xform(df, subjects_col, time_series_ref_cols: list, time_series_data_cols):
    df_ts = pd.DataFrame(columns=list(subjects_col) + time_series_ref_cols + time_series_data_cols)
    subjects_list = list(set(df[subjects_col]))
    df_ts[subjects_col] = subjects_list

    for subject in subjects_list:

        for time_series_col in time_series_ref_cols + time_series_data_cols:
            df_ts.at[df_ts.index[df_ts[subjects_col] == subject], time_series_col] = list(
                df.loc[df[subjects_col] == subject].sort_values(by=time_series_ref_cols)[time_series_col])

    return df_ts


'''
df_time_series = time_series_xform(df=df_origin, subjects_col='stu_uuid',
                                   time_series_ref_cols=['Timestamp_1', 'Timestamp_2'],
                                   time_series_data_cols=['Feature_1', 'Feature_2'])

model = dtw_kmedoids.fit(df=df_time_series, n_clusters=5, uni_dim_cols=['Feature_1'],
                         multi_dim_cols=[['Feature_2', 'Feature_3']], max_iterations=10, wss_threshold=0.1)
'''
