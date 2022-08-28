import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# Option 1: Use the KNNImputer to impute the missing values

def knn_imputation(df, cols_with_nas, n_neighbors):
    knn_imp = KNNImputer(n_neighbors=n_neighbors)
    data_mat = df.loc[:, cols_with_nas]
    df.loc[:, cols_with_nas] = knn_imp.fit_transform(data_mat)
    return df


'''
df = pd.read_csv('space.csv')
columns_with_nas = df.columns[1:len(df.columns)]
print(knn_imputation(df, columns_with_nas))
'''


# Option 2: Use the Collaborative filtration algorithm to impute the missing values

def nom_euclidean(nom_1, nom_2):
    if nom_1 == nom_2:
        return 0

    else:
        return 1


def mix_euclidean(mix_1, mix_2, squared=False):
    if type(mix_1) == str:
        return nom_euclidean(nom_1=mix_1, nom_2=mix_2)

    else:
        mix_euclidean_squared = mix_1 ** 2 + mix_2 ** 2

        if squared:
            return mix_euclidean_squared

        else:
            return np.sqrt(mix_euclidean_squared)


def euclidean_sim(squared_dist):
    return 1 / (1 + squared_dist)


def integ_dist(df, row_1_idx, row_2_idx, sim_cols, squared=False):
    s = 0
    for sim_col in sim_cols:
        s += mix_euclidean(mix_1=df.loc[row_1_idx, sim_col], mix_2=df.loc[row_2_idx, sim_col], squared=True)

    if squared:
        return s

    else:
        return np.sqrt(s)


# 'sim_cols' cannot have any NAs
def collab_fltr(df, subjects_col, sim_cols, ca_col_with_nas, val_col_with_nas, n_neighbors):
    init_target_df = df.loc[:, [subjects_col, ca_col_with_nas, val_col_with_nas]]

    subjects_list = init_target_df.loc[:, subjects_col].unique()
    ca_col_with_nas_set = set(init_target_df.loc[:, ca_col_with_nas].unique())

    for subject in subjects_list:
        subject_idx = df.loc[:, subjects_col] == subject
        actl_cas_set = set(df.loc[subject_idx, ca_col_with_nas])
        ref_ca = list(actl_cas_set)[0]  # The reference category is the target subject's first category by default
        ref_ca_idx = df.loc[:, ca_col_with_nas] == ref_ca

        # Only the subjects with missing value categories are needed to be imputed
        if actl_cas_set != ca_col_with_nas_set:
            diff_cas_set = ca_col_with_nas_set - actl_cas_set

            for diff_ca in diff_cas_set:

                integ_sims_list = []  # Data structure to store the neighbours and their similarity to target subject

                # Collect the neighbours and their similarity
                for subject_2 in subjects_list:

                    subject_2_idx = df.loc[:, subjects_col] == subject_2
                    valid_sub_id_2_flag = set(ref_ca).union(diff_ca).issubset(
                        set(df.loc[subject_2_idx, ca_col_with_nas]))

                    if valid_sub_id_2_flag and subject_2 != subject:
                        integ_distance = integ_dist(df=df,
                                                    row_1_idx=subject_idx & ref_ca_idx,
                                                    row_2_idx=subject_2_idx & ref_ca_idx,
                                                    sim_cols=sim_cols, squared=True)
                        integ_similarity = euclidean_sim(squared_dist=integ_distance)
                        integ_sims_list.append([subject_2, integ_similarity])

                integ_sims_list.sort(key=lambda x: x[1], reverse=True)  # Calculate the top number of neighbours

                # The weighted average of the top n_neighbors' value of the target subject's missing value category
                numerator, denominator = 0, 0
                for i in range(0, n_neighbors):
                    numerator += integ_sims_list[i][1] * df.loc[
                        df.loc[:, subjects_col] == integ_sims_list[i][0], val_col_with_nas]
                    denominator += integ_sims_list[i][1]
                predicted_value = numerator / denominator

                # Append the missing value for each category of each subject to the target dataframe
                init_target_df = pd.concat([init_target_df, pd.DataFrame(
                    {subjects_col: [subject], ca_col_with_nas: [diff_ca], val_col_with_nas: [predicted_value]})],
                                           ignore_index=True)

    final_target_df = init_target_df

    return final_target_df


'''
df = collab_fltr(df=df, subjects_col='stu_uuid', sim_cols=['Feature_1', 'Feature_2'], ca_col_with_nas='year_level',
                 val_col_with_nas='percentile', n_neighbors=5)
df = pd.pivot_table(df, values='percentile', index=['stu_uuid'], columns=['year_level'], aggfunc=np.sum)
'''
