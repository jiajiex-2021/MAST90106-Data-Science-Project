import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mix_dtw
import random
import statistics as stat
import time


class DtwKmedoids:
    def __init__(self, clusters, centroids, centroids_detail, integ_dist_mat_sq, wss, s_coef, total_iterations,
                 wss_list):
        self.clusters = clusters
        self.centroids = centroids
        self.centroids_detail = centroids_detail
        self.integ_dist_mat_square = integ_dist_mat_sq
        self.within_cluster_sst = wss
        self.silhouette_coef = s_coef
        self.total_iterations = total_iterations
        self.wss_list = wss_list


def fit(df, n_clusters, uni_dim_cols=None, multi_dim_cols=None, max_iterations=None, wss_threshold=None):
    if uni_dim_cols is None:
        uni_dim_cols = []

    if multi_dim_cols is None:
        multi_dim_cols = []

    current_iteration = 1

    wss_list = []

    n_rows = len(df.index)
    init_centroids_idx_list = random.sample(range(n_rows), n_clusters)

    integ_dist_mat_sq = integ_dist_matrix(df=df, uni_dim_cols=uni_dim_cols, multi_dim_cols=multi_dim_cols, squared=True)

    init_clusters = assign_to_centroids(centroids_idx_list=init_centroids_idx_list, integ_dist_mat_sq=integ_dist_mat_sq)

    current_clusters = init_clusters
    current_centroids_idx_list = init_centroids_idx_list
    current_clusters_wss = within_cluster_sst(clusters=init_clusters, integ_dist_mat_sq=integ_dist_mat_sq)
    wss_list.append(current_clusters_wss)

    for new_iterations in range(max_iterations - 1):
        if current_clusters_wss < wss_threshold:
            break

        current_iteration = new_iterations + 1 + 1

        new_centroids_idx_list = genr_new_centroids(clusters=current_clusters, integ_dist_mat_sq=integ_dist_mat_sq)
        new_clusters = assign_to_centroids(centroids_idx_list=current_centroids_idx_list,
                                           integ_dist_mat_sq=integ_dist_mat_sq)
        new_clusters_wss = within_cluster_sst(clusters=new_clusters, integ_dist_mat_sq=integ_dist_mat_sq)

        current_clusters = new_clusters
        current_centroids_idx_list = new_centroids_idx_list
        current_clusters_wss = new_clusters_wss
        wss_list.append(current_clusters_wss)

    current_centroids_detail = extr_centroids_detail(df=df, uni_dim_cols=uni_dim_cols, multi_dim_cols=multi_dim_cols,
                                                     clusters=current_clusters)
    current_s_coef = silhouette_coef(clusters=current_clusters, integ_dist_mat_sq=integ_dist_mat_sq)

    return DtwKmedoids(clusters=current_clusters, centroids=current_centroids_idx_list,
                       centroids_detail=current_centroids_detail, integ_dist_mat_sq=integ_dist_mat_sq,
                       wss=current_clusters_wss, s_coef=current_s_coef, total_iterations=current_iteration,
                       wss_list=wss_list)


def integ_dist(df, row_1_idx, row_2_idx, uni_dim_cols, multi_dim_cols, squared=False):
    s = 0
    for uni_dim_col in uni_dim_cols:

        if type(df.loc[row_1_idx, uni_dim_col]) == list:
            s += mix_dtw.uni_dim_dist(ud_mix_t1=df.loc[row_1_idx, uni_dim_col],
                                      ud_mix_t2=df.loc[row_2_idx, uni_dim_col], squared=True)

        else:
            s += mix_dtw.mix_euclidean(mix_1=df.loc[row_1_idx, uni_dim_col], mix_2=df.loc[row_2_idx, uni_dim_col],
                                       squared=True)

    for multi_dim_cols_pair in multi_dim_cols:
        s += mix_dtw.multi_dim_mix_dist(md_mix_t1=list(df.loc[row_1_idx, multi_dim_cols_pair]),
                                        md_mix_t2=list(df.loc[row_2_idx, multi_dim_cols_pair]), squared=True)

    if squared:
        return s

    else:
        return np.sqrt(s)


def integ_dist_matrix(df, uni_dim_cols, multi_dim_cols, squared=False):
    n_rows = len(df.index)
    integ_dist_mat = np.empty(n_rows, n_rows)

    for row_i_idx in range(n_rows):

        for row_j_idx in range(n_rows):
            integ_dist_mat[row_i_idx][row_j_idx] = integ_dist(df, row_i_idx, row_j_idx, uni_dim_cols, multi_dim_cols,
                                                              squared=True)

    if squared:
        return integ_dist_mat

    else:
        return np.sqrt(integ_dist_mat)


def assign_to_centroids(centroids_idx_list, integ_dist_mat_sq):
    n_rows = len(integ_dist_mat_sq)

    clusters = dict((centroid_idx, []) for centroid_idx in centroids_idx_list)

    for row_idx in range(n_rows):

        if row_idx not in centroids_idx_list:  # Only assign non-centroids to clusters

            min_integ_dist_sq = integ_dist_mat_sq[row_idx][centroids_idx_list[0]]
            target_centroid_idx = centroids_idx_list[0]

            for centroid_idx in centroids_idx_list:

                if centroid_idx != centroids_idx_list[0]:  # Skip the first centroid examination

                    if integ_dist_mat_sq[row_idx][centroid_idx] < min_integ_dist_sq:
                        min_integ_dist_sq = integ_dist_mat_sq[row_idx][centroid_idx]
                        target_centroid_idx = centroid_idx

            clusters[target_centroid_idx].append(row_idx)

    return clusters


def genr_new_centroids(clusters, integ_dist_mat_sq):
    new_centroids_idx_list = []

    for centroid_idx in clusters.keys():

        alter_centroid_idx = centroid_idx

        if len(clusters[centroid_idx]) > 0:  # Exam the clusters with non-centroid elements to the list

            min_wss = integ_dist_mat_sq[centroid_idx][clusters[centroid_idx]]

            for non_centroid in clusters[centroid_idx]:

                current_wss = integ_dist_mat_sq([non_centroid][centroid_idx]) + sum(
                    integ_dist_mat_sq[non_centroid][clusters[centroid_idx]])

                if current_wss < min_wss:
                    min_wss = current_wss
                    alter_centroid_idx = non_centroid

        new_centroids_idx_list.append(alter_centroid_idx)

    return new_centroids_idx_list


def extr_centroids_detail(df, uni_dim_cols, multi_dim_cols, clusters):
    df_columns = uni_dim_cols + [sub_multi_dim_cols_elem for sub_multi_dim_cols in multi_dim_cols for
                                 sub_multi_dim_cols_elem in sub_multi_dim_cols]

    centroids_detail = [df.loc[centroid_idx][df_columns] for centroid_idx in clusters.keys()]

    return centroids_detail


def silhouette_coef(clusters, integ_dist_mat_sq):
    n_rows = len(integ_dist_mat_sq)

    integ_dist_mat_sr = np.sqrt(integ_dist_mat_sq)

    s_coef_sum = 0

    for centroid_idx in clusters.keys():

        if len(clusters[centroid_idx]) > 0:  # The s_coef of a cluster with no non-centroid elements is 0
            # Calculate the s_coef of the centroid
            a = stat.mean(integ_dist_mat_sr[centroid_idx][clusters[centroid_idx]])

            b_list = []
            for other_centroid_idx in clusters.keys():
                if other_centroid_idx != centroid_idx:
                    b_list.append((integ_dist_mat_sr[centroid_idx][other_centroid_idx] + sum(
                        integ_dist_mat_sr[centroid_idx][clusters[other_centroid_idx]])) / (
                                          1 + len(clusters[other_centroid_idx])))
            b = min(b_list)

            s_coef_sum += (b - a) / max(a, b)
            # Calculate the s_coef of the non-centroid elements
            for non_centroid_idx in clusters[centroid_idx]:

                a = (integ_dist_mat_sr[non_centroid_idx][centroid_idx] + sum(
                    integ_dist_mat_sr[non_centroid_idx][clusters[centroid_idx]])) / len(clusters[centroid_idx])

                b_list = []
                for other_centroid_idx in clusters.keys():
                    if other_centroid_idx != centroid_idx:
                        b_list.append((integ_dist_mat_sr[non_centroid_idx][other_centroid_idx] + sum(
                            integ_dist_mat_sr[non_centroid_idx][clusters[other_centroid_idx]])) / (
                                              1 + len(clusters[other_centroid_idx])))
                b = min(b_list)

                s_coef_sum += (b - a) / max(a, b)

    s_coef = s_coef_sum / n_rows

    return s_coef


def within_cluster_sst(clusters, integ_dist_mat_sq):
    wss = 0

    for centroid_idx in clusters.keys():

        if len(clusters[centroid_idx]) > 0:  # Skip clusters with no non-centroid elements
            wss += sum(integ_dist_mat_sq[centroid_idx][clusters[centroid_idx]])

    return wss
