import numpy as np
import dtaidistance as dt


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


def uni_dim_dist(ud_mix_t1, ud_mix_t2, squared=False):
    if type(ud_mix_t1[0]) == str:  # Nominal Uni-dimensional DTW

        ud_mix_t1_len = len(ud_mix_t1)
        ud_mix_t2_len = len(ud_mix_t2)

        ud_mix_dwt_mat = np.empty(shape=(ud_mix_t1_len, ud_mix_t2_len))

        for i in range(ud_mix_t1_len):

            for j in range(ud_mix_t2_len):

                nom_euclidean_dist = nom_euclidean(nom_1=ud_mix_t1[i], nom_2=ud_mix_t2[j])

                if i == 0 and j == 0:
                    ud_mix_dwt_mat[i][j] = nom_euclidean_dist

                elif i == 0 and j != 0:
                    ud_mix_dwt_mat[i][j] = ud_mix_dwt_mat[i][j - 1] + nom_euclidean_dist

                elif i != 0 and j == 0:
                    ud_mix_dwt_mat[i][j] = ud_mix_dwt_mat[i - 1][j] + nom_euclidean_dist

                else:
                    ud_mix_dwt_mat[i][j] = min(ud_mix_dwt_mat[i - 1][j], ud_mix_dwt_mat[i][j - 1],
                                               ud_mix_dwt_mat[i - 1][j - 1]) + nom_euclidean_dist

        uni_dim_dist_squared = ud_mix_dwt_mat[ud_mix_t1_len - 1][ud_mix_t2_len - 1]

        if squared:
            return uni_dim_dist_squared

        else:
            return np.sqrt(uni_dim_dist_squared)

    else:  # Numerical Uni-dimensional DTW

        uni_dim_dist_orig = dt.dtw.distance(ud_mix_t1, ud_mix_t2)

        if squared:
            return uni_dim_dist_orig ** 2

        else:
            return uni_dim_dist_orig


def multi_dim_mix_dist(md_mix_t1, md_mix_t2, squared=False):
    dim = np.shape(md_mix_t1)[0]
    md_mix_t1_len = np.shape(md_mix_t1)[1]
    md_mix_t2_len = np.shape(md_mix_t2)[1]

    mix_dtw_mat = np.empty(shape=(md_mix_t1_len, md_mix_t2_len))

    for i in range(md_mix_t1_len):

        for j in range(md_mix_t2_len):

            s = 0
            for k in range(dim):
                s += mix_euclidean(mix_1=md_mix_t1[k][i], mix_2=md_mix_t2[k][j], squared=True)

            if i == 0 and j == 0:
                mix_dtw_mat[i][j] = s

            elif i == 0 and j != 0:
                mix_dtw_mat[i][j] = mix_dtw_mat[i][j - 1] + s

            elif i != 0 and j == 0:
                mix_dtw_mat[i][j] = mix_dtw_mat[i - 1][j] + s

            else:
                mix_dtw_mat[i][j] = min(mix_dtw_mat[i - 1][j], mix_dtw_mat[i][j - 1],
                                        mix_dtw_mat[i - 1][j - 1]) + s

    multi_dim_mix_dist_squared = mix_dtw_mat[md_mix_t1_len - 1][md_mix_t2_len - 1]

    if squared:
        return multi_dim_mix_dist_squared

    else:
        return np.sqrt(multi_dim_mix_dist_squared)
