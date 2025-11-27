import numpy as np

def build_dcm(x):
    """
    x: shape (N_patients, max_visits, n_codes)
    return: DCM (n_codes x n_codes)
    """

    n_patients, max_visits, n_codes = x.shape
    dcm = np.zeros((n_codes, n_codes), dtype=np.float32)

    for p in range(n_patients):
        for v in range(max_visits):
            visit = x[p, v]
            indexes = np.where(visit == 1)[0]

            for i in indexes:
                for j in indexes:
                    if i != j:
                        dcm[i, j] += 1

    # normalize
    dcm = dcm / dcm.max()
    return dcm
