import numpy as np

def build_dcm(x):
    """
    x: numpy array (patients, visits, n_codes)
    """
    print("✔ DCM builder: using NUMPY raw x (one-hot)")
    _, _, n_codes = x.shape

    dcm = np.zeros((n_codes, n_codes), dtype=np.float32)

    for p in range(x.shape[0]):
        for v in range(x.shape[1]):
            visit = x[p, v]
            indexes = np.where(visit == 1)[0]
            for i in indexes:
                dcm[i,i] += 1
                for j in indexes:
                    if i != j:
                        dcm[i,j] += 1

    # normalize
    dcm = dcm / dcm.max()

    print(f"✔ DCM built: {n_codes} x {n_codes}")
    return dcm
