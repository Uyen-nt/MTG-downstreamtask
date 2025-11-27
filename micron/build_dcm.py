import numpy as np

def build_dcm(x):
    """
    chấp nhận 2 dạng:
    - numpy: (N, V, C)
    - list: list[list[list[int]]]
    """

    # CASE 1 — numpy
    if hasattr(x, "shape"):
        n_patients, max_visits, n_codes = x.shape

        dcm = np.zeros((n_codes, n_codes), dtype=np.float32)

        for p in range(n_patients):
            for v in range(max_visits):
                visit = x[p, v]
                indexes = np.where(visit == 1)[0]

                for i in indexes:
                    dcm[i,i] += 1
                    for j in indexes:
                        if i != j:
                            dcm[i, j] += 1

    # CASE 2 — list-based
    else:
        print("⚠️ DCM builder: using LIST mode")

        n_patients = len(x)
        n_codes = len(x[0][0])

        dcm = np.zeros((n_codes, n_codes), dtype=np.float32)

        for patient in x:
            for visit in patient:
                indexes = [i for i,v in enumerate(visit) if v==1]

                for i in indexes:
                    dcm[i,i] += 1
                    for j in indexes:
                        if i != j:
                            dcm[i,j] += 1

    # normalize
    dcm /= dcm.max()

    print(f"✔ DCM built: {n_codes} x {n_codes}")
    return dcm
