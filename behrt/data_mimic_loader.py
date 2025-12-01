import numpy as np
import torch

class RealBEHRTDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, max_len=512):
        data = np.load(npz_path)
        self.x = data["x"]          # shape (N, V, C)
        self.lens = data["lens"]    # shape (N)
        self.max_len = max_len
        self.num_codes = self.x.shape[2]

    def __len__(self):
        return len(self.x)

    def pad(self, arr):
        if len(arr) >= self.max_len:
            return arr[:self.max_len]
        else:
            return torch.cat([arr, torch.zeros(self.max_len - len(arr), dtype=torch.long)])

    def __getitem__(self, idx):

        visits = []
        seg_ids = []
        pos_ids = []
        age_ids = []
        position_counter = 0

        num_visits = self.lens[idx]

        # nếu chỉ có 1 visit → bỏ
        if num_visits < 2:
            return self.__getitem__((idx+1) % len(self.x))

        # ===============================
        # 🟩 build input sequence
        #    visits[0] → visits[last−1]
        # ===============================
        for v in range(num_visits - 1):
            codes = self.x[idx, v]            # vector multi-hot length num_codes
            code_ids = np.nonzero(codes)[0]   # convert to sparse list
            for c in code_ids:
                visits.append(c)
                seg_ids.append(v)
                pos_ids.append(position_counter)
                age_ids.append(0)
                position_counter += 1

        # ===============================
        # 🟥 build label (target)
        #   predict next visit
        # ===============================
        codes_next = self.x[idx, num_visits - 1]
        label_ids = np.nonzero(codes_next)[0]     # các mã bệnh ở visit tiếp theo

        label_tensor = torch.zeros(self.num_codes)
        label_tensor[label_ids] = 1

        # chuyển sang tensor
        visits = torch.tensor(visits, dtype=torch.long)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)
        pos_ids = torch.tensor(pos_ids, dtype=torch.long)
        age_ids = torch.tensor(age_ids, dtype=torch.long)
        attention_mask = torch.ones(len(visits), dtype=torch.long)

        # padding đến max_len
        visits = self.pad(visits)
        seg_ids = self.pad(seg_ids)
        pos_ids = self.pad(pos_ids)
        age_ids = self.pad(age_ids)
        attention_mask = self.pad(attention_mask)

        # ===============================
        # RETURN — giống hệt synthetic
        # ===============================
        return visits, age_ids, seg_ids, pos_ids, attention_mask, label_tensor
