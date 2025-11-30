import numpy as np
import torch

class SyntheticBEHRTDataset(torch.utils.data.Dataset):
    def __init__(self, synth_path):
        data = np.load(synth_path)
        self.x = data["x"]       # (1500, 34, 2869)
        self.lens = data["lens"] # (1500,)

        self.max_position = 1500  

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        visits = []
        seg_ids = []
        pos_ids = []
        age_ids = []

        label = []
        position_counter = 0
        num_visits = self.lens[idx]

        # Nếu chỉ có 1 visit thì bỏ (không có next-visit)
        if num_visits < 2:
            return self.__getitem__((idx+1) % len(self.x))

        ### Làm input:
        for v in range(num_visits-1):  # đến visit t-1
            codes = self.x[idx, v]
            code_ids = np.nonzero(codes)[0]

            for c in code_ids:
                visits.append(c)
                seg_ids.append(v)
                pos_ids.append(position_counter)
                age_ids.append(0)
                position_counter+=1

        ### Làm label:
        codes_next = self.x[idx, num_visits-1]    # visit t
        label = np.nonzero(codes_next)[0]

        # convert to torch
        visits = torch.tensor(visits, dtype=torch.long)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)
        pos_ids = torch.tensor(pos_ids, dtype=torch.long)
        age_ids = torch.tensor(age_ids, dtype=torch.long)
        attention_mask = torch.ones_like(visits)

        # Labels dạng multi-hot
        label_tensor = torch.zeros(2871)
        label_tensor[label] = 1

        return visits, age_ids, seg_ids, pos_ids, attention_mask, label_tensor
