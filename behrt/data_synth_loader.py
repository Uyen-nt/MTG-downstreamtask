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

        position_counter = 0

        num_visits = self.lens[idx]

        for v in range(num_visits):
            codes = self.x[idx, v]               # (2869,)
            code_ids = np.nonzero(codes)[0]      # indices of ones

            for c in code_ids:
                visits.append(c)
                seg_ids.append(v)
                pos_ids.append(position_counter)
                age_ids.append(0)   # all zeros, no age used

                position_counter += 1

        visits = torch.tensor(visits, dtype=torch.long)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)
        pos_ids = torch.tensor(pos_ids, dtype=torch.long)
        age_ids = torch.tensor(age_ids, dtype=torch.long)

        attention_mask = torch.ones_like(visits)

        return visits, age_ids, seg_ids, pos_ids, attention_mask
