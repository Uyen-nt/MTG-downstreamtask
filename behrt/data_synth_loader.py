import numpy as np
import torch

class SyntheticBEHRTDataset(torch.utils.data.Dataset):
    def __init__(self, synth_path, max_len=512):
        data = np.load(synth_path)
        self.x = data["x"]
        self.lens = data["lens"]

        self.max_len = max_len
        self.num_codes = 2871

    def __len__(self):
        return len(self.x)

    def pad(self, arr):
        # arr = tensor 1D
        if len(arr) >= self.max_len:
            return arr[:self.max_len]
        else:
            pad_len = self.max_len - len(arr)
            return torch.cat([arr, torch.zeros(pad_len, dtype=torch.long)])

    def __getitem__(self, idx):
        visits = []
        seg_ids = []
        pos_ids = []
        age_ids = []
        position_counter = 0
        num_visits = self.lens[idx]

        if num_visits < 2:
            return self.__getitem__((idx+1) % len(self.x))

        # build input tokens
        for v in range(num_visits-1):
            codes = self.x[idx, v]
            code_ids = np.nonzero(codes)[0]
            for c in code_ids:
                visits.append(c)
                seg_ids.append(v)
                pos_ids.append(position_counter)
                age_ids.append(0)
                position_counter += 1

        # build label
        codes_next = self.x[idx, num_visits-1]
        label_ids = np.nonzero(codes_next)[0]
        label_tensor = torch.zeros(self.num_codes)
        label_tensor[label_ids] = 1

        # convert to tensor
        visits = torch.tensor(visits, dtype=torch.long)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)
        pos_ids = torch.tensor(pos_ids, dtype=torch.long)
        age_ids = torch.tensor(age_ids, dtype=torch.long)
        attention_mask = torch.ones(len(visits), dtype=torch.long)

        # PAD EVERYTHING
        visits = self.pad(visits)
        seg_ids = self.pad(seg_ids)
        pos_ids = self.pad(pos_ids)
        age_ids = self.pad(age_ids)
        attention_mask = self.pad(attention_mask)

        return visits, age_ids, seg_ids, pos_ids, attention_mask, label_tensor
