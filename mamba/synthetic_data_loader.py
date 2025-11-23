import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

class SyntheticDataset(Dataset):
    def __init__(self, npz_path, code_map_path, max_seq_len=50):
        """
        Load synthetic EHR data với code mapping
        """
        # Load synthetic data
        data = np.load(npz_path)
        self.x = data['x']  # (num_patients, max_visits, code_num)
        self.lens = data['lens']  # (num_patients,)
        
        # Load code map
        with open(code_map_path, 'rb') as f:
            self.code_map = pickle.load(f)
        
        # Create reverse mapping
        self.index_to_code = {idx: code for code, idx in self.code_map.items()}
        self.code_num = len(self.code_map)
        
        print(f"✅ Loaded synthetic data: {self.x.shape}")
        print(f"✅ Loaded code map: {len(self.code_map)} codes")
        print(f"✅ Sample codes: {list(self.code_map.items())[:5]}")
        
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sequence = self.x[idx]  # (max_visits, code_num)
        seq_len = min(self.lens[idx], self.max_seq_len)
        
        # Truncate to actual sequence length
        sequence = sequence[:seq_len]
        
        # Pad if necessary
        if seq_len < self.max_seq_len:
            padded_sequence = np.zeros((self.max_seq_len, self.code_num))
            padded_sequence[:seq_len] = sequence
            sequence = padded_sequence
        else:
            sequence = sequence[:self.max_seq_len]
            
        #return torch.FloatTensor(sequence), seq_len
        return torch.tensor(sequence, dtype=torch.float32), seq_len
    
    def get_icd9_codes(self, prediction_probs, threshold=0.5):
        """
        Convert prediction probabilities to ICD9 codes
        prediction_probs: (code_num,) probabilities from model
        returns: list of (code, probability) tuples above threshold
        """
        predicted_codes = []
        for idx, prob in enumerate(prediction_probs):
            if prob > threshold and idx in self.index_to_code:
                predicted_codes.append((self.index_to_code[idx], float(prob)))
        
        # Sort by probability descending
        predicted_codes.sort(key=lambda x: x[1], reverse=True)
        return predicted_codes
    
    def decode_patient_sequence(self, patient_idx):
        """
        Decode a patient's entire sequence to ICD9 codes
        """
        sequence = self.x[patient_idx]
        seq_len = self.lens[patient_idx]
        
        decoded_visits = []
        for visit_idx in range(seq_len):
            visit_codes = []
            for code_idx in range(self.code_num):
                if sequence[visit_idx, code_idx] == 1:
                    visit_codes.append(self.index_to_code[code_idx])
            decoded_visits.append(visit_codes)
        
        return decoded_visits

def create_synthetic_data_loader(npz_path, code_map_path, batch_size=32, shuffle=True):
    dataset = SyntheticDataset(npz_path, code_map_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.code_num, dataset.index_to_code
