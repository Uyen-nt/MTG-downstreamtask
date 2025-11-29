# gpt/data_preprocessing.py
import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

data_dir = "data/mimic3/raw"
output_dir = "gpt/result"

os.makedirs(output_dir, exist_ok=True)

admissionFile = f"{data_dir}/ADMISSIONS.csv"
diagnosisFile = f"{data_dir}/DIAGNOSES_ICD.csv"
yaml_file = f"{data_dir}/hcup_ccs_2015_definitions_benchmark.yaml"   # <-- ĐẶC BIỆT QUAN TRỌNG

# Kiểm tra file tồn tại
for f in [admissionFile, diagnosisFile, yaml_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Không tìm thấy file: {f}")

print("Loading CSVs Into Dataframes")
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['ADMITTIME'] = pd.to_datetime(admissionDf['ADMITTIME'])
admissionDf = admissionDf.sort_values('ADMITTIME').reset_index(drop=True)

diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("HADM_ID")
diagnosisDf = diagnosisDf[diagnosisDf['ICD9_CODE'].notnull()][['ICD9_CODE']]

print("Building Dataset")
data = {}
for row in tqdm(admissionDf.itertuples(), total=len(admissionDf)):
    hadm_id = row.HADM_ID
    subject_id = row.SUBJECT_ID
    
    diagnoses = diagnosisDf.loc[[hadm_id]]["ICD9_CODE"].tolist() if hadm_id in diagnosisDf.index else []
    diagnoses = list(set(diagnoses))
    
    if subject_id not in data:
        data[subject_id] = {'visits': [diagnoses]}
    else:
        data[subject_id]['visits'].append(diagnoses)

# Vocab
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v]))
np.random.shuffle(all_codes)
code_to_index = {c: i for i, c in enumerate(all_codes)}
index_to_code = {v: k for k, v in code_to_index.items()}
print(f"VOCAB SIZE: {len(code_to_index)}")

data = list(data.values())

# Labels (HCUP CCS)
print("Adding Labels")
with open(yaml_file) as f:
    definitions = yaml.full_load(f)

code_to_group = {}
for group in definitions:
    if not definitions[group].get('use_in_benchmark', False):
        continue
    for code in definitions[group]['codes']:
        code_to_group[code] = group

id_to_group = sorted([k for k, v in definitions.items() if v.get('use_in_benchmark')])
group_to_id = {g: i for i, g in enumerate(id_to_group)}

for p in data:
    label = np.zeros(len(group_to_id))
    for v in p['visits']:
        for c in v:
            if c in code_to_group:
                label[group_to_id[code_to_group[c]]] = 1
    p['labels'] = label

# Convert to indices
for p in data:
    p['visits'] = [[code_to_index[c] for c in visit] for visit in p['visits']]

# Stats
print(f"MAX LEN: {max(len(p['visits']) for p in data)}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL: {sum(1 for p in data if len(p['visits']) > 1)}")

# Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# LƯU VÀO THƯ MỤC result
pickle.dump(code_to_index, open(f"{output_dir}/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open(f"{output_dir}/indexToCode.pkl", "wb"))
pickle.dump(id_to_group, open(f"{output_dir}/idToLabel.pkl", "wb"))
pickle.dump(train_data, open(f"{output_dir}/trainDataset.pkl", "wb"))
pickle.dump(val_data, open(f"{output_dir}/valDataset.pkl", "wb"))
pickle.dump(test_data, open(f"{output_dir}/testDataset.pkl", "wb"))

print("PREPROCESSING HOÀN TẤT! Dữ liệu đã lưu vào /kaggle/working/gpt/result/")
