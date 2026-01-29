import os
import pandas as pd

cwd = os.getcwd()

if not cwd.endswith("unsloth-learning"):
    raise ValueError("Please run this notebook in the unsloth-learning directory")
else:
    base_dir = cwd   

save_dir = os.path.join(base_dir, "data", "training_data")
os.makedirs(save_dir, exist_ok=True)

dataset_dir = os.path.join(base_dir, "data/mai_task/military/military_vocab")

dataset_files_list = os.listdir(dataset_dir)
# Only keep .xlsx files
dataset_files_list = [f for f in dataset_files_list if f.endswith('.xlsx')]

print(dataset_files_list)

df = pd.read_excel(os.path.join(dataset_dir, dataset_files_list[0]), index_col=0)

for i in range(1, len(dataset_files_list)):
    df_tem = pd.read_excel(os.path.join(dataset_dir, dataset_files_list[i]), index_col=0)
    df = pd.concat([df, df_tem], ignore_index=True)

df = df.rename(columns={
    '범주': 'Category',
    '표제어': 'Headword',
    '원어': 'Original_Word',
    '한자': 'Hanja',
    '약어': 'Abbreviation',
    '의미': 'Meaning',
    '상위어': 'Hypernym',
    '하위어': 'Hyponym',
    '동의어': 'Synonym',
    '관련어': 'Related_Word',
    '사전명': 'Dictionary_Name',
    '출전': 'Source',
    '등록일': 'Registration_Date'
})

print("Found {} entries.".format(len(df)))
print(f"Found {df.duplicated().sum()} duplicated entries. Dropping duplicates...")

df = df.drop_duplicates()

df.to_csv(os.path.join(save_dir, "dataset.csv"), index=False)

eval_df = pd.read_excel(os.path.join(base_dir, "data/mai_task/gemma3_evaluation.xlsx"))

test_data_dir = os.path.join(base_dir, "data", "test_data")
os.makedirs(test_data_dir, exist_ok=True)

output_path_l = os.path.join(test_data_dir, "gemma3_evaluation.jsonl")
eval_df.to_json(output_path_l, orient="records", lines=True, force_ascii=False)
