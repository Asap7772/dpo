import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict


# Read the CSV file
csv_path = '/u/asingh15/asap7772/dpo/dataset.csv'
df = pd.read_csv(csv_path)
all_columns = df.columns

short_level_map={'elementary':0, 'middle':1, 'high':2, 'college':3, 'expert':4, 'unprompted':5}
level_map={}

# Map the level to integer
for x in df.level.unique():
    for y in short_level_map:
        if y in x.lower():
            level_map[x]=short_level_map[y]
            break

df['level_int'] = df['level'].map(level_map)

# First create SFT split with unprompted level
df_unprompted = df[df.level_int==5]
dataset_sft = Dataset.from_pandas(df_unprompted)
def remap_keys(example):
    new_example = dict(
        x=example['question'],
        y=example['output'],
        model=example['model']
    )
    return new_example
dataset_sft = dataset_sft.map(remap_keys, batched=True, remove_columns=['question', 'output', 'model', 'level', 'output', 'api_key', 'raw_output', 'level_int', '__index_level_0__',])
print(dataset_sft[0])

# Then create AutoLabel with Mistral Model
# collums to keep: question, output, model, level_int
df = df[['question', 'output', 'model', 'level', 'level_int']]
df_autolabel = df[df.model == 'mistralai/Mistral-7B-Instruct-v0.2']
df_autolabel = df_autolabel[df_autolabel.level_int != 5]
df_autolabel = df_autolabel.merge(df_autolabel, on='question')
df_autolabel = df_autolabel[df_autolabel.output_x != df_autolabel.output_y]
df_autolabel = df_autolabel[df_autolabel.level_int_x != df_autolabel.level_int_y]
df_autolabel = df_autolabel.loc[df_autolabel.index.repeat(5)]
df_autolabel['level_int'] = np.tile([0,1,2,3,4], len(df_autolabel)//5)
df_autolabel['preferred'] =  abs(df_autolabel.level_int_y - df_autolabel.level_int) < abs(df_autolabel.level_int_x - df_autolabel.level_int)
df_autolabel['preferred'] = df_autolabel['preferred'].astype(int) + 1
# remove the _x and _y suffixes
df_autolabel.columns = df_autolabel.columns.str.replace('_x', '_1')
df_autolabel.columns = df_autolabel.columns.str.replace('_y', '_2')

dataset_autolabel = Dataset.from_pandas(df_autolabel)
def remap_keys(example):
    if example['preferred'] == 1:
        new_example = dict(
            x=example['question'],
            yw=example['output_1'],
            model_yw=example['model_1'],
            level_yw=example['level_1'],
            yl=example['output_2'],
            model_yl=example['model_2'],
            level_yl=example['level_2'],
            level=example['level_int']
        )
    else:
        new_example = dict(
            x=example['question'],
            yw=example['output_2'],
            model_yw=example['model_2'],
            level_yw=example['level_2'],
            yl=example['output_1'],
            model_yl=example['model_1'],
            level_yl=example['level_1'],
            level=example['level_int']
        )
    return new_example
dataset_autolabel = dataset_autolabel.map(remap_keys, batched=True, remove_columns=['question', 'output_1', 'output_2', 'model_1', 'model_2', 'level_1', 'level_2', 'level_int', 'level_int_1', 'level_int_2', 'preferred', '__index_level_0__'])
print(dataset_autolabel[0])

# create train test split
dataset_sft = dataset_sft.train_test_split(test_size=0.2, seed=42)
dataset_autolabel = dataset_autolabel.train_test_split(test_size=0.2, seed=42)

# save the datasets
token='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'
dataset_sft.push_to_hub('Asap7772/education_sft', token=token)
dataset_autolabel.push_to_hub('Asap7772/education_autolabel', token=token)