# from google.colab import drive
# drive.mount('/content/drive')

# !mkdir -p set_transformer
# !wget https://raw.githubusercontent.com/juho-lee/set_transformer/master/models.py -O set_transformer/models.py


import sys
sys.path.append('/content/set_transformer')


from models import SetTransformer


model = SetTransformer(dim_input=768, num_outputs=1, dim_output=768)


import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/BIO_NLP/seek_file_features/seqs_biom_files/132218_seqs_all.csv")

import numpy as np
import torch
from models import SetTransformer
from tqdm import tqdm
import pandas as pd

# sample 단위로 그룹화
grouped = df.groupby("sample_name")

sample_embeddings = []
sample_ids = []



import torch


# Set Transformer 인스턴스 (공통으로 사용)
model = SetTransformer(dim_input=768, num_outputs=1, dim_output=768)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()



for sample_name, group in tqdm(grouped):
    repeated_embeddings = []

    for _, row in group.iterrows():
        emb_vector = row[[f"emb_{i}" for i in range(768)]].values.astype(np.float32)
        abundance = int(row["abundance"])
        repeated_embeddings.extend([emb_vector] * abundance)

    if not repeated_embeddings:
        continue

    input_tensor = torch.tensor(repeated_embeddings).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    sample_embeddings.append(output.squeeze(0).squeeze(0).cpu().numpy())
    sample_ids.append(sample_name)


# 결과 DataFrame
df_sample_emb = pd.DataFrame(sample_embeddings, index=sample_ids, columns=[f"emb_{i}" for i in range(768)])
df_sample_emb.reset_index(inplace=True)
df_sample_emb.rename(columns={"index": "sample_name"}, inplace=True)


df_sample_emb.to_csv("/content/drive/MyDrive/BIO_NLP/seek_file_features/seqs_biom_files/132218_seqs_SetTransformer_emb.csv", index=False)
