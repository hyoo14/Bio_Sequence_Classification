# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/BIO_NLP/seek_file_features/seqs_biom_files/132218_seqs_all.csv")

import numpy as np
import torch

from tqdm import tqdm
import pandas as pd

# sample 단위로 그룹화
grouped = df.groupby("sample_name")

sample_embeddings = []
sample_ids = []



for sample_name, group in tqdm(grouped):
    # 임베딩만 추출 (abundance 무시)
    emb_matrix = group[[f"emb_{i}" for i in range(768)]].values.astype(np.float32)

    if len(emb_matrix) == 0:
        continue

    # 평균 벡터 계산
    mean_embedding = emb_matrix.mean(axis=0)

    sample_embeddings.append(mean_embedding)
    sample_ids.append(sample_name)

# 결과 DataFrame
df_sample_emb = pd.DataFrame(sample_embeddings, index=sample_ids, columns=[f"emb_{i}" for i in range(768)])
df_sample_emb.reset_index(inplace=True)
df_sample_emb.rename(columns={"index": "sample_name"}, inplace=True)


df_sample_emb.to_csv("/content/drive/MyDrive/BIO_NLP/seek_file_features/seqs_biom_files/132218_seqs_Average_emb.csv", index=False)
