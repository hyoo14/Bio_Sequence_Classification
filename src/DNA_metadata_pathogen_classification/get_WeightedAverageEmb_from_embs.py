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
    # (N, 768) 임베딩 배열
    emb_matrix = group[[f"emb_{i}" for i in range(768)]].values.astype(np.float32)

    # (N,) abundance 배열
    weights = group["abundance"].values.astype(np.float32)

    if len(emb_matrix) == 0 or weights.sum() == 0:
        continue

    # abundance 기반 가중 평균
    weighted_mean = np.average(emb_matrix, axis=0, weights=weights)

    sample_embeddings.append(weighted_mean)
    sample_ids.append(sample_name)

# 결과 DataFrame 생성
df_sample_emb = pd.DataFrame(sample_embeddings, index=sample_ids, columns=[f"emb_{i}" for i in range(768)])
df_sample_emb.reset_index(inplace=True)
df_sample_emb.rename(columns={"index": "sample_name"}, inplace=True)



df_sample_emb.to_csv("/content/drive/MyDrive/BIO_NLP/seek_file_features/seqs_biom_files/132218_seqs_WeightedAverage_emb.csv")
