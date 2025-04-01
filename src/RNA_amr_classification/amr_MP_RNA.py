# from google.colab import drive
# drive.mount("/content/drive")

# !pip install ViennaRNA
# !pip install datasets

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일 경로
train_path = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_train_dc.csv"
valid_path = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_val_dc.csv"
test_path  = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
label_column = "Drug Class"

# 데이터 불러오기
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df  = pd.read_csv(test_path)

# RNA 전처리 함수 (U → T)
def preprocess_rna(seq):
    return seq.upper().replace("U", "T")

for df in [train_df, valid_df, test_df]:
    df["sequence"] = df["DNA Sequence"].apply(preprocess_rna)

# 라벨 인코딩
label_to_id = {label: i for i, label in enumerate(train_df[label_column].unique())}
id_to_label = {v: k for k, v in label_to_id.items()}
for df in [train_df, valid_df, test_df]:
    df["label"] = df[label_column].map(label_to_id)

# Hugging Face Dataset 변환
dataset_train = Dataset.from_dict({'text': train_df['sequence'], 'label': train_df['label']})
dataset_valid = Dataset.from_dict({'text': valid_df['sequence'], 'label': valid_df['label']})
dataset_test  = Dataset.from_dict({'text': test_df['sequence'],  'label': test_df['label']})

# MP-RNA Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("yangheng/MP-RNA")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_tok = dataset_train.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_tok = dataset_valid.map(tokenize_fn, batched=True, remove_columns=["text"])
test_tok  = dataset_test.map(tokenize_fn,  batched=True, remove_columns=["text"])

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn as nn

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("yangheng/MP-RNA")

# 기본 MP-RNA 모델 로드
base_model = AutoModel.from_pretrained("yangheng/MP-RNA")


#### v1. one layer finetune
# # 커스텀 분류 모델 정의
# class CustomMPRNAForSequenceClassification(nn.Module):
#     def __init__(self, base_model, num_labels):
#         super().__init__()
#         self.base_model = base_model
#         self.num_labels = num_labels
#         # MP-RNA의 hidden_size를 확인해야 함 (예: 768)
#         self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[0][:, 0, :]  # [CLS] 토큰의 출력 사용 (모델 구조에 따라 다를 수 있음)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}

# # 모델 인스턴스 생성
# model = CustomMPRNAForSequenceClassification(base_model, num_labels=len(label_to_id)).to(device)

## v2. weighted label one layer fine tune -> much much better performance
# 클래스별 샘플 수
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


class CustomMPRNAForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, class_weights=None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}


model = CustomMPRNAForSequenceClassification(base_model, num_labels=len(label_to_id), class_weights=class_weights_tensor).to(device)






# 평가 지표
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# Trainer 설정
# training_args = TrainingArguments(
#     output_dir="./mp_rna_output",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-4,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=64,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1_macro",
#     logging_dir="./logs"
# )
training_args = TrainingArguments(
    output_dir="./mp_rna_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4, #5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=5, #3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# 훈련
trainer.train()

# 테스트 예측
predictions = trainer.predict(test_tok)
preds = np.argmax(predictions.predictions, axis=1)
true = predictions.label_ids

# 결과 출력
print("Accuracy:", accuracy_score(true, preds))
print("Balanced Accuracy:", balanced_accuracy_score(true, preds))
print("F1 (Macro):", f1_score(true, preds, average="macro"))
print("Precision:", precision_score(true, preds, average="macro"))
print("Recall:", recall_score(true, preds, average="macro"))

# Confusion matrix
true_str = [id_to_label[i] for i in true]
pred_str = [id_to_label[i] for i in preds]
labels_order = sorted(id_to_label.values(), key=lambda x: label_to_id[x])

cm = confusion_matrix(true_str, pred_str, labels=labels_order)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("mp_rna_confusion_matrix.png")
plt.show()
