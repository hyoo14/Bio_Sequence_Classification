# from google.colab import drive
# drive.mount("/content/drive")

# !pip install datasets
# !pip install Bio

# Install dependencies (uncomment if needed)
# !pip install transformers[torch] peft datasets biopython

from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from Bio.Seq import Seq
import pandas as pd
import re
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV files
train_df = pd.read_csv("/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_train_dc.csv")
valid_df = pd.read_csv("/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_val_dc.csv")
test_df = pd.read_csv("/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv")

test_label_name = "Drug Class"

# Translate DNA to protein and preprocess
def translate_dna_to_protein(seq):
    return str(Seq(seq).translate(to_stop=True))

def preprocess_protein_sequence(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    return ' '.join(list(seq))

for df in [train_df, valid_df, test_df]:
    df['Protein Sequence'] = df['DNA Sequence'].str.upper().apply(translate_dna_to_protein)
    df['Protein Sequence'] = df['Protein Sequence'].apply(preprocess_protein_sequence)

# Label encoding
label_to_id = {label: idx for idx, label in enumerate(train_df[test_label_name].unique())}
id_to_label = {v: k for k, v in label_to_id.items()}
train_df['label'] = train_df[test_label_name].map(label_to_id)
valid_df['label'] = valid_df[test_label_name].map(label_to_id)
test_df['label']  = test_df[test_label_name].map(label_to_id)

# Load ProtBert
model_name = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id))

# Create HuggingFace datasets
dataset_train = Dataset.from_dict({'data': train_df['Protein Sequence'], 'labels': train_df['label']})
dataset_valid = Dataset.from_dict({'data': valid_df['Protein Sequence'], 'labels': valid_df['label']})
dataset_test  = Dataset.from_dict({'data':  test_df['Protein Sequence'], 'labels':  test_df['label']})

def tokenize_function(examples):
    return tokenizer(examples["data"], padding="max_length", truncation=True)

tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=["data"])
tokenized_valid = dataset_valid.map(tokenize_function, batched=True, remove_columns=["data"])
tokenized_test  = dataset_test.map(tokenize_function, batched=True, remove_columns=["data"])

# Apply PEFT
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=1,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
model = get_peft_model(model, peft_config)
model.to(device)

# Training
training_args = TrainingArguments(
    output_dir="/content/protbert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir="/content/logs"
)

def compute_metrics(eval_pred):
    preds = torch.argmax(torch.tensor(eval_pred.predictions), dim=1).numpy()
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluation
model.eval()
data_collator = DataCollatorWithPadding(tokenizer)
test_loader = DataLoader(tokenized_test, collate_fn=data_collator, batch_size=32)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
        preds = torch.argmax(logits, axis=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

# Classification Report
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Macro:", f1_score(all_labels, all_preds, average='macro'))

# Confusion matrix
true_str = [id_to_label[i] for i in all_labels]
pred_str = [id_to_label[i] for i in all_preds]
labels_order = sorted(id_to_label.values(), key=lambda x: label_to_id[x])
cm = confusion_matrix(true_str, pred_str, labels=labels_order)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
