


# !pip install transformers[torch]
# !pip install accelerate -U
# !pip install peft
# !pip install datasets
# !pip install huggingface_hub

# Imports
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from peft import LoraConfig, TaskType
from peft import get_peft_model
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from huggingface_hub import HfApi, login
from huggingface_hub import create_repo
from datetime import datetime

# Define the working device
device = torch.device("cuda")


""" Data load  """

train_target = "dc"

if train_target == "dc": test_label_name, df_postfix, test_input_01, test_input_02 = "Drug Class", "dc","Gene Family",  "Resistance Mechanism"
elif train_target == "gf": test_label_name, df_postfix, test_input_01, test_input_02 = "Gene Family", "gf", "Drug Class",  "Resistance Mechanism"
elif train_target == "rm": test_label_name, df_postfix, test_input_01, test_input_02 = "Resistance Mechanism", "rm", "Drug Class", "Gene Family"

# Paths for the new datasets
train_dir = "/content/drive/MyDrive/RDL/prj/data/train.csv"
test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
valid_dir = "/content/drive/MyDrive/RDL/prj/data/dev.csv"
aug_dir = "none"

NT_save_dir = "content"




train_df = pd.read_csv(f"{train_dir}")
test_df = pd.read_csv(f"{test_dir}")
valid_df = pd.read_csv(f"{valid_dir}")



""" model define """

model_name = "PoetschLab/GROVER"#"InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")#("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
max_len = tokenizer.model_max_length #512      1000

# Load the model
num_labels_mechanism = len(train_df["label"].unique()) #len(label_to_id)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels_mechanism)
model = model.to(device)

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs



""" Data preprocess  """


train_df['sequence'] = train_df['sequence'].str.upper()
test_df['sequence'] = test_df['sequence'].str.upper()
valid_df['sequence'] = valid_df['sequence'].str.upper()

# 'DNA Sequence' length limit to 1000
train_df['sequence'] = train_df['sequence'].apply(lambda x: x[:max_len])
test_df['sequence'] = test_df['sequence'].apply(lambda x: x[:max_len])
valid_df['sequence'] = valid_df['sequence'].apply(lambda x: x[:max_len])

label_to_id = {label: i for i, label in enumerate(train_df['label'].unique())}


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
valid_dataset = Dataset.from_pandas(valid_df)

# Convert labels to ids
train_dataset = train_dataset.map(lambda example: {'label': label_to_id[example['label']]})
test_dataset = test_dataset.map(lambda example: {'label': label_to_id[example['label']]})
valid_dataset = valid_dataset.map(lambda example: {'label': label_to_id[example['label']]})

# Get training data from your dataframe
train_sequences = train_dataset['sequence']
train_labels = train_dataset['label']

validation_sequences = valid_dataset['sequence']
validation_labels = valid_dataset['label']

test_sequences = test_dataset['sequence']
test_labels = test_dataset['label']

print(test_sequences)
print(test_labels)



# drug dataset
ds_train = Dataset.from_dict({"data": train_sequences,'labels':train_labels})
ds_validation = Dataset.from_dict({"data": validation_sequences,'labels':validation_labels})
ds_test = Dataset.from_dict({"data": test_sequences,'labels':test_labels})

# Creating tokenized dataset
tokenized_datasets_train = ds_train.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_datasets_validation = ds_validation.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_datasets_test = ds_test.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)

print(len(tokenized_datasets_test))



""" Lora model define """



peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha= 32, lora_dropout=0.1, target_modules= ["query", "value"]
)


lora_classifier = get_peft_model(model, peft_config)
lora_classifier.print_trainable_parameters()
lora_classifier.to(device)









""" model train """

batch_size = 8
model_name='nucleotide-transformer'
args_enhancers = TrainingArguments(
    output_dir=f"{NT_save_dir}",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps= 1,
    per_device_eval_batch_size= 64,
    num_train_epochs= 2,
    logging_steps= 100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps= 1000
)



def compute_metrics_macro_f1(eval_pred):
    """Computes macro-average F1 score for multiclass classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids

    f1_macro = f1_score(references, predictions, average="macro")

    return {'f1_macro': f1_macro}

trainer = Trainer(
    lora_classifier,
    args_enhancers,
    train_dataset= tokenized_datasets_train,
    eval_dataset= tokenized_datasets_validation,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_macro_f1,
)

train_results = trainer.train()

print(len(tokenized_datasets_test))




""" model validation """


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to GPU
model.to(device)
model.eval()



# create DataLoader 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size (adjustable)

all_predictions = []
all_true_labels = []

# list to save logits value
all_logits_values = []  # <-- 

# prediction
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = model(**inputs).logits

        # save logits values to all_logits_values list
        all_logits_values.append(logits.cpu().numpy())  # <-- 

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)
        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())


# compare results with lists
print(all_predictions)
print(len(all_predictions))




# Accuracy
accuracy = accuracy_score(all_true_labels, all_predictions)

# Balanced Accuarcy
balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predictions)

# F1 score (macro average)
f1 = f1_score(all_true_labels, all_predictions, average='macro')

# Precision (macro average)
precision = precision_score(all_true_labels, all_predictions, average='macro')

# Recall (macro average)
recall = recall_score(all_true_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")



# label_to_id func for converting text label
id_to_label = {id: label for label, id in label_to_id.items()}

# number label to text label
true_labels_str = [id_to_label[label] for label in all_true_labels]
predicted_labels_str = [id_to_label[label] for label in all_predictions]

# get the label list(order protected)
unique_labels = sorted(list(id_to_label.values()), key=lambda x: label_to_id[x])

# Confusion matrix calcualte
cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=unique_labels)

# visualize
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# # save image file
# from datetime import datetime
# now = datetime.now()
# name_part = NT_save_dir.split("/models/")[-1]
# plt.savefig(f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.png')

# # 파일 이름 정의 (예: results.txt)
# filename = f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.txt'

# # 결과를 파일에 작성
# with open(filename, "w") as file:
#     file.write(f"Accuracy: {accuracy:.4f}\n")
#     file.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
#     file.write(f"F1 Score (Macro): {f1:.4f}\n")
#     file.write(f"Precision (Macro): {precision:.4f}\n")
#     file.write(f"Recall (Macro): {recall:.4f}\n")



""" save and upload model to hf """




# save model
lora_classifier.save_pretrained(f"{NT_save_dir}")
tokenizer.save_pretrained(f"{NT_save_dir}")

# Hugging Face login
login("your hf api key")

# name your repository 
repo_name = "YOUR_REPO_NAME"
api = HfApi()
api.create_repo(repo_name, private=False)  # private=True menas private

# uploade your model and tokenizer to hf repo
lora_classifier.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)




""" hf repo model evaluation """

# load model
model = AutoModelForSequenceClassification.from_pretrained("hyoo14/GROVER_PD")

# to GPU
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("hyoo14/GROVER_PD")

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  

all_predictions = []
all_true_labels = []
all_logits_values = [] 

# prediction
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU로 
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = model(**inputs).logits

        # save logits values in all_logits_values 
        all_logits_values.append(logits.cpu().numpy())  # 

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)
        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())

# comparisoon results with list comparing
print(all_predictions)
print(len(all_predictions))

# Accuracy
accuracy = accuracy_score(all_true_labels, all_predictions)

# Balanced Accuarcy
balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predictions)

# F1 score (macro average)
f1 = f1_score(all_true_labels, all_predictions, average='macro')

# Precision (macro average)
precision = precision_score(all_true_labels, all_predictions, average='macro')

# Recall (macro average)
recall = recall_score(all_true_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")

# label_to_id func for converting text label
id_to_label = {id: label for label, id in label_to_id.items()}

# number label to text label
true_labels_str = [id_to_label[label] for label in all_true_labels]
predicted_labels_str = [id_to_label[label] for label in all_predictions]

# get the label list(order protected)
unique_labels = sorted(list(id_to_label.values()), key=lambda x: label_to_id[x])

# Confusion matrix calcualte
cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=unique_labels)

# visualize
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# save image file

# now = datetime.now()
# name_part = NT_save_dir.split("/models/")[-1]
# plt.savefig(f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.png')

# # 파일 이름 정의 (예: results.txt)
# filename = f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.txt'

# # 결과를 파일에 작성
# with open(filename, "w") as file:
#     file.write(f"Accuracy: {accuracy:.4f}\n")
#     file.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
#     file.write(f"F1 Score (Macro): {f1:.4f}\n")
#     file.write(f"Precision (Macro): {precision:.4f}\n")
#     file.write(f"Recall (Macro): {recall:.4f}\n")
