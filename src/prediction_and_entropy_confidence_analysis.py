
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
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from peft import LoraConfig, TaskType
from peft import get_peft_model
from sklearn.metrics import f1_score
import numpy as np
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import HfApi, login
from huggingface_hub import create_repo
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Define the working device
device = torch.device("cuda")





""" Data load  """

train_target = "dc"

if train_target == "dc": test_label_name, df_postfix, test_input_01, test_input_02 = "Drug Class", "dc","Gene Family",  "Resistance Mechanism"
elif train_target == "gf": test_label_name, df_postfix, test_input_01, test_input_02 = "Gene Family", "gf", "Drug Class",  "Resistance Mechanism"
elif train_target == "rm": test_label_name, df_postfix, test_input_01, test_input_02 = "Resistance Mechanism", "rm", "Drug Class", "Gene Family"

# Paths for the new datasets
train_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_train_dc.csv"
test_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
valid_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_val_dc.csv"

aug_dir = "none"

NT_save_dir = "content"



train_df = pd.read_csv(f"{train_dir}")
test_df = pd.read_csv(f"{test_dir}")
valid_df = pd.read_csv(f"{valid_dir}")

target = 'label'


""" model define """

# model_name = "PoetschLab/GROVER"#"InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
# tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")#("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

# def tokenize_function(examples):
#     outputs = tokenizer(examples["data"])
#     return outputs

max_len = 1000 #tokenizer.model_max_length #512      #
# Load the model
num_labels_mechanism = len(train_df["label"].unique()) #len(label_to_id)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels_mechanism)
# model = model.to(device)



tokenizer = AutoTokenizer.from_pretrained("hyoo14/GROVER_AMR")#("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")#("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs




""" Data preprocess  """


# train_df['sequence'] = train_df['sequence'].str.upper()
# test_df['sequence'] = test_df['sequence'].str.upper()
# valid_df['sequence'] = valid_df['sequence'].str.upper()
train_df['DNA Sequence'] = train_df['DNA Sequence'].str.upper()
test_df['DNA Sequence'] = test_df['DNA Sequence'].str.upper()
valid_df['DNA Sequence'] = valid_df['DNA Sequence'].str.upper()


# 'DNA Sequence' length limit to 1000
# train_df['sequence'] = train_df['sequence'].apply(lambda x: x[:max_len])
# test_df['sequence'] = test_df['sequence'].apply(lambda x: x[:max_len])
# valid_df['sequence'] = valid_df['sequence'].apply(lambda x: x[:max_len])
train_df['DNA Sequence'] = train_df['DNA Sequence'].apply(lambda x: x[:max_len])
test_df['DNA Sequence'] = test_df['DNA Sequence'].apply(lambda x: x[:max_len])
valid_df['DNA Sequence'] = valid_df['DNA Sequence'].apply(lambda x: x[:max_len])

# label_to_id = {label: i for i, label in enumerate(train_df['label'].unique())}
drug_id_to_label = train_df.drop_duplicates().set_index('label')[test_label_name].to_dict()
label_to_id = {v: k for k, v in drug_id_to_label.items()}








train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
valid_dataset = Dataset.from_pandas(valid_df)

# Convert labels to ids
# train_dataset = train_dataset.map(lambda example: {'label': label_to_id[example['label']]})
# test_dataset = test_dataset.map(lambda example: {'label': label_to_id[example['label']]})
# valid_dataset = valid_dataset.map(lambda example: {'label': label_to_id[example['label']]})
train_dataset = train_dataset.map(lambda example: {test_label_name: label_to_id[example[test_label_name]]})
test_dataset = test_dataset.map(lambda example: {test_label_name: label_to_id[example[test_label_name]]})
valid_dataset = valid_dataset.map(lambda example: {test_label_name: label_to_id[example[test_label_name]]})

# Get training data from your dataframe
# train_sequences = train_dataset['sequence']
# train_labels = train_dataset['label']

# validation_sequences = valid_dataset['sequence']
# validation_labels = valid_dataset['label']

# test_sequences = test_dataset['sequence']
# test_labels = test_dataset['label']
train_sequences = train_dataset['DNA Sequence']
train_labels = train_dataset[test_label_name]

validation_sequences = valid_dataset['DNA Sequence']
validation_labels = valid_dataset[test_label_name]

test_sequences = test_dataset['DNA Sequence']
test_labels = test_dataset[test_label_name]

print(test_sequences)
print(test_labels)


# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
#tokenizer = AutoTokenizer.from_pretrained("hyoo14/NucletideTransformer_PD")




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



from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# create DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size (adjustable)




""" hf model evaluation """

num_labels_mechanism = len(train_df["label"].unique()) #len(label_to_id)
model = AutoModelForSequenceClassification.from_pretrained("hyoo14/GROVER_AMR", num_labels=num_labels_mechanism)
tokenizer = AutoTokenizer.from_pretrained("hyoo14/GROVER_AMR")

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs

# model to GPU
model.to(device)
model.eval()

# createe DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size 조절 가능

all_predictions = []
all_true_labels = []

#
all_logits_values = []  #

# prediciton
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = model(**inputs).logits

        #
        all_logits_values.append(logits.cpu().numpy())  #

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)
        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())

"""### logit info"""

print(all_predictions)
print(len(all_predictions))

"""## Nucleotide result"""



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


##################################
# entropy and confidence analysis#
##################################


from scipy.special import softmax
from scipy.stats import entropy

def get_entropy_and_confidence(logits_batch):
    probs = softmax(logits_batch, axis=-1)
    entropies = entropy(probs, axis=-1)  # shape: (batch_size,)
    top1_conf = np.max(probs, axis=-1)   # shape: (batch_size,)
    return entropies, top1_conf



""" hf model evaluation """

num_labels_mechanism = len(train_df["label"].unique()) #len(label_to_id)
model = AutoModelForSequenceClassification.from_pretrained("hyoo14/GROVER_AMR", num_labels=num_labels_mechanism)
tokenizer = AutoTokenizer.from_pretrained("hyoo14/GROVER_AMR")

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs

# model to GPU
model.to(device)
model.eval()

# createe DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size 조절 가능

all_predictions = []
all_true_labels = []
all_entropies = []
all_confidences = []
#
all_logits_values = []  #

# prediciton
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = model(**inputs).logits

        #
        all_logits_values.append(logits.cpu().numpy())  #

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)

        # 추가: entropy 및 confidence 계산
        logits_np = logits.cpu().numpy()
        ent_batch, conf_batch = get_entropy_and_confidence(logits_np)
        all_entropies.extend(ent_batch)
        all_confidences.extend(conf_batch)

        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())

"""### logit info"""

print(all_predictions)
print(len(all_predictions))

"""## Nucleotide result"""



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

print(f"Mean Entropy: {np.mean(all_entropies):.4f}")
print(f"Std Entropy: {np.std(all_entropies):.4f}")
print(f"Mean Top-1 Confidence: {np.mean(all_confidences):.4f}")
print(f"Std Top-1 Confidence: {np.std(all_confidences):.4f}")


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

plt.hist(all_entropies, bins=30, alpha=0.7)
plt.title("Histogram of Prediction Entropy")
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.show()

plt.hist(all_confidences, bins=30, alpha=0.7)
plt.title("Histogram of Top-1 Confidence")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()

