

# !pip install transformers
# !pip install datasets
# !pip install triton==2.0.0.dev20221202
# !pip install einops
# !pip install accelerate -U
# !pip install peft


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertConfig
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime





# Define the working device
device = torch.device("cuda")

"""## Data"""

#model_name = "zhihan1996/DNABERT-2-117M"
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

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

train_df = pd.read_csv(f"{train_dir}")
test_df = pd.read_csv(f"{test_dir}")
valid_df = pd.read_csv(f"{valid_dir}")

if aug_dir != "none":
    aug_df = pd.read_csv(f"{aug_dir}")
    train_df = pd.concat([train_df, aug_df])
    print("---------------------yes augmentation")
else:
    print("---------------------no augmentation")

train_df['sequence'] = train_df['sequence'].str.upper()
test_df['sequence'] = test_df['sequence'].str.upper()
valid_df['sequence'] = valid_df['sequence'].str.upper()
# 'DNA Sequence' seq limitation as 510 because total 512 but it includes cls token and  sep token
# 'DNA Sequence' length limit to 1000
train_df['sequence'] = train_df['sequence'].apply(lambda x: x[:510])
test_df['sequence'] = test_df['sequence'].apply(lambda x: x[:510])
valid_df['sequence'] = valid_df['sequence'].apply(lambda x: x[:510])

label_to_id = {label: i for i, label in enumerate(train_df['label'].unique())}


from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

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





tokenizer = AutoTokenizer.from_pretrained("hyoo14/DNABERT2_PD", model_max_length=512)
def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs



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






# print(valid_dataset)

num_labels_mechanism = len(label_to_id)

# Load the model
# load Config for setting (mendatory for DNABERT2)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
config.num_labels = len(label_to_id)  # 레이블의 수를 설정합니다.

# model load by using config for concordance(DNABERT2 mendatory를)
model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", config=config)
model.to('cuda')  # to GPU 


from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha= 32, lora_dropout=0.1, target_modules= ["query", "value"],
    #modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
)

from peft import get_peft_model

lora_classifier = get_peft_model(model, peft_config) # transform our classifier into a peft model
lora_classifier.print_trainable_parameters()
lora_classifier.to(device) # Put the model on the GPU

# Load the tokenizer


tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", model_max_length=512)





# 1Base Model Freeze 
for param in lora_classifier.base_model.parameters():
    param.requires_grad = False  # base model freeze

#  LoRA weight only (trainable)
for name, param in lora_classifier.named_parameters():
    if "lora" in name:  # LoRA only
        param.requires_grad = True



""" train model """

import torch
import random
import numpy as np

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

batch_size = 8
model_name='dnabert-2'
args_enhancers = TrainingArguments(
    #f"{model_name}-finetuned-lora-NucleotideTransformer",
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
    load_best_model_at_end=True,  # Keep the best model according to the evaluation
    metric_for_best_model="f1_macro",#"mcc_score", # The mcc_score on the evaluation dataset used to select the best model
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps= 1000,
    seed = seed

)

from sklearn.metrics import f1_score
import numpy as np

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
    #compute_metrics=compute_metrics_mcc,
    compute_metrics=compute_metrics_macro_f1,
)

train_results = trainer.train()




print(len(tokenized_datasets_test))


""" evaluate the model """


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model to GPU
lora_classifier.to(device)
lora_classifier.eval()

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# DataLoader 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size 

all_predictions = []
all_true_labels = []

# 
all_logits_values = []  # <-- 

# prediction
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = lora_classifier(**inputs).logits

        # logits value save list
        all_logits_values.append(logits.cpu().numpy())  #

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)
        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())

"""### logit info

"""

print(all_predictions)

print(len(all_predictions))

"""## Nucleotide result"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
from datetime import datetime
now = datetime.now()
name_part = NT_save_dir.split("/models/")[-1]
plt.savefig(f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.png')


# 
filename = f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.txt'

# 
with open(filename, "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"F1 Score (Macro): {f1:.4f}\n")
    file.write(f"Precision (Macro): {precision:.4f}\n")
    file.write(f"Recall (Macro): {recall:.4f}\n")



""" save the model(save the state dict) """

from safetensors.torch import save_file, load_file


# save dir setting
save_path = "/content/content_lora/lora_state_dict.safetensors"

# save model weight as a state dict
save_file(lora_classifier.state_dict(), save_path)

print(f"LoRA model state_dict saved to {save_path}")

# Hugging Face login
from huggingface_hub import HfApi, login


login("YOUR KEY")



# Hugging Face repo 
repo_id = "hyoo14/DNABERT2_PD"

# upload file path
local_file_path = "/content/content_lora/lora_state_dict.safetensors"

# Hugging Face Hub repo update
api = HfApi()
api.upload_file(
    path_or_fileobj=local_file_path,  # local dir
    path_in_repo="lora_state_dict.safetensors",  # saved file name and dir
    repo_id=repo_id
)

print(f"File successfully uploaded to Hugging Face Hub: {repo_id}/lora_state_dict.safetensors")

tokenizer.push_to_hub("hyoo14/DNABERT2_PD")



""" load the model and state dict """

# Hugging Face login
from huggingface_hub import HfApi, login

login("YOUR KEY")

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, load_file

#  `lora_state_dict.safetensors` download
downloaded_file = hf_hub_download(repo_id="hyoo14/DNABERT2_PD",
                                  filename="lora_state_dict.safetensors",
                                  local_dir="/content/")

print(f"File successfully downloaded to: {downloaded_file}")


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from peft import get_peft_model, LoraConfig, TaskType


hf_model_name = "zhihan1996/DNABERT-2-117M"  # DNABERT-2 model

# BertConfig load
config = BertConfig.from_pretrained(hf_model_name)
config.num_labels = len(label_to_id)

# DNABERT-2 model load
base_model3 = AutoModelForSequenceClassification.from_pretrained(hf_model_name, config=config)



peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=1,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

lora_classifier3 = get_peft_model(base_model3, peft_config)
lora_classifier3.print_trainable_parameters()


# 1Base Model Freeze 
for param in lora_classifier3.base_model.parameters():
    param.requires_grad = False  # base model freezing

#  LoRA weight is made trainable
for name, param in lora_classifier3.named_parameters():
    if "lora" in name:  # LoRA weight only
        param.requires_grad = True


# load state_dict
load_path = "/content/lora_state_dict.safetensors"
loaded_state_dict3 = load_file(load_path)
lora_classifier3.load_state_dict(loaded_state_dict3)


""" evaluate again """


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to GPU
lora_classifier3.to(device)
lora_classifier3.eval()

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# DataLoader create
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_datasets_test, collate_fn=data_collator, batch_size=32)  # batch_size adjustable

all_predictions = []
all_true_labels = []

# logits values save list
all_logits_values = []  # <-- 


# prediction
with torch.no_grad():
    for batch in test_dataloader:
        # data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}

        logits = lora_classifier3(**inputs).logits

        # list for saving logits values
        all_logits_values.append(logits.cpu().numpy())  # 

        predicted_labels_batch = np.argmax(logits.cpu().numpy(), axis=-1)
        all_predictions.extend(predicted_labels_batch)
        all_true_labels.extend(batch["labels"].cpu().numpy())

"""### logit info

"""

print(all_predictions)

print(len(all_predictions))

"""## Nucleotide result"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
from datetime import datetime
now = datetime.now()
name_part = NT_save_dir.split("/models/")[-1]
plt.savefig(f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.png')


# 
filename = f'{name_part}_{now.strftime("%Y_%m_%d__%H_%M_%S")}.txt'

# 
with open(filename, "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"F1 Score (Macro): {f1:.4f}\n")
    file.write(f"Precision (Macro): {precision:.4f}\n")
    file.write(f"Recall (Macro): {recall:.4f}\n")



