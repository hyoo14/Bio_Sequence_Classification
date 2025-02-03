
# from google.colab import drive
# drive.mount("/content/drive")

# # library
# !pip install transformers
# !pip install datasets
# !pip install triton==2.0.0.dev20221202
# !pip install einops
# !pip install accelerate -U
# !pip install peft


from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, BertConfig
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib.pyplot as plt
from datetime import datetime
import random
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import HfApi, login, hf_hub_download
from safetensors.torch import save_file, load_file



# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 및 Tokenizer 설정



# Constants




## PD
# task_name = "PD"
# test_file_path = test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# train_dir = "/content/drive/MyDrive/RDL/prj/data/train.csv"
# test_label_name = "label"

# train_df = pd.read_csv(train_dir)
# label_to_id = {label: i for i, label in enumerate(train_df[test_label_name].unique())}# Create label-to-id mapping


#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER              OK!!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2        OK!!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER             DONE???
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2       OK!!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT      DONE???

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2          OK!!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT



## AMR
task_name = "AMR"
test_file_path = test_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
train_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_train_dc.csv"
test_label_name = "Drug Class"

train_df = pd.read_csv(train_dir)
drug_id_to_label = train_df.drop_duplicates().set_index('label')[test_label_name].to_dict()
label_to_id = {v: k for k, v in drug_id_to_label.items()}


#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2     OK!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2     OK!!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2       DONE???
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT      DONE???





tokenizer = AutoTokenizer.from_pretrained(model_name)


if model_name == "hyoo14/DNABERT2_PD" or model_name == "hyoo14/DNABERT2_AMR":
    # Hugging Face에 로그인합니다.


    # Hugging Face 웹사이트에서 생성한 토큰을 사용해 로그인합니다.
    login("YOUR KEY")

    # Hugging Face에서 `lora_state_dict.safetensors` 다운로드
    downloaded_file = hf_hub_download(repo_id=model_name,
                                      filename="lora_state_dict.safetensors",
                                      local_dir="/content/")
    print(f"File successfully downloaded to: {downloaded_file}")

    NT_save_dir = "content_lora"  # 저장된 모델 경로
    hf_model_name = "zhihan1996/DNABERT-2-117M"  # DNABERT-2 원본 모델

    # BertConfig 불러오기 (레이블 수 설정)
    config = BertConfig.from_pretrained(hf_model_name)
    config.num_labels = len(label_to_id)

    # DNABERT-2 모델 불러오기
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

    # 1Base Model Freeze (기본 모델 고정)
    for param in lora_classifier3.base_model.parameters():
        param.requires_grad = False  # 기본 모델의 모든 가중치 동결

    #  LoRA 가중치만 학습 가능하게 설정
    for name, param in lora_classifier3.named_parameters():
        if "lora" in name:  # LoRA 관련 가중치만 업데이트
            param.requires_grad = True

    # 저장된 state_dict 불러오기
    load_path = "/content/lora_state_dict.safetensors"
    loaded_state_dict3 = load_file(load_path)
    lora_classifier3.load_state_dict(loaded_state_dict3)

    model = lora_classifier3
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id)).to(device)


output_dir = "content"






# 데이터 전처리 함수
def preprocess_data(file_path, max_len):
    """Load and preprocess data from a CSV file."""
    df = pd.read_csv(file_path)
    df[input_name] = df[input_name].str.upper()
    df[input_name] = df[input_name].apply(lambda x: x[:max_len])
    return df

# Dataset 및 DataLoader 생성 함수
def create_dataloader(df, label_to_id, batch_size):
    """Create a tokenized DataLoader from a dataframe."""
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {test_label_name: label_to_id[x[test_label_name]]})
    dataset = Dataset.from_dict({"data": dataset[input_name], 'labels': dataset[test_label_name]})

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["data"], padding=True, truncation=True),
        batched=True,
        remove_columns=["data"]
    )

    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        collate_fn=data_collator,
        batch_size=batch_size
    )
    return dataloader

# Prediction 함수
def predict(model, dataloader, device):
    """Run prediction on the dataloader and return predictions and true labels."""
    model.to(device)
    model.eval()
    all_predictions, all_true_labels, all_logits = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != 'labels'}

            logits = model(**inputs).logits
            predictions = np.argmax(logits.cpu().numpy(), axis=-1)

            all_predictions.extend(predictions)
            all_true_labels.extend(batch['labels'].cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return all_predictions, all_true_labels, np.concatenate(all_logits)

# 평가 및 시각화 함수
def evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir):
    """Evaluate performance metrics and visualize confusion matrix."""
    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    # Confusion Matrix
    id_to_label = {v: k for k, v in label_to_id.items()}
    true_labels_str = [id_to_label[label] for label in true_labels]
    predicted_labels_str = [id_to_label[label] for label in predictions]
    unique_labels = sorted(label_to_id.keys(), key=lambda x: label_to_id[x])

    cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=unique_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Save image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    now = datetime.now()
    plt.savefig(f'{output_dir}/confusion_matrix_{now.strftime("%Y%m%d_%H%M%S")}.png')

    # Save metrics to file
    with open(f'{output_dir}/results_{now.strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"F1 Score (Macro): {f1:.4f}\n")
        f.write(f"Precision (Macro): {precision:.4f}\n")
        f.write(f"Recall (Macro): {recall:.4f}\n")




# ATTACK


# Define mutation functions
def nucleotide_mutation(sequence, mutation_rate=0.1):
    """Introduce random mutations into a nucleotide sequence."""
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice('ATCG')
    return ''.join(sequence)

def nucleotide_attack(sequences, mutation_rate=0.1, iteration=1):
    """Apply mutations to a list of sequences."""
    mutated_sequences = sequences.copy()  # copy original
    for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: nucleotide_mutation(seq, mutation_rate)
        )
    return mutated_sequences


def codon_mutation(sequence, mutation_rate=0.1):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    for i in range(len(codons)):
        if random.random() < mutation_rate:
            codons[i] = ''.join(random.choices('ATCG', k=3))
    return ''.join(codons)


def codon_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_mutation(seq, mutation_rate)
        )
  return mutated_sequences




# DNA codon to amino acid translation table
dna_to_aa_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

# Reverse translation table from amino acids to DNA codons
aa_to_dna_table = {v: [k for k in dna_to_aa_table if dna_to_aa_table[k] == v] for v in set(dna_to_aa_table.values())}

def dna_to_aa(sequence):
    return ''.join(dna_to_aa_table.get(sequence[i:i+3], 'X') for i in range(0, len(sequence), 3))

def aa_to_dna(aa_sequence):
    return ''.join(np.random.choice(aa_to_dna_table[aa]) for aa in aa_sequence if aa in aa_to_dna_table)

def back_translation(sequence):
    aa_sequence = dna_to_aa(sequence)
    #print(aa_sequence)
    translated_dna = aa_to_dna(aa_sequence)
    #print(translated_dna)
    return translated_dna

def backtranslation_attack(sequences, mutation_rate=None, iteration=None):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: back_translation(seq)
        )
  return mutated_sequences




def update_adv_datasets_cascading(original_predictions, original_labels, adv_files, output_files, max_len):
    """
    Cascadingly update ADV datasets to ensure successful adversarial attacks are consistent.

    Parameters:
    - original_predictions: List of predictions from the original test set.
    - original_labels: List of true labels from the original test set.
    - adv_files: List of ADV dataset file paths (ADV 0.1 ~ 0.5).
    - output_files: List of output file paths for updated ADV datasets.
    - max_len: Maximum sequence length.

    Returns:
    - None (Saves updated datasets to disk)
    """

    # Track successful adversarial attack sequences
    successful_sequences = {}

    # Process ADV datasets step by step
    for idx, (adv_file, output_file) in enumerate(zip(adv_files, output_files)):
        print(f"Processing {adv_file}...")

        # Load current ADV dataset
        adv_data = preprocess_data(adv_file, max_len)

        # For ADV 0.1, compare with original predictions
        if idx == 0:
            for i, (pred, label) in enumerate(zip(original_predictions, original_labels)):
                if pred == label:  # Originally correct
                    adv_pred = predict_single_sequence(adv_data[input_name].iloc[i], model, tokenizer, device)
                    if adv_pred != label:  # ADV 0.1에서 공격 성공
                        successful_sequences[adv_data[input_name].iloc[i]] = label
        else:
            # For ADV 0.2 onwards, compare with previous ADV results
            for i, seq in enumerate(adv_data[input_name]):
                if seq in successful_sequences:
                    # If sequence already marked as successful, retain it
                    adv_data.at[i, input_name] = seq
                    adv_data.at[i, test_label_name] = successful_sequences[seq]
                else:
                    # Check if attack succeeds in this step
                    adv_pred = predict_single_sequence(seq, model, tokenizer, device)
                    if adv_pred != adv_data[test_label_name].iloc[i]:  # 공격 성공
                        successful_sequences[seq] = adv_data[test_label_name].iloc[i]

        # Save updated dataset
        adv_data.to_csv(output_file, index=False)
        print(f"Updated dataset saved to {output_file}")

    print("All ADV datasets updated successfully!")





# 적대적 공격 성공률 계산 함수
def calculate_attack_success_rate(true_labels, original_predictions, attack_predictions):
    """
    Calculate the success rate of adversarial attacks.

    성공 정의:
    - 원본 데이터에서 예측 레이블 == 실제 레이블 (정확하게 분류됨)
    - 적대적 공격 후 예측 레이블 != 실제 레이블 (오분류됨)
    """
    # 원래 모델이 정확히 분류했던 샘플 중 공격 후 오분류된 샘플 수
    successful_attacks = sum(
        1 for true, orig_pred, atk_pred in zip(true_labels, original_predictions, attack_predictions)
        if true == orig_pred and true != atk_pred
    )
    # 원래 모델이 정확히 분류했던 샘플 수
    total_correct = sum(1 for true, orig_pred in zip(true_labels, original_predictions) if true == orig_pred)

    success_rate = successful_attacks / total_correct if total_correct > 0 else 0.0

    print(f"Adversarial Attack Success Rate: {success_rate:.4f}")
    return success_rate




def update_sequences_with_attack_single(sequences, true_labels, model, tokenizer, device, mutation_rate, max_iterations, save_intervals, path):
    """
    Perform iterative adversarial attacks on sequences one by one using predict_single_sequence.

    Parameters:
    - sequences: Original sequences (Pandas Series).
    - true_labels: True labels corresponding to the sequences.
    - model: Trained model for predictions.
    - tokenizer: Tokenizer for the model.
    - device: Device (CPU/GPU).
    - mutation_rate: Mutation rate for adversarial attacks.
    - max_iterations: Total number of attack iterations.
    - save_intervals: Iteration intervals for saving results.

    Returns:
    - None (Saves intermediate and final results to disk).
    """
    successful_sequences = {}  # Dictionary to track successful attacks
    current_sequences = sequences.copy()  # Current sequences to mutate
    all_results = []  # To store results for saving

    for iteration in range(1, max_iterations + 1):
        print(f"Starting iteration {iteration}/{max_iterations}...")

        for idx in range(len(current_sequences)):
            seq = current_sequences.iloc[idx]

            # Skip already successful sequences
            if seq in successful_sequences:
                continue

            # Apply mutation to the sequence
            if attack_name =="nucl":
              mutated_seq = nucleotide_mutation(seq, mutation_rate)
            elif attack_name =="codon":
              mutated_seq = codon_mutation(seq, mutation_rate)
            elif attack_name =="bt":
              mutated_seq = back_translation(seq)

            # Predict the mutated sequence
            pred = predict_single_sequence(mutated_seq, model, tokenizer, device)

            # Check for adversarial success
            if pred != true_labels.iloc[idx]:  # If attack is successful
                successful_sequences[seq] = true_labels.iloc[idx]  # Mark as successful
                current_sequences.iloc[idx] = mutated_seq  # Save mutated version

            # GPU 메모리 정리
            torch.cuda.empty_cache()

        # Save intermediate results at specified intervals
        if iteration in save_intervals:
            save_results(current_sequences, successful_sequences, iteration, mutation_rate, path)
            print(f"Iteration {iteration}: Results saved!")

    print("Adversarial attack completed.")

def predict_single_sequence(sequence, model, tokenizer, device):
    """
    Predict a single sequence using the model.

    Parameters:
    - sequence: Input sequence.
    - model: Model for prediction.
    - tokenizer: Tokenizer for the model.
    - device: Device (CPU/GPU).

    Returns:
    - Predicted label.
    """
    model.to(device)
    model.eval()
    tokens = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
    return torch.argmax(logits, axis=-1).item()

def save_results(sequences, successful_sequences, iteration, mutation_rate, path):
    """
    Save intermediate results to a CSV file.

    Parameters:
    - sequences: Current sequences.
    - successful_sequences: Dictionary of successful sequences.
    - iteration: Current iteration.
    - mutation_rate: Mutation rate.
    """
    result_df = pd.DataFrame({
        input_name: sequences,
        'success': sequences.apply(lambda seq: seq in successful_sequences)
    })
    file_path = f"{path}{iteration * 10}_rate{mutation_rate}.csv"
    result_df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")






print(f"{attack_name}, {task_name}, {model_short_name}")
if __name__ == "__main__":
    # Load original test data

    test_data = preprocess_data(test_file_path, max_len)
    sequences = test_data[input_name]
    true_labels = test_data[test_label_name]

    # Mutation and iteration settings
    mutation_rate = 0.1
    max_iterations = 50
    save_intervals = [10, 20, 30, 40, 50]  # Save results at these iterations

    path = f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter"  # 저장 경로

    # Perform adversarial attack (single sequence processing)
    update_sequences_with_attack_single(sequences, true_labels, model, tokenizer, device, mutation_rate, max_iterations, save_intervals, path)



# iteration 에서 어택 잘 된 시퀀스가 다시 바뀌는 문제를 해결하기위한 방편책코드 추가함Feb 3rd, 2025  
base_path = "/content/drive/MyDrive/RDL/prj/data/" #"data/"

iteration_files = [
    f"test_{task_name}_atk_{model_short_name}_nucl_iter100_rate0.1.csv",
    f"test_{task_name}_atk_{model_short_name}_nucl_iter200_rate0.1.csv",
    f"test_{task_name}_atk_{model_short_name}_nucl_iter300_rate0.1.csv",
    f"test_{task_name}_atk_{model_short_name}_nucl_iter400_rate0.1.csv",
    f"test_{task_name}_atk_{model_short_name}_nucl_iter500_rate0.1.csv",
]

# 원본 데이터 로드 및 DataLoader 생성
test_df = preprocess_data(test_dir, max_len)
test_dataloader = create_dataloader(test_df, label_to_id, batch_size)

print("Running prediction for original test data...")
original_predictions, true_labels, logits = predict(model, test_dataloader, device)

# 공격 성공한 시퀀스를 추적할 딕셔너리 (idx 기준)
successful_attacks = {}

# 각 iteration 파일을 순차적으로 업데이트
for i, file in enumerate(iteration_files):
    print(f"Processing {base_path}{file}...")
    
    df_adv = preprocess_data(base_path + file, max_len)
    adv_test_dataloader = create_dataloader(df_adv, label_to_id, batch_size)
    adv_predictions, _, _ = predict(model, adv_test_dataloader, device)
    
    # 이전 단계에서 성공한 공격을 유지
    for idx in successful_attacks:
        df_adv.iloc[idx] = successful_attacks[idx]
    
    # 새로운 공격 성공한 시퀀스 탐색
    for idx in range(len(test_df)):
        orig_label = original_predictions[idx]  # 원본 예측
        adv_label = adv_predictions[idx]  # 현재 iteration 예측

        # 원본에서는 정확히 분류되었지만, 이번 iteration에서 잘못 분류된 경우 공격 성공
        if orig_label == true_labels[idx] and adv_label != true_labels[idx]:
            successful_attacks[idx] = df_adv.iloc[idx]

    # 업데이트된 파일 저장
    df_adv.to_csv(base_path + "updated_" + file, index=False)
    print(f"Updated {file} with successful attacks.")

print("All iteration files updated successfully!")
#



# Main 실행에 추가된 부분
if __name__ == "__main__":
    # 원본 테스트 데이터 로드 및 전처리
    for test_atk_dir in [f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter{i}00_rate0.1.csv" for i in range(1, 6)]:
    #for test_atk_dir in [f"/content/drive/MyDrive/RDL/prj/data/test_atk_{task_name}_{model_short_name}_{attack_name}_iter1_rate0_{i}.csv" for i in range(1, 6)]:
    #test_atk_dir = "/content/drive/MyDrive/RDL/prj/data/test_atk_nucl_iter1_rate0_1.csv"

        train_df = preprocess_data(train_dir, max_len)
        test_df = preprocess_data(test_dir, max_len)
        test_atk_df = preprocess_data(test_atk_dir, max_len)

        test_atk_df[test_label_name] = test_df[test_label_name]

        # 라벨-아이디 매핑 생성
        # label_to_id = {label: i for i, label in enumerate(train_df[test_label_name].unique())}
        print(f"Label-to-ID Mapping: {label_to_id}")

        # DataLoader 생성
        test_dataloader = create_dataloader(test_df, label_to_id, batch_size)
        test_atk_dataloader = create_dataloader(test_atk_df, label_to_id, batch_size)

        # 원본 데이터 예측
        print("Running prediction for original test data...")
        predictions, true_labels, logits = predict(model, test_dataloader, device)

        # 적대적 공격 데이터 예측
        print("Running prediction for adversarial attack data...")
        atk_predictions, atk_true_labels, atk_logits = predict(model, test_atk_dataloader, device)

        # 평가 및 시각화 (원본 데이터)
        print("\nEvaluation for original test data:")
        evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir)

        # 평가 및 시각화 (공격 데이터)
        print("\nEvaluation for original test data:")
        evaluate_and_visualize(true_labels, atk_predictions, label_to_id, output_dir)

        # 적대적 공격 성공률 계산
        print("\nEvaluating adversarial attack success rate:")
        success_rate = calculate_attack_success_rate(true_labels, predictions, atk_predictions)

        # 성공률 저장
        now = datetime.now()
        with open(f'{output_dir}/attack_success_rate_{now.strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
            f.write(f"Adversarial Attack Success Rate: {success_rate:.4f}\n")





print(f"{attack_name}, {task_name}, {model_short_name}")



# Main 실행
if __name__ == "__main__":
    # Paths for ADV datasets
    # FOR NUCL


    adv_files = [
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{attack_name}_iter1_rate0_1.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{attack_name}_iter1_rate0_2.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{attack_name}_iter1_rate0_3.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{attack_name}_iter1_rate0_4.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{attack_name}_iter1_rate0_5.csv"
    ]

    # Output paths for updated ADV datasets
    output_files = [
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_1.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_2.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_3.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_4.csv",
        f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_5.csv"
    ]

    # Load and preprocess original data

    train_df = preprocess_data(train_dir, max_len)
    test_df = preprocess_data(test_dir, max_len)



    # Create original test DataLoader
    test_dataloader = create_dataloader(test_df, label_to_id, batch_size)

    # Predict original test set
    original_predictions, original_labels, _ = predict(model, test_dataloader, device)

    # Update ADV datasets with cascading logic
    update_adv_datasets_cascading(original_predictions, original_labels, adv_files, output_files, max_len)




print(f"{attack_name}, {task_name}, {model_short_name}")


# Main 실행에 추가된 부분
if __name__ == "__main__":
    # 원본 테스트 데이터 로드 및 전처리

    for test_atk_dir in [f"/content/drive/MyDrive/RDL/prj/data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_{i}.csv" for i in range(1, 6)]:
    #test_atk_dir = "/content/drive/MyDrive/RDL/prj/data/test_atk_nucl_iter1_rate0_1.csv"

        train_df = preprocess_data(train_dir, max_len)
        test_df = preprocess_data(test_dir, max_len)
        test_atk_df = preprocess_data(test_atk_dir, max_len)

        # 라벨-아이디 매핑 생성
        #label_to_id = {label: i for i, label in enumerate(train_df[test_label_name].unique())}
        print(f"Label-to-ID Mapping: {label_to_id}")

        # DataLoader 생성
        test_dataloader = create_dataloader(test_df, label_to_id, batch_size)
        test_atk_dataloader = create_dataloader(test_atk_df, label_to_id, batch_size)

        # 원본 데이터 예측
        print("Running prediction for original test data...")
        predictions, true_labels, logits = predict(model, test_dataloader, device)

        # 적대적 공격 데이터 예측
        print("Running prediction for adversarial attack data...")
        atk_predictions, atk_true_labels, atk_logits = predict(model, test_atk_dataloader, device)

        # 평가 및 시각화 (원본 데이터)
        print("\nEvaluation for original test data:")
        evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir)

        # 평가 및 시각화 (공격 데이터)
        print("\nEvaluation for original test data:")
        evaluate_and_visualize(true_labels, atk_predictions, label_to_id, output_dir)

        # 적대적 공격 성공률 계산
        print("\nEvaluating adversarial attack success rate:")
        success_rate = calculate_attack_success_rate(true_labels, predictions, atk_predictions)

        # 성공률 저장
        now = datetime.now()
        with open(f'{output_dir}/attack_success_rate_{now.strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
            f.write(f"Adversarial Attack Success Rate: {success_rate:.4f}\n")



