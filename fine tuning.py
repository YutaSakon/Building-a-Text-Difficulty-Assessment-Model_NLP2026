#SET UP

import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 47
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

TEST_SIZE = 0.2
N_SPLITS = 5

MODEL_ID = "indolem/indobert-base-uncased"
MODEL_NAME = "IndoBERT"

NUM_EPOCHS = 25
LR = 1e-5
TRAIN_BS = 8
EVAL_BS = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LEN = 512

USE_FREEZE = True
FREEZE_LAYER_UNTIL = 5

save_suffix = "_noaug"
FINAL_SAVE_DIR = f"./indonesian_readability_classifier_final_{MODEL_NAME}{save_suffix}" #Designed for Google Colab


#Dataset Preparation
csv_file_path = "XXX.csv"

df = pd.read_csv(csv_file_path, header=None)

all_texts_raw = df.iloc[:, 0].astype(str).tolist()
all_labels_raw = df.iloc[:, 1].tolist()

GROUP_LABELS = True #If you want to merge the labels, set this to True; if you want to use the textbook’s original 7-level labels as-is, set this to False.

valid_texts = []
valid_labels = []

for text, label in zip(all_texts_raw, all_labels_raw):
    try:
        label_int = int(label)
        target_label = None

        if GROUP_LABELS:
            if label_int in [2, 3]:
                target_label = 'A'
            elif label_int in [4, 5]:
                target_label = 'B'
            elif label_int in [6, 7]:
                target_label = 'C'
        else:
            if label_int in [2, 3, 4, 5, 6, 7]:
                target_label = str(label_int)

        if target_label:
            valid_texts.append(text)
            valid_labels.append(target_label)

    except ValueError:
        continue
"""
#For the Kompas analysis, replace the code above with the following.
for text, label in zip(all_texts_raw, all_labels_raw):
    try:
        label_int = int(label)
    except ValueError:
        continue

    if GROUP_LABELS:
        if label_int in [1, 2]:
            valid_texts.append(text); valid_labels.append("A")
        elif label_int in [3]:
            valid_texts.append(text); valid_labels.append("B")
        elif label_int in [4,5]:
            valid_texts.append(text); valid_labels.append("C")
    else:
        if label_int in [1,2, 3, 4, 5]:
            valid_texts.append(text); valid_labels.append(str(label_int))
"""

unique_labels = sorted(list(set(valid_labels)))
num_classes = len(unique_labels)

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Number of detected classes: {num_classes}")
print(f"Label definitions: {unique_labels}")
print(f"ID mapping: {label2id}")

train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    valid_texts,
    valid_labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=valid_labels
)

print(f"\nSplit fixed test:")
print(f"  train_val: {len(train_val_texts)}")
print(f"  test     : {len(test_texts)}")

# testは“原文”で評価するのが基本（増強しない）
test_labels_ids = [label2id[l] for l in test_labels]

#Function definition
class ReadabilityClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def freeze_lower_layers(model, freeze_until_layer: int = 5):
    """
    embeddings と encoder.layer.0〜freeze_until_layer をfreezeする（BERT系向け）
    """
    base_prefix = getattr(model, "base_model_prefix", "bert")
    base = getattr(model, base_prefix, None)
    if base is None:
        print("[Warning] base model not found; skip freezing.")
        return

    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        for i, layer in enumerate(base.encoder.layer):
            if i <= freeze_until_layer:
                for p in layer.parameters():
                    p.requires_grad = False

#Five-fold cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

cv_f1s, cv_accs = [], []

print("\n" + "="*80)
print(f"StratifiedKFold CV on train_val ({N_SPLITS}-fold)  |  model={MODEL_ID}")
print("="*80)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_val_texts, train_val_labels), start=1):
    print(f"\n--- CV Fold {fold}/{N_SPLITS} ---")

    tr_texts = [train_val_texts[i] for i in tr_idx]
    tr_labs  = [train_val_labels[i] for i in tr_idx]
    va_texts = [train_val_texts[i] for i in va_idx]
    va_labs  = [train_val_labels[i] for i in va_idx]

    y_tr = [label2id[l] for l in tr_labs]
    y_va = [label2id[l] for l in va_labs]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_classes, id2label=id2label, label2id=label2id
    ).to(device)

    if USE_FREEZE:
        freeze_lower_layers(model, freeze_until_layer=FREEZE_LAYER_UNTIL)

    train_ds = ReadabilityClassificationDataset(tr_texts, y_tr, tokenizer, max_len=MAX_LEN)
    val_ds   = ReadabilityClassificationDataset(va_texts, y_va, tokenizer, max_len=MAX_LEN)

    out_dir = f"./cv_tmp/{MODEL_NAME}{save_suffix}/fold_{fold}"
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    trainer.train()
    metrics = trainer.evaluate()

    cv_f1 = float(metrics["eval_f1"])
    cv_acc = float(metrics["eval_accuracy"])
    cv_f1s.append(cv_f1)
    cv_accs.append(cv_acc)

    print(f"CV Fold {fold} | Val F1: {cv_f1:.4f} | Val Acc: {cv_acc:.4f}")

cv_f1_mean = float(np.mean(cv_f1s))
cv_f1_std  = float(np.std(cv_f1s, ddof=1))
cv_acc_mean = float(np.mean(cv_accs))
cv_acc_std  = float(np.std(cv_accs, ddof=1))

print("\n" + "-"*80)
print(f"CV mean F1 : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
print(f"CV mean Acc: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
print("-"*80)

print("\n" + "="*80)
print("Final training on ALL train_val ... (no test used)")
print("="*80)

final_train_texts = list(train_val_texts)
final_train_labels = list(train_val_labels)

final_train_labels_ids = [label2id[l] for l in final_train_labels]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=num_classes, id2label=id2label, label2id=label2id
).to(device)

if USE_FREEZE:
    freeze_lower_layers(final_model, freeze_until_layer=FREEZE_LAYER_UNTIL)

final_train_ds = ReadabilityClassificationDataset(final_train_texts, final_train_labels_ids, tokenizer, max_len=MAX_LEN)
test_ds = ReadabilityClassificationDataset(test_texts, test_labels_ids, tokenizer, max_len=MAX_LEN)

os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

final_args = TrainingArguments(
    output_dir=os.path.join(FINAL_SAVE_DIR, "trainer_out"),
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    eval_strategy="no", 
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=final_train_ds,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

final_trainer.train()

#Evaluate accuracy on the test data
print("\n" + "="*80)
print("Final evaluation on FIXED test set (only once)")
print("="*80)

test_output = final_trainer.predict(test_ds)
test_preds = np.argmax(test_output.predictions, axis=1)

test_acc = accuracy_score(test_labels_ids, test_preds)
test_f1 = precision_recall_fscore_support(
    test_labels_ids, test_preds, average="weighted", zero_division=0
)[2]

print(f"TEST Accuracy: {test_acc:.4f}")
print(f"TEST F1(w)   : {test_f1:.4f}")

print("\n" + "="*80)
print(f"Saving final model to: {FINAL_SAVE_DIR}")
print("="*80)

final_model.save_pretrained(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)

print("Saved successfully.")
