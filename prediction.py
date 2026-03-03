#Extract all features

import os
import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Settings (same data preparation as above) ---
csv_file_path = "XXX.csv"  # your data
saved_model_path = "/content/indonesian_readability_classifier_final_IndoBERT_noaug"  # saved model

df = pd.read_csv(csv_file_path, header=None)

all_texts_raw = df.iloc[:, 0].astype(str).tolist()
all_labels_raw = df.iloc[:, 1].tolist()

valid_texts = []
valid_labels = []

for text, label in zip(all_texts_raw, all_labels_raw):
    try:
        label_int = int(label)
        if label_int in [2, 3]:
            valid_texts.append(text)
            valid_labels.append('A')
        elif label_int in [4, 5]:
            valid_texts.append(text)
            valid_labels.append('B')
        elif label_int in [6, 7]:
            valid_texts.append(text)
            valid_labels.append('C')
    except ValueError:
        continue

train_val_texts, test_texts_orig, train_val_labels, test_labels_orig = train_test_split(
    valid_texts, valid_labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=valid_labels
)

train_texts_orig, val_texts_orig, train_labels_orig, val_labels_orig = train_test_split(
    train_val_texts, train_val_labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=train_val_labels
)

train_texts = list(train_texts_orig)
train_labels_str = list(train_labels_orig)

test_texts = list(test_texts_orig)
test_labels_str = list(test_labels_orig)

label2id = {'A': 0, 'B': 1, 'C': 2}
train_labels_ids = [label2id[l] for l in train_labels_str]
test_labels_ids = [label2id[l] for l in test_labels_str]

texts = train_texts + test_texts
labels = train_labels_ids + test_labels_ids
split_boundary = len(train_texts)

print("Data split completed (leakage prevented)")
print(f"Train: {len(train_texts)} (orig: {len(train_texts_orig)})")
print(f"Test : {len(test_texts)} (orig: {len(test_texts_orig)})")
print(f"Label mapping: {label2id}")

# --- Hand-crafted feature extraction (Stanza + Imperial + Syllable) ---
print("\nStep 1/3: Extracting hand-crafted features (Stanza + Imperial + Syllable)...")
imperial_extractor = ImperialFeaturesExtractor()
imperial_new_extractor = ImperialNewFeaturesExtractor()  # added: new features
syllable_extractor = SyllablePatternExtractor()

handcrafted_features = []

for text in tqdm(texts):
    # 1. Stanza (11 dims)
    stanza_feats = extract_stanza_features(str(text))

    # 2. Imperial (15 dims)
    imperial_feats = imperial_extractor.extract(str(text))

    # 3. Imperial New (4 dims)
    imperial_new_feats = imperial_new_extractor.extract(str(text))

    # 4. Syllable Patterns (11 dims)
    syll_feats = syllable_extractor.extract(str(text))

    # Concatenate (11 + 15 + 4 + 11 = 41 dims)
    combined_manual = stanza_feats + imperial_feats + imperial_new_feats + syll_feats
    handcrafted_features.append(combined_manual)

X_handcrafted = np.array(handcrafted_features)
print(f"Hand-crafted features shape: {X_handcrafted.shape}")

# --- BERT embedding extraction ---
print("\nStep 2/3: Extracting BERT embeddings...")

bert_model_path = saved_model_path
X_bert = extract_bert_embeddings(texts, bert_model_path)
print(f"BERT embeddings shape: {X_bert.shape}")

# --- Combine features and prepare for training ---
X_hc_train = X_handcrafted[:split_boundary]
X_hc_test = X_handcrafted[split_boundary:]

X_bert_train = X_bert[:split_boundary]
X_bert_test = X_bert[split_boundary:]

y_train = labels[:split_boundary]
y_test = labels[split_boundary:]

scaler = StandardScaler()
X_hc_train_scaled = scaler.fit_transform(X_hc_train)
X_hc_test_scaled = scaler.transform(X_hc_test)

X_train = np.concatenate((X_bert_train, X_hc_train_scaled), axis=1)
X_test = np.concatenate((X_bert_test, X_hc_test_scaled), axis=1)

print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

#Prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 1. Split the data (fix: prevent leakage)
print("Splitting data using slicing to prevent leakage...")

# split_boundary was computed in the previous cell (number of augmented train samples)
if 'split_boundary' not in globals():
    split_boundary = 144  # fallback if not found

X_bert_train = X_bert[:split_boundary]
X_bert_test = X_bert[split_boundary:]

X_hc_train = X_handcrafted[:split_boundary]
X_hc_test = X_handcrafted[split_boundary:]

y_train = labels[:split_boundary]
y_test = labels[split_boundary:]

print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

# 2. Scale the features (fit scaler on Train only -> apply to Test)
scaler = StandardScaler()
X_hc_train_scaled = scaler.fit_transform(X_hc_train)
X_hc_test_scaled = scaler.transform(X_hc_test)

# 3. Prepare feature sets
datasets = {
    "Hand-crafted Only": (X_hc_train_scaled, X_hc_test_scaled),
    "BERT Only": (X_bert_train, X_bert_test),
    "Combined": (
        np.concatenate((X_bert_train, X_hc_train_scaled), axis=1),
        np.concatenate((X_bert_test, X_hc_test_scaled), axis=1)
    )
}

# 4. Define classifiers
classifiers = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
}

results = []

print("Running comparative experiments...")

# 5. Loop (feature set × classifier)
for feature_name, (X_tr, X_te) in datasets.items():
    print(f"\n--- Evaluating: {feature_name} (Train Shape: {X_tr.shape}) ---")

    for model_name, clf in classifiers.items():
        # Train
        clf.fit(X_tr, y_train)

        # Predict
        y_pred = clf.predict(X_te)

        # Compute scores
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store results
        results.append({
            'Feature Set': feature_name,
            'Model': model_name,
            'Accuracy': acc,
            'F1 Score': f1
        })
        print(f"  > {model_name}: F1 = {f1:.4f}")

# 6. Summarize and display results
results_df = pd.DataFrame(results)

# Pivot table
pivot_table = results_df.pivot(index='Model', columns='Feature Set', values='F1 Score')

print("\n" + "=" * 40)
print("Comparison of F1 Scores (Leakage-Free)")
print("=" * 40)
display(pivot_table)

# 7. Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y='F1 Score', hue='Feature Set')
plt.title('Model Performance Comparison (F1 Score) - No Leakage')
plt.ylim(0, 1.05)
plt.ylabel('Weighted F1 Score')
plt.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Visualizing feature importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if 'X_handcrafted' in globals():
    scaler_feat = StandardScaler()
    X_handcrafted_scaled = scaler_feat.fit_transform(X_handcrafted)
    print("Created X_handcrafted_scaled from X_handcrafted.")
else:
    print("Error: X_handcrafted not found. Please run feature extraction first.")
    X_handcrafted_scaled = np.zeros((10, 41))

feature_names = [
    'Affix Ratio', 'Avg Syllables (Stanza)', 'Noun Ratio', 'Verb Ratio',
    'Adj Ratio', 'Adv Ratio', 'Pron Ratio', 'Conj Ratio',
    'Redup Ratio', 'TTR', 'Lexical Density',
    'Avg Sentence Length', 'Avg Syllables/Word', '% Complex Words',
    'Flesch Ease', 'Flesch Grade', 'Gunning Fog', 'ARI', 'Coleman-Liau', 'SMOG',
    'LIX', 'RIX', 'Consonant Ratio',
    'Avg Chars/Word', 'Linsear Write', 'Non-Common Ratio',
    'Word Entropy', 'Char Entropy', 'Root TTR', 'Log TTR',
    'V', 'VC', 'CV', 'CVC', 'CCV', 'CCVC', 'CVCC', 'CCCV', 'CCCVC', 'CCVCC', 'Other Pattern'
]

if X_handcrafted_scaled.shape[1] != len(feature_names):
    print(f"Warning: Feature dimension mismatch. Data: {X_handcrafted_scaled.shape[1]}, Names: {len(feature_names)}")
    print("Using generic names for plot.")
    plot_feature_names = [f"Feat {i}" for i in range(X_handcrafted_scaled.shape[1])]
else:
    plot_feature_names = feature_names

X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
    X_handcrafted_scaled, labels, test_size=0.5, random_state=47
)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_fs, y_train_fs)
rf_importance = rf.feature_importances_

scaler_plot = MinMaxScaler()

df_imp = pd.DataFrame({
    'Feature': plot_feature_names,
    'Random Forest (Imp)': rf_importance
})

cols_to_plot = ['Random Forest (Imp)']
df_imp[cols_to_plot] = scaler_plot.fit_transform(df_imp[cols_to_plot])

plt.figure(figsize=(10, 12))
sns.barplot(data=df_imp, y='Feature', x='Random Forest (Imp)', color='teal')
plt.title('Feature Importance: Random Forest')
plt.xlabel('Relative Importance (0-1 Scaled)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("--- Top 10 Features (Random Forest) ---")
display(df_imp.sort_values('Random Forest (Imp)', ascending=False).head(10))

#Heatmap of true vs. predicted labels
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

X_tr = X_bert_train
X_te = X_bert_test
y_tr = y_train
y_te = y_test

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)

acc = accuracy_score(y_te, y_pred)
f1 = f1_score(y_te, y_pred, average='weighted')
print("Model: Random Forest (BERT Only)")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_te, y_pred)

labels = sorted(label2id.keys(), key=lambda x: label2id[x])

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=labels, yticklabels=labels
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Random Forest (BERT Only)')
plt.show()
