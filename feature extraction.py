#!pip install stanza transformers torch scikit-learn pandas tqdm

#SET UP
import torch
import sys

import stanza

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

stanza.download('id', verbose=False)
nlp = stanza.Pipeline('id', processors='tokenize,lemma,pos', use_gpu=True, verbose=False)

#Linguistic Feature
def extract_stanza_features(text):
    if not text or not isinstance(text, str):
        return [0.0] * 11

    try:
        doc = nlp(text)
    except Exception:
        return [0.0] * 11

    total_words = 0
    complex_words = 0
    pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0, 'CONJ': 0}
    total_syllables = 0
    reduplication_count = 0
    unique_lemmas = set()

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'PUNCT': 
                continue  # ignore punctuation

            total_words += 1
            unique_lemmas.add(word.lemma)

            # Feature 1: Affix Ratio (words with affixes)
            if word.text.lower() != word.lemma.lower():
                complex_words += 1

            # Feature 2: Avg. Syllables
            syllables = len(re.findall(r'[aiueo]', word.text.lower()))
            total_syllables += max(1, syllables)

            # Features 3–8: POS ratios (noun/verb/adj/adv/pron/conj)
            if word.upos in pos_counts:
                pos_counts[word.upos] += 1
            elif word.upos in ['CCONJ', 'SCONJ']:
                pos_counts['CONJ'] += 1

            # Feature 9: Reduplication (contains a hyphen and length >= 4)
            # A stricter implementation should check the lemma, but this is a simplified version.
            if '-' in word.text and len(word.text) > 3:
                reduplication_count += 1

    if total_words == 0:
        return [0.0] * 11

    # Feature 10: TTR
    ttr = len(unique_lemmas) / total_words

    # Feature 11: Lexical Density (content words / total words)
    content_words = pos_counts['NOUN'] + pos_counts['VERB'] + pos_counts['ADJ'] + pos_counts['ADV']
    lexical_density = content_words / total_words

    return [
        complex_words / total_words,       # Affix Ratio
        total_syllables / total_words,     # Avg Syllables
        pos_counts['NOUN'] / total_words,  # Noun Ratio
        pos_counts['VERB'] / total_words,  # Verb Ratio
        pos_counts['ADJ'] / total_words,   # Adj Ratio
        pos_counts['ADV'] / total_words,   # Adv Ratio
        pos_counts['PRON'] / total_words,  # Pron Ratio
        pos_counts['CONJ'] / total_words,  # Conj Ratio
        reduplication_count / total_words, # Redup Ratio
        ttr,                               # TTR
        lexical_density                    # Lexical Density
    ]

# Imperial-based features (15 features)
import re
import math
import requests
from bs4 import BeautifulSoup

class ImperialFeaturesExtractor:
    def __init__(self):
        self.common_words = self._fetch_common_words()

    def _count_syllables(self, word):
        return len(re.findall(r'[aiueo]', word.lower()))

    def _fetch_common_words(self):
        """
        Fetch common Indonesian words from the specified website.
        Using the raw GitHub URL to get the clean list.
        """
        # Use this word list as the definition of basic/common words.
        # Lanin, I., Geovedi, J., & Soegijoko. W. (2013). Perbandingan distribusi frekuensi kata bahasa Indonesia di Kompas, Wikipedia, Twitter, dan Kaskus.
        # In Proceedings of Konferensi Linguistik Tahunan Atma Jaya Kesebelas (KOLITA11) (pp. 249-252).
        url = "https://raw.githubusercontent.com/ardwort/freq-dist-id/master/data/kaskus.1gram"
        print(f"Fetching common Indonesian words from {url}...")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                content = response.text
                words = set()
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if parts:
                        words.add(parts[0].lower())

                print(f"Successfully loaded {len(words)} unique words from the site.")
                return words
            else:
                print(f"Failed to access site. Status code: {response.status_code}")
                return set()
        except Exception as e:
            print(f"Error fetching common words: {e}")
            return set()

    def extract(self, text):
        if not text or not isinstance(text, str):
            return [0.0] * 15

        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        num_sentences = max(1, len(sentences))

        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words_list = clean_text.split()
        num_words = max(1, len(words_list))

        total_syllables = 0
        complex_words_count = 0
        long_words_count = 0

        non_common_count = 0
        if self.common_words:
            for w in words_list:
                if w not in self.common_words:
                    non_common_count += 1

        alpha_chars = re.sub(r'[^a-z]', '', clean_text)
        num_alpha = len(alpha_chars)
        num_vowels = len(re.findall(r'[aiueo]', alpha_chars))
        num_consonants = num_alpha - num_vowels

        num_chars = len(clean_text.replace(" ", ""))

        for word in words_list:
            syll = max(1, self._count_syllables(word))
            total_syllables += syll
            if syll >= 3:
                complex_words_count += 1

            if len(word) > 6:
                long_words_count += 1

        # Feature 12: Average Sentence Length
        asl = num_words / num_sentences
        # Feature 13: Average Syllables per Word
        asw = total_syllables / num_words
        # Feature 14: Complex Word Percentage (>= 3 syllables)
        pct_complex = (complex_words_count / num_words) * 100
        # Feature 15: Consonant Ratio
        consonant_ratio = num_consonants / num_alpha if num_alpha > 0 else 0.0

        # Feature 16: Non-Common Word Ratio (non-basic words)
        non_common_ratio = non_common_count / num_words if num_words > 0 else 0.0

        # Feature 17: Flesch Reading Ease
        flesch_ease = 206.835 - (1.015 * asl) - (84.6 * asw)
        # Feature 18: Flesch-Kincaid Grade
        flesch_grade = (0.39 * asl) + (11.8 * asw) - 15.59
        # Feature 19: Gunning Fog Index
        gunning_fog = 0.4 * (asl + pct_complex)

        # Feature 20: Average Characters per Word
        avg_chars_per_word = num_chars / num_words
        # Feature 21: Automated Readability Index (ARI)
        ari = (4.71 * avg_chars_per_word) + (0.5 * asl) - 21.43

        # Feature 22: Coleman–Liau Index
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        coleman_liau = (0.0588 * L) - (0.296 * S) - 15.8

        # Feature 23: SMOG
        smog = 1.0430 * math.sqrt(complex_words_count * (30 / num_sentences)) + 3.1291

        # Feature 24: LIX (Läsbarhetsindex): ASL + %LongWords (> 6 chars)
        lix = asl + ((long_words_count / num_words) * 100)

        # Feature 25: RIX: LongWords / Sentences
        rix = long_words_count / num_sentences

        # Feature 26: Linsear Write Formula
        # Standard: (Easy_words + 3 * Hard_words) / Sentences
        # Easy = < 3 syllables, Hard = >= 3 syllables
        easy_words = num_words - complex_words_count
        lw_raw = (easy_words + (3 * complex_words_count)) / num_sentences

        if lw_raw > 20:
            linsear_write = lw_raw / 2
        else:
            linsear_write = (lw_raw - 2) / 2

        return [
            asl, asw, pct_complex, flesch_ease, flesch_grade, gunning_fog,
            ari, coleman_liau, smog, lix, rix, consonant_ratio,
            avg_chars_per_word, linsear_write, non_common_ratio
        ]
import re
import math
from collections import Counter

class ImperialNewFeaturesExtractor:
    def _calculate_entropy(self, elements):
        if not elements:
            return 0.0
        counts = Counter(elements)
        total = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log(p, 2)
        return entropy

    def extract(self, text):
        if not text or not isinstance(text, str):
            return [0.0] * 4

        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        words_list = clean_text.split()
        num_words = max(1, len(words_list))

        # Feature 27: Word Entropy
        word_entropy = self._calculate_entropy(words_list)

        # Feature 28: Character Entropy (excluding spaces)
        chars = list(clean_text.replace(" ", ""))
        char_entropy = self._calculate_entropy(chars)

        # Feature 29: Root TTR (Guiraud's Index): V / sqrt(N)
        unique_words = set(words_list)
        root_ttr = len(unique_words) / math.sqrt(num_words) if num_words > 0 else 0.0

        # Feature 30: Log TTR (Herdan's C): log(V) / log(N)
        log_ttr = math.log(len(unique_words)) / math.log(num_words) if num_words > 1 and len(unique_words) > 0 else 0.0

        return [word_entropy, char_entropy, root_ttr, log_ttr]
import re

# Features 31–41: Syllable patterns
class SyllablePatternExtractor:
    def __init__(self):
        self.vowels = set('aiueo')
        self.target_patterns = [
            'V', 'VC', 'CV', 'CVC', 'CCV', 'CCVC', 'CVCC', 'CCCV', 'CCCVC', 'CCVCC'
        ]

    def _is_vowel(self, char):
        return char.lower() in self.vowels

    def _to_cv_pattern(self, word):
        pattern = []
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        for char in clean_word:
            if self._is_vowel(char):
                pattern.append('V')
            else:
                pattern.append('C')
        return "".join(pattern)

    def _syllabify_cv(self, cv_string):
        syllables = []
        current_syllable = ""

        v_indices = [i for i, char in enumerate(cv_string) if char == 'V']

        if not v_indices:
            return [cv_string] if cv_string else []

        start = 0
        for i in range(len(v_indices) - 1):
            curr_v = v_indices[i]
            next_v = v_indices[i + 1]

            dist = next_v - curr_v

            if dist == 1:
                # V-V (e.g., sa-at) -> split right after the first vowel
                split_point = curr_v + 1
            elif dist == 2:
                # V-C-V (e.g., ma-kan) -> split before the consonant (ma-kan)
                split_point = curr_v + 1
            elif dist == 3:
                # V-CC-V (e.g., ban-tu) -> split between the two consonants (ban-tu)
                split_point = curr_v + 2
            elif dist == 4:
                # V-CCC-V (e.g., in-struk-si) -> split after the first consonant (in-struk-si)
                # This is a simplified rule; assumes C-CCV is more common than CC-CV.
                split_point = curr_v + 2
            else:
                split_point = curr_v + 2

            syllables.append(cv_string[start:split_point])
            start = split_point

        syllables.append(cv_string[start:])

        return syllables

    def extract(self, text):
        if not text or not isinstance(text, str):
            return [0.0] * (len(self.target_patterns) + 1)

        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        total_syllables = 0
        pattern_counts = {p: 0 for p in self.target_patterns}
        pattern_counts['OTHER'] = 0

        for word in words:
            cv_str = self._to_cv_pattern(word)
            sylls = self._syllabify_cv(cv_str)
            for s in sylls:
                if s in pattern_counts:
                    pattern_counts[s] += 1
                else:
                    pattern_counts['OTHER'] += 1
                total_syllables += 1

        if total_syllables == 0:
            return [0.0] * (len(self.target_patterns) + 1)

        features = []
        for p in self.target_patterns:
            features.append(pattern_counts[p] / total_syllables)

        features.append(pattern_counts['OTHER'] / total_syllables)

        return features
#BERT embedding features
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

def extract_bert_embeddings(text_list, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embs = []
    for text in tqdm(text_list):
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
        embs.append(cls)

    return np.array(embs)
