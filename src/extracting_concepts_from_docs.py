"""
Extracting concepts from documents using a lexical database and sentence embeddings.

Author: Cem Rıfkı Aydın
Date: 25/09/2025

"""

import ast
import os
import sys
import json
from collections import Counter
import re
import warnings

import torch
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add project src directory to path
sys.path.append("src")

# Suppress warnings
warnings.filterwarnings("ignore")

LANG_MODEL = "tr_core_news_trf"  # spaCy language sbert_model

# Attempt to load or download the sbert_model
try:
    nlp = spacy.load(LANG_MODEL)
except OSError:
    print(f"Downloading {LANG_MODEL}...")
    spacy.cli.download(LANG_MODEL)
    nlp = spacy.load(LANG_MODEL)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load sentence-transformers sbert_model
sbert_model = SentenceTransformer(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
).to(device)


def read_txt(file_path: str) -> str:
    """Read text from a .txt file robustly."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_turkish_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\(I+\)", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zçğıöşüûâî\.\?\!\s]", " ", text)  # keep Turkish chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_document(doc_text: str, df_tr_hypernyms: pd.DataFrame, top_k: int = 1):
    # Process the document with spaCy
    doc = nlp(doc_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    n_sentences = len(sentences)

    if n_sentences == 0:
        return None

    # Encode all sentences
    embeddings = sbert_model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    centrality_scores = sim_matrix.sum(axis=1)

    # Positional weights: sentences in the middle get lower weights, since they might be less important and
    # be more related to details. Sentences at the beginning and end get higher weights, because they often contain
    # important information (e.g., introduction, conclusion).
    pos_weights = np.array(
        [
            max(abs((i - (n_sentences - 1) / 2) / ((n_sentences - 1) / 2)), 0.1)
            for i in range(n_sentences)
        ]
    )
    weighted_scores = centrality_scores * pos_weights

    # Top-k sentences
    top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
    top_sentences = [sentences[i] for i in top_indices]


    # =======================================================
    # Using the document to extract keywords
    # =======================================================

    keywords = [
        clean_turkish_text(tok.lemma_.lower())
        for sentence in top_sentences
        for tok in nlp(sentence)
        if tok.tag_.lower() in ["noun", "propn"]
        and not tok.is_stop
        and tok.lemma_.strip()
    ]
    counter = Counter(keywords)
    most_common_keywords = [k for k, v in counter.most_common(10)]


    # =======================================================
    # Using the lexical database to extract concepts
    # =======================================================

    weights = list(reversed(range(1, top_k + 1)))
    doc_hypernyms_weighted = []

    for idx, sentence in enumerate(top_sentences):
        weight = weights[idx] if idx < len(weights) else 1
        keywords = [
            clean_turkish_text(tok.lemma_.lower())
            for tok in nlp(sentence)
            if tok.tag_.lower() in ["noun", "propn"]
            and not tok.is_stop
            and tok.lemma_.strip()
        ]
        for keyword in keywords:
            root = keyword
            if root in df_tr_hypernyms["name_root"].values:
                hypernyms = df_tr_hypernyms[
                    df_tr_hypernyms["name_root"] == root
                ]["hypernym"].values[0]
                if isinstance(hypernyms, str):
                    hypernyms = ast.literal_eval(hypernyms)
                if root not in hypernyms:
                    hypernyms.append(root)
            else:
                hypernyms = [root]
            doc_hypernyms_weighted.extend(hypernyms * weight)

    counter_weighted = Counter(doc_hypernyms_weighted)
    most_common_concepts_weighted = [k for k, v in counter_weighted.most_common(10)]

    return {
        "top_sentences": top_sentences,
        "keywords": most_common_keywords,
        "concepts": most_common_concepts_weighted,
    }

def main(args=None):
    
    df_tr_hypernyms = pd.read_csv(args.lexical_db)

    for root, _, files in os.walk(args.dataset):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path, args.dataset)

            # Split top-level category (sports, laws, news)
            category = rel_path.split(os.sep)[0]
            rest_path = os.sep.join(rel_path.split(os.sep)[1:])

            # Create category folder with prefix "concepts_"
            output_category = f"concepts_{category}"
            output_dir = os.path.join(args.output, output_category, os.path.dirname(rest_path))
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, fname.rsplit(".", 1)[0] + ".json")

            print(f"\n=== Processing {file_path} ===")
            try:
                doc_text = read_txt(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            results = process_document(doc_text, df_tr_hypernyms, args.top_k)
            if results:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved results → {output_file}")
            else:
                print("No sentences found, skipping output.")

def main_tmp(args=None):
    
    df_tr_hypernyms = pd.read_csv(args.lexical_db)

    for root, _, files in os.walk(args.dataset):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path, args.dataset)

            # Split top-level category (sports, laws, news)
            category = rel_path.split(os.sep)[0]
            rest_path = os.sep.join(rel_path.split(os.sep)[1:])

            # Create category folder with prefix "concepts_"
            output_category = f"concepts_{category}"
            output_dir = os.path.join(args.output, output_category, os.path.dirname(rest_path))
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, fname.rsplit(".", 1)[0] + ".json")

            print(f"\n=== Processing {file_path} ===")
            try:
                doc_text = read_txt(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            results = process_document(doc_text, df_tr_hypernyms, args.top_k)
            if results:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved results → {output_file}")
            else:
                print("No sentences found, skipping output.")


if __name__ == "__main__":
    main(args=None)