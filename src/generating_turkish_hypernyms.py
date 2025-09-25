"""
This is the main module that performs concept mining for Turkish. The official Turkish dictionary
is parsed to extract words and their meanings. A pre-trained BERT model for Turkish is then used
to predict hypernyms for each word based on its meaning. The results are saved in a CSV file.
However, I am not allowed to share the whole Turkish dictionary due to copyright issues. 

Author: Cem Rifki Aydin
Date: 23.08.2025

"""

import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import re
from itertools import chain
from collections import Counter

import xml.etree.ElementTree as ET

import torch
from transformers import BertTokenizer, BertForMaskedLM


import warnings
warnings.filterwarnings("ignore")


LANG_MODEL = "tr_core_news_trf" 

# Attempt to download the model (only if not already installed)
try:
    nlp = spacy.load(LANG_MODEL)
except OSError:
    print(f"Downloading {LANG_MODEL} model...")
    spacy.cli.download(LANG_MODEL)
    nlp = spacy.load(LANG_MODEL)

# Load NLTK stopwords based on the language
lang_stopwords = set(stopwords.words('turkish'))


def clean_turkish_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\(I+\)", "", text)
    
    # Lowercase text    
    text = text.lower()

    # Remove punctuation and special characters (keep Turkish letters)
    text = re.sub(r"[^a-zçğıöşüûâî\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(args=None):
    # Path to your folder containing XML files
    folder_path = os.path.join("resources", "dictionary")
    xml_files = glob.glob(os.path.join(folder_path, "**", "HARF*.xml"), recursive=True)

    # List to store all entries
    data = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for entry in root.findall("entry"):
            name = entry.findtext("name").strip()
            affix = entry.findtext("affix").strip()
            lex_class = entry.findtext("lex_class").strip()
            stress = entry.findtext("stress").strip()
            
            # Each entry can have multiple meanings
            for meaning in entry.findall("meaning"):
                meaning_class = meaning.findtext("meaning_class").strip()
                meaning_text = meaning.findtext("meaning_text").strip()
                
                # Some meanings have quotations
                quotations = meaning.findall("quotation")
                if quotations:
                    for quotation in quotations:
                        author = quotation.findtext("author")
                        quotation_text = quotation.findtext("quotation_text")
                        
                        # Append a row for each quotation
                        data.append({
                            "name": name,
                            "affix": affix,
                            "lex_class": lex_class,
                            "stress": stress,
                            "meaning_class": meaning_class,
                            "meaning_text": meaning_text,
                            "author": author,
                            "quotation_text": quotation_text
                        })
                else:
                    # No quotation for this meaning
                    data.append({
                        "name": name,
                        "affix": affix,
                        "lex_class": lex_class,
                        "stress": stress,
                        "meaning_class": meaning_class,
                        "meaning_text": meaning_text,
                        "author": None,
                        "quotation_text": None
                    })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(df.columns)
    # Index(['name', 'affix', 'lex_class', 'stress', 'meaning_class', 'meaning_text',
    #        'author', 'quotation_text'],
    #       dtype='object')



    hypernyms_list = []
    letter_dict = {}

    # df = df.iloc[:50]
    df = df[df.apply(lambda row: "isim" in row["lex_class"], axis=1)]

    tqdm.pandas()  # Enable progress_apply

    df["name_root"] = df["name"].progress_apply(lambda x: " ".join([tok.lemma_.lower() for tok in nlp(x)]))  #tok.lemma_.lower(), tok.pos_ , tok.is_alpha, tok.is_stop)

    df.to_csv("tmp_Turkish_words_meanings.csv", index=False, encoding="utf-8")


    # -----------------------------
    # Load pre-trained BERT (masked LM)
    # -----------------------------
    model_name = "dbmdz/bert-base-turkish-cased"  # or "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    model.eval()  # set to evaluation mode


    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for index, row in tqdm(df.iterrows(), total=len(df)):
        letter = row['name'][0].upper()
        if letter not in letter_dict:
            letter_dict[letter] = 1
            print(f"Processing letter: {letter}")

        # -----------------------------
        # Example sentence with a [MASK]
        # -----------------------------
        template_filled = f"{row['meaning_text']}. Buna göre, {row['name']} bir [MASK] idir."

        inputs = tokenizer(template_filled, return_tensors="pt").to(device)
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # -----------------------------
        # Predict masked token
        # -----------------------------
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predicted token, select the top 5 predictions
        mask_token_logits = logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        hypernyms = []

        for token_id in top_5_tokens:
            token = tokenizer.decode([token_id])
            token = clean_turkish_text(token)
            hypernyms.append(token)

        hypernyms_roots = hypernyms  
        hypernyms_list.append(hypernyms_roots)


    df["generic_hypernyms"] = hypernyms_list


    # Flatten the list of lists into a single list
    flat_hypernymy_list = list(chain.from_iterable(hypernyms_list))

    # Create a counter
    counter = Counter(flat_hypernymy_list)

    # Get the 34 most common words that had better be eliminated because they are too generic
    most_common = counter.most_common(34)
    most_common_words = [word for word, count in most_common]
    print(most_common)


    words_to_eliminate = tokenizer.all_special_tokens + most_common_words + list(lang_stopwords)

    # Some other very generic words / noise to eliminate
    specific_toks = ["dakika", "saat", "tarih", "zaman", "sene", "yıl", "vakit", "sezon", "ay", "bir",
                    "te", "de", "da", "in", "ın", "nin", "nın", "fi", "se"]

    words_to_eliminate = list(set(words_to_eliminate + specific_toks))

    df["hypernym"] = df["generic_hypernyms"].apply(lambda x: [word for word in x if word.lower() not in words_to_eliminate])       

    df = df.drop(columns=["generic_hypernyms"])

    df["hypernym"] = df.apply(lambda x: [clean_turkish_text(x["name_root"])] if x["hypernym"] == [] else x["hypernym"], axis=1)
    print(df[["name", "meaning_text", "hypernym"]].head(5))

    df.to_csv("Turkish_words_meanings_and_hypernyms.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main(args=None)

