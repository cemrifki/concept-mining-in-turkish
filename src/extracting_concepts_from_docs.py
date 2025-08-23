import ast
import os
import sys
from collections import Counter
import re
from copy import deepcopy
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


SBERT_LANG_MODEL = "tr_core_news_trf" # Load the appropriate spaCy language model


# Attempt to download the model (only if not already installed)
try:
    nlp = spacy.load(SBERT_LANG_MODEL)
except OSError:
    print(f"Downloading {SBERT_LANG_MODEL} model...")
    spacy.cli.download(SBERT_LANG_MODEL)
    nlp = spacy.load(SBERT_LANG_MODEL)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Load sentence-transformers model
# ------------------------------
model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr').to(device)

# Example document / text (in Turkish)
document = """

Patent, buluş sahibinin, buluş konusu ürünü 3. kişilerin belirli bir süre üretme, kullanma, satma veya ithal etmesini engelleme hakkı olan belgedir. Buluşu yapılan neredeyse her şey patent koruması kapsamına dahildir. Buluşu yapılan bir ürün ya da sistemin bütün hakları patent sahibine ait olur ve ondan izinsiz kullanılamaz.

Patent, ürün veya buluş sahibine, icat ettiği ürünün satışı, pazarlanması, çoğaltılması, bir benzerinin üretilmesi gibi alanlarda ayrıcalıklar getiren resmi bir belge ve unvandır.

Makineler, araçlar, aygıtlar, kimyasal bileşikler ve işlemleri ile her türlü üretim yöntemleri, patent korumasının kapsamındadır.

Patent Yasalarının amacı; buluş yapmayı, yenilikleri ve yaratıcı fikri faaliyetleri teşvik etmek için gerekli olan korumayı ve buluşlarla elde edilen teknik çözümlerin sanayide uygulanmasını sağlamaktır. Verilen patentler ve bunların sanayide uygulanması ile teknik, ekonomik ve sosyal ilerlemenin gerçekleşmesi sağlanır. Sanayi alanında gelişmiş ülkelerde verilen patent sayılarının yüksekliği bu düşüncenin doğruluğunu kanıtlamaktadır.
Patent verilmeyecek konular ve buluşlar

Bulduğumuz ürünün veya yöntemin, her ne ise, buluş basamağı içerip içermediğini anlamak için kullandığımız kriter şöyle izah edilebilir:

    Her buluş teknik bir probleme çözüm içermektedir. Her buluşçu da böyle bir probleme çözüm öneren kişi olarak karşımıza çıkmaktadır. İşte bu önerilen çözüm, tekniğin bu alanında uzman bir kişiye aşikar bir çözüm ise buluş aşaması içermediği kabul edilir ve patent alamaz. Kullanılan kriter kısaca böyle anlatılabilir.

Buluş niteliğinde olmadığı için Kanun Hükmünde Kararname kapsamı dışında kalanlar

    Keşifler, bilimsel teoriler, matematik metotları
    Zihni, ticari ve oyun faaliyetlerine ilişkin plan, usul ve kurallar
    Edebiyat ve sanat eserleri, bilim eserleri, estetik niteliği olan yaratmalar, bilgisayar yazılımları
    Bilginin derlenmesi, düzenlenmesi, sunulması ve iletilmesi ile ilgili teknik yönü bulunmayan usuller
    İnsan veya hayvan vücuduna uygulanacak cerrahi ve tedavi usulleri ile insan, hayvan vücudu ile ilgili teşhis usulleri

    Bu maddenin birinci fıkrâsının (e) bendindeki hüküm bu usûllerin herhangi birinde kullanılan terkip ve maddeler ile bunların üretim usullerine uygulanmaz.

Bu maddenin birinci fıkrasında sayılanlar için münhasıran koruma talep edilmesi halinde patent verilmez
Patent verilerek korunamayan buluşlar

    Konusu kamu düzenine veya genel ahlaka aykırı olan buluşlar.
    Bitki veya hayvan türleri veya önemli ölçüde biyolojik esaslara dayanan bitki veya hayvan yetiştirilmesi usulleri.
"""

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


def main():
    # Process the document
    doc = nlp(document)

    # Split document into sentences
    sentences = [sent.text.strip() for sent in doc.sents]

    n_sentences = len(sentences)

    # Encode all sentences into embeddings
    embeddings = model.encode(sentences)

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Compute centrality score for each sentence (sum of similarities to others)
    centrality_scores = sim_matrix.sum(axis=1)

    # Get indices of top k central sentences (highest scores)
    top_k = 1

    # Positional weights: sentences in the middle get lower weights, since they might be less important and
    # be more related to details. Sentences at the beginning and end get higher weights, because they often contain
    # important information (e.g., introduction, conclusion).

    pos_weights = np.array([max(abs((i - (n_sentences-1)/2)/((n_sentences-1)/2)), 0.1) 
                            for i in range(n_sentences)])

    weighted_scores = centrality_scores * pos_weights

    # # Identify the most central sentences
    top_indices = np.argsort(weighted_scores)[-top_k:][::-1]  # descending order

    # Retrieve sentences and embeddings
    top_sentences = [sentences[i] for i in top_indices]
    top_embeddings = [embeddings[i] for i in top_indices]

    print(f"Top {top_k} most important sentences:")
    for s in top_sentences:
        print("-", s)

    # =======================================================
    # Using the lexical dataset to extract keywords
    # =======================================================

    print("Embedding shape:", top_embeddings[0].shape)

    keywords = [clean_turkish_text(tok.lemma_.lower()) 
                for sentence in top_sentences for tok in nlp(sentence) 
                if tok.tag_.lower() in ["noun", "propn"] and not tok.is_stop and tok.lemma_.strip()]

    # Create a counter
    counter = Counter(keywords)

    # Get the 10 most common words
    most_common_keywords = [k for k, v in counter.most_common(10)]
    print("The most important keywords of the document:", most_common_keywords)


    # =======================================================
    # Using the lexical dataset to extract concepts
    # =======================================================

    df_tr_hypernyms = pd.read_csv(os.path.join("resources", "lexical_database", 
                                               "Turkish_words_and_hypernyms.csv"))
    top_k = 3  # number of top sentences to consider

    # Example: top_k = 3 sentences already sorted by centrality
    weights = list(reversed(range(1, top_k + 1)))  # weight for 1st, 2nd, 3rd sentences; adjust as needed

    doc_hypernyms_weighted = []

    for idx, sentence in enumerate(top_sentences):
        weight = weights[idx] if idx < len(weights) else 1  # default weight 1
        keywords = [
            clean_turkish_text(tok.lemma_.lower())
            for tok in nlp(sentence)
            if tok.tag_.lower() in ["noun", "propn"] and not tok.is_stop and tok.lemma_.strip()
        ]

        for keyword in keywords:
            root = keyword
            if root in df_tr_hypernyms["name_root"].values:
                hypernyms = df_tr_hypernyms[df_tr_hypernyms["name_root"] == root]["hypernym"].values[0]
                if isinstance(hypernyms, str):
                    hypernyms = ast.literal_eval(hypernyms)
                if root not in hypernyms:
                    hypernyms.append(root)
            else:
                hypernyms = [root]

            # Append each hypernym with the sentence weight
            doc_hypernyms_weighted.extend(hypernyms * weight)

    # Count frequency with weights
    counter_weighted = Counter(doc_hypernyms_weighted)

    # Most common hypernyms (e.g., top 10)
    most_common_concepts_weighted = [k for k, v in counter_weighted.most_common(10)]

    # print("Weighted Counter:", counter_weighted)
    print("Most Common Concepts (weighted):", most_common_concepts_weighted)

    
if __name__ == "__main__":
    main()
