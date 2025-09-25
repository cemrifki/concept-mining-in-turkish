# Dictionary-Based Concept Mining in Turkish Leveraging BERT Models
This repository performs concept mining for Turkish. It leverages the Turkish BERT model and utilizes the official Turkish dictionary. Ultimately, concepts / hypernyms are extracted from Turkish documents. The template for the Turkish BERT model can be adjusted to generate more plausible tokens according to your requirements. 

Unfortunately, I cannot share the full Turkish dictionary due to copyright restrictions, so only a small portion is provided. For now, the functionality for extracting both keywords and concepts has been integrated. In the meantime, I will also keep trying to improve the performance of the BERT model in generating the hypernymy dictionary.


## Requirements

- Python 3.9 (or a newer version)
- pandas
- torch
- transformers
- gensim
- spacy
- thinc
- scipy
- scikit-learn 
- sentence_transformers
- nltk
- numpy
- huggingface-hub
- lxml 

 The code can work through `Python 3.9` or a newer version thereof. I tested it relying on `Python 3.9`. In this project, `python3` and `pip3`. We leveraged two datasets, which are sentiment and spam corpora and which can be found in the input folder.

## Project Structure

```bash
concept-mining-in-turkish/
│
├── src/
│   ├── extracting_concepts_from_docs.py    # Module to extract concepts from documents
│   └── generating_turkish_hypernyms.py     # Module for generating Turkish hypernyms using the official Turkish dictionary and the Turkish BERT model
│                                           
├── datasets/
│   ├── forensic_decisions/                 # Dataset containing forensic decisions
│   ├── forensic_news/                      # Dataset containing forensic news articles
│   └── sports_news/                        # Dataset containing sports news articles
│
├── resources/
│   ├── dictionary/                         # Dictionary resources for NLP tasks
│   └── lexical_database/                   # Lexical database containing hypernyms for linguistic reference
│
├── LICENSE                                  # License file for the project
├── README.md                                # Project documentation (this file)
├── requirements.txt                         # Python dependencies for the project
└── main.py                                  # Entrypoint script for training/evaluation
```

 ## Execution

Execute the file `main.py` to generate a hypernymy dictionary and to extract keywords from documents.

#### Setup with virtual environment (Python 3):

-  `python3 -m venv my_venv`
-  `source my_venv/bin/activate`

Install the requirements:

-  `pip3 install -r requirements.txt`

If everything works well, you can run the example usage given below.

### Example Usage:

- The following guide shows an example usage of the model performing training and evaluation for the aspect-based sentiment analysis task.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the exemplary main.py file.

Example:

```
python main.py \
  --dataset datasets \
  --lexical-db resources/lexical_database/Turkish_words_and_hypernyms.csv \
  --output output \
  --top-k 3
```
Another example usage, where the Turkish hypernym lexical database is regenerated in advance:

```
python main.py \
  --dataset ./datasets \
  --lexical-db ./resources/lexical_database/Turkish_words_and_hypernyms.csv \
  --output ./output \
  --top-k 5 \
  --generate-hypernyms
```
## Citation
If you find this code useful, please cite the following in your work:
```
@misc{concept-mining-in-turkish,
  author       = {Cem Rifki Aydin},
  title        = {Concept Mining in Turkish using BERT},
  howpublished = {\url{https://github.com/cemrifki/concept-mining-in-turkish}},
  year         = {2025},
  note         = {Accessed: 2025-08-23}
}
```
## Credits
- All the code has been written by Cem Rifki Aydin
