# concept-mining-in-turkish
This repository performs concept mining for Turkish. It leverages the Turkish BERT model and utilizes the official Turkish dictionary. Ultimately, concepts / hypernyms are extracted from Turkish documents. The template for the Turkish BERT model can be adjusted to generate more plausible tokens according to your requirements. 

Unfortunately, I cannot share the full Turkish dictionary due to copyright restrictions, so only a small portion is provided. I will also soon add another module that takes Turkish documents as input and extracts a candidate list of concepts. For now, the functionality of extracting keywords is integrated. In the meantime, 
I will also keep trying to improve the performance of the BERT model in generating the hypernymy dictionary.


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
python3 main.py
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
