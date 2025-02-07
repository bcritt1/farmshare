# Import packages
import os
import json
from transformers import pipeline
from transformers.models.roberta import RobertaTokenizer, RobertaForTokenClassification
import torch
# Read in corpus
user = os.getenv('USER')
corpusdir = '/farmshare/home/groups/srcc/cesta_workshop/corpus/'
#corpusdir = '/scratch/users/{}/corpus/'.format(user)
corpus = []
for infile in os.listdir(corpusdir):
    with open(corpusdir+infile, errors='ignore') as fin:
        corpus.append(fin.read())

# Import language models and pipeline elements
tokenizer = RobertaTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

# Process corpus
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
entities = nlp(corpus)

# Export data to json
with open('/scratch/users/{}/outputs/dataRoberta.json'.format(user), 'w', encoding='utf-8') as f:
    json.dump(str(entities), f, ensure_ascii=False, indent=4)
