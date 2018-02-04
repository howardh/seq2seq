import csv
from tqdm import tqdm

sentences_file = "./data/sentences.csv"
links_file = "./data/links.csv"
langs = ['eng', 'fra']

sentence_ids = set()
sentence_ids_by_lang = dict()
for l in langs:
    sentence_ids_by_lang[l] = set()
with open(sentences_file, "r") as sentences_csvfile:
    reader = csv.reader(sentences_csvfile, delimiter='\t')
    for sentence_id, lang, sentence in tqdm(reader):
        if lang in langs:
            sentence_ids.add(sentence_id)
            sentence_ids_by_lang[lang].add(sentence_id)

pairs = []
translated_sentence_ids = set()
with open(links_file, "r") as links_csvfile:
    reader = csv.reader(links_csvfile, delimiter='\t')
    for id1,id2 in tqdm(reader):
        if id1 in sentence_ids and id2 in sentence_ids:
            if id1 in sentence_ids_by_lang[langs[0]]:
                pairs.append((id1,id2))
            else:
                pairs.append((id2,id1))
            translated_sentence_ids.add(id1)
            translated_sentence_ids.add(id2)

translated_sentences = dict()
with open(sentences_file, "r") as sentences_csvfile:
    reader = csv.reader(sentences_csvfile, delimiter='\t')
    for sentence_id, lang, sentence in tqdm(reader):
        if sentence_id in translated_sentence_ids:
            translated_sentences[sentence_id] = sentence

output = [(translated_sentences[p[0]],translated_sentences[p[1]]) for p in tqdm(pairs)]
