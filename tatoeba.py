import os
import csv
import dill
from tqdm import tqdm

sentences_file = "./data/sentences.csv"
links_file = "./data/links.csv"
langs = ['eng', 'fra']
pickled_file = "./data/eng-fra.pkl"

def parse_data(sentences_file, links_file, langs):
    # Get IDs of all sentences that are of the correct language
    global sentence_ids_by_lang
    sentence_ids = set()
    sentence_ids_by_lang = dict()
    for l in langs:
        sentence_ids_by_lang[l] = set()
    with open(sentences_file, "r") as sentences_csvfile:
        reader = csv.reader(sentences_csvfile, delimiter='\t')
        for sentence_id, lang, sentence in tqdm(reader,"Reading Sentence IDs"):
            if lang in langs:
                sentence_ids.add(sentence_id)
                sentence_ids_by_lang[lang].add(sentence_id)

    # Get every translated pair of sentence IDs where the sentences are of the
    # correct language
    # Ensure the pair is in the right language order
    pairs = []
    translated_sentence_ids = set()
    with open(links_file, "r") as links_csvfile:
        reader = csv.reader(links_csvfile, delimiter='\t')
        for id1,id2 in tqdm(reader,"Reading Translation Pairs"):
            if id1 in sentence_ids and id2 in sentence_ids:
                # Note: There are 'translations' that match equivalent
                # sentences from the same language
                pair = None
                if id1 in sentence_ids_by_lang[langs[0]] and id2 in sentence_ids_by_lang[langs[1]]:
                    pair = (id1,id2)
                elif id1 in sentence_ids_by_lang[langs[1]] and id2 in sentence_ids_by_lang[langs[0]]:
                    pair = (id2,id1)
                if pair is not None:
                    pairs.append(pair)
                    translated_sentence_ids.add(id1)
                    translated_sentence_ids.add(id2)

    # Read the sentences of interest from file
    global translated_sentences
    translated_sentences = dict()
    with open(sentences_file, "r", encoding='utf-8') as sentences_csvfile:
        reader = csv.reader(sentences_csvfile, delimiter='\t')
        for sentence_id, lang, sentence in tqdm(reader,"Reading Translated Sentences"):
            if sentence_id in translated_sentence_ids:
                translated_sentences[sentence_id] = sentence

    output = [(translated_sentences[p[0]],translated_sentences[p[1]]) for p in
            tqdm(pairs,"Matching Sentences to IDs")]
    with open(pickled_file,"wb") as f:
        dill.dump(output, f)
    return output

def get_data():
    if not os.path.isfile(pickled_file):
        parse_data(sentences_file, links_file, langs)
    with open(pickled_file,"rb") as f:
        return dill.load(f)

def get_short_data(max_len=10):
    data = get_data()
    def check_len(pair):
        return len(pair[0].split(' ')) <= max_len and len(pair[1].split(' ')) <= max_len
    return [d for d in tqdm(data,"Filtering len <= %d"%max_len) if check_len(d)]

if __name__=="__main__":
    data = parse_data(sentences_file, links_file, langs)
