from tqdm import tqdm
import torch

import tatoeba

data = tatoeba.get_data()

class Language:
    def __init__(self, name):
        self.__name = name
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def process_sentence(lang_name, sentence):
    """
    Preprocess the sentence from the given language by:
    - Removing leading/trailing spaces
    - Removing invalid characters
    - Separating out punctuation
    - Splitting contractions
    """
    if lang_name == "eng":
        # Find punctuations
        punctuation = ['.',',','!','?']
        sentence = sentence.lower().strip()
        return sentence

def compute_language(data, langs):
    lang1 = Language(langs[0])
    lang2 = Language(langs[1])
    for pairs in tqdm(data,"Computing Language"):
        lang1.addSentence(pairs[0])
        lang2.addSentence(pairs[1])
    return lang1,lang2

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def sentence_to_variable(language, sentence):
    """
    Given a sentence and its associated Language object, return a vector of
    word indices corresponding to that sentence.
    The sentence can either be a string where each word is separated by a
    space, or a list of tokens.
    """
    if type(sentence) is str:
        tokens = sentence.split(' ')
    elif type(sentence) is list:
        tokens = sentence
    return [language.word2index[t] for t in tokens]

if __name__=="__main__":
    eng_lang, fra_lang = compute_language(data, ["eng","fra"])
