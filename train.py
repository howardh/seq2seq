from tqdm import tqdm
import numpy as np
import time
import string
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import tatoeba

use_cuda = False

SOS = 0
EOS = 1 

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {SOS: "SOS", EOS: "EOS"}
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
    - Make everything lowercase
    - Removing invalid characters
    - Separating out punctuation
    - Splitting contractions
    """
    # Replace non-breaking spaces with spaces
    sentence = sentence.replace(u'\xa0', u' ')
    sentence = sentence.replace(u'\u202f', u' ')
    if lang_name == "eng":
        # Remove leading/training spaces and change to lowercase
        sentence = sentence.lower().strip()
        # Remove invalid characters
        valid_chars = string.printable
        sentence = ''.join((x for x in sentence if x in valid_chars))
        # Find punctuations and separate them
        punctuation = ['.',',','!','?','"']
        for p in punctuation:
            sentence = sentence.replace(p," "+p)
        # Split contractions
        sentence = sentence.replace("'"," ")

        return sentence

    if lang_name == "fra":
        # Remove leading/training spaces and change to lowercase
        sentence = sentence.lower().strip()
        # Find punctuations and separate them
        punctuation = ['.',',','!','?','"','«','»']
        for p in punctuation:
            sentence = sentence.replace(p," "+p)
        # Split contractions
        sentence = sentence.replace("'"," ")

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

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def sentence_to_indices(language, sentence):
    """
    Given a sentence and its associated Language object, return a vector of
    word indices corresponding to that sentence.
    The sentence can either be a string where each word is separated by a
    space, or a list of tokens.
    """
    if type(sentence) is str:
        sentence = process_sentence(language.name, sentence)
        tokens = sentence.split(' ')
    elif type(sentence) is list:
        tokens = sentence
    return [language.word2index[t] for t in tokens if t != ''] + [EOS]

def sentence_to_variable(language, sentence, use_cuda=False):
    indices = sentence_to_indices(language, sentence)
    result = Variable(torch.LongTensor(indices).view(-1,1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt,
        criterion, encoder_hidden=None, teacher_forcing_ratio=0.5):
    if encoder_hidden is None:
        encoder_hidden=encoder.init_hidden()
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_length = input_var.size()[0]
    target_length = target_var.size()[0]

    # Encode input
    encoder_outputs = []
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_var[ei], encoder_hidden)
        encoder_outputs.append(encoder_output[0][0])

    # Decode
    decoder_input = Variable(torch.LongTensor([[SOS]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    loss = 0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_var[di])
            decoder_input = target_var[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_var[di])
            if ni == EOS:
                break

    loss.backward()

    encoder_opt.step()
    decoder_opt.step()

    return loss.data[0] / target_length

def test(input_var, target_var, encoder, decoder,
        criterion, encoder_hidden=None):
    if encoder_hidden is None:
        encoder_hidden=encoder.init_hidden()

    input_length = input_var.size()[0]
    target_length = target_var.size()[0]

    # Encode input
    encoder_outputs = []
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_var[ei], encoder_hidden)
        encoder_outputs.append(encoder_output[0][0])

    # Decode
    decoder_input = Variable(torch.LongTensor([[SOS]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        loss += criterion(decoder_output, target_var[di])
        if ni == EOS:
            break

    return loss.data[0] / target_length

def test_all(encoder, decoder, data, input_lang, output_lang, criterion):
    loss = 0
    for data_pair in data:
        input_variable = sentence_to_variable(input_lang, data_pair[0])
        target_variable = sentence_to_variable(output_lang, data_pair[1])
        loss += test(input_variable, target_variable, encoder,
                     decoder, criterion)
    return loss/len(data)

def trainIters(encoder, decoder, train_data, test_data, input_lang, output_lang, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    n_iters = 100
    print_every = 10
    plot_every = 10
    for iter in range(1, n_iters + 1):
        training_pair = random.choice(train_data)
        input_variable = sentence_to_variable(input_lang, training_pair[0])
        target_variable = sentence_to_variable(output_lang, training_pair[1])

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            test_loss = test_all(encoder,decoder,test_data,input_lang,output_lang,criterion)
            print('%.4f\t %.4f' % (print_loss_avg, test_loss))

def train_test_split(data, test_percent=0.1):
    test_size = int(len(data)*test_percent)
    test_indices = np.random.choice(range(len(data)), size=test_size, replace=False)
    train = [data[i] for i in range(len(data)) if i not in test_indices]
    test = [data[i] for i in range(len(data)) if i in test_indices]
    return train,test

if __name__=="__main__":
    data = tatoeba.get_data()[:100]
    data = [(process_sentence("eng",p0),process_sentence("fra",p1)) for p0,p1
            in tqdm(data,"Preprocessing Sentences")]
    train_data, test_data = train_test_split(data, 0.1)
    eng_lang, fra_lang = compute_language(data, ["eng","fra"])

    hidden_size = 256
    encoder = EncoderRNN(eng_lang.n_words, hidden_size)
    decoder = DecoderRNN(hidden_size, fra_lang.n_words)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    trainIters(encoder, decoder, train_data, test_data, eng_lang, fra_lang, 0.01)
