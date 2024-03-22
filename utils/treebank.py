#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "utils/datasets/stanfordSentimentTreebank" # path to datatset folder

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        # checks if self_tokens is already initialized
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens
        
        '''
        It goes through self._sentences list by list. Also, at the end it adds extra one toekn 'UNK'
        self._tokens = assings as unique id to every new token it encounters, a dictionary
        self._tokenfreq = a dictionary conataining the frequency of every token
        self._word_count = total count of words in the doc (= self._cumsentlen[-1] + 1 (for 'UNK'))
        self._revtokens - list of unique tokens in the file + 'UNK' (at the end)
        '''
        
        
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0
                
        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1
        

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        # check if self._sentences is already initialized
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        '''
        Read every sentence datasetSentences.txt, except the first sentence (since it's the attributes name) line by line, split the sentence into a list of words (all small lettered) and append to the sentences list - list of list 
        
        sentences = self._sentences = [['the', 'rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'century', "'s", 'new', '``', 'conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'arnold', 'schwarzenegger', ',', 'jean-claud', 'van', 'damme', 'or', 'steven', 'segal', '.'], ['the', 'gorgeously', 'elaborate', 'continuation', 'of', '``', 'the', 'lord', 'of', 'the', 'rings', "''", 'trilogy', 'is', 'so', 'huge', 'that', 'a', 'column', 'of', 'words', 'can', 'not', 'adequately', 'describe', 'co-writer\\/director', 'peter', 'jackson', "'s", 'expanded', 'vision', 'of', 'j.r.r.', 'tolkien', "'s", 'middle-earth', '.']])
        
        self._sentlengths = list containin the number of tokes for each sentence
        self._cumsentlen = cum sum of  self._sentlengths i.e last nu,ber represents total no. of tokens in the text file
        
        '''
        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)
        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        # check if self._allsentences is already initialized
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        
        '''
        for every sentence it samples tokens depending on the weights assigned in self.rejectProb()
        If the count of a particular token <= threshold. it would sample always otherwise it samples from a uniform distrbution
        
        These are stored in self._allsentences, list of list containing sampled tokens for each sentence
        '''
        allsentences = [[w for w in s
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        # samples a sentence from  seld._allsentences
        sentID = random.randint(0, len(allsent) - 1) 
        sent = allsent[sentID]
        
        # samples a center word
        wordID = random.randint(0, len(sent) - 1)
        
        # for context words it seens numbre of words before the center word, it is less that window size ut takes everything other takes C umber of words. Same for other side of center word        
        context = sent[max(0, wordID - C):wordID]
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]
        
        # if no context words present, sample the center word and repeat the process again
        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            sent_labels[i] = labels[dictionary[full_sent]]

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in range(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        # Checks if self._sampletable is already defined
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable
        
        '''
        samplingFreq = U(w)**0.75/Z where U(W) is the frequency of every token and 
        Z = total number of wrods
        '''
        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        
        # last term of  np.cumsum(samplingFreq) is 1. We multiply by a constant self.table_size = 1000000
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize
        self._sampleTable = [0] * self.tablesize
        
        '''
        we create a list self_sampleTable where elements are tokenID. 
        We loop over number 1 to 1000000 and and only when it becomes greater tsamplingFreq[j] we increase the token id i.i we are creating a table which assigns a token id for for every CDF values from 0 to 1000000
        '''
        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        # Chcek if self._rejectProb is already initizalized
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount # different probability weight if count of a token is  < threshold

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,)) # list containing weigh tvalues for every token 
        # loop through every unique token and asssign a wt. depending on its frequency
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        # return self._rejectProb
        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        '''
        Samples a token id based on a CDF. Sample a number between 0 to 1000000, and use the number to ken the token id 
        '''
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]