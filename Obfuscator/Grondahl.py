# Obfuscator for Grondahl

import csv
import sys
import random
import sent2vec
#import numpy as np
import json
import os
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
import emoji
import wordsegment
wordsegment.load()
from nltk.corpus import wordnet, words


class Grondahl:
    
    def __init__(self, obf_method = 'add_space'):
        self.obf_method = obf_method
        
        
    def selectMethod(self, obf_method):
        self.obf_method = obf_method


    def obfuscate(self, query):
        if(self.obf_method == 'add_space'):
            return self.add_space(query)
        elif(self.obf_method == 'remove_space'):
            return self.remove_space(query)
        elif(self.obf_method == 'add_love'):
            return self.add_love(query)
        elif(self.obf_method == 'remove_space_add_love'):
            return self.add_love(self.remove_space(query))
            
    # simulates Grondahl attack of adding space randomly between each word to split words
    def add_space(self, query):
        
        obf_query = []

        for word in query.split():
            split_pos = random.randrange(0, len(word))
            
            obf_word = word[:split_pos] + ' ' + word[split_pos:]
            
            obf_query.append(obf_word)
            
        return ' '.join(obf_query)


    # Grondahl attack of removing all white space
    def remove_space(self, query):
        obf_query = query.replace(' ', '')
        
        return obf_query
            

    # Grondahl attack of adding 'love' to text to mitigate hate speech detectors
    def add_love(self, query):
        obf_query = query + ' love'
        
        return obf_query

            

# Walk through a file where each line is text which should be obfuscated.
def main(conversion_file, obf_alg = 'add_space'):
    
    all_tweets = {}

    # load in text to be obfuscated
    with open(conversion_file, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            all_tweets[tweet[0]] = tweet


    # walk through tweets and obfuscate 
    obfuscator = Grondahl(obf_method = obf_alg)

    out_file = open('obfuscatedVsGrondahl_' + obf_alg + '_' + conversion_file, 'w')
    outCSV = csv.writer(out_file, delimiter = '\t')
    for tweet in all_tweets:
        all_tweets[tweet][1] = obfuscator.obfuscate(all_tweets[tweet][1])

        out_tweet = all_tweets[tweet]
        
        outCSV.writerow(out_tweet)




if __name__ == "__main__":
    if(len(sys.argv) == 3):
        main(sys.argv[1], sys.argv[2])
    
