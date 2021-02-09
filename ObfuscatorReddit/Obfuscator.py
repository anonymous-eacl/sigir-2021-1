# Obfuscator for REDDIT comments
# Author: 

import csv
import sys
import random
sys.path.append('../DetectorsReddit/')
sys.path.append('AttnReddit/')
from BLSTM_Attention import Masker, EncoderRNN, Attn, AttnClassifier
from NULI import NULI, BertForSequenceClassification
from MIDAS import MIDAS, CNN, BLSTM, BLSTM_GRU
import sent2vec
import numpy as np
import json
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import emoji
import wordsegment
wordsegment.load()
from nltk.corpus import wordnet, words
import nltk

class Obfuscator:
    
    def __init__(self, obf_method = 'GS_GR', detection_alg = 'NULI', embedding_file = 'glove/glove.twitter.27B.100d.txt', sent_emb_file = 'sent2vecModels/twitter_bigrams.bin', filter_min = 3, filtered_embs = None, alg_version = 0):
        self.obf_method = obf_method

        if(detection_alg == 'NULI'):
            self.detector = NULI(train_data= '../DetectorsReddit/NULI/1_train_CensoredRedditData_ratio_15.0.tsv', trained_model = '../DetectorsReddit/NULI/NULI.pt', params_file = '../DetectorsReddit/NULI/NULI_params.json')
        elif(detection_alg == 'vradivchev'):
            self.detector = vradivchev(train_data= '../DetectorsReddit/vradivchev_anikolov/1_train_CensoredRedditData_ratio_15.0.tsv', trained_model = '../DetectorsReddit/vradivchev_anikolov/vradivchev.pt', params_file = '../DetectorsReddit/vradivchev_anikolov/vradivchev_params.json')
        elif(detection_alg == 'MIDAS'):
            self.detector =  MIDAS(train_data= '../DetectorsReddit/MIDAS/TRUEis1_1_train_CensoredRedditData_ratio_15.0.tsv', trained_cnn_model = '../DetectorsReddit/MIDAS/MIDAS_CNN.pt', trained_blstm_model = '../DetectorsReddit/MIDAS/MIDAS_BLSTM.pt', trained_blstmGru_model = '../DetectorsReddit/MIDAS/MIDAS_BLSTM-GRU.pt')
        #elif(detection_alg == 'Perspective'):
        #    self.detector = Perspective(threshold = 0.5)
        #elif(detection_alg == 'LexiconDetect'):
        #    self.detector = LexiconDetect(offensive_word_file = '../Detectors/LexiconDetect/abusive_words.txt')
        else:
            self.detector = None
            print('Detection algorithm not available.')
        
        # load in embeddings, differs for methods as GR uses word embeddings and EC uses sentence embeddings
        if(obf_method == 'GS_GR'):
            self.embeddingDict = {}
            embeddings = open(embedding_file, 'r')

            for embedding in embeddings:
                embedding = embedding.strip().split()
                self.embeddingDict[embedding[0]] = [float(x) for x in embedding[1:]]

            embeddings.close()

        elif(obf_method == 'GS_EC' or obf_method == 'GS_EC_MAX'):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(sent_emb_file)
            
            # convert glove embedding to word2vec embedding
            #glove_file = datapath(os.path.abspath())
            #tmp_file = get_tmpfile(os.path.abspath('glove/word2vec.twitter.27B.100d.txt'))
            #_ = glove2word2vec(glove_file, tmp_file)
            self.w2v_model = KeyedVectors.load_word2vec_format(embedding_file)
        
        elif(obf_method == 'AT_EC_MAX'):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(sent_emb_file)
            
            # convert glove embedding to word2vec embedding
            #glove_file = datapath(os.path.abspath())
            #tmp_file = get_tmpfile(os.path.abspath('glove/word2vec.twitter.27B.100d.txt'))
            #_ = glove2word2vec(glove_file, tmp_file)
            self.w2v_model = KeyedVectors.load_word2vec_format(embedding_file)
            
            self.masker = Masker(method = 'attention', target = 'TRUE', freq_thresh = 5, train_data = 'AttnReddit/1_train_CensoredRedditData_ratio_15.0.tsv', encoder = 'AttnReddit/Encoder.pt', classifier = 'AttnReddit/Classifier.pt')
            self.tokenize = lambda x: nltk.word_tokenize(x.lower())
        
        
        # refers to which algorithm should be used with the filter embbeding file
        # 0 = remove all with less frequency than filter_min (note if no file is passed for filtered_embs, no words are removed)
        # 1 = 0 + sort candidate words by highest frequency
        # 2 = 0 + sort candidate words by lowest frequency
        # 3 = 0 + rerank candidate list by sim_pos + freq_pos where sim_pos = 1 if most_similar and freq_pos = 1 if most frequent
        self.alg_version = int(alg_version)
        
        # if filter embedding files, read in and store
        filter_min = int(filter_min)
        self.filtered_embs = {}
        if(filtered_embs):
            filtercsv = csv.reader(open(filtered_embs), delimiter = ',')
            for cur_line in filtercsv:
                word = cur_line[0].strip()
                self.filtered_embs[word] = []
                
                if(self.alg_version == 0):
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            self.filtered_embs[word].append(sim.strip())

                elif(self.alg_version == 1):
                    tmp_sims = {}
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                        
                    self.filtered_embs[word].extend(sorted(tmp_sims, key=tmp_sims.get, reverse = True))

                elif(self.alg_version == 2):
                    tmp_sims = {}
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                        
                    self.filtered_embs[word].extend(sorted(tmp_sims, key=tmp_sims.get))
                
                else: 
                    tmp_sims = {}
                    tmp_pos = {}
                    cur_pos = 1
                    for cur_sim in cur_line[1:]:
                        split_sim = cur_sim.strip().split('-')
                        sim = '-'.join(split_sim[:-1])
                        count = split_sim[-1]
                        # add in similar words with at least a count of filter_min
                        if(int(count) >= filter_min):
                            tmp_sims[sim.strip()] = int(count)
                            tmp_pos[sim.strip()] = cur_pos
                            cur_pos += 1
                    
                    tmp_freqs = {}
                    cur_freq = 1
                    for cur in sorted(tmp_sims, key = tmp_sims.get, reverse = True):
                        tmp_freqs[cur] = cur_freq
                        cur_freq += 1
                        
                    # join sim and freq rankings together
                    final_ranks = {}
                    for cur in tmp_pos:
                        final_ranks[cur] = tmp_pos[cur] + tmp_freqs[cur]
                        
                    self.filtered_embs[word].extend(sorted(final_ranks, key=final_ranks.get))
                    
        #print(self.filtered_embs['dishonest'])
        

    def selectMethod(self, obf_method):
        self.obf_method = obf_method


    def preProcessText(self, text):
        text = text.lower().strip()

        # change emojis to readable text
        # segment text
        text = ' '.join(wordsegment.segment(emoji.demojize(text)))

        return text


    # determines word to be chosen via greedy select (checking probability changes) and greedy replaces (random replacement)
    def GS_GR(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations
        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
                prob_diffs.append(initial_prob - cur_prob)
            
            
        replace_pos = prob_diffs.index(max(prob_diffs))
        
        # get a random word from vocab to replace word
        rand_pos = random.randint(0, len(self.embeddingDict))
        replace_word = list(self.embeddingDict.keys())[rand_pos]
        replaced = [rand_pos]

        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

        # if the query without the word is not offensive randomly choose until you find a non offensive replacement
        if(orig_pred[replace_pos] == 'NOT'):
            # keep randomly replacing while prediction is TRUE
            new_pred, _ = self.detector.predict(obf_query)
            while(new_pred == 'TRUE'):
                # if all embeddings attempted, break out
                if(len(replaced) == len(self.embeddingDict)):
                    break

                rand_pos = random.choice([x for x in range(0, len(self.embeddingDict)) if x not in replaced])
                replaced.append(rand_pos)
                replace_word = list(self.embeddingDict.keys())[rand_pos]

                obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])
                new_pred, _ = self.detector.predict(obf_query)


        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])
        return obf_query

    
    # determines word to be chosen via greedy select (checking probability changes) and replaces using constraints on the embedding
    def GS_EC(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)

        
        replace_pos = prob_diffs.index(max(prob_diffs))
            
        
        # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
        #l1_dict = {}
        #orig_emb = self.model.embed_sentence(query)
        #for can_word in self.vocab:
        #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
        #    new_emb = self.model.embed_sentence(cur_obf)

        #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
        #    l1_dict[can_word] = l1_dist

        # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
        #for can_word in sorted(l1_dict, key=l1_dict.get):
        #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
        #    obf_pred, _ = self.detector.predict(cur_obf)
        #    replace_word = can_word
        #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'TRUE'):
        #        break


        # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
        #print(l1_dict)

        candidate_words = []

        # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
        if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
            candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
        else:
            # use word2vec to get list of closest words to the word to be replaced
            try:
                candidate_words = self.w2v_model.most_similar(split_query[replace_pos])
            except: #if replacement word does not exist in the vocabulary generate random list
                w2v_vocab = list(self.w2v_model.vocab.keys())
                for _ in range(10):
                    rand_pos = random.randint(0, len(w2v_vocab))
                    candidate_words.append((w2v_vocab[rand_pos], 0))

            tmp_words = candidate_words.copy()
            # filter out any non words from candidate list
            #for candidate in candidate_words:
            #    temp_candidate = candidate[0]
            #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
            #        candidate_words.remove(candidate)
                
            # if no english words, use anyways
            #if len(candidate_words) == 0:
            #    candidate_words = tmp_words.copy()

        lowest_distance = float('inf')
        replace_word = ''
        found_replace = False
        cands_probs = {}
        cands_dists = {}

        orig_emb = self.model.embed_sentence(query)
        # choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message TRUE
        for cand in candidate_words:
            if(type(cand) == tuple):
                can_word = cand[0]
            else:
                can_word = cand
            cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            obf_pred, obf_prob = self.detector.predict(cur_obf)
            cands_probs[can_word] = obf_prob

            new_emb = self.model.embed_sentence(cur_obf)

            l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            cands_dists[can_word] = l1_dist

            if(l1_dist < lowest_distance and obf_pred != 'TRUE'):
                lowest_distance = l1_dist
                replace_word = can_word
                found_replace = True

        # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest TRUE score
        if(not found_replace):
            replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                    
                
        

        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])
        
        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'l1 dist:', cands_dists[replace_word], 'replacement:', replace_word)
        return obf_query



    # determines MULTIPLE words to be chosen via greedy select (checking probability changes until no longer offensive) and replaces using constraints on the embedding
    def GS_EC_MAX(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations

        orig_pred, var_probs = self.detector.predictMultiple(variations)
        
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        
        finished = False
        remaining_diffs = []
        for x in prob_diffs:
            remaining_diffs.append(x)

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(remaining_diffs) == 0):
                finished = True
                break

            
            replace_pos = prob_diffs.index(max(remaining_diffs))
            remove_pos = remaining_diffs.index(max(remaining_diffs))
            remaining_diffs.pop(remove_pos)

            # find closest embedding in vocab to replace word, previous measurements are store to reduce runtime
            #l1_dict = {}
            #orig_emb = self.model.embed_sentence(query)
            #for can_word in self.vocab:
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    new_emb = self.model.embed_sentence(cur_obf)

            #    l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
            #    l1_dict[can_word] = l1_dist

            # if the query without the word is not offensive choose minimum distance until you find a non offensive replacement
            #for can_word in sorted(l1_dict, key=l1_dict.get):
            #    cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
            #    obf_pred, _ = self.detector.predict(cur_obf)
            #    replace_word = can_word
            #    if(obf_pred == 'NOT' or orig_pred[replace_pos] == 'TRUE'):
            #        break


            # sort by minimum l1 distance, if sentence is not offensive use, else move to next candidate word
            #print(l1_dict)


            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
                candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            else:
                # use word2vec to get list of closest words to the word to be replaced
                try:
                    candidate_words = self.w2v_model.most_similar(split_query[replace_pos])
                except: #if replacement word does not exist in the vocabulary generate random list
                    w2v_vocab = list(self.w2v_model.vocab.keys())
                    for _ in range(10):
                        rand_pos = random.randint(0, len(w2v_vocab))
                        candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}

            orig_emb = self.model.embed_sentence(query)

            # choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message TRUE 
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand

                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                obf_pred, obf_prob = self.detector.predict(cur_obf)
                cands_probs[can_word] = obf_prob

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                if(l1_dist < lowest_distance and obf_pred != 'TRUE'):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                    

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest TRUE score
            if(not found_replace):
                replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                finished = False


            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query



# determines MULTIPLE words to be chosen via attention select (checking attentions until no longer offensive) and replaces using constraints on the embedding
    def AT_EC_MAX(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        pred, prob, atts = self.masker.predict(query)
        
        split_query = self.tokenize(query)
        
        
        finished = False

        orig_atts = atts.copy()

        while(not finished):
            finished = True
            
            # if all positions have been replaced, break out
            if(len(atts) == 0):
                finished = True
                break

            
            replace_pos = orig_atts.index(max(atts))
            remove_pos = atts.index(max(atts))
            atts.pop(remove_pos)



            candidate_words = []

            # if the current word to be replaced exists in the filtered_embs, use those as candidates, else get from word2vec
            if(split_query[replace_pos].lower() in self.filtered_embs and len(self.filtered_embs[split_query[replace_pos].lower()]) > 0):
                candidate_words.extend(self.filtered_embs[split_query[replace_pos].lower()])
            else:
                # use word2vec to get list of closest words to the word to be replaced
                try:
                    candidate_words = self.w2v_model.most_similar(split_query[replace_pos])
                except: #if replacement word does not exist in the vocabulary generate random list
                    w2v_vocab = list(self.w2v_model.vocab.keys())
                    for _ in range(10):
                        rand_pos = random.randint(0, len(w2v_vocab))
                        candidate_words.append((w2v_vocab[rand_pos], 0))

                #tmp_words = candidate_words.copy()
                # filter out any non words from candidate list
                #for candidate in candidate_words:
                #    temp_candidate = candidate[0]
                #    if(not wordnet.synsets(temp_candidate) and not temp_candidate in words.words()):
                #        candidate_words.remove(candidate)

                # if no english words, use anyways
                #if len(candidate_words) == 0:
                #    candidate_words = tmp_words.copy()


            lowest_distance = float('inf')
            replace_word = ''
            found_replace = False
            cands_probs = {}
            cands_dists = {}

            orig_emb = self.model.embed_sentence(query)
            #print(candidates_words)

            # choose candidate word which creates minimum l1 distance (checking via sentence emb) and does not make message OFF 
            for cand in candidate_words:
                if(type(cand) == tuple):
                    can_word = cand[0]
                else:
                    can_word = cand
            
                cur_obf = ' '.join(split_query[:replace_pos] + [can_word] + split_query[replace_pos+1:])
                if(self.detector):
                    obf_pred, obf_prob = self.detector.predict(cur_obf)
                else:
                    obf_pred, obf_prob, _ = self.masker.predict(cur_obf)

                cands_probs[can_word] = obf_prob

                new_emb = self.model.embed_sentence(cur_obf)

                l1_dist = sum(np.abs(orig_emb[0] - new_emb[0]))
                cands_dists[can_word] = l1_dist

                # BLSTM returns 0 or 1
                if(self.detector):
                    label = 'TRUE'
                else:
                    label = 1

                if(l1_dist < lowest_distance and obf_pred != label):
                    lowest_distance = l1_dist
                    replace_word = can_word
                    found_replace = True
                    

            # if unable to find a replacement such that the obfuscation is not offensive, choose the one which resulted in lowest OFF score
            if(not found_replace):
                replace_word = sorted(cands_probs, key = cands_probs.get)[0]
                finished = False


            obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

            #print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos], 'l1 dist:', cands_dists[replace_word])

            split_query = obf_query.split()

        return obf_query




    
    # determines word to be chosen via greedy select (checking probability changes) and replaces with explitive letters (!#@*%)
    def GS_PR(self, query):
        query = self.preProcessText(query)

        # get inital probability for query 
        _, initial_prob = self.detector.predict(query)
        
        split_query = query.split()
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            variations.append(modified_query)

        # get probabilities for all variations
        orig_pred, var_probs = self.detector.predictMultiple(variations)
            
        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)

            
        replace_pos = prob_diffs.index(max(prob_diffs))
        
        punctAvail = ['!', '#', '@', '*', '%', '$']

        # replace word with combination of punctuation characters
        replace_word = split_query[replace_pos][0] + ''.join([random.choice(punctAvail) for _ in range(len(split_query[replace_pos]) - 1)])

        
        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos+1:])

        print('initial prob:', initial_prob, 'max diff prob:', max(prob_diffs), 'word replaced:', split_query[replace_pos])
        return obf_query

        


    def obfuscate(self, query):
        if(self.obf_method == 'GS_GR'):
            return self.GS_GR(query)
        elif(self.obf_method == 'GS_EC'):
            return self.GS_EC(query)
        elif(self.obf_method == 'GS_EC_MAX'):
            return self.GS_EC_MAX(query)
        elif(self.obf_method == 'GS_PR'):
            return self.GS_PR(query)
        elif(self.obf_method == 'AT_EC_MAX'):
            return self.AT_EC_MAX(query)


# Walk through a file where each line is text which should be obfuscated.
def main(conversion_file, detector_name, obf_alg = 'GS_GR', embedding_loc = 'glove/glove.twitter.27B.100d.txt', filter_min = 3, filtered_embs = None, alg_version = 0):
    
    all_tweets = {}

    # load in text to be obfuscated
    with open(conversion_file, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            all_tweets[tweet[0]] = tweet

    if(filtered_embs == 'None'):
        filtered_embs = None


    # walk through tweets and obfuscate 
    obfuscator = Obfuscator(obf_method = obf_alg, detection_alg=detector_name, embedding_file = embedding_loc, filter_min = filter_min, filtered_embs = filtered_embs, alg_version = alg_version)

    out_beg = '-'.join(embedding_loc.split('/')[-1].split('.')[:-1])
    if(obf_alg == 'AT_EC_MAX' and detector_name == 'None'):
        detector_name = 'BLSTM'

    if(filtered_embs):
        out_file = open(out_beg + '_obfuscatedVs-' + detector_name + '_' + 'filtered-' + alg_version + '_' + obf_alg + '_' + conversion_file, 'w')
    else:
        out_file = open(out_beg + '_obfuscatedVs-' + detector_name + '_' + obf_alg + '_' + conversion_file, 'w')
    outCSV = csv.writer(out_file, delimiter = '\t')
    for tweet in all_tweets:
        all_tweets[tweet][1] = obfuscator.obfuscate(all_tweets[tweet][1])

        out_tweet = all_tweets[tweet]
        
        outCSV.writerow(out_tweet)




if __name__ == "__main__":
    if(len(sys.argv) == 3):
        main(sys.argv[1], sys.argv[2])
    elif(len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif(len(sys.argv) == 5):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif(len(sys.argv) == 7):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    
            
