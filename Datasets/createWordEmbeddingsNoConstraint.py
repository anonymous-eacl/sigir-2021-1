# trains word embeddings from file
from gensim.models import Word2Vec, KeyedVectors
import wordsegment
import csv
import sys
from nltk.corpus import wordnet, words


def createWordEmbeddings(inFileName, num_epochs = 1):
    sentences = []
    wordsegment.load()
    num_epochs = int(num_epochs)

    with open(inFileName, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter = '\t')
        for tweet in tweetreader:
            try:
                temp_segs = tweet[1].lower().strip().split()
                for seg in range(len(temp_segs)):
                    if('http' in temp_segs[seg] or '@' in temp_segs[seg]):
                        temp_segs.pop(seg)
                    #elif(not (wordnet.synsets(temp_segs[seg]) or temp_segs[seg] in words.words())):
                    #    temp_segs.pop(seg)

                temp_segs = wordsegment.segment(' '.join(temp_segs))

                sentences.append(temp_segs)
            except Exception as e:
                print(e)
                print(tweet)
                continue

    model = Word2Vec(sentences, min_count = 1, iter = num_epochs)
    model.save('nonOffensiveModel-NoConstraint_' + str(num_epochs) + 'epoch.bin')


    

createWordEmbeddings(sys.argv[1], sys.argv[2])
