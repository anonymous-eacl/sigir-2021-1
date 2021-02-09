# trains word embeddings from file
from gensim.models import Word2Vec, KeyedVectors
import wordsegment
import csv
import sys
from nltk.corpus import wordnet, words

# example from: https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights
def updateWordEmbeddings(tweetFile, oldEmbeddingFile, num_epochs = 1):
    num_epochs = int(num_epochs)
    
    # load in sentences to update vocab
    sentences = []
    wordsegment.load()
    
    with open(tweetFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter = '\t')
        for tweet in tweetreader:
            try:
                temp_segs = tweet[1].lower().strip().split()
                for seg in range(len(temp_segs)):
                    if('http' in temp_segs[seg] or '@' == temp_segs[seg][0]):
                        temp_segs.pop(seg)
                    #elif(not (wordnet.synsets(temp_segs[seg]) or temp_segs[seg] in words.words())): #Only include english words
                    #    temp_segs.pop(seg)
                
                temp_segs = wordsegment.segment(' '.join(temp_segs))
                # remove urls
                sentences.append(temp_segs)
            except:
                continue

    
    #update vocab and retrain
    model_2 = Word2Vec(size=100, min_count=1)
    model_2.build_vocab(sentences)
    total_examples = model_2.corpus_count
    
    model = KeyedVectors.load_word2vec_format(oldEmbeddingFile)
    
    model_2.build_vocab([list(model.vocab.keys())], update = True)
    model_2.intersect_word2vec_format(oldEmbeddingFile, lockf=1.0)
    #model_2.save('pretrained_model.bin')
    model_2.train(sentences, total_examples = total_examples, epochs=num_epochs)

    #model_2.save('updatedNonOffensiveModel-NoConstraint_' + str(num_epochs) + 'epoch.bin')
    model_2.save('updatedDeletedModel-NoConstraint_' + str(num_epochs) + 'epoch.bin')
    

updateWordEmbeddings(sys.argv[1], sys.argv[2], sys.argv[3])
