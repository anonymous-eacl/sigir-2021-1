#
# sigir-2021-1
#
#

#############
Obfuscator/
#############
	
Obfuscator.py - program to set up and run obfuscation. 
Grondahl.py - obfuscates using grondahl techniques
viper_*.py - VIPER program, WRITTEN BY VIPER AUTHORS full code found: https://github.com/UKPLab/naacl2019-like-humans-visual-attacks



#############
ObfuscatorReddit/
#############	

Obfuscator.py - program to set up and run obfuscation for reddit data. 



############
Detectors/
############

NULI.py - implementation of NULI offenseval system

vradivchev.py - implementation of vradivchev offenseval 2019 system

MIDAS.py - implementation of MIDAS offenseval 2019 system

Perspective.py - Perspective offense classifier implementation, api key needed

LexiconDetect.py - Lexicon offense classifier.



##########
DetectorsReddit/
##########

NULI.py - implementation of NULI trained on reddit data

MIDAS.py - implementation of MIDAS trained on reddit data



##########
Datasets/
##########

CensoredRedditData_ratio_15.0.tsv - dataset for reddit
1_train_CensoredRedditData_ratio_15.0.tsv - training split for reddit results
1_test_CensoredRedditData_ratio_15.0.tsv - testing split for reddit results

NonOffensiveTweets - Final list of non offensive tweets as deemed by classifiers (Make up evasion dataset)
createWordEmbeddingsNoConstraint.py - script to create NEW embeddings
trainWordEmbeddingsNoConstraint.py - script to fine tune (FT) glove embedding
(NOTE trained glove embeddings too large for github will link another way)

subset_OFF_testset-taska-gold.tsv - subset of offensive examples from test set from offenseval, used in results
offenseval-training-v1.tsv - training set from offenseval, used to train models in results
