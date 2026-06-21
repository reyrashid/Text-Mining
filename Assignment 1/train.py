##############################################################
# helper functions for feature extraction and model training #
#############################################################

import util
import sklearn_crfsuite
import pickle

from sklearn.feature_extraction import DictVectorizer
import feature_extraction

import spacy
from spacy.tokens import Doc
import benepar
from nltk.tree import Tree

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def extract_features(infile, all_sentences = None, all_cues = None):
    """Extracts the features given an input file.

    args:
        infile: input conll file
        all_sentences: List of tokens for each sentence
        all_cues: List of negation cues for each token in a list of sentences

    Either infile (conll format) or sentences/cues if available can be specified. Not both.
        
    returns: List of dict where each dict is a feature dictionary containing features for each sentence.
    """
    all_features = list()
    if infile:
        all_sentences = util.conll_sentences_as_token_lists(infile, token_col=3)  # col 3 is for tokens
        all_cues = util.conll_sentences_as_token_lists(infile, token_col=7)  # col 7 is for neg cues
    
    num_sentences = len(all_sentences)

    # These features are extracted outside the loop since it takes the file as input
    
    token_itself= list()
    cue_token_position= list()
    cue_token_same_phrase_and_BIE = list()
    negation_cue_type = list()
    is_cue_and_token_in_diff_S_or_SBAR_clause = list()
    boundary_indicator = list()
    dependency_relations = list()
    pos = list()
    bidirectional_dependency_distance = list()

    # Loop over all sentences one by one to build the features for each sentence
    # These features are extracted on sentence level as constituency and dependency features require full sentence, not just a token
    for i in range(num_sentences):
        tokens = all_sentences[i]
        cues = all_cues[i]
        doc = Doc(nlp.vocab, words=tokens)
        doc = nlp(doc)

        cue_token_position += feature_extraction.cue_token_position(tokens, cues)
        cue_token_same_phrase_and_BIE += feature_extraction.cue_token_same_phrase_and_BIE (tokens, cues)
        negation_cue_type += feature_extraction.negation_cue_type(tokens, cues)
        is_cue_and_token_in_diff_S_or_SBAR_clause += feature_extraction.is_cue_and_token_in_diff_S_or_SBAR_clause(tokens, cues)

        boundary_indicator += feature_extraction.extract_boundary_indicator(tokens)
        dependency_relations += feature_extraction.extract_dependency_relations(tokens)

        pos += feature_extraction.extract_pos_tags_from_doc(doc)
        bidirectional_dependency_distance += feature_extraction.extract_bidirectional_dependency_distance(doc, cues)
        for token in tokens:
            token_itself.append(token)

    # List of the feature dictionary per token will be the complete feature list for all tokens.
    total_tokens = len(cue_token_position)
    for i in range(total_tokens):
        feature_dict = { 'token': token_itself [i],
            # carmen
            'negation_cue_type': negation_cue_type[i],
            'is_cue_and_token_in_diff_S_or_SBAR_clause': is_cue_and_token_in_diff_S_or_SBAR_clause[i],
            # keze
            'cue_token_position': cue_token_position[i],
            'cue_token_same_phrase_and_BIE': cue_token_same_phrase_and_BIE[i],
            # manar
            'boundary_indicator': boundary_indicator[i],
            'dependency_relations': dependency_relations[i],
            # rey
            'pos': pos[i],
            'bidirectional_dependency_distance': bidirectional_dependency_distance[i],
        }
        all_features.append(feature_dict)
    
    return all_features

def extract_labels(inputfile):
    """Extracts the labels given an input file.

    args:
        infile: input conll file

    returns: List of negation scope label corresponding to each token for each sentence.
    """
    gold= []
    
    with open (inputfile, 'r', encoding = 'utf-8') as file:
        sents= file.read().strip().split('\n\n')
            
        for sent in sents: 
            gold_per_sent =[]  #collecting gold per sent
            lines= sent.split('\n')
        
            for line in lines:
                data= line.split('\t')
                gold_per_sent.append(data[8]) #extracting token per line
            gold.append( gold_per_sent)
        
    return gold
    

def train(features, labels, model_output_path):
    """Trains a CRF model for negation scope detection task. First converts features from the flat list of dict into nexted lists.
    The model is saved in the given output path directory.

    args:
        features: List of all feature dictionaries
        labels: List of all negation scope labels for each token per sentence (nested lists)
        model_output_path: Output directory path to save the trained model and vectorizer

    returns: model and vectorizer
    """
    # transform the feature from a list of dict into nested lists
    nested_features= []
    slices = 0
    for label_per_sent in labels:
        sent_len= len(label_per_sent)
        feature_by_sent= features[slices:slices+sent_len]
        nested_features.append(feature_by_sent)
        slices += sent_len  #updating slices after per sent

    # Train model.
    crf = sklearn_crfsuite.CRF()
    crf.fit(nested_features, labels)

    # Save model and vectorizer.
    with open(f"{model_output_path}/model.pkl", "wb") as f:
        pickle.dump(crf, f)
    
    return crf
