##############################################################
      # Feature extraction code per group member        #
#############################################################

import spacy
from spacy.tokens import Doc
import benepar
from nltk.tree import Tree

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

### Manar 
# Manar Feature 1: Boundary indicator
def extract_boundary_indicator(tokens):
    """
    Extract a boolean Boundary Token Indicator for each token.

    Parameters
    ----------
    tokens : list[str]
        Tokenized sentence.

    Returns
    -------
    list[bool]
        True if token is a strong surface boundary, else False.
    """

    PUNCT_BOUNDARIES = {",", ".", ";", ":", "?", "!"}
    DISCOURSE_BOUNDARIES = {
        "but", "however", "although", "though", "yet", "and", "or"
    }

    boundary_flags = []

    for tok in tokens:
        is_boundary = (
            tok in PUNCT_BOUNDARIES or
            tok.lower() in DISCOURSE_BOUNDARIES
        )
        boundary_flags.append(is_boundary)

    return boundary_flags

# Manar Feature 2: Dependancy Relation label

def extract_dependency_relations(tokens):
    """
    Extract dependency relation labels for each token.

    Parameters
    ----------
    tokens : list[str]
        Tokenized sentence.

    Returns
    -------
    list[str]
        Dependency relation label per token (e.g., 'nsubj', 'obj', 'ROOT').
    """

    doc = Doc(nlp.vocab, words=tokens)
    doc = nlp(doc)

    return [token.dep_ for token in doc]
    
#########################################
### Keze
#Keze Feature 1: Relevant Position
def cue_token_position (token, cue_labels):
    """
    extract the relevant position of a token w.r.t the negation cue token in the sentence, using the values below:
    0 == the sentence has *no negation cues*
    1 == the token is *before* the token
    2 == the token *is / part of* the negation cue token(s)
    3 == the token is *after* the negation token
    Parameters
    ----------
    token: a list of string
    cue_labels: a list of strings with binary cue-labels ('_'/'x')
        Path to the input file with the 10-column format conll file.
        
    Returns
    -------
    list
        A list of numericposition values mentioned above
    """
    
    position= []
          
    #finding the cue index (i.e. the position of the (first) cue token in a sent)    
    cue_index= -1 #setting index value to -1 at default, meaning that the sent has no neg cues
    for i, cue_label in enumerate(cue_labels):
        if cue_label != '_':
            #only captures the first occurrence of a cue, and overwrites the cue_index
            cue_index= i
            break 
                
    for i, (token, cue_label) in enumerate (zip(token,cue_labels)):
        #if a sent has no neg cues (index== -1), position value= 0)
        if cue_index == -1:
            position.append(0)
        #addressing cues with any number of words: as long as cue label is not _, the position value is always 2 (the cue itself)
        elif cue_label != '_':
            position.append (2)  
        #tokens after are assigned with 3    
        elif i< cue_index:
            position.append(1)
        #tokens after are assigned with 3
        elif i> cue_index:
            position.append(3)
    return position

#Keze Feature 2 is_same_phrase and BIE
def cue_token_same_phrase_and_BIE (token, cue_labels):
    """
    extract the least common ancestor between a token and a cue at the phrasal level, as 
    well as the phrasal boundaries:
    O= No negation cue in the sentence/ No LCA
    B-XP= The token is at the beginning of the LCA phrase
    I-XP= The token is inside the LCA phrase
    E-XP= The token marks the end of the LCA phrase
    Parameters
    ----------
    token: a list of string
    cue_labels: a list of strings with binary cue-labels ('_'/'x')
    
        
    Returns
    -------
    list
        A list of categorical values mentioned above
    """    
    lca= []
    
    # preparing the Spacy doc by feeding in sent as a list of tokens
    ###fix suggested by Gemini regarding the error of NonConstituentException due to the strage quotation marks, last accessed on 15 Jan
    clean_tokens = [t.replace("``", '"').replace("''", '"') for t in token]
    ###
    doc = Doc(nlp.vocab, words=clean_tokens)
    pro_doc = nlp(doc)
    cue_indices= []
    #extracting cue indices
    for i, label in enumerate(cue_labels):
        if label != '_':
            cue_indices.append(i)
    
        #for sents without negation cues, a O is given.
    if not cue_indices:
        lca.extend(['O'] * len(token)) # Changed token_per_sent to token
    else:
    
        #defining the negation cue boundary
        cue_start = min(cue_indices)
        cue_end = max(cue_indices) + 1
        
        for i in range (len(token)):
            phrase = None
            lca_label = 'O'
            ###fix suggested by Gemini on  NonConstituentException, obtained on 15 Jan
            for s in pro_doc.sents:
            ###
                ### span methods inspired by the official Github of the parser (https://github.com/mrdrozdov/self-attentive-parser-with-extra-features),
                #last accessed: 15 Jan
                for constituent in s._.constituents:
                    if (constituent.start <= cue_start and constituent.end >= cue_end and constituent.start <= i < constituent.end):
                        if len(constituent)== len(doc): #skipping the root 
                            continue
                        if phrase is None or len(constituent) < len(phrase): #iterating over the smallest consti unit
                            phrase = constituent
                            lca_label = constituent._.labels[0] if constituent._.labels else "XP"
                ###
            if phrase: #Checks if the target token and the cue in the same phrase
                if lca_label =='S' or lca_label == 'SBAR': #the feature is phrase-oriented, so 'S' and 'SBAR' labels are converted to O
                    lca_bie= ('O')
                else:
                    #assigning Beginning/End/Inside depending on the position
                    if i == phrase.start:
                        lca_bie= (f'B-{lca_label}')
                    elif i == phrase.end-1:
                        lca_bie= (f'E-{lca_label}')
                    else:
                        lca_bie= (f'I-{lca_label}')
        
            else:
                lca_bie=('O')
                    
            lca.append(lca_bie)
    
        
    return lca


#########################################
### Rey
#Rey Feature 1: POS
def extract_pos_tags_from_doc(doc):
    """
    Extract POS tags for each token from a spaCy Doc object (already parsed).
    Returns POS tage (list of strings)
    """
    return [token.pos_ for token in doc]


# Rey Feature 2: Bidirectional dependency distance
def get_dependency_distance_length(token1, token2):
    """
    Gets shortest distance between two tokens in the dependency tree. 
    Treats the dependency tree as undirected. It can go from a token to its head or to its children (bidirectional)
    
    Parameters
    ----------
    token1: the first token (starting point)
    token2: the second token (target token)
    
    Returns
    -------
    int: number of steps between the two tokens in the dependency tree.
        0 if token1 and token2 are the same token.
        999 if token2 is not reachable (unlikely)
    """
    #If the two tokens are the same, distance is zero
    if token1 == token2:
        return 0

    #Queue to track tokens to visit, starting from token1
    queue = [(token1, 0)]
    visited = set([token1.i]) #Set to track visited tokens

    while queue:
        current, dist = queue.pop(0)  # pop first element in queue
        
        # Neighbors: children + head (bidirectional)
        neighbors = list(current.children)
        if current.head != current:
            neighbors.append(current.head)
        #Explore neighbors
        for neighbor in neighbors:
            if neighbor == token2:
                return dist + 1   # Found target token
            if neighbor.i not in visited:
                visited.add(neighbor.i)
                queue.append((neighbor, dist + 1))
    
    #If token2 is unreachable (unlikely)
    return 999


def extract_bidirectional_dependency_distance(doc, cue_labels):
    """
    Compute bidirectional dependency distance from each token to the nearest cue in the dependency tree.

    Parameters
    ----------
    doc: the parsed sentence as a spaCy Doc object.
    cue_labels: cue labels for each token in the sentence (list of str)

    Returns
    -------
    Distances for each token in the sentence (list of int)
        0 for cue tokens
        positive integer for non-cue tokens
        -1 for tokens in sentences without any cues

    """
    #Find indices of all cue tokens
    cue_indices = [i for i, label in enumerate(cue_labels) if label != '_'] #'_' for tokens that are not cues
    distances = []
    # Loop through all tokens
    for i, token in enumerate(doc):
        if not cue_indices:
            distances.append(-1) #-1 for no cue in the sentence 
        elif i in cue_indices:
            distances.append(0)  #the token is the cue
        else:
            #shortest distance to any cue
            min_dist = min(
                get_dependency_distance_length(token, doc[cue_idx])
                for cue_idx in cue_indices
            )
            distances.append(min_dist)
    return distances
    
#########################################

### Carmen
# Carmen Feature 1 - Negation Cue Type
def negation_cue_type(sentence, neg_cues, non_cue_marker = '_'):
    """Feature that extracts the type of the negation cue:
        - NO_CUE
        - SINGLE : e.g. not
        - PREFIX : e.g. (un)important
        - POSTFIX : e.g. care(less)
        - INFIX : e.g. care(less)ness
        - MULTI : e.g. neither ... nor

    Assumes that the neg_cues only carries one negation cue in a sentence (single or multi word).

    args:
        sentence: List of string tokens, e.g. ['I', 'do', 'not', 'like', 'apples', '.']
        neg_cues: List which is negation cue token for negation cue and '_' otherwise. e.g. ['_', '_', 'not', '_', '_', '_']
        non_cue_marker: Marker in neg cue list for non-cue tokens, e.g. '_' by default.

    returns: type of the negation cue encoded as a string.
    """
    neg_cue_type = 'NO_CUE'
    sentence_length = len(sentence) 

    # Check if number of non negation cue markers are one less than total tokens, ie there is one negation cue.
    if neg_cues.count(non_cue_marker) == sentence_length - 1:
        neg_cue_index = -1

        # Find the index of negation cue in the sentence.
        for index, token in enumerate(neg_cues):
            if token != non_cue_marker:
                neg_cue_index = index

        neg_cue = neg_cues[neg_cue_index]
        neg_token = sentence[neg_cue_index]
        
        # Check if the token at the negation cue index in sentence is equal to negation cue.
        if neg_token == neg_cue:
            neg_cue_type = 'SINGLE'
        # Check if the token at the negation cue index in sentence starts with negation cue.
        elif neg_token.startswith(neg_cue):
            neg_cue_type = 'PREFIX'
        # Check if the token at the negation cue index in sentence ends with negation cue.
        elif neg_token.endswith(neg_cue):
            neg_cue_type = 'POSTFIX'
        # Check if the token at the negation cue index in sentence contains negation cue in the middle.
        elif neg_cue in neg_token:
            neg_cue_type = 'INFIX'
        # Check if the token at the negation cue index in sentence has no overlap with negation cue.
        # Possibly due to unclean or incorrect data.
        else:
            neg_cue_type = 'UNKNOWN'
    
    # If there are more than one negation cues, then is a multi-word negation cue type.
    elif neg_cues.count(non_cue_marker) < sentence_length - 1:
        neg_cue_type = 'MULTI'
        
    return [neg_cue_type] * sentence_length


# Carmen Feature 2 - If cue and token are in same S/SBAR Clause
def is_cue_and_token_in_diff_S_or_SBAR_clause(sentence, neg_cues, non_cue_marker = '_'):
    """Feature that indicates whether each token of the sentence belongs to different S/SBAR clause compared to negation cue,
        ie, whether the token and cue cross the clause boundary.

    Assumes that the neg_cues only carries one negation cue in a sentence (single or multi word).

    args:
        sentence: List of string tokens, e.g. ['I', 'do', 'not', 'like', 'apples', '.']
        neg_cues: List which is negation cue token for negation cue and '_' otherwise. e.g. ['_', '_', 'not', '_', '_', '_']
        non_cue_marker: Marker in neg cue list for non-cue tokens, e.g. '_' by default.

    returns: List of strings where for each token the value is 'DIFF_S_SBAR' if the token is in different S/SBAR than negation cue,
        else it is 'SAME_S_SBAR'. e.g. ['DIFF_S_SBAR', 'DIFF_S_SBAR', 'DIFF_S_SBAR', 'DIFF_S_SBAR', 'DIFF_S_SBAR', 'DIFF_S_SBAR'].
        If there is no negation cue, the value for a token is 'NO_NEG_CUE'.
    """
    feature = list()
    # Spacy constituency parsing (used from the class notebook).
    spacy_doc = Doc(nlp.vocab, words=sentence)
    processed_doc = nlp(spacy_doc)
    sentence_length = len(sentence)
    # display(Tree.fromstring(tree_str))

    neg_cue_index = -1
    
    # Find the index of negation cue in the sentence.
    for index, token in enumerate(neg_cues):
        if token != non_cue_marker:
            neg_cue_index = index

    # If there is no negation cue, then for each token in sentence, the category is 'NO_NEG_CUE'.
    if neg_cue_index == -1:
        return ['NO_NEG_CUE'] * sentence_length

    token_clause_ids = list()
    # Process each separate tree in the parsed sentence.
    # Some sentences may have more than one tree (e.g. "What! you don't mean to say" has two trees)
    for processed_sentence in list(processed_doc.sents):
        tree_str = processed_sentence._.parse_string
        tree = Tree.fromstring(tree_str)
        # Find the leaf position of each token in the tree and parse the tree upwards to find the closest S/SBAR node.
        for index in range(len(processed_sentence)):
            leaf = tree.leaf_treeposition(index)
            found = None
            # Parse the tree upwards
            for i in range(len(leaf), 0, -1):
                parent = leaf[:i-1]
                node = tree[parent]
                if node.label() in {"S", "SBAR"}:
                    found = parent
                    break
            # Append the id of the closest S/SBAR node of the token to corresponding token index.
            token_clause_ids.append(found)

    # For each token in the sentence, check if the S/SBAR node id is different for token and neg cue.
    for index in range(sentence_length):
        if token_clause_ids[index] != token_clause_ids[neg_cue_index]:
            feature.append('DIFF_S_SBAR')
        else:
            feature.append('SAME_S_SBAR')

    return feature
                            
