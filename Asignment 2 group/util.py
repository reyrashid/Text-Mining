##############################################################
#             helper functions for corpus processing         #
#############################################################
from collections import Counter

def extracting_sent_token_before_preprocessing(inputfile):
    """
    Extract sentence blocks and tokens from a raw CoNLL-style file 

    Assumptions consistent with CoNLL conventions:
    - One token per line
    - Tab-separated columns
    - Sentences separated by a blank line

    Notes
    -----
    This function does not modify the input. It reads sentence blocks as raw
    text and extracts tokens based on a fixed column position.

    Parameters
    ----------
    inputfile : str
        Path to the raw input file.

    Returns
    -------
    tuple(list, list)
        sents : list of str
            Raw sentence blocks (each block corresponds to one sentence).
        all_tokens : list
            Flat list of tokens extracted from each token line.
    """
    sents= []
    all_tokens= []
    with open (inputfile, 'r', encoding = 'utf-8') as file:
        # CoNLL-style sentence segmentation: blank line separates sentences
        sents= file.read().strip().split('\n\n')
        for sent in sents:
            lines= sent.split('\n')
            for line in lines:
                # to skip empty lines
                if not line.strip():
                    continue
                # Token extraction based on fixed position. 
                cols = line.split('\t')
                if len (cols) <4:
                    continue
                all_tokens.append (cols[3].strip())
    return  sents, all_tokens


def format_transformation (inputfile, outputfile):
    """
    Transform the original annotation format into a 10-column CoNLL-style file.

    CoNLL-style properties enforced by this function:
    - One token per line
    - Tab-separated columns
    - Blank line separates sentences
    - Exactly 10 columns in the output (task-specific schema)

    Task-specific output schema (10 columns)
    ---------------------------------------
    This code constructs a 10-column representation where:
    - Columns 1–6 are copied from the input (token metadata)
    - Column 7 is set to '_' (placeholder; parsing info excluded)
    - Column 8 encodes the negation cue token (or '_' if none)
    - Columns 9–10 encode scope membership as binary markers:
        'x' = token is in scope
        '_' = token is out of scope / not applicable

    Multiple cues
    -------------
    If a sentence contains multiple negation cues, the sentence is duplicated
    once per cue so each output sentence instance contains exactly one cue.
    Sentence IDs are modified by appending _{i}.

    Parameters
    ----------
    inputfile : str
        Path to the input file.
    outputfile : str
        Path to the output file.

    Returns
    -------
    list
        A list of processed token rows (lists of strings), with empty lists
        inserted to mark sentence boundaries.
    """
    duplicated_output= []
    with open (inputfile, 'r', encoding = 'utf-8') as file:
        sents= file.read().strip().split('\n\n')
        for sent in sents:
            lines= sent.split('\n')
            seg=[]
            for line in lines:
                # Each token line is split into columns (tab-separated)
                seg.append(line.split('\t'))

            # Number of cues inferred from column layout:
            # - first 7 columns are fixed
            # - each cue contributes 3 additional columns
            nr_cues= (len(seg[0])-7)//3

            # Case 1: no negation cue
            if nr_cues ==0:
                for line in seg:
                    # Keep columns 0–5, replace parsing info with '_'
                    data = line[0:6] + ['_']  # parsing info excluded/placeholder

                    # Add task-specific negation columns (cue + 2 scope cols)
                    cue_col= ['_', '_', '_']
                    new_data= data + cue_col
                    duplicated_output.append(new_data)
                duplicated_output.append([])  # sentence boundary

            # Case 2: exactly one negation cue
            elif nr_cues ==1:
                for line in seg:
                    data = line[0:6] + ['_']

                    # Column 8: cue token/form (from input column 7)
                    data.append(line[7])

                    # Columns 9–10: binary scope encoding derived from input
                    for cue_info in line [8: 10]:
                            if cue_info != '_':
                                # Mark tokens in scope with 'x'
                                data.append('x')
                            else:
                                data.append('_')
                    duplicated_output.append(data)
                duplicated_output.append([])

            # Case 3: multiple negation cues → duplicate sentence per cue
            else:
                for i in range (nr_cues):
                    for line in seg:
                        data = line[0:6] + ['_']

                        # Make sentence id unique per cue instance
                        new_sent_id = f'{line[1]}_{i}'
                        data[1] = new_sent_id

                        # Locate cue columns for cue i
                        neg_cue_index = 7+i*3

                        # Column 8: cue token/form
                        data.append(line[neg_cue_index])

                        # Columns 9–10: scope markers for cue i
                        for cue_info in line [neg_cue_index+1: neg_cue_index+3]:
                            if cue_info != '_':
                                data.append('x')
                            else:
                                data.append('_')
                        duplicated_output.append(data)
                    duplicated_output.append([])

    # Write output as CoNLL-style TSV with blank-line sentence boundaries
    with open (outputfile, 'w', encoding = 'utf-8') as file2:
        for processed_data in duplicated_output:
            file2.write('\t'.join(processed_data)+ '\n')
    return duplicated_output


def extracting_info_after_preprocessing (inputfile):
    """
    Extract analysis-oriented information from a preprocessed 10-column
    CoNLL-style negation file.

    Assumptions
    -----------
    - One token per line
    - Sentences separated by blank lines
    - Negation cue is stored in the third-to-last column (data[-3])
    - Scope membership marker is stored in the second-to-last column (data[-2])
      where 'x' marks tokens inside the scope.

    Outputs
    -------
    - Sentence blocks as raw text
    - All tokens (column index 3)
    - Negation cues per sentence (multi-word cues merged)
    - Count of in-scope tokens per sentence
    """
    sents= []
    tokens= []
    cues= []
    tokens_in_scopes_count=[]
    with open (inputfile, 'r', encoding = 'utf-8') as file:
        sents= file.read().strip().split('\n\n')
        for sent in sents:
            cues_per_sent= []  # per-sentence cues (supports multi-word cues)
            token_scope_count_per_sent=0
            lines= sent.split ('\n')
            for line in lines:
                data= line.split ('\t')

                # Token column (index 3) appended to flat token list
                tokens.append(data[3])

                # Count tokens inside scope
                if data[-2] == 'x':
                    token_scope_count_per_sent +=1

                # Collect cue tokens (may be multi-token cue)
                if data[-3] != '_':
                    cues_per_sent.append(data[-3].strip())

            tokens_in_scopes_count.append(token_scope_count_per_sent)

            # Merge multi-token cues into a single string per sentence
            if cues_per_sent:
                cues.append(' '.join(cues_per_sent))

    return sents, tokens, cues, tokens_in_scopes_count


def statistic_after_trans (sents, tokens, cues, tokens_in_scopes_count, set_name= 'training'):
    """
    """
    ###adapted from Assignment 1
#nr of sents and tokens
    print (f'The number of sentences in the {set_name} set after transformation: {len(sents)}')
    print (f'The number of tokens in the {set_name} set after transformation: {len(tokens)}') 
    
    #nr of cues
    print(f'The number of all cue instances in the {set_name} set after transformation: {len(cues)}')
    
    #counting the number of tokens in all cues
    cues_total_tokens = 0
    for cue in cues:
        cues_total_tokens += len(cue.split())
    print(f'The number of word tokens in all negation cues in the {set_name} data set after transformation: {cues_total_tokens}')
    
    #finding the unique cue forms, as well as their distribution
    lower_af_cues= []
    for cue in cues:
        lower_af_cues.append(cue.lower()) #lowercasing the cues to count 'Not' and 'not' as the same, for instance
    print(f'The number of unique cue forms in the {set_name} set after transformation: {len(set(lower_af_cues))}')
    print(f'Cue Frequency distribution: {Counter(lower_af_cues)}')
    
    #tokens in/outside scopes
    print(f'The number of tokens in scope after {set_name} set transformation: {sum(tokens_in_scopes_count)}')
    print(f'The number of tokens outside the scope after {set_name} set transformation: {len(tokens)-sum(tokens_in_scopes_count)}')

def conll_sentences_as_token_lists(inputfile, token_col=3, keep_empty=False):
    """
    Produce a Python list of sentences (each sentence is a list of tokens)
    from a CoNLL-style file.

    Assumes:
    - One token per line
    - Tab-separated columns
    - Sentences separated by a blank line

    Parameters
    ----------
    inputfile : str
        Path to the CoNLL-style file.
    token_col : int, default=3
        Index of the token column (0-based).
    keep_empty : bool, default=False
        If True, keeps empty sentences as [].

    Returns
    -------
    list[list[str]]
        A list of sentences, where each sentence is a list of tokens.
    """
    sentences = []
    current_tokens = []

    with open(inputfile, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Blank line => end of sentence
            if not line.strip():
                if current_tokens or keep_empty:
                    sentences.append(current_tokens)
                current_tokens = []
                continue

            cols = line.split("\t")

            # Skip malformed lines safely
            if len(cols) <= token_col:
                continue

            tok = cols[token_col].strip()

            # Skip placeholder/empty tokens (optional)
            if tok == "":
                continue

            current_tokens.append(tok)

    # Catch final sentence if file does not end with a blank line
    if current_tokens or keep_empty:
        sentences.append(current_tokens)

    return sentences


def conll_sent_list_literal(inputfile, token_col=3):
    """
    Convenience helper: returns a *string* formatted like:

    sent_list = [
        ["Token1", "Token2", ...],
        ["Token1", "Token2", ...],
    ]

    Useful if you want to paste the result directly into code.
    """
    sents = conll_sentences_as_token_lists(inputfile, token_col=token_col)

    lines = ["sent_list = ["]
    for tokens in sents:
        # Use repr() to safely escape quotes/backslashes inside tokens
        token_items = ", ".join(repr(t) for t in tokens)
        lines.append(f"    [{token_items}],")
    lines.append("]")

    return "\n".join(lines)


def extracting_col_and_gold (inputfile):
    """
    Extract data columns and gold labels from the file after transformation 

    Parameters
    ----------
    inputfile : str
        Path to the raw input file.

    Returns
    -------
    tuple(list, list)
        all_tokens : list
            Flat list of tokens extracted from each token line.
        PoS : list
            Flat list of Pos extracted from each token line.
        Cue_labels: list
            Flat list of Cue_labels extracted from each token line.
        gold: list
            Flat list of scope labels extracted from each token line.
    """
    tokens = []
    pos= []
    cue_labels= []
    gold= []
    with open(inputfile, 'r', encoding ='utf8') as infile:
        ###code inspired by dataset processing in ML4NLP, last accssed on 15 Jan
        for line in infile:
            components = line.rstrip('\n').split()
        ###
            if len(components) > 0:
                tokens.append(components[3])
                pos.append(components[5])
                cue_labels.append(components[7])
                gold.append(components[8])
    return tokens, pos, cue_labels, gold
    
def preprocessing_neg_bert(infile):
    """Preprocesses the input CoNLL file for the negation scope detection BERT task.

    It uses the replace method to preprocess and replaces the negation cue with:
         - [CUE_0]: an affixial cue
    - [CUE_1]: a single-word cue
    - [CUE_2]: part of multiword cue


    In addition, it uses the binary encoding scheme for the negation soope labels: 0 if the token is not in scope
    and 1 if the token is in scope.

    args:
        infile: input conll file

    returns:
        sentences: List of sentences where each sentence is a list of tokens
        negation scope labels: List of encoded label list for each sentence
    """
    with open(infile, 'r', encoding = 'utf-8') as file:
        all_sentences = list()
        all_labels = list()
        sents = file.read().strip().split('\n\n')
        for sent in sents:
            lines = sent.split('\n')
            sentence = list()
            label = list()
            segment = list()
            neg_cue_count = 0
            for line in lines:
                parts = line.split('\t')
                token = parts[3]
                neg_cue = parts[7]
                neg_scope = parts[8]
                segment.append((token, neg_cue, neg_scope))
                if neg_cue != '_': # to check if is a multi neg cue
                    neg_cue_count += 1

            for token, neg_cue, neg_scope in segment:
                if neg_cue == '_':
                    sentence.append(token)
                else:
                    if neg_cue_count == 1 and neg_cue == token:  # normal cue
                        sentence.append("[CUE_1]")
                    elif neg_cue_count > 1:
                        sentence.append("[CUE_2]")  # multi word cue
                    else:
                        sentence.append("[CUE_0]")  # affix cue

                if neg_scope == '_':
                    label.append(0)
                else:
                    label.append(1)

            all_sentences.append(sentence)
            all_labels.append(label)

        return all_sentences, all_labels


# Standalone function to predict negation scope given a sentence and negation cue.
def preprocessing_neg_bert_standalone(sentence, cue):
    """Preprocesses the input sentence for the negation scope detection BERT task.

    It uses the replace method to preprocess and replaces the negation cue with:
         CUE_0 - Affix
         CUE_1 - Normal Cue
         CUE_2 - Part of a multiword cue

    args:
        sentence: Single sentence as list of tokens for standalone method
        cue: Negation cue encoded with the cue word and _

    returns:
        sentence: processed sentence
    """
    processed_sentence = list()
    neg_cue_count = 0
    
    for neg in cue:
        if neg != '_':
            neg_cue_count += 1

    for i, token in enumerate(sentence):
        if cue[i] == '_':
            processed_sentence.append(token)
        else:
            if neg_cue_count == 1 and cue[i] == token:  # normal cue
                processed_sentence.append("[CUE_1]")
            elif neg_cue_count > 1:
                processed_sentence.append("[CUE_2]")  # multi word cue
            else:
                processed_sentence.append("[CUE_0]")  # affix cue

    return processed_sentence
