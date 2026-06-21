##############################################################
#             helper functions for corpus processing         #
#############################################################

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
