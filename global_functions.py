#%% Imports
import spacy
import pandas as pd
from nltk.corpus import stopwords

#%% Constants
SPACY_MODEL = "en_core_web_trf" # Word model for SpaCy
STOPWORDS = stopwords.words("english")
STOPWORDS += ["patient", "outcome", "mortality", "year", "month", "day", "hour", "predict", "factor", "follow", \
    "favorable", "adult", "difference", "tbi", "score", "auc", "risk", "head", "associate", \
    "significantly", "group", "unfavorable", "outcome", "accuracy", "probability", "median", "mean", \
    "average", "high", "analysis",] # List of other stop words to include 


#%% Functions 
def importData(data, col = "Abstract", filt = None, filt_col = "Abstract", skiprows = 0) -> list:
    """Returns entire processed DF based on imported Excel data filterd using preliminary str filter
    data: Filepath to Excel file containing data
    col: String of column that contains content
    filt: String that will filter the abstracts
    filt_col: String of column to apply filter to
    skiprows: number of rows to skip when processing data
    """
    if (data.endswith(".xls") or data.endswith(".xlsx")):
        raw = pd.read_excel(data, skiprows = skiprows)
    elif (data.endswith(".csv")):
        raw = pd.read_csv(data, skiprows = skiprows)
    else:
        print("Invalid filetype")
        return
    # raw = raw.drop_duplicates(subset="Title") # drop duplicates based on title 
    # raw = raw.dropna(subset = [col]) # Drop if this column is empty
    raw = raw[raw[filt_col].str.contains(r"[A-Za-z]", regex = True) == True] # Only allow non-empty strings through
    raw = raw[raw[col].str.contains(r"[A-Za-z]", regex = True) == True] # Only allow non-empty strings through
    if filt != None:
        raw = raw.loc[raw[filt_col].str.contains(filt)] # Filters abstracts based on a str
    filtered = raw[["Title", col, "Tags"]] # Output columns
    filtered = filtered.reset_index(drop = True) # to re-index dataframe so it becomes iterable again, drop variable to avoid old index being added as a column
    return filtered

def lemmatizeText(texts, pos_tags=["NOUN", "ADJ", "VERB", "ADV"]) -> list:
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    texts_out = []
    count = 1
    for text in texts:
        print(count)
        if type(text) == str:
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in pos_tags and token.lemma_ not in STOPWORDS and token.text not in STOPWORDS:
                    new_text.append(token.lemma_)
            final = " ".join(new_text)
            texts_out.append(final)
        count += 1
    return (texts_out)