#%% Imports
from typing import Union
import os

import spacy
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords


#%% Constants
SPACY_MODEL = "en_core_web_trf" # Word model for SpaCy
STOPWORDS = stopwords.words("english")
STOPWORDS += ["patient", "outcome", "mortality", "year", "month", "day", "hour", "predict", "factor", "follow", \
    "favorable", "adult", "difference", "tbi", "score", "auc", "risk", "head", "associate", \
    "significantly", "group", "unfavorable", "outcome", "accuracy", "probability", "median", "mean", \
    "average", "high", "analysis",] # List of other stop words to include 


#%% Functions 
def mergeDfSlices(prefix: str, dir: str = os.getcwd()):
    """
    Merges Excel and CSV of a certain root prefix in a given directory into one Excel file in the same directory

    Args:
        prefix (str): Prefix of filenames to be merged, will be used to search 
        dir (str, optional): Relative path to directory containing files to be merged Defaults to os.getcwd().
    """
    root, dirs, files = list(os.walk(dir))[0] # os.walk yields generator that should only get one item, use index 0 to obtain it
    df_merged = pd.DataFrame()
    for file_name in files:
        if file_name.startswith(prefix) and file_name.endswith((".xlsx", ".xls", ".csv")):
            df = importData(os.path.join(dir, file_name), preprocess=False) # Has support for both XLS and CSV
            df_merged = pd.concat([df_merged, df])

    df_merged.to_excel(os.path.join(dir, f"{prefix}_merged.xlsx"))

def importData(file_path: Union[str, bytes, os.PathLike],
               preprocess: bool = True,
               col: str = "Abstract",
               filt: str = "",
               filt_col: str = "Abstract",
               skiprows: int = 0,
               ) -> DataFrame:
    """
    Returns entire processed DF based on imported Excel data filterd using preliminary str filter
    If 
    
    file_path: Filepath to Excel file containing data
    col: String of column that contains content
    filt: String that will filter the abstracts
    filt_col: String of column to apply filter to
    skiprows: number of rows to skip when processing data
    """
    try:
        if (file_path.endswith(".xls") or file_path.endswith(".xlsx")):
            df = pd.read_excel(file_path, skiprows = skiprows)
        elif (file_path.endswith(".csv")):
            df = pd.read_csv(file_path, skiprows = skiprows)
        elif (file_path == ""):
            print("Empty file path, returning empty DataFrame")
            return DataFrame() # Return empty dataframe to maintain type consistency
        else:
            print("Invalid filetype, returning empty DataFrame")
            return DataFrame() # Return empty dataframe to maintain type consistency
        if preprocess:
            # df = df.drop_duplicates(subset="Title") # drop duplicates based on title 
            df = df.dropna(subset = [col]) # Drop if this column is empty
            df = df[df[col].str.contains(r"[A-Za-z]", regex = True) == True] # Only allow non-empty strings through
            if filt:
                df = df[df[filt_col].str.contains(r"[A-Za-z]", regex = True) == True] # Only allow non-empty strings through
                df = df.loc[df[filt_col].str.contains(filt)] # Filters abstracts based on a str
            df = df[["Title", col, "Tags"]] # Output columns
            df: DataFrame = df.reset_index(drop = True) # to re-index dataframe so it becomes iterable again, drop variable to avoid old index being added as a column
        return df
    except:
        print("Error during import, returning empty DataFrame")
        return DataFrame() # Return empty dataframe to maintain type consistency 
        

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