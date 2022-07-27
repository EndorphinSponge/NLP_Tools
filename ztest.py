#%% Similarity scoring using sets

print(set("snpsc").issubset(set("snp")))
print(set("snp").issubset(set("snpsc")))

#%% Behaviour of instances in sets 
class A:
    def __init__(self, data) -> None:
        self.data = data
        
a = A("test")
b = A("test")
c = A("test")

print(a == b)
print(set([a,b,c]))

#%% Amount of overlap needed for converting between abbreviations 
from difflib import SequenceMatcher

a = "gos"
b = "gose"
a = "glasgow outcome score extended"
b = "glasgow outcome score"
a = "gos"
b = "gcs"
a = "mtbi"
b = "stbi"
SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


#%% Check spacy pipeline components 
import spacy
from scispacy.abbreviation import AbbreviationDetector # Added via NLP.add_pipe("abbreviation_detector")

NLP = spacy.load("en_core_sci_scibert")
NLP.add_pipe("abbreviation_detector") # Requires AbbreviationDetector to be imported first 
print(NLP.components) # Components in general will fetch all components, including disabled
print(NLP.component_names)
print(NLP.pipeline) # Pipeline components refer to active components
print(NLP.pipe_names)
print(NLP.pipe_factories)


#%% Converting text to JSON strings within each row
import json

import pandas as pd
from pandas import DataFrame


df: DataFrame = pd.read_excel("testdata.xlsx", engine="openpyxl")
df_extracted = DataFrame()
# print(df["Extracted_Text"])
for index, row in df.iterrows():
    print(index)
    row_text: str = row["Extracted_Text"]
    processed_text = json.dumps(row_text.split("|"))
    print(processed_text)
    new_row = DataFrame({"testcol": [processed_text]})
    df_extracted = pd.concat([df_extracted, new_row])
df_extracted = df_extracted.reset_index(drop = True) # Drop variable avoids adding old index as a column
df = pd.concat([df, df_extracted], axis = 1) # Concat on columns instead of rows
df.to_excel("testdataOUT.xlsx")



# %%
