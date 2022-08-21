# Post-processing component for keyword associations in documents
#%% Imports
import os, json
from typing import Union

import pandas as pd
from pandas import DataFrame

from internals import importData

#%% 

# Keyword list obtained from MeSH standardized terms 
MODALITIES = ["brain depth stimulation",
              "transcranial magnetic stimulation",
              "vagus nerve stimulation"]

DISEASES_BROAD = ["drug resistant epilepsy",
            "focal epilepsy",
            "intractable epilepsy",
            "focal epilepsy",
            "generalized epilepsy"
            ]

DISEASES_NARROW = ["complex partial seizure",
                   "frontal lobe epilepsy",
                   "gelastic seizure",
                   "panayiotopoulos syndrome",
                   "Rasmussen syndrome",
                   "rolandic epilepsy",
                   "simple partial seizure",
                   "temporal lobe epilepsy"] \
                + ["absence",
                   "Alpers disease",
                   "atonic seizure",
                   "benign childhood epilepsy",
                   "clonic seizure",
                   "grand mal epilepsy",
                   "hypsarrhythmia",
                   "infantile spasm",
                   "Lennox Gastaut syndrome",
                   "MERRF syndrome",
                   "myoclonic astatic epilepsy",
                   "myoclonus",
                   "myoclonus epilepsy",
                   "myoclonus seizure",
                   "nodding syndrome",
                   "tonic clonic seizure",
                   "tonic seizure"]
                # Split into focal and generalized epilepsy terms
                
def procKeywordsEpilep(df_path: Union[str, bytes, os.PathLike],
                    col: str = "MH",
                    delim: str = "\n\n"
                    ):
    # Processing of OVID output format only
    root_name = os.path.splitext(df_path)[0]
    df = importData(df_path)
    
    df_keyword_info = DataFrame()
    for ind, row in df.iterrows():
        keyword_str: str = row[col]
        keywords: list[str] = keyword_str.split(delim)
        for i, kw in enumerate(keywords): # Remove modifiers
            if kw.find("/") != -1: # Split based on presence of forwardslash which precedes the modifier
                keywords[i] = kw.split("/")[0] # Only store the main keyword, exclude the modifier
        

        modalities = {kw.strip("*") for kw in keywords if kw.strip("*") in MODALITIES}
        broad = {kw.strip("*") for kw in keywords if kw.strip("*") in DISEASES_BROAD}
        narrow = {kw.strip("*") for kw in keywords if kw.strip("*") in DISEASES_NARROW}
        
        if 0: # Use these generators to get main keywords only
            modalities = {kw.strip("*") for kw in keywords 
                          if kw.startswith("*") and kw.strip("*") in MODALITIES}
            broad = {kw.strip("*") for kw in keywords 
                     if kw.startswith("*") and kw.strip("*") in DISEASES_BROAD}
            narrow = {kw.strip("*") for kw in keywords
                      if kw.startswith("*") and kw.strip("*") in DISEASES_NARROW}

        new_entry = DataFrame({
            "modalities": [json.dumps(list(modalities))],
            "diseases_broad": [json.dumps(list(broad))],
            "diseases_narrow": [json.dumps(list(narrow))],
        })
        new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
        df_keyword_info = pd.concat([df_keyword_info, new_entry])
    
    df_merged = pd.concat([df, df_keyword_info], axis=1)
    df_merged.to_csv(F"{root_name}_kw.csv")



    