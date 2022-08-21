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

import matplotlib.pyplot as plt

def visHeatmapParams(df_path: Union[str, bytes, os.PathLike],
                    col: str,
                    col_sampsize: str = "",
                    xlabel: str = "",
                    ):
    """
    Visualize numerical information contained inside a Doc custom extension using a heatmap
    col: column containing the parameter to be visualized (e.g., frequency)
    extension: the exact identifier of the Doc extension containing the numberical information to be visualized
    xlabel: label for the x-axis of the output
    
    """
    root_name = os.path.splitext(df_path)[0]
    df = importData(df_path)
    points: list[tuple[float, float, float]] = [] # In format of X, Y, Size
    
    for ind, row in df.iterrows():
        params_json: str = row[col]
        sample_size = row[col_sampsize] if col_sampsize else 1
        params: dict[str, list[str]] = json.loads(params_json)
        if params["fl"] != []:
            for fl in params["fl"]:
                points.append((float(fl), 0, sample_size*30))
        if params["rg"] != []:
            for (low, high) in params["rg"]:
                points.append(((float(low)+float(high))/2, 0, sample_size*30)) # Takes the average of the min and max of the range 
        if params["cp"] != []:
            for (op, num) in params["cp"]:
                points.append((float(num), 0, sample_size*30)) # Takes specified number in the comparator expression, will have to refine later
        
        

    plt.figure(figsize=(15, 7.5))
    plot = plt.scatter(x=[p[0] for p in points],
                       y=[p[1] for p in points],
                       s=[p[2] for p in points],
                       alpha = 0.15) # Alpha to set transparency, otherwise points are overlapping
    # plt.xlim(-10,800) # Manually set axes to exclude outliers
    # plt.ylim(-1,1)
    plt.xlabel(xlabel)
    plot.axes.get_yaxis().set_visible(False)
    plt.savefig(f"{root_name}_heatmap_{col}.png", bbox_inches="tight", dpi=300) # Save function has to be called before the show() function