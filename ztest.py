#%%
from models_spacy import GeneralExtractor

extractor = GeneralExtractor()
extractor.addPipeGenes()

#%%
doc_str = "The gene ACF was found"
doc = extractor.NLP(doc_str)
print(doc.user_data)

#%% Parsing genenames.json
import json
import pandas as pd


with open("data/genenames.json", "r", encoding = "utf-8") as file:
    genes_raw_json = json.load(file)
    genes_json = genes_raw_json["response"]["docs"]
    
gene_symbols = {gene["symbol"] for gene in genes_json}
gene_alias_dict = {gene["symbol"]: gene["alias_symbol"] for gene in genes_json if "alias_symbol" in gene}
gene_aliases = {alias for group in gene_alias_dict.values() for alias in group} # Flatten

targets = set.union(gene_symbols, gene_aliases) # Merge standardized name and their aliasis into one set to search through


#%% Networkx inheritance
import networkx as nx
a = nx.Graph()
b = nx.DiGraph()
c = nx.MultiGraph()
d = nx.MultiDiGraph()

print([isinstance(i, nx.Graph) for i in [a, b, c, d]])
print([isinstance(i, nx.DiGraph) for i in [a, b, c, d]])
print([isinstance(i, nx.MultiGraph) for i in [a, b, c, d]])


#%% Pandas concatentation 
import pandas as pd
df = pd.DataFrame({
    "col1": ["series1", "series2", "series3", "series4"],
    "col2": ["series1", float("nan"), "series3", "series4"],
    "col3": ["series1", "series2", "series3", "series4"],
})
print(df)
df = df[df["col2"].str.contains(r"[A-Za-z]", regex=True) == True]
print(df)

for ind, row in df.iterrows():
    print(ind)
    new_entry = pd.DataFrame({"col4": [F"data for series{ind}"]})
    new_entry.index = pd.RangeIndex(ind, ind+1, 1)
    df = pd.concat([df, new_entry], axis=1)
print(df)


#%% Networkx node degree
import networkx as nx

a = nx.MultiDiGraph()
a.add_edge("A", "B", width=1)
a.add_edge("B", "A", width=1)
a.add_edge("B", "C", width=1)
a.add_edge("B", "D", width=1)
a.add_edge("C", "D", width=1)



#%% Regex
import re
    
if re.search(R"F_entsF_t\d+$", "test_gpt3F_entsF_t420"): # If it has matched naming conventions (remember to stay model agnostic)
    thresh = re.search(R"F_entsF(_t\d+)$", "test_gpt3F_entsF_t420").group(1)
    root_name = re.sub(R"F_entsF_t\d+$", "", "test_gpt3F_entsF_t420") # Remove annotations to leave root E.g., test_gpt3F as root
    root_name = root_name + thresh # Add threshold annotation to root (can't use lookahead with variable length search)
    print(root_name)

#%% Counters
from collections import Counter
a = Counter()
a.update({1,4,5,2,2})
a.update({1,4,5,2,2})
print(a.items())


#%% Permutations and combinations of lists
from itertools import product, combinations, permutations
a = ["foo", "bar", "baz"]
b = ["1", "2"]
c = []


# print(list(product(a, b)))
# print(list(product(b, a)))
# print(list(product(c, a)))

# for item1, item2 in product(a, b):
#     print(item1, item2)

print(list(permutations(b, 2)))
#%% Sorting things 
import json
with open("gpt3_output_abrvs_rfn.json", "r") as file:
    json_obj: list[tuple[tuple[str, str], int]] = json.load(file)

json_obj.sort(key=lambda x: (x[1], len(x[0][1])), reverse=True)
ABRVS = {abrv[0][1]: abrv[0][0] for abrv in json_obj}
for abrv in ABRVS:
    print(abrv)
#%% Set operations

s1 = set([1,2,3])
s2 = set([3,4,5])

print(set.intersection(s1, s2))
print(s1 | s2)
print(s1)
s1 = s1 - s2
print(s1)

#%% Hashing Abrv class
class Abrv: 
    """
    Class for making abbreviation parsing more readable after being unpacked from JSON
    Corresponds with output of extractAbrvCont method 
    """
    def __init__(self, abrv: list[list[str, str], int]) -> None:
        self.original = abrv # Store original container for easy back and forth conversion
        self.short: str = abrv[0][0]
        self.long: str = abrv[0][1]
        self.count: int = abrv[1]
    def __hash__(self) -> int: # Required for being put in a set, seems like it is overwritten when __eq__ is changed
        return hash((self.short, self.long)) # Use the tuple of short and long for hash, ignore count
    def __eq__(self, __o: object) -> bool: # Used for set comparison
        return self.__hash__() == __o.__hash__()
    def __ne__(self, __o: object) -> bool: # Add reverse just in case
        return self.__hash__() != __o.__hash__() 

a = Abrv([["test2", "long1"], 99])
b = Abrv([["test2", "long2"], 1])
c = Abrv([["test2", "long"], 1])
d = Abrv([["test2", "long"], 11])

obj_list = set([a, b, c, d])
print(b == c)

print(obj_list)
print([(obj.short, obj.long, obj.count) for obj in obj_list])
print(a == d)
#%% Modifying class hash to enable set operations 
class A:
    def __init__(self, data) -> None:
        self.data = data
    def __hash__(self) -> int: # Required for being put in a set, seems like it is overwritten when __eq__ is changed
        return hash(self.data)
    # def __eq__(self, __o: object) -> bool: # Use for actual set comparison
    #     return self.__hash__() == __o.__hash__()
    # def __ne__(self, __o: object) -> bool:
    #     return self.__hash__() != __o.__hash__() # Add reverse just in case


a = A("aaaa")
b = A("lmao")
c = A("aaaa")
d = A("lmao")

obj_list = set([a, b, c, d])
print(obj_list)
print([obj.data for obj in obj_list])

#%% Removing class instances

class A:
    def __init__(self, data) -> None:
        self.data = data
    

def remObj(obj, obj_list: list):
    obj_list.remove(obj)
    return obj_list

a = A("aaaa")
b = A("asdfasdfasdf")
c = A("aaaa")
d = A("lmao")

obj_list = [a, b, c, d]

print([obj.data for obj in obj_list])
remObj(a, obj_list)
print([obj.data for obj in obj_list])

#%% Similarity scoring using sets

print(set("snpsc").issubset(set("snp")))
print(set("snp").issubset(set("snpsc")))

#%% Behaviour of instances in sets 
class A:
    def __init__(self, data) -> None:
        self.data = data
        
a = A("aaaa")
b = A("asdfasdfasdf")
c = A("sdoi2ng")
print(a == b)
print(set([a,b,c]))
obj_list = [a, b, c]
obj_list.sort(key=lambda x: len(x.data), reverse=False)
print([obj.data for obj in obj_list])

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
a = "test"
b = "test"
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
