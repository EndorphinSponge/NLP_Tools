#%%
import re
from typing import Iterable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pandas import DataFrame, Series


from internals import importData, LOG

#%%
DF_PATH = "pre_univariate_regression.csv"



class Term:
    def __init__(self, original_str: str) -> None:
        self.original_str = original_str
        
        modifier: list[str] = re.findall(R"\(.*\)", original_str)
        if modifier: # If no match, should be empty str
            self.root = original_str.replace(modifier[0], "").lower().strip() # Remove modifier to get root
            self.modifier_str = modifier[0].strip("()")
            self.modifiers: set[str] = {i.strip() for i in self.modifier_str.split(",")}

        else:
            self.root = original_str.lower()
            self.modifier_str = ""
            self.modifiers: set[str] = set()

        
        mod_time = [i for i in self.modifiers if "post-injury" in i] # Should probably use regex for post-injury
        # Can do more processing with regex to get more resolution on modifier
                
        self.mod_time = mod_time
        self.mod_ineq = ""
        
    
    def __hash__(self) -> int:
        copy_modifiers = list(self.modifiers.copy())
        copy_modifiers.sort() # Sort modifiers so they hash in the same way
        return hash((self.root, tuple(copy_modifiers)))
    
    def __eq__(self, __o: object) -> bool: # Recall that eq and ne need to be defined for comparison in sets/dicts
        return self.__hash__() == __o.__hash__()
    
    def __ne__(self, __o: object) -> bool:
        return self.__hash__() != __o.__hash__()
    
    def __repr__(self) -> str:
        return F"{self.root} - {self.modifier_str}"
    
class StatSig(Enum):
    NO = 0
    YES = 1
    UNKNOWN = -1

@dataclass(frozen=True)
class RelRow:
    factor: str
    outcome: str
    significant: str
    analysis: str
    

#%%
df = importData(DF_PATH)
# %%

factors = set()
outcomes = set()
analysis = set()

for ind, row in df.iterrows():
    contains_include = row["Include?"] == "Yes"
    contains_core = not pd.isna(row["Prognostic factors and outcomes analyses"])
    
    if contains_include and contains_core:
        item_table: str = row["Prognostic factors and outcomes analyses"]
        items_raw = item_table.split("\n")
        invalid_entries = [i for i in items_raw if i and i.count("|") != 3]
        if invalid_entries: # Screen for entries where rows don't contain 3 separators
            LOG.debug(ind)
            LOG.debug(invalid_entries)
            LOG.debug(row["Prognostic factors and outcomes analyses"])
        items_fmt = [[e.strip() for e in i.split("|")] for i in items_raw if i.count("|") == 3] # Get rid of empty strings and split by "|" and strip whitespace
        items = [RelRow(factor=i[0],
                        outcome=i[1],
                        significant=i[2],
                        analysis=i[3]
                        ) for i in items_fmt]
        for item in items:
            factors.add(Term(item.factor))
            outcomes.add(Term(item.outcome))
            analysis.add(Term(item.analysis))


# %%
