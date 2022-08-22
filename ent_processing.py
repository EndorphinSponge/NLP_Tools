import os, re, json
from typing import Union
from difflib import SequenceMatcher
from itertools import combinations
from collections import Counter

import pandas as pd
from pandas import DataFrame

from internals import importData, LOG

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

class EntProcessorCore:
    # Core component of EntProcessor, is inherited by different core classes of different components to return their own exclusions and translations
    def __init__(self,
                 abrv_path: Union[str, bytes, os.PathLike],
                 common_trans_path: Union[str, bytes, os.PathLike],
                 ):
        with open(abrv_path, "r") as file:
            abrv_json: list[list[list[str, str], int]] = json.load(file)
        self.abrv_json = abrv_json

        with open(common_trans_path, "r") as file:
            trans_json: dict[str, str] = json.load(file)
        self.trans_json = trans_json
        
        self.abbreviations: dict[str, str] = {abrv[0][1]: abrv[0][0] for abrv in abrv_json} # Unpack abrvs with LONG AS KEY and short as value (reversed order compared to original tuples)
        self.exclusions: dict[str, set[str]] = dict()
        self.translations: dict[str, dict[str, str]] = dict()

class EntProcessor:
    def __init__(self,
                 ent_processor_core: EntProcessorCore,
                 ) -> None:
        self.abbreviations: dict[str, str] = ent_processor_core.abbreviations
        self.exclusions: dict[str, set[str]] = ent_processor_core.exclusions
        self.translations: dict[str, dict[str, str]] = ent_processor_core.translations
        self.proc_ents: dict[str, set[str]] = dict() # Tracks processed ents of each type, initialize a set for each type of ent
        self.abrv_log = Counter()
        self.trans_log = Counter()
        self.conf_ent_log = Counter()
    # Ignore and translation containers will be dictionaries with labels of types of nodes that it applies to 
    def procDfEnts(self,
                   df_path: Union[str, bytes, os.PathLike],
                   col: str = "Ents",
                   col_out: str = "Processed_ents"):
        root_name = os.path.splitext(df_path)[0]
        if root_name.endswith("entsR"): # Replace raw with fmt if it exists at end of filename
            new_name = re.sub(R"entsR$", "entsF", root_name)
        else: # Otherwise append fmt to end
            new_name = root_name + "_entsF"
        df = importData(df_path, screen_text=[col]) # Screen col for text
        
        df_out = DataFrame()        
        for ind, row in df.iterrows():
            LOG.info(F"Processing ents for: {ind}")
            ents_json: list[dict[str, list[str]]] = json.loads(row[col]) # List of ent dicts containing list of ent (value) for each ent type (key)
            
            list_ents = self._procEnts(ents_json)
            
            new_row = DataFrame({col_out: [list_ents]})
            new_row.index = pd.RangeIndex(start=ind, stop=ind+1, step=1) # Assign corresponding index to new row
            df_out = pd.concat([df_out, new_row])
        df_merged = pd.concat([df, df_out], axis=1)
            
        
        LOG.info("Separating overlapping ents")
        for ind, row in df_merged.iterrows(): # Iterate through merged df to resolve any overlaps between node types
            list_ents: list[dict[str, list[str]]] = row[col_out] # Will not be serialized into json yet
            
            list_ents = self._sepConfEnts(list_ents)
            df_merged.loc[ind, col_out] = json.dumps(list_ents) # Need loc function and original df to modify in place, can't just give index
        df_merged.to_excel(f"{new_name}.xlsx")
        LOG.info(f"Exported processed ents to {new_name}.xlsx")
    
    def _procEnts(self,
                  list_ents: list[dict[str, list[str]]],
                  igno_type: list[str] = []
                  ) -> list[dict[str, list[str]]]:
        
        def _abrvEnts(self: EntProcessor, ent: str, thresh: int = 0.95) -> str:
            for long_form in self.abbreviations: # Abbreviations ordered by most common and then by longest long form
                if SequenceMatcher(a=ent.lower(), b=long_form.lower()).ratio() > thresh:
                    short_form = self.abbreviations[long_form]
                    self.abrv_log[(short_form, ent)] += 1 # Log abbreviation mapping
                    return short_form # Return corresponding short form in abbreviation
                    # Other short form is either merged by set or translated and merged in transEnts
            return ent # If no fuzzy matches, return input unchanged 
        
        def _transEnts(self: EntProcessor, ent: str, ent_type: str) -> str:
            # Different logic from map abrv which uses fuzzy matching 
            type_specific_trans = self.translations[ent_type]
            if ent in type_specific_trans:
                self.trans_log[(type_specific_trans[ent], ent)] += 1 # Log translation 
                return type_specific_trans[ent]
            else: 
                return ent
            
        for ent_dict in list_ents:
            for ent_type in [t for t in ent_dict if t not in igno_type]: 
                ents = set(ent_dict[ent_type]) # Convert list from JSON to set
                
                ents = {_abrvEnts(self, ent) for ent in ents}
                ents = {ent for ent in ents if ent not in self.exclusions[ent_type]}
                ents = {_transEnts(self, ent, ent_type) for ent in ents}
                
                ent_dict[ent_type] = list(ents) # Update container with new contents of ents, changes will propagate to entry ents list
                if ent_type not in self.proc_ents: 
                    self.proc_ents[ent_type] = set() # Initialize ent type in tracker if it doesn't exist                
                self.proc_ents[ent_type].update(ents) # Track change by adding ents to its corresponding type in tracker
                
        return list_ents
    
    def _sepConfEnts(self,
                     list_ents: list[dict[str, list[str]]],
                     igno_type: list[str] = []
                     ) -> list[dict[str, list[str]]]:
        # Should resolve overlap between any number of groups 
        # Needs to be run after procEnts has been run on all ents
        common_ents: set[str] = set()
        for ent_type1, ent_type2 in combinations(self.proc_ents, 2):
            if ent_type1 not in igno_type and ent_type2 not in igno_type: # If neither types are being ignored
                overlap = set.intersection(self.proc_ents[ent_type1], self.proc_ents[ent_type2])
                common_ents.update(overlap) # Add overlap between group to all common ents
        for ent_dict in list_ents:
            for ent_type in [t for t in ent_dict if t not in igno_type]:
                for ent in ent_dict[ent_type].copy():
                    if ent in common_ents:
                        ind = ent_dict[ent_type].index(ent) # Get index of ent within its list
                        ent_dict[ent_type][ind] = ent + f" ({ent_type})" # Replace value with annotated version
                        self.conf_ent_log[ent] += 1 # Log conflict resolution
        return list_ents

    def printLogs(self):
        LOG.info(F"Abbreviations: {self.abrv_log}")
        LOG.info(F"Translations: {self.trans_log}")
        LOG.info(F"Conflicts: {self.conf_ent_log}")

#%% Independent functions for abbreviations
def checkAbrvs(json_path: Union[str, bytes, os.PathLike], diff_thresh = 0.9):
    with open(json_path, "r") as file:
        abrv_json: list[list[list[str, str], int]] = json.load(file) # List of tuples of the counted object (tuple[str, str]) and its count
        abrv_set = {Abrv(abrv) for abrv in abrv_json} # Convert to Abrv object for better readability 
    short_forms = [abrv.short for abrv in abrv_set]
    long_forms = [abrv.long for abrv in abrv_set]
    
    short_conf: set[str] = set()
    for short in short_forms:
        if short_forms.count(short) > 1: # Check if entry occurs more than once
            short_conf.add(short)
            
    long_conf: set[str]  = set()
    long_warn: list[tuple[str, set[str]]] = []
    for full in long_forms:
        if long_forms.count(full) > 1: # Check if entry occurs more than once
            long_conf.add(full)
        else:
            similar_terms = set()
            for term in [long for long in set(long_forms) if long != full]: # Compare against each term excluding itself
                similarity = SequenceMatcher(a=full.lower(), b=term.lower()).ratio()
                if similarity > diff_thresh:
                    similar_terms.add(term)
            if similar_terms:
                long_warn.append((full, similar_terms))

    LOG.info("Short form conflicts: ========================================")
    for conflict in short_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.short == conflict]
        conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
        LOG.info(conf_abrvs)
        
    LOG.info("Long form conflicts: ========================================")
    for conflict in long_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.long == conflict]
        conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
        LOG.info(conf_abrvs)
    
    LOG.info("Long form warnings: ========================================")
    for term, similars in long_warn:
        term_abrv = [abrv for abrv in abrv_set if abrv.long == term] # Extract full abbreviation 
        term_abrv = [(abrv.short, abrv.long, abrv.count) for abrv in term_abrv] # Unpack Abrv object
        LOG.info(f">>>> Similar terms for {term_abrv} <<<<")
        for similar in similars:
            similarity = SequenceMatcher(a=term.lower(), b=similar.lower()).ratio()
            conf_abrvs = [abrv for abrv in abrv_set if abrv.long == similar] # Should only return one item if long forms are all unique
            conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
            LOG.info(F"{round(similarity, 3)} | {conf_abrvs}")

def refineAbrvs(json_path: Union[str, bytes, os.PathLike],
              l_thresh: float = 0.9,
              ):
    root_name = os.path.splitext(json_path)[0]
    with open(json_path, "r") as file:
        abrv_json: list[list[list[str, str], int]] = json.load(file) # List of tuples of the counted object (tuple[str, str]) and its count
        abrv_set = {Abrv(abrv) for abrv in abrv_json} # Convert to Abrv object for better readability 
    short_forms = [abrv.short for abrv in abrv_set]
    long_forms = [abrv.long for abrv in abrv_set]
    trans_conversions: list[tuple[str, str, int]] = []
            
    LOG.info("Short form major conflicts: ========================================")
    
    short_conf: set[str] = set()
    for short in short_forms:
        if short_forms.count(short) > 1: # Check if entry occurs more than once
            short_conf.add(short) # Add to set so that repeats of same term are ignored 
    
    for conflict in short_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.short == conflict] # Collect corresponding abrvs
        warnings = set()
        for abrv1, abrv2 in combinations(conf_abrvs, 2): # Find all combinations of comparing abrvs using class
            l_similarity = SequenceMatcher(a=abrv1.long.lower(), b=abrv2.long.lower()).ratio() # Compare long forms
            if l_similarity < l_thresh: # If low similarity between long forms, make warning, otherwise ignore (no else statement)
                abrv1_data = (abrv1.short, abrv1.long, abrv1.count) # Need to unpack data from classes for sets (since different instances are not treated equal even if same data)
                abrv2_data = (abrv2.short, abrv2.long, abrv2.count) 
                warnings.add(abrv1_data) # Don't need to add both nodes together as a single item since it generates too many combinations for largers conflict containers
                warnings.add(abrv2_data)
        if warnings:
            LOG.info(tuple(warnings))
    
    LOG.info("Long form major conflict conversions: ========================================")
    
    long_conf: set[str]  = set()
    for abrv in abrv_set.copy(): # Use full Abrv objects for parsing long forms
        if long_forms.count(abrv.long) > 1: # Check if long form occurs more than once
            long_conf.add(abrv.long)

    for conflict in long_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.long == conflict]
        _mergeConf(conf_abrvs, abrv_set, trans_conversions) # Will not be left with any more conflicts after merge, only similar abbreviation long forms
    
    
    LOG.info("Long form minor conflict conversions: ========================================")
    for i in range(1): # Can run for multiple iterations 
        LOG.info(F"Iteration no {i+1}")
        warnings: set[frozenset] = set()
        conversions_made = 0
        for abrv1, abrv2 in combinations(abrv_set, 2): # Find all combinations of comparing abrvs using class
            l_similarity = SequenceMatcher(a=abrv1.long.lower(), b=abrv2.long.lower()).ratio()
            if l_similarity > l_thresh: # If long forms are similar
                if abrv1.short == abrv2.short:
                    pass # Ignore, since we would want similar long forms to converge on the same short form
                elif set(abrv1.short).issubset(set(abrv2.short)) or \
                    set(abrv2.short).issubset(set(abrv1.short)): # Use sets to check if one short form is a subset of another (rearrangements also qualify)
                    _mergeConf([abrv1, abrv2], abrv_set, trans_conversions)
                    conversions_made += 1
                else: # Assume they are dissimilar and make a warning 
                    abrv1_data = (abrv1.short, abrv1.long, abrv1.count) # Need to unpack data from classes for sets (since different instances are not treated equal even if same data)
                    abrv2_data = (abrv2.short, abrv2.long, abrv2.count) 
                    warnings.add(frozenset([abrv1_data, abrv2_data])) # Add conflict as frozenset so that order doesn't matter, needs to be frozen to be hashable by outer set
        LOG.info(f"Conversions made: {conversions_made}")

    
    LOG.info("Long form major conflicts: ========================================")
    warnings_list: list[tuple] = [tuple(warning) for warning in warnings]
    for warning in warnings_list:
        LOG.info(warning)
        

    trans_counter = Counter()
    for dest, key, count in trans_conversions:
        trans_counter[(dest, key)] += count # Add count as weight to this particular translation 
    
    LOG.info("Short form translations: ========================================")
    continue_conversion = True
    while continue_conversion: # Keep running until there are no more conflicts
        conversions_made = 0
        a = trans_counter.copy()
        for trans1, trans2 in combinations(trans_counter.items(), 2):
            if set(trans1[0]) == set(trans2[0]): # Check if items within the counter items are the same (will filter for translations that are the inverse of each other)
                items = [trans1, trans2]
                items.sort(key=lambda x: x[1]) # Sort by count, ascending
                LOG.info(items)
                del trans_counter[items[0][0]] # Remove the less common translation of the pair
                conversions_made += 1
        LOG.info(f"Conversions made: {conversions_made}")
        if conversions_made == 0:
            continue_conversion = False
            
    trans_final: dict[str, str] = {item[0][1]: item[0][0] for item in trans_counter.items()} # Note that items are stored in (alternate, common) format
    
        
    with open(f"{root_name}_trans.json", "w") as file:
        json.dump(trans_final, file)
    LOG.info(f"Exported alternative translations to {root_name}_trans.json")
    
    abrv_json_new = [[[abrv.short, abrv.long], abrv.count] for abrv in abrv_set] # Repack into hashable json obj
    abrv_json_new.sort(key=lambda x: (x[1], len(x[0][1])), reverse=True) # Sort by counts and then by length of long form, will be translated in this priority
    with open(f"{root_name}_rfn.json", "w") as file:
        json.dump(abrv_json_new, file)
    LOG.info(f"Exported refined abbreviations to {root_name}_rfn.json")

def _mergeConf(abrv_confs: list[Abrv], abrv_set: set[Abrv], trans_conversions: list[tuple[str, str, int]]):
    # Goal is to merge long forms into a single one short
    # Will only change short form destination of abrvs, doesn't modify long form 
    # Modifies trans_conversions by adding short form translations to it 
    abrv_confs.sort(key=lambda x: x.count, reverse=True) # Sort by count, descending
    shorts = [abrv.short for abrv in abrv_confs]
    shorts.sort(key=lambda x: len(x)) # Sort short forms by ascending length
    longs = [abrv.long for abrv in abrv_confs]
    if len(set(shorts)) > 1: # If more than one unique short form, pick the most common
        common_short = abrv_confs[0].short # Take short form of the most common abrv
    else: # Assume abrvs are similarly common or there's only one unique short form
        common_short = shorts[0] # Take the shortest short form (better to be more general than wrongly specific)
        
    alt_shorts = set([alt_short for alt_short in shorts if alt_short != common_short]) # Get list of unique alternative short forms
    alt_list: list[tuple[str, str, int]] = [] 
    for alt_short in alt_shorts:
        total_converted = sum([abrv.count for abrv in abrv_confs if abrv.short == alt_short]) # Get count of entries with same short form, is an estimate of how many abbrevations will be converted by this action
        alt_list.append((common_short, alt_short, total_converted))
    trans_conversions += alt_list
    
    
    if len(set(longs)) == 1: # If there is only one unique long form in abrv_confs
        for abrv_conf in abrv_confs:
            abrv_set.remove(abrv_conf) # Remove conflicts from master abrv list
        uniq_long = longs[0] # First long form should be representative of rest of long forms
        total_count = sum([abrv.count for abrv in abrv_confs if abrv.long == uniq_long]) # Get count of entries with same long form
        abrv_set.add(Abrv([[common_short, uniq_long], total_count])) # Add new merged entry, Abrv.short and Abrv.long are used for comparison via hash
        LOG.info(f"New entry {(common_short, uniq_long, total_count)} <==== {[(abrv.count, abrv.short, abrv.long) for abrv in abrv_confs]}")
    else: # Should be no duplicates of long forms at this point 
        assert len(abrv_confs) == 2 # Assume we are only comparing two abrvs (since comparisons with different long forms will only be done with two abrvs at a time)
        for abrv_conf in abrv_confs:
            new_abrv = Abrv([[common_short, abrv_conf.long], abrv_conf.count]) # Replace short form with common short form
            if new_abrv != abrv_conf: # Only need modification if new entry will be different from current one (i.e., the short forms don't match since that's the only part being changed)
                abrv_origin = [abrv for abrv in abrv_set if abrv.long == abrv_conf.long] # Use long form to find original abrv since it may have been modified 
                abrv_origin = abrv_origin[0] # Should only have one 
                abrv_set.remove(abrv_origin)
                abrv_set.add(new_abrv)
                LOG.info(f"Modified entry of {(new_abrv.short, new_abrv.long, new_abrv.count)} <=== {(abrv_conf.short, abrv_conf.long, abrv_conf.count)}")
            else:
                LOG.info(f"Kept: {(abrv_conf.short, abrv_conf.long, abrv_conf.count)}")
                
        LOG.info("------------")


if __name__ == "__main__":
    # checkAbrvs("test_fmt_abrvs.json")
    refineAbrvs("gpt3_output_abrvs.json")
