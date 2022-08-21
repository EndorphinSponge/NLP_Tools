import os, re, json
from typing import Union
from difflib import SequenceMatcher
from itertools import combinations
from collections import Counter

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


#%% Independent functions
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

    print("Short form conflicts: ========================================")
    for conflict in short_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.short == conflict]
        conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
        print(conf_abrvs)
        
    print("Long form conflicts: ========================================")
    for conflict in long_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.long == conflict]
        conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
        print(conf_abrvs)
    
    print("Long form warnings: ========================================")
    for term, similars in long_warn:
        term_abrv = [abrv for abrv in abrv_set if abrv.long == term] # Extract full abbreviation 
        term_abrv = [(abrv.short, abrv.long, abrv.count) for abrv in term_abrv] # Unpack Abrv object
        print(f">>>> Similar terms for {term_abrv} <<<<")
        for similar in similars:
            similarity = SequenceMatcher(a=term.lower(), b=similar.lower()).ratio()
            conf_abrvs = [abrv for abrv in abrv_set if abrv.long == similar] # Should only return one item if long forms are all unique
            conf_abrvs = [(abrv.short, abrv.long, abrv.count) for abrv in conf_abrvs] # Unpack Abrv object
            print(round(similarity, 3), conf_abrvs)

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
            
    print("Short form major conflicts: ========================================")
    
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
            print(tuple(warnings))
    
    print("Long form major conflict conversions: ========================================")
    
    long_conf: set[str]  = set()
    for abrv in abrv_set.copy(): # Use full Abrv objects for parsing long forms
        if long_forms.count(abrv.long) > 1: # Check if long form occurs more than once
            long_conf.add(abrv.long)

    for conflict in long_conf:
        conf_abrvs = [abrv for abrv in abrv_set if abrv.long == conflict]
        _mergeConf(conf_abrvs, abrv_set, trans_conversions) # Will not be left with any more conflicts after merge, only similar abbreviation long forms
    
    
    print("Long form minor conflict conversions: ========================================")
    for i in range(1): # Can run for multiple iterations 
        print("Iteration no ", i+1)
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
        print("Conversions made: ", conversions_made)

    
    print("Long form major conflicts: ========================================")
    warnings_list: list[tuple] = [tuple(warning) for warning in warnings]
    for warning in warnings_list:
        print(warning)
        

    trans_counter = Counter()
    for dest, key, count in trans_conversions:
        trans_counter[(dest, key)] += count # Add count as weight to this particular translation 
    
    print("Short form translations: ========================================")
    continue_conversion = True
    while continue_conversion: # Keep running until there are no more conflicts
        conversions_made = 0
        a = trans_counter.copy()
        for trans1, trans2 in combinations(trans_counter.items(), 2):
            if set(trans1[0]) == set(trans2[0]): # Check if items within the counter items are the same (will filter for translations that are the inverse of each other)
                items = [trans1, trans2]
                items.sort(key=lambda x: x[1]) # Sort by count, ascending
                print(items)
                del trans_counter[items[0][0]] # Remove the less common translation of the pair
                conversions_made += 1
        print("Conversions made: ", conversions_made)
        if conversions_made == 0:
            continue_conversion = False
            
    trans_final: dict[str, str] = {item[0][1]: item[0][0] for item in trans_counter.items()} # Note that items are stored in (alternate, common) format
    
        
    with open(f"{root_name}_trans.json", "w") as file:
        json.dump(trans_final, file)
    print(f"Exported alternative translations to {root_name}_trans.json")
    
    abrv_json_new = [[[abrv.short, abrv.long], abrv.count] for abrv in abrv_set] # Repack into hashable json obj
    abrv_json_new.sort(key=lambda x: (x[1], len(x[0][1])), reverse=True) # Sort by counts and then by length of long form, will be translated in this priority
    with open(f"{root_name}_rfn.json", "w") as file:
        json.dump(abrv_json_new, file)
    print(f"Exported refined abbreviations to {root_name}_rfn.json")

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
        print(f"New entry {(common_short, uniq_long, total_count)} <==== {[(abrv.count, abrv.short, abrv.long) for abrv in abrv_confs]}")
    else: # Should be no duplicates of long forms at this point 
        assert len(abrv_confs) == 2 # Assume we are only comparing two abrvs (since comparisons with different long forms will only be done with two abrvs at a time)
        for abrv_conf in abrv_confs:
            new_abrv = Abrv([[common_short, abrv_conf.long], abrv_conf.count]) # Replace short form with common short form
            if new_abrv != abrv_conf: # Only need modification if new entry will be different from current one (i.e., the short forms don't match since that's the only part being changed)
                abrv_origin = [abrv for abrv in abrv_set if abrv.long == abrv_conf.long] # Use long form to find original abrv since it may have been modified 
                abrv_origin = abrv_origin[0] # Should only have one 
                abrv_set.remove(abrv_origin)
                abrv_set.add(new_abrv)
                print(f"Modified entry of {(new_abrv.short, new_abrv.long, new_abrv.count)} <=== {(abrv_conf.short, abrv_conf.long, abrv_conf.count)}")
            else:
                print(f"Kept: {(abrv_conf.short, abrv_conf.long, abrv_conf.count)}")
                
        print("------------")


if __name__ == "__main__":
    # checkAbrvs("test_fmt_abrvs.json")
    refineAbrvs("gpt3_output_abrvs.json")
