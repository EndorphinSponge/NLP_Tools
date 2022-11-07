# Pipeline component to extract stimulation location names using neuronames database
#%% Imports
import json, pickle, os

import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, Doc
from spacy.language import Language

from internals import LOG

#%%

DEPRAC = False # Guard while keeping linting active
# Reminder that set_extension is a classmethod, will affect all instances of Doc
if DEPRAC:
    Doc.set_extension("targets_spans", default = [], force = True) # Force true to avoid having to restart kernel every debug cycle
    Doc.set_extension("targets_text", default = [], force = True)



CACHE_LOCATION = "data/component_location.cache"

if os.path.exists(CACHE_LOCATION): # Used cached version if possible
    with open(CACHE_LOCATION, "rb") as file:
        matcher_cache = pickle.load(file)
        matcher, targets_syn_dict = matcher_cache # Unpack into variables
    LOG.info(F"Cache found at {CACHE_LOCATION}")
else: # Generate from scatch
    # NLP = spacy.load("en_core_web_trf") # Loads in model as an object which can be used as a function to analyze other strings 
    # NLP = spacy.load("en_core_web_lg") # Loads in model as an object which can be used as a function to analyze other strings 
    NLP = spacy.load("en_core_web_sm")
    
    with open("data/NeuroNames.json", "r", encoding = "utf-8") as file:
        targets_json = json.load(file)
        
    targets_main = set([entry["standardName"] for entry in targets_json])
    targets_syn_groups = [{syn["synonymName"] for syn in entry["synonyms"] 
                        if syn["synonymLanguage"] in ["English", "Latin"]} # Filter to only get English and Latin names
                        for entry in targets_json 
                        if "synonyms" in entry] # Only get groups with synonyms 
    targets_syn_dict: dict[str, set[str]] = {entry["standardName"]: {syn["synonymName"] 
                                                                        for syn in entry["synonyms"] 
                                                                        if syn["synonymLanguage"] in ["English", "Latin"]} 
                                                for entry in targets_json 
                                                if "synonyms" in entry} # Used to map synonyms to their standard names
    targets_syn = set()
    for group in targets_syn_groups:
        targets_syn.update(group) # Unpack synonym groups into targets
    targets = set.union(targets_main, targets_syn) # Merge standardized name and their aliasis into one set to search through
    # Remove extraneous entries
    targets.discard("")
    targets.discard("brain")
    targets.discard("nervous system")
    targets.discard("central nervous system")
    targets.discard("40") # Is a synonym for area PF
        
    # JSON Internal modifications:
        # "temporal cortex (rodent)" renamed to "temporal cortex" as it had temporal lobe as its synonym which was common 

    targets_pattern = list(NLP.tokenizer.pipe(targets))
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER") # Matches based on LOWER attribute to make matches case-insensitive 
    matcher.add("TARGETS", targets_pattern)
    
    matcher_cache = (matcher, targets_syn_dict)
    
    with open(CACHE_LOCATION, "w+b") as file:
        pickle.dump(matcher_cache, file)

@Language.component("extractCnsLocations")
def extractCnsLocations(doc: Doc):
    global matcher
    global targets_syn_dict
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="TARGET") 
             for (match_id, start, end) in matches] # Convert matches to spans
    doc.user_data["cns_locs_spans"] = [span.text for span in spans] # Convert to text to serlialize into JSON
    spans_unique = [*{*[span.text for span in spans]}] # Get str and remove duplicates
    for ind, span in enumerate(spans_unique): # Replace any synonyms with their standard terms
        for term_std, term_syns in targets_syn_dict.items():
            if span.lower() in term_syns: # Search for matched span in synonyms 
                spans_unique[ind] = term_std # Replace span with standard term
                
    spans_unique = {*spans_unique} # To get rid of any new duplicate terms
    doc.user_data["cns_locs"] = [*spans_unique]
    #FIXME Will have to concatenate nested spans (e.g., thalamus in anterior thalamus)
    #FIXME Add another doc extension to process synonyms, using array of strings instead of spans which are harder to manipulate 
    return doc
#%% Debug
# NLP.add_pipe("extractTargets")
