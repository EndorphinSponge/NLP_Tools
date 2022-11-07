# Pipeline component to extract gene names using genenames.org database, basic structure copied from component_location.py
#%% Imports
import json, pickle, os

import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, Doc
from spacy.language import Language

from internals import LOG

#%%


CACHE_LOCATION = "data/component_gene.cache"

if os.path.exists(CACHE_LOCATION): # Used cached version if possible
    with open(CACHE_LOCATION, "rb") as file:
        matcher_cache = pickle.load(file)
        matcher, gene_alias_dict = matcher_cache # Unpack into variables
    LOG.info(F"Cache found at {CACHE_LOCATION}")
else: # Generate from scatch
    # NLP = spacy.load("en_core_web_trf") # Loads in model as an object which can be used as a function to analyze other strings 
    # NLP = spacy.load("en_core_web_lg") # Loads in model as an object which can be used as a function to analyze other strings 
    NLP = spacy.load("en_core_web_sm")
    
    with open("data/genenames.json", "r", encoding = "utf-8") as file:
        genes_raw_json = json.load(file)
        genes_json = genes_raw_json["response"]["docs"]
        
    gene_symbols = {gene["symbol"] for gene in genes_json}
    gene_alias_dict = {gene["symbol"]: gene["alias_symbol"] for gene in genes_json if "alias_symbol" in gene}
    gene_aliases = {alias for group in gene_alias_dict.values() for alias in group} # Flatten

    genes = set.union(gene_symbols, gene_aliases) # Merge standardized name and their aliasis into one set to search through
    
    # Remove extraneous entries
    genes.discard("")

        
    # JSON Internal modifications:
        # "temporal cortex (rodent)" renamed to "temporal cortex" as it had temporal lobe as its synonym which was common 

    genes_pattern = list(NLP.tokenizer.pipe(genes))
    matcher = PhraseMatcher(NLP.vocab, attr="ORTH") # Matches verbatim text
    matcher.add("GENE", genes_pattern)
    
    matcher_cache = (matcher, gene_alias_dict)
    
    with open(CACHE_LOCATION, "w+b") as file:
        pickle.dump(matcher_cache, file)

@Language.component("extractGenes")
def extractGenes(doc: Doc):
    global matcher
    global gene_alias_dict
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="GENE") 
             for (match_id, start, end) in matches] # Convert matches to spans
    doc.user_data["gene_spans"] = [span.text for span in spans] # Convert to text to serlialize into JSON
    spans_unique = [*{*[span.text for span in spans]}] # Get str and remove duplicates
    for ind, span in enumerate(spans_unique): # Replace any synonyms with their standard terms
        for term_std, term_syns in gene_alias_dict.items():
            if span in term_syns: # Search for matched span in synonyms 
                spans_unique[ind] = term_std # Replace span with standard term
                
    spans_unique = {*spans_unique} # To get rid of any new duplicate terms
    doc.user_data["genes"] = [*spans_unique]
    return doc
#%% Debug
# NLP.add_pipe("extractTargets")
