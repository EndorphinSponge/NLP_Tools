#%% Imports 
# General
from collections import Counter
from itertools import combinations
from difflib import SequenceMatcher
from typing import Union
import re, os, json

# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Internal imports
from components_tbi import factors_ignore, factors_trans, outcomes_ignore, outcomes_trans
from internal_globals import importData

#%% Constants
# NLP = spacy.load("en_core_sci_scibert") # Requires GPU
# NLP.add_pipe("abbreviation_detector") # Requires AbbreviationDetector to be imported first 
with open("gpt3_output_abrvs_rfn.json", "r") as file:
    abrv_json: list[list[list[str, str], int]] = json.load(file)

with open("gpt3_output_abrvs_trans.json", "r") as file:
    trans_json: dict[str, str] = json.load(file)

ABRVS = {abrv[0][1]: abrv[0][0] for abrv in abrv_json} # Unpack abrvs with LONG AS KEY and short as value (reversed order compared to original tuples)

EXCLUDE = {
    "factor": factors_ignore,
    "outcome": outcomes_ignore,
}

TRANSLATE = {
    "factor": factors_trans | trans_json, # Merge custom translations with automatically generated ones
    "outcome": outcomes_trans | trans_json,
}

#%% Classes & Functions

        
class GraphBuilder:
    """
    Takes GPT-3 output triples and builds graph GPT-3 output 
    
    Takes structured statements and generates a NetworkX graph depending 
    on the method 
    """
    def __init__(self, abrv_cont = None, ):
        self.abrvs = abrv_cont # Is only needed for NLP processing, not needed beyond populating counters
        self.df = DataFrame()
        self.graph = nx.Graph()
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        common_ignore = ["patient", "patient\'", "patients", "rate", "associated", "hour", "day", "month", "year", "level", 
            "favorable", "favourable", "good", "high", "low", "prevalence", "presence", "result", "ratio", "in-hospital",
            "decrease", "bad", "poor", "unfavorable", "unfavourable", "reduced", "use of", "development",
            "clinical trial", "significance", "finding", "score", "analysis", "isolate"
            "early", "adult", "study", "background", "conclusion", "compare", "time"
            "hours", "days", "months", "years", "rates",
            ] # Words common to both factors and outcomes
        common_tbi_ignore = ["tbi", "mtbi", "stbi", "csf", "serum", "blood", "plasma", "mild",
            "moderate", "severe", "concentration", "risk", "traumatic", "finding", "post-injury",
            "injury", "injuries",
            ] # Specific to TBI 
        self.factors_ignore = ["problem","mortality rate"] + common_ignore + common_tbi_ignore
        self.outcomes_ignore = ["age", "improved", "reduced", "trauma", "s100b"] + common_ignore + common_tbi_ignore
        self.factors_trans = {
            "snps": "snp",
            "rotterdam ct score": "rotterdam",
            "rotterdam score": "rotterdam",
            "marshall ct score": "marshall",
            "marshall score": "marshall",

        }
        self.outcomes_trans = {
            "hospital mortality": "in-hospital mortality",
            "clinical outcome": "outcome",
            "death": "mortality",
            "mortality rate": "mortality",
            "survival": "mortality",
            "functional": "functional outcome",
        }



    def populateCounters(self, df_path, col = "Ents"):
        """
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        """
        # Reset counters
        self.resetCounters()
        
        set_mast_factors = set() # Master set of factors, used for finding intersection
        set_mast_outcomes = set() # Master set of outcomes, used for finding intersection 
        
        list_articles: list[list[tuple[set[str], set[str]]]] = [] # Stores output of each row/article as lists of factors and outcomes for each statement
        # Has shape of list(list(tuple(set(factor), set(outcome)))), outer list for article, inner list for statements within articles
        
        
        for index, row in df_path.iterrows():
            print("NLP processing: " + str(index))
            article_statements: list[tuple[set[str], set[str]]] = [] # List of tuples (per statement) of set containing entities from each individual statement
            text = row[col]
            # Return a list of statements for each row/article
            statements = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
            for statement in statements: # Iterate through each statement of GPT3 output
                factor, outcome, *size = list(filter(None, statement.split("|"))) # Filter with none to get rid of empty strings, should only return 3 items corresponding to the 3 columns of output
                """CAN ADD MORE FACTORS HERE IF GPT3 OUTPUT HAS MORE COLUMNS, *size is a placeholder for extra variables"""
                if re.search(r"\w", factor) != None: # Given that this cell is not empty
                    set_factors = nlpString(factor)
                    set_factors = self.postprocessEntitySet(set_factors, "factor")
                    set_mast_factors.update(set_factors) # Update master list
                else:
                    set_factors = set() # Otherwise, assign empty set to maintain same data type
                if re.search(r"\w", outcome) != None:
                    set_outcomes = nlpString(outcome)
                    set_outcomes = self.postprocessEntitySet(set_outcomes, "outcome")
                    set_mast_outcomes.update(set_outcomes) # Update master list
                else:
                    set_outcomes = set() # Otherwise, assign empty set to maintain same data type
                article_statements.append((set_factors, set_outcomes)) # Append tuple of set with entities for this statement 
            list_articles.append(article_statements)

        set_common_ents = set_mast_factors.intersection(set_mast_outcomes) # Get intersection between master lists

        # This set of loops uses the post-processed entities 
        num_article = 0
        for list_statements in list_articles:
            print("Appending to counter: " + str(num_article))
            # Use sets for containers so multiple mentions within same paper are not recounted 
            set_article_factors = set()
            set_article_outcomes = set()
            set_article_relationships = set()            
            for tuple_columns in list_statements:
                set_factors, set_outcomes = tuple_columns 
                """CAN UNPACK MORE VARIABLES HERE IF OUTPUT HAS MORE VARIABLES"""
                # Can add additional resolution parsing within the if statements
                if len(set_factors) != 0: # Given that the set is not empty
                    for ent in set_factors.copy(): # Copy set and modify original
                        if ent in set_common_ents: # Check against common list
                            set_factors.remove(ent)
                            set_factors.add(ent + " (factor)") # Modify with suffix if there are conflicts
                    set_article_factors.update(set_factors)
                if len(set_outcomes) != 0:
                    for ent in set_outcomes.copy(): # Copy set and modify original
                        if ent in set_common_ents: # Check against common list
                            set_outcomes.remove(ent)
                            set_outcomes.add(ent + " (outcome)") # Modify with suffix if there are conflicts
                    set_article_outcomes.update(set_outcomes)
                if len(set_factors) != 0 and len(set_outcomes) != 0: # Given that the statement has at least one of each node
                    for str_factor in set_factors: # Add connection between a factor and all outcomes
                        for str_outcome in set_outcomes:
                            if str_factor != str_outcome: # So that you don't get self connections
                                set_article_relationships.add((str_factor, str_outcome))
                                set_article_relationships.add((str_outcome, str_factor)) # Add bidirectional relationship
                                # Remember to enumerate here to avoid repeating connections
            # Update global counter with information from sets; each entity will only be entered once since all entities are in sets 
            for factor in set_article_factors:
                self.factor_counter[factor] += 1
            for outcome in set_article_outcomes:
                self.outcome_counter[outcome] += 1
            for edge in set_article_relationships:
                self.edge_counter[edge] += 1
            num_article += 1
        return None
    
    def buildGraph(self, thresh = 1, ):
        """
        Builds Networkx graph with the populated counters and a threshold for node count
        ---
        thresh: lower threshold for number of counts needed for each node (exclusive)
        """
        # Reminder that nx nodes can have abitrary attributes that don't contribute to rendering, need to manually adjust visual parameters with drawing methods
        # nx.Graph is just a way to store data, data can be stored in node attributes         
        self.graph = nx.Graph() # Reset graph
        for entity in self.factor_counter:
            count = self.factor_counter[entity]
            if count > thresh: # Only add if there is more than 1 mention
                self.graph.add_node(entity, color = "#8338ec", size = count) # Color is in #RRGGBBAA format (A is transparency)
        for entity in self.outcome_counter:
            count = self.outcome_counter[entity]
            if count > thresh:
                self.graph.add_node(entity, color = "#f72585", size = count) # Color is in #RRGGBBAA format (A is transparency)
        for (node1, node2) in self.edge_counter:
            count = self.edge_counter[(node1, node2)]
            if (self.factor_counter[node1] > thresh or self.outcome_counter[node1] > thresh) and\
                (self.factor_counter[node2] > thresh or self.outcome_counter[node2] > thresh): # Need to each node in all sets of counters
                self.graph.add_edge(node1, node2, width = count) # "width" attribute affects pyvis rendering, pyvis doesn't support edge opacity
                print(node1, node2)
        return None

    def exportGraph(self, path):
        """
        Exports currently stored graph to an XML file with the specified path 
        """
        # Export graph , https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.write_graphml.html#networkx.readwrite.graphml.write_graphml
        # Graphml documentation: https://networkx.org/documentation/stable/reference/readwrite/graphml.html
        nx.write_graphml(self.graph, path)
        return None
        
    def resetCounters(self):
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        return None

    def resetGraph(self,):
        self.graph = nx.Graph()

def compareStrings(str1, str2):
    return SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio()

def nlpString(string):
    """
    Takes a string and returns a set of entities, also combines
    any abbreviation definitions into the short form (deletes long form to
    avoid duplication)
    """
    doc = NLP(string.strip()) # Need to strip whitespace, otherwise recognition is suboptimal esp for shorter queries
    if len(doc) > 1: # Only process if there is more than one token
        ents = set()
        for ent in list(doc.ents):
            if len(ent.text.lower().strip()) <= 5:
                ents.add(ent.text.lower().strip())
            else: # Only add lemma if word is bigger than 5 characters (lemmas on abbreviations tend to be buggy)
                ents.add(ent.lemma_.lower().strip())
        abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
        for abrv, full in abrvs:
            for ent in ents.copy(): # Iterate over a copy of the set while changing the original
                if compareStrings(full, ent) > 0.9: # Find ent matching with full form of abbreviation
                    ents.remove(ent) # Remove full form
                    ents.add(abrv) # Add abbreviated form
        return ents
    else: # Otherwise there will be only one token, return its lemma 
        if len(doc[0].text.lower().strip()) <= 5:
            return {doc[0].text.lower().strip(),}
        else: # Only add lemma if word is bigger than 5 characters (lemmas on abbreviations tend to be buggy)
            return {doc[0].lemma_.strip().lower(),} 

class EntProcessor:
    def __init__(self,
                 abrv_cont: dict[str, str] = ABRVS,
                 exclude_cont: dict[str, set[str]] = EXCLUDE,
                 trans_cont: dict[str, dict[str, str]] = TRANSLATE,
                 ) -> None:
        self.abbreviations = abrv_cont
        self.exclusions = exclude_cont
        self.translations = trans_cont
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
        if root_name.endswith("ents"): # Replace raw with fmt if it exists at end of filename
            new_name = re.sub(R"ents$", "fmtents", root_name)
        else: # Otherwise append fmt to end
            new_name = root_name + "fmtents"
        df = importData(df_path, screen_text=[col]) # Screen col for text
        
        df_out = DataFrame()        
        for ind, row in df.iterrows():
            print("Processing ents for: ", ind)
            ents_json: list[dict[str, list[str]]] = json.loads(row[col]) # List of ent dicts containing list of ent (value) for each ent type (key)
            
            list_ents = self._procEnts(ents_json)
            
            new_row = DataFrame({col_out: [list_ents]})
            new_row.index = pd.RangeIndex(start=ind, stop=ind+1, step=1) # Assign corresponding index to new row
            df_out = pd.concat([df_out, new_row])
        df_merged = pd.concat([df, df_out], axis=1)
            
        
        print("Separating overlapping ents")
        for ind, row in df_merged.iterrows(): # Iterate through merged df to resolve any overlaps between node types
            list_ents: list[dict[str, list[str]]] = row[col_out] # Will not be serialized into json yet
            
            list_ents = self._sepConfEnts(list_ents)
            
            row[col_out] = json.dumps(list_ents)
        df_merged.to_excel(f"{new_name}.xlsx")
    
    def _procEnts(self,
                  list_ents: list[dict[str, list[str]]],
                  igno_type: list[str] = []
                  ) -> list[dict[str, list[str]]]:
        
        def abrvEnts(self: EntProcessor, ent: str, thresh: int = 0.95) -> str:
            for long_form in self.abbreviations:
                if SequenceMatcher(a=ent.lower(), b=long_form.lower()).ratio() > thresh:
                    short_form = self.abbreviations[long_form]
                    self.abrv_log[(short_form, ent)] += 1 # Log abbreviation mapping
                    return short_form # Return corresponding short form in abbreviation
            return ent # If no fuzzy matches, return input unchanged 
        
        def transEnts(self: EntProcessor, ent: str, ent_type: str) -> str:
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
                
                ents = {abrvEnts(self, ent, thresh=0.9) for ent in ents}
                ents = {ent for ent in ents if ent not in self.exclusions[ent_type]}
                ents = {transEnts(self, ent, ent_type) for ent in ents}
                
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
        print("Abbreviations: ", self.abrv_log)
        print("Translations: ", self.trans_log)
        print("Conflicts: ", self.conf_ent_log)

def mapAbrv(string, abrv_container, threshold = 0.9):
    """
    Checks if there is an abbreviation in a string given an abbreviation
    container and maps it to the abbreviation if present
    Returns original string if no matches
    """
    for abrv, full in abrv_container:
        if compareStrings(full, string) > threshold:
            print("Mapped " + full + " to " + abrv)
            return abrv
    return string

def transEnts(string, trans_dict):
    # Different logic from map abrv which uses fuzzy matching 
    if string in trans_dict.keys(): # If the string matches a translation key
        print("Translated " + string + " to " + trans_dict[string])
        return trans_dict[string]
    else:
        return string
    
def extractAbrvs(string: str) -> list:
    """
    Takes a string and returns a set of all the abbreviations
    """
    doc = NLP(string.strip()) # Need to strip whitespace, otherwise recognition is suboptimal esp for shorter queries
    abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
    return abrvs 

def extractAbrvCont(df, col_input = "Extracted_Text"):
    """
    Separated from main counter process because some tasks may want to use
    NLP to look ahead and pre-process all documents in a corpora for 
    metadata (e.g., corpora-wide abbreviations) that can be used to help
    process subsets of the corpora 
    """
    abrv_container = set()
    for index, row in df.iterrows():
        print(index)
        text = row[col_input]
        items = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
        for item in items: # Collect abbreviations 
            abrv_container.update(extractAbrvs(item))
    return abrv_container



if False:
    #%%
    builder = GraphBuilder()
    builder.importGraph("tbi_topic0_graph.xml")
    builder.renderGraphNX(cmap = True)
    #%%
    builder = GraphBuilder()
    builder.importGraph("tbi_topic10_t1_graph.xml")
    builder.renderGraphNX(cmap = True)

    #%%
    builder = GraphBuilder()
    builder.importGraph("tbi_ymcombined_t5_graph.xml")
    builder.renderGraphNX(cmap = True)
    #%% Build abbreviations
    df_origin = pd.read_excel("gpt3_output_formatted_annotated25.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
    abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
    #%% Build graph 
    builder = GraphBuilder(abrvs)
    builder.populateCounters(df_origin)
    builder.buildGraph(thresh = 1)
    builder.exportGraph("tempgraph.xml")
    builder.renderGraphNX(cmap = True)

    #%%
    g1 = nx.read_graphml("tbi_ymcombined_t15_graph.xml")
    print(g1.nodes(data = "size"))
    sorted_sizes = sorted(list(g1.nodes(data = "size")), key = lambda x: x[1])
    sorted_edges = sorted(list(g1.edges(data = "width")), key = lambda x: x[2])
    print(sorted_sizes)
    print(sorted_edges)

    #%% Preview distributions contained within an array
    data = [] # Container for data
    bins = np.arange(min(data), max(data), 1) # fixed bin size
    plt.xlim([min(data), max(data)])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('Test')
    plt.xlabel('variable X')
    plt.ylabel('count')

    plt.show()
# %%
