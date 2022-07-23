#%% Imports 
# General
from collections import Counter
import re
import difflib
import os

# Data science
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt

# NLP
import spacy 
from scispacy.abbreviation import AbbreviationDetector # Needed for adding "abbreviation_detector"

from spacy import displacy
from scispacy.linking import EntityLinker

#%% Constants
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector") # Requires AbbreviationDetector to be imported first 

#%% Classes & Functions

        
class GraphBuilder:
    """
    Takes GPT-3 output triples and builds graph GPT-3 output 
    """
    def __init__(self, abrv_cont = None, ):
        self.abrvs = abrv_cont # Is only needed for NLP processing, not needed beyond populating counters
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



    def populateCounters(self, df, col_input = "Extracted_Text"):
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
        
        
        for index, row in df.iterrows():
            print("NLP processing: " + str(index))
            article_statements: list[tuple[set[str], set[str]]] = [] # List of tuples (per statement) of set containing entities from each individual statement
            text = row[col_input]
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

    def postprocessEntitySet(self, set_entities, entity_type):
        """
        entity_type: type of entities contained by the set (e.g., factor vs outcome), specifies
        which containers will be used for processing 
        """
        # Order matters: ignore, translate, then map
        if entity_type == "factor":
            set_entities = {ent for ent in set_entities if ent not in self.factors_ignore}
            set_entities = {transEnts(ent, self.factors_trans) for ent in set_entities}
            set_entities = {mapAbrv(ent, self.abrvs) for ent in set_entities} # Map any abbreviable strings to their abbreviations
            return set_entities
        elif entity_type == "outcome":
            set_entities = {ent for ent in set_entities if ent not in self.outcomes_ignore}
            set_entities = {transEnts(ent, self.outcomes_trans) for ent in set_entities}
            set_entities = {mapAbrv(ent, self.abrvs) for ent in set_entities} # Map any abbreviable strings to their abbreviations
            return set_entities
        # Can add other entity pipelines here
        
    def resetCounters(self):
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        return None

    def resetGraph(self,):
        self.graph = nx.Graph()

def compareStrings(str1, str2):
    return difflib.SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio()

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

if __name__ == "__main__":
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