#%% Imports 
# General
from collections import Counter
from itertools import combinations, product, permutations
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
with open("test/gpt3_output_fmt_abrvs_rfn.json", "r") as file:
    abrv_json: list[list[list[str, str], int]] = json.load(file)

with open("test/gpt3_output_fmt_abrvs_trans.json", "r") as file:
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
    def __init__(self):
        self.graph = nx.Graph()
        self.node_counters: dict[str, Counter] = dict()
        self.edge_counters: dict[(str, str), Counter] = dict()


    def popCountersMulti(self, df_path, col = "Processed_ents"):
        """
        For DFs containing multiple entity types (i.e., distinguishes between node types for node and edge enties)
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        """

        df = importData(df_path, screen_text=[col])
        proto_stmt: dict[str, list[str]] = json.loads(df[col].iloc[0])[0] # Get prototypical statement (obtains first statment from first row of col of interest)
        ent_types = [key for key in proto_stmt]
        edge_types = list(permutations(proto_stmt, 2))
        
        for index, row in df.iterrows():
            print("Populating counters from: ", index)
            list_statements: list[dict[str, list[str]]] = json.loads(row[col])
            article_nodes: dict[str, set[str]] = {t: set() for t in ent_types} # Initialize node container
            article_edges = {(t[0], t[1]): set() for t in edge_types} # Initialize edge container for all types
            for statement in list_statements:
                for ent_type in statement: # Parsing for each type of entity type
                    ents = statement[ent_type]
                    article_nodes[ent_type].update(ents) # Add ent to corresponding ent tracker set
                for ent_type1, ent_type2 in combinations(statement, 2):
                    # Can add if statement to screen out relationship between certain types of nodes 
                    for ent1, ent2 in product(statement[ent_type1], statement[ent_type2]):
                        article_edges[(ent_type1, ent_type2)].add((ent1, ent2)) 
                        article_edges[(ent_type2, ent_type1)].add((ent2, ent1)) # Add reverse relationship 
            for ent_type in article_nodes:
                if ent_type not in self.node_counters: # Instantiate node counter if not already instantiated
                    self.node_counters[ent_type] = Counter()
                self.node_counters[ent_type].update(article_nodes[ent_type]) # Add all nodes of the ent type to counter
            for edge_type in article_edges:
                if edge_type not in self.edge_counters: # Instantiate edge type counter if it doesn't exist
                    self.edge_counters[edge_type] = Counter()
                self.edge_counters[edge_type].update(article_edges[edge_type]) # Add all edges of this type to counter

    
    def buildGraph(self, thresh = 1,):
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
        for (node1, node2) in self.edge_counters:
            count = self.edge_counters[(node1, node2)]
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
        self.edge_counters = Counter()
        return None

    def resetGraph(self,):
        self.graph = nx.Graph()

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
            new_name = root_name + "_fmtents"
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
            for long_form in self.abbreviations: # Abbreviations ordered by most common and then by longest long form
                if SequenceMatcher(a=ent.lower(), b=long_form.lower()).ratio() > thresh:
                    short_form = self.abbreviations[long_form]
                    self.abrv_log[(short_form, ent)] += 1 # Log abbreviation mapping
                    return short_form # Return corresponding short form in abbreviation
                    # Other short form is either merged by set or translated and merged in transEnts
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
                
                ents = {abrvEnts(self, ent) for ent in ents}
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
