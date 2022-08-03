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


# Internal imports
from internal_globals import importData

#%% Constants


#%% Classes & Functions

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
            df_merged.loc[ind, col_out] = json.dumps(list_ents) # Need loc function and original df to modify in place, can't just give index
        df_merged.to_excel(f"{new_name}.xlsx")
        print(f"Exported processed ents to {new_name}.xlsx")
    
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
        print("Abbreviations:", self.abrv_log)
        print("Translations:", self.trans_log)
        print("Conflicts:", self.conf_ent_log)

        
class GraphBuilder:
    """
    Takes GPT-3 output triples and builds graph GPT-3 output 
    
    Takes structured statements and generates a NetworkX graph depending 
    on the method 
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.node_counters: dict[str, Counter[str]] = dict()
        self.edge_counters: dict[tuple[str, str], Counter[tuple[str, str]]] = dict()
        self.df_root_name: str = "" # Populated by the root name of the last df that was used to populate counters
        self.graph_root_name: str = "" # Derived from df_root_name, includes information about thresholding


    def popCountersMulti(self, df_path, col = "Processed_ents",
                         col_sub = "", subset = ""):
        """
        For DFs containing multiple entity types (i.e., distinguishes between node types for node and edge enties)
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        DOES NOT RESET THE COUNTERS
        """
        self.df_root_name = os.path.splitext(df_path)[0] # Store root name
        df = importData(df_path, screen_text=[col])
        if col_sub and subset:
            pass
        proto_stmt: dict[str, list[str]] = json.loads(df[col].iloc[0])[0] # Get prototypical statement (obtains first statment from first row of col of interest)
        ent_types = [key for key in proto_stmt]
        edge_types = list(permutations(proto_stmt, 2))
        print("Populating counters from DF...") # Below code runs quite fast so we don't need to track progress
        for index, row in df.iterrows():
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

    def printCounters(self):
        print("Entities ==============================================")
        for ent_type in self.node_counters:
            print("------------ Entity type ", ent_type, " ------------")
            list_items = list(self.node_counters[ent_type].items())
            list_items.sort(key=lambda x: x[1], reverse=True)
            print(list_items)
            # Print nodes sorted by count
        print("Edges ==============================================")
        for edge_type in self.edge_counters:
            print("------------ Edge type ", edge_type, " ------------")
            list_items = list(self.edge_counters[edge_type].items())
            list_items.sort(key=lambda x: x[1], reverse=True)
            print(list_items)
            # Print edges sorted by count

    
    def buildGraph(self, thresh = 1, multidi = False):
        """
        Builds Networkx graph with the populated counters and a threshold for node count
        ---
        thresh: lower threshold for number of counts needed for each node (exclusive)
        """
        # Reminder that nx nodes can have abitrary attributes that don't contribute to rendering, need to manually adjust visual parameters with drawing methods
        # nx.Graph is just a way to store data, data can be stored in node attributes         
        if multidi:
            self.graph = nx.MultiDiGraph() # Instantiate multidigraph
            # Allows parallel and directed relationships to be rendered
        else:
            self.graph = nx.Graph() # Instantiate regular graph
        print("Building graph from counters...")
        for ent_type in self.node_counters:
            node_counter = self.node_counters[ent_type]
            
            # Can add specific data into into node depending on node type
            if ent_type == "factor":
                pass
            elif ent_type == "outcome":
                pass
            else: 
                pass 
                
            for ent in node_counter:
                count = node_counter[ent]
                if count > thresh:
                    self.graph.add_node(ent, size=count, ent_type=ent_type)
                pass
        
        for edge_type in self.edge_counters:
            edge_counter = self.edge_counters[edge_type]
            node_s_counter = self.node_counters[edge_type[0]] # Retrieve node counter for source
            node_t_counter = self.node_counters[edge_type[1]] # Retrieve node counter for target
            
            if edge_type == ("factor", "outcome"): # Demo for different styling
                pass
            elif edge_type == ("outcome", "factor"):
                pass
            elif "factor" in edge_type and "outcome" in edge_type: # Can also find bi-directional relationships
                pass
            
            for edge in edge_counter:
                count = edge_counter[edge]
                node_s = edge[0]
                node_t = edge[1]
                if (node_s_counter[node_s] > thresh and node_t_counter[node_t] > thresh): # Don't need to check in both counters like before since node type is already specified by edge_type
                    node_s_mentions = node_s_counter[node_s]
                    probability = count/node_s_mentions # Probability of this connection is number of articles supporting this connection divided by total number of articles mentioning source
                    self.graph.add_edge(node_s, node_t, width=count, prob=probability) # "width" attribute affects pyvis rendering, pyvis doesn't support edge opacity
        self.graph_root_name = self.df_root_name + f"_t{thresh}" # Add threshold information 

    def exportGraph(self, path: Union[str, bytes, os.PathLike] = ""):
        """
        Exports currently stored graph to an XML file with the specified path 
        """
        if path:
            file_name = path
        else:
            file_name = self.graph_root_name + ".xml"
        # Export graph , https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.write_graphml.html#networkx.readwrite.graphml.write_graphml
        # Graphml documentation: https://networkx.org/documentation/stable/reference/readwrite/graphml.html
        nx.write_graphml(self.graph, file_name)
        print("Exported graph to", file_name)
        
    def resetCounters(self):
        self.node_counters: dict[str, Counter[str]] = dict()
        self.edge_counters: dict[tuple[str, str], Counter[tuple[str, str]]] = dict()

    def resetGraph(self):
        self.graph = nx.Graph()

if __name__ == "__main__": # For testing purposes
    b = GraphBuilder()
    b.popCountersMulti("test/gpt3_output_fmt_fmtents.xlsx")
    b.buildGraph()
    b.exportGraph()

