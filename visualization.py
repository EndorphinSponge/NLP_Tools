#%% Imports 
# General
from collections import Counter
from math import log
import re
import difflib

# Data science
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt

# NLP
import spacy 
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

#%% Constants
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector")

#%% Functions & classes 

def compareStrings(str1, str2):
    return difflib.SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio()

def extractAbrvs(string) -> list:
    """
    Takes a string and returns a set of all the abbreviations
    """
    doc = NLP(string.strip()) # Need to strip whitespace, otherwise recognition is suboptimal esp for shorter queries
    abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
    return abrvs

def nlpString(string) -> list:
    """
    Takes a string and returns a tuple of a set of entities with abbreviation mapping
    """
    doc = NLP(string.strip()) # Need to strip whitespace, otherwise recognition is suboptimal esp for shorter queries
    if len(doc) > 1: # Only process if there is more than one token
        ents = {ent.lemma_.lower().strip() for ent in list(doc.ents)} # Need to convert to str first, otherwise causes problems with subsequent functions which only take strings
        abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
        for abrv, full in abrvs:
            for ent in ents.copy(): # Iterate over a copy of the set while changing the original
                if compareStrings(full, ent) > 0.9: # Find ent matching with full form of abbreviation
                    ents.remove(ent) # Remove full form
                    ents.add(abrv) # Add abbreviated form                
        return ents
    else:
        return {doc[0].lemma_.strip().lower(),} # Otherwise there will be only one token, return its lemma 

def mapAbrv(string, abrv_container, threshold = 0.9):
    """
    Checks if there is an abbreviation in a string given an abbreviation
    container and maps it to the abbreviation if present
    Returns original string if no matches
    """
    for abrv, full in abrv_container:
        if compareStrings(full, string) > threshold:
            return abrv
    return string

def transEnts(string, trans_dict):
    if string in trans_dict.keys(): # If the string matches a translation key
        return trans_dict[string]
    else:
        return string

def extractAbrvCont(df, col_input = "Extracted_Text"):
    abrv_container = set()
    for index, row in df.iterrows():
        print(index)
        text = row[col_input]
        items = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
        for item in items: # Collect abbreviations 
            abrv_container.update(extractAbrvs(item))
    return abrv_container
        
class GraphBuilder:
    """
    Contains 
    """
    def __init__(self, abrv_cont, ):
        self.abrvs = abrv_cont    
        self.graph = nx.Graph()
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        common_ignore = ["patient", "patient\'", "patients", "rate", "associated", "hour", "day", "month", "year", "level", 
            "favorable", "favourable", "good", "prevalence", "presence", "result", "ratio", "in-hospital",
            "decrease", "bad", "poor", "unfavorable", "unfavourable", "reduced", "use of", "development",
            "clinical trial", "significance", "finding", "score", "analysis",
            "early", "adult",
            ] # Words common to both factors and outcomes
        common_tbi_ignore = ["tbi", "mtbi", "stbi", "csf", "serum", "blood", "plasma", "mild",
            "moderate", "severe", "concentration", "risk", "traumatic", "finding", "post-injury",
            ] # Specific to TBI 
        self.factors_ignore = ["problem",] + common_ignore + common_tbi_ignore
        self.outcomes_ignore = ["age", "improved", "reduced", "trauma", "s100b"] + common_ignore + common_tbi_ignore
        self.factors_trans = {
            "gcs": "gcs (factor)",
            "injury": "injury (factor)",
            "depression": ""

        }
        self.outcomes_trans = {
            "gcs": "gcs (outcome)",
            "injury": "injury (outcome)",
            "hospital mortality": "in-hospital mortality",
            "clinical outcome": "outcome",
            "death": "mortality",
            "morality rate": "mortality",
            "survival": "mortality"

        }
    def resetCounters(self):
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        return None

    def populateCounters(self, df, col_input = "Extracted_Text"):
        """
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        """
        # Reset counters
        self.resetCounters()
        for index, row in df.iterrows():
            # Use sets for containers so multiple mentions within same paper are not recounted 
            print(index)
            factors = set()
            outcomes = set()
            relationships = set()
            text = row[col_input]
            statement = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
            for item in statement: # Counting items
                factor, outcome, size = list(filter(None, item.split("|"))) # Filter with none to get rid of empty strings
                # Can add additional resolution parsing within the if statements
                if re.search(r"\w", factor) != None: # Given that this cell is not empty
                    factor_ents = nlpString(factor)
                    factor_ents = {mapAbrv(ent, self.abrvs) for ent in factor_ents} # Map any abbreviable strings to their abbreviations
                    factor_ents = {ent for ent in factor_ents if ent not in self.factors_ignore}
                    factor_ents = {transEnts(ent, self.factors_trans) for ent in factor_ents}
                    factors.update(factor_ents)
                if re.search(r"\w", outcome) != None:
                    outcome_ents = nlpString(outcome)
                    outcome_ents = {mapAbrv(ent, self.abrvs) for ent in outcome_ents} # Map any abbreviable strings to their abbreviations
                    outcome_ents = {ent for ent in outcome_ents if ent not in self.outcomes_ignore}
                    outcome_ents = {transEnts(ent, self.outcomes_trans) for ent in outcome_ents}
                    outcomes.update(outcome_ents)
                if re.search(r"\w", factor) != None and re.search(r"\w", outcome) != None:
                    for factor_ent in factor_ents: # Add connection between a factor and all outcomes
                        for outcome_ent in outcome_ents:
                            if factor_ent != outcome_ent: # So that you don't get self connections
                                relationships.add((factor_ent, outcome_ent))
                                relationships.add((outcome_ent, factor_ent)) # Add bidirectional relationship
                                # Remember to enumerate here to avoid repeating connections
            for factor in factors:
                self.factor_counter[factor] += 1
            for outcome in outcomes:
                self.outcome_counter[outcome] += 1
            for edge in relationships:
                self.edge_counter[edge] += 1
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

    def importGraph(self, path):
        """
        Imports an NX graph from an XML file into the object
        """
        # Import graph https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.read_graphml.html#networkx.readwrite.graphml.read_graphml
        self.graph = nx.read_graphml(path)
        return None

    def resetGraph(self,):
        self.graph = nx.Graph()

    def renderGraphNX(self, 
        width_log = 2, width_min = 0.2, 
        alpha_max = 0.8, alpha_min = 0.01, alpha_root = 1, 
        save_prefix = False, cmap= True, 
        ):
        """
        Renders the graph contained within the object using NX
        ----
        save_prefix: prefix for saving figures
        cmap: use color mapping in the stead of transparency 
        """
        node_sizes = [size for (node, size) in self.graph.nodes(data="size")]
        node_colors = [color for (node, color) in self.graph.nodes(data="color")]
        edge_width_true = [width for (node1, node2, width) in self.graph.edges(data="width")]
        edge_widths = [log(width, width_log) for width in edge_width_true]
        edge_widths = np.clip(edge_widths, width_min, None) # Set lower bound of width to 1
        edge_transparency = [alpha_max*(width/max(edge_width_true))**(1/alpha_root) for width in edge_width_true] # Scaled to max width times 0.7 to avoid solid lines, cube root if you want to reduce right skewness 
        edge_transparency = np.clip(edge_transparency, alpha_min, None) # Use np to set lower bound for edges
        label_sizes = node_sizes

        #%% Networkx visualization (multiple elements)
        # nx uses matplotlib.pyplot for figures, can use plt manipulation to modify size
        plt.figure(1, figsize = (12, 12), dpi = 600)
        plt.clf() # Clear figure, has to be done AFTER setting figure size/DPI, otherwise this information is no assigned properly
        layout = nx.kamada_kawai_layout(self.graph) # Different position solvers available: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html
        nx.draw_networkx_nodes(self.graph, 
            pos = layout,
            alpha = 0.8,
            node_size = node_sizes,
            node_color = node_colors,
            )
        ## Manually draw labels with different sizes: https://stackoverflow.com/questions/62649745/is-it-possible-to-change-font-sizes-according-to-node-sizes
        for node, (x, y) in layout.items():
            label_size = log(self.graph.nodes[node]["size"], 2) # Retrieve size information via node identity in graph
            plt.text(x, y, node, fontsize = label_size, ha = "center", va = "center") # Manually draw text
        if cmap: # Will map values (in proportion to min/max) to a color spectrum
            nx.draw_networkx_edges(self.graph,
                pos = layout,
                # alpha = edge_transparency, # Can add transparency on top to accentuate
                edge_color = edge_transparency,
                width = edge_widths,
                edge_cmap = plt.cm.summer, 
                )
            # Available colormaps: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
            # Tested colormaps: GnBu is too similar to node color scheme, [YlOrRd, PuRd, Wistia] makes small edges too light, 
        else:
            nx.draw_networkx_edges(self.graph,
                pos = layout,
                alpha = edge_transparency,
                width = edge_widths,
                )

        if save_prefix:
            plt.savefig(f"net_{save_prefix}_(width[log{str(width_log)}_min{str(width_min)}]alpha[max{str(alpha_max)}min{str(alpha_min)}root{str(alpha_root)}]).png")
        else:
            plt.show()
        return None

    def renderGraphPyvis(self, path = "pyvis_network.html", solver = "repulsion"):
        """
        Builds graph from counters and renders it using Pyvis
        """
        graphpy = Network()
        graphpy.from_nx(self.graph)

        graphpy.toggle_physics(True)
        if solver == "repulsion":
            graphpy.repulsion()
        elif solver == "atlas":
            graphpy.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65) # For smaller graphs 
            # graphpy.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100) # For larger graphs 
        else:
            graphpy.barnes_hut()
        graphpy.inherit_edge_colors(False)
        graphpy.show_buttons(filter_=['physics'])
        graphpy.show(path)




#%% Execution 
if __name__ == "__main__":
    df_origin = pd.read_excel("gpt3_output_formatted_annotated.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
    abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
    builder = GraphBuilder(abrvs)
    for topic_num in range(0, 11): # May want to automate topic detection 
        df_subset = df_origin[df_origin["Topic"]==topic_num]
        builder.populateCounters(df_subset, col_input = "Extracted_Text")
        builder.buildGraph(thresh = 1)
        builder.renderGraphNX(save_prefix = f"tbi_topic{topic_num}_t1", cmap = True)
        # builder.renderGraphNX()

#%%
df_origin = pd.read_excel("gpt3_output_formatted_annotated25.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
#%%
builder = GraphBuilder(abrvs)
#%%
builder.populateCounters(df_origin)
builder.buildGraph(thresh = 1)
builder.renderGraphNX(cmap = True)
#%%

#%% Preview distributions contained within an array
data = [] # Container for data
bins = np.arange(min(data), max(data), 1) # fixed bin size
plt.xlim([min(data), max(data)])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Test')
plt.xlabel('variable X')
plt.ylabel('count')

plt.show()