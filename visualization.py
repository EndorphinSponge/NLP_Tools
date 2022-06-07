#%% Imports 
# General
from collections import Counter
from math import log
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
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

#%% Constants
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector")

#%% Functions & classes 

def roundNum(num, base):
    """
    Rounds a number to a base, will return 1/5 of the base if the number
    is less than half the base (i.e., rounding with min being half of base)
    """
    if num > base/2:
        return base * round(num/base)
    elif num <= base/2 and num > base/5: # Catch cases in between and round up (normally it would round down)
        return base
    else:
        return int(base/5)

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
            print("Mapped " + full + " to " + abrv)
            return abrv
    return string

def transEnts(string, trans_dict):
    if string in trans_dict.keys(): # If the string matches a translation key
        print("Translated " + string + " to " + trans_dict[string])
        return trans_dict[string]
    else:
        return string

def extractAbrvs(string) -> list:
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
        
class GraphBuilder:
    """
    Contains 
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
            "gc",
            ] # Words common to both factors and outcomes
        common_tbi_ignore = ["tbi", "mtbi", "stbi", "csf", "serum", "blood", "plasma", "mild",
            "moderate", "severe", "concentration", "risk", "traumatic", "finding", "post-injury",
            "injury",
            ] # Specific to TBI 
        self.factors_ignore = ["problem",] + common_ignore + common_tbi_ignore
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
            "morality rate": "mortality",
            "survival": "mortality",
            "functional": "fo",


        }
    def resetCounters(self):
        self.factor_counter = Counter()
        self.outcome_counter = Counter()
        self.edge_counter = Counter()
        return None

    def postprocessEntitySet(self, set_entities, entity_type):
        """
        entity_type: type of entities contained by the set (e.g., factor vs outcome), specifies
        which containers will be used for processing 
        """
        if entity_type == "factor":
            set_entities = {mapAbrv(ent, self.abrvs) for ent in set_entities} # Map any abbreviable strings to their abbreviations
            set_entities = {ent for ent in set_entities if ent not in self.factors_ignore}
            set_entities = {transEnts(ent, self.factors_trans) for ent in set_entities}
            return set_entities
        elif entity_type == "outcome":
            set_entities = {mapAbrv(ent, self.abrvs) for ent in set_entities} # Map any abbreviable strings to their abbreviations
            set_entities = {ent for ent in set_entities if ent not in self.outcomes_ignore}
            set_entities = {transEnts(ent, self.outcomes_trans) for ent in set_entities}
            return set_entities
        # Can add other entity pipelines here


    def populateCounters(self, df, col_input = "Extracted_Text"):
        """
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        """
        # Reset counters
        self.resetCounters()
        # NLP processing
        set_mast_factors = set() # Set of all post-processed factors
        set_mast_outcomes = set() # Set of all post-processed outcomes
        list_articles = [] # Stores output of each row/article as lists of factors and outcomes for each statement
        # Has shape of list(list(tuple(set(factor), set(outcome))))        
        for index, row in df.iterrows():
            print("NLP processing: " + str(index))
            article_statements = [] # List of tuples (per statement) of set containing entities from each individual statement
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
        alpha_max = 0.95, alpha_min = 0.01, alpha_root = 1.5, 
        save_prefix = False, cmap= True, fig_size = 10,
        ):
        """
        Renders the graph contained within the object using NX
        ----
        save_prefix: prefix for saving figures
        cmap: use color mapping in the stead of transparency 
        """
        dict_sizes = dict(self.graph.nodes(data="size")) # Convert node data to dict
        scaling = fig_size*15/log(sum(dict_sizes.values()), 5) # Log the sum of the node sizes for scaling factor, fig_size taken into account, constant made based off manual tweaking
        scaling = fig_size*40/max(list(dict_sizes.values())) # Use max node value 
        node_sizes = [scaling*size for (node, size) in self.graph.nodes(data="size")]
        node_colors = [color for (node, color) in self.graph.nodes(data="color")]
        edge_width_true = [width for (node1, node2, width) in self.graph.edges(data="width")]
        edge_widths = [log(scaling*width, width_log) for width in edge_width_true]
        edge_widths = np.clip(edge_widths, width_min, None) # Set lower bound of width to 1
        edge_transparency = [alpha_max*(width/max(edge_width_true))**(1/alpha_root) for width in edge_width_true] # Scaled to max width times 0.7 to avoid solid lines, cube root if you want to reduce right skewness 
        edge_transparency = np.clip(edge_transparency, alpha_min, None) # Use np to set lower bound for edges
        edge_zeroes = [0 for i in edge_width_true]

        #%% Networkx visualization (multiple elements)
        # nx uses matplotlib.pyplot for figures, can use plt manipulation to modify size
        fig = plt.figure(1, figsize = (fig_size * 1.1, fig_size), dpi = 800)
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
            label_size = log(scaling*self.graph.nodes[node]["size"], 2) # Retrieve size information via node identity in graph
            plt.text(x, y, node, fontsize = label_size, ha = "center", va = "center", alpha = 0.7) # Manually draw text

        # Draw legend: https://stackoverflow.com/questions/29973952/how-to-draw-legend-for-scatter-plot-indicating-size
        # Same scaling factor but different rounding thresholds
        d1 = roundNum(0.02*max(node_sizes)/scaling, 5) # Reference of 5 for max of 250
        d2 = roundNum(0.08*max(node_sizes)/scaling, 10) # Reference of 20 for max of 250 
        d3 = roundNum(0.4*max(node_sizes)/scaling, 20) # Reference of 100 for max of 250
        d4 = roundNum(max(node_sizes)/scaling, 50) # Reference of 250 for max of 250
        p1 = plt.scatter([],[], s=d1*scaling, marker='o', color='#8338ec', alpha = 0.8)
        p2 = plt.scatter([],[], s=d2*scaling, marker='o', color='#8338ec', alpha = 0.8)
        p3 = plt.scatter([],[], s=d3*scaling, marker='o', color='#8338ec', alpha = 0.8)
        p4 = plt.scatter([],[], s=d4*scaling, marker='o', color='#8338ec', alpha = 0.8)

        def genLabel(num):
            """
            Reverts scaled number into original scale and rounds off decimal,
            then converts to string
            """
            return str(roundNum(num/scaling, 1))

        plt.legend((p1, p2, p3, p4), 
            (d1, d2, d3, d4), # Divide by scaling to convert back to normal size
            scatterpoints=1,loc='lower left', title = "Number of articles with factor/outcome", 
            ncol=4, prop={'size': fig_size}, title_fontsize = fig_size, borderpad = 0.8,
            )
        # scatterpoints = number of points in each size demo
        # ncol = number of columns that each size demo will be split into
        # prop = container deciding properties of text relating to the size demos
        # 10 is default font size
        
        if cmap: # Will map values (in proportion to min/max) to a color spectrum
            edges = nx.draw_networkx_edges(self.graph,
                pos = layout,
                alpha = edge_transparency, # Can add transparency on top to accentuate
                edge_color = edge_transparency,
                width = edge_widths,
                edge_cmap = plt.cm.summer, 
                )
            edges = nx.draw_networkx_edges(self.graph, # Dummy variable for if color assignment is cube rooted, (transparency set to zero)
                pos = layout,
                alpha = edge_zeroes, # Array of zeroes
                edge_color = edge_width_true, # NOTE THAT THIS IS NOT EXACTLY THE SAME SCALE (due to cube root)
                edge_cmap = plt.cm.summer, 
                )
            # Colorbar legend solution: https://groups.google.com/g/networkx-discuss/c/gZmr-YgvIQs
            # Alternative solution using FuncFormatter here: https://stackoverflow.com/questions/38309171/colorbar-change-text-value-matplotlib
            plt.sci(edges)
            plt.colorbar().set_label("Number of articles supporting connection")
            # Available colormaps: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
            # Tested colormaps: GnBu is too similar to node color scheme, [YlOrRd, PuRd, Wistia] makes small edges too light, 
        else:
            edges = nx.draw_networkx_edges(self.graph,
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

#%% Rendering from exported XMLs
DIR = r"figures\network_v0.9.1 (colorbar, node legend, no auto size)\exportXML"
builder = GraphBuilder()
for i in range(0, 5): # Rendering with threshold = 1
    builder.importGraph(os.path.join(DIR, f"tbi_topic{i}_t3_graph.xml"))
    builder.renderGraphNX(save_prefix = f"tbi_topic{i}_t3_graph.xml", cmap = True)
for i in range(5, 6): # Rendering with threshold = 2
    builder.importGraph(os.path.join(DIR, f"tbi_topic{i}_t2_graph.xml"))
    builder.renderGraphNX(save_prefix = f"tbi_topic{i}_t2_graph.xml", cmap = True)
for i in range(6, 11): # Rendering with threshold = 3
    builder.importGraph(os.path.join(DIR, f"tbi_topic{i}_t1_graph.xml"))
    builder.renderGraphNX(save_prefix = f"tbi_topic{i}_t1_graph.xml", cmap = True)

#%% Execution 
if __name__ == "__main__":
    df_origin = pd.read_excel("gpt3_output_formatted_annotated.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
    abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
    builder = GraphBuilder(abrvs)
    builder.populateCounters(df_origin, col_input = "Extracted_Text")
    builder.buildGraph(thresh = 10)
    builder.exportGraph(f"tbi_ymcombined_t10_graph.xml")
    builder.renderGraphNX(save_prefix = f"tbi_ymcombined_t10", alpha_root = 3, cmap = True)
    builder.buildGraph(thresh = 15)
    builder.exportGraph(f"tbi_ymcombined_t15_graph.xml")
    builder.renderGraphNX(save_prefix = f"tbi_ymcombined_t15", alpha_root = 3, cmap = True)
    builder.buildGraph(thresh = 20)
    builder.exportGraph(f"tbi_ymcombined_t20_graph.xml")
    builder.renderGraphNX(save_prefix = f"tbi_ymcombined_t20", alpha_root = 3, cmap = True)
    for topic_num in range(0, 11): # May want to automate topic detection 
        df_subset = df_origin[df_origin["Topic"]==topic_num]
        builder.populateCounters(df_subset, col_input = "Extracted_Text")
        builder.buildGraph(thresh = 1)
        builder.exportGraph(f"tbi_topic{topic_num}_t1_graph.xml")
        builder.renderGraphNX(save_prefix = f"tbi_topic{topic_num}_t1", alpha_root = 3, cmap = True)
        builder.buildGraph(thresh = 2)
        builder.exportGraph(f"tbi_topic{topic_num}_t2_graph.xml")
        builder.renderGraphNX(save_prefix = f"tbi_topic{topic_num}_t2", alpha_root = 3, cmap = True)
        builder.buildGraph(thresh = 3)
        builder.exportGraph(f"tbi_topic{topic_num}_t3_graph.xml")
        builder.renderGraphNX(save_prefix = f"tbi_topic{topic_num}_t3", alpha_root = 3, cmap = True)

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

#%% Preview distributions contained within an array
data = [] # Container for data
bins = np.arange(min(data), max(data), 1) # fixed bin size
plt.xlim([min(data), max(data)])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Test')
plt.xlabel('variable X')
plt.ylabel('count')

plt.show()