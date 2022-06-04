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

#%% Functions

def compareStrings(str1, str2):
    return difflib.SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio()

def extractAbrvs(string) -> list:
    """
    Takes a string and returns a set of all the abbreviations
    """
    doc = NLP(string)
    abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
    return abrvs

def nlpString(string) -> list:
    """
    Takes a string and returns a tuple of a set of entities with abbreviation mapping
    """
    doc = NLP(string)
    ents = {ent.text.lower().strip() for ent in list(doc.ents)} # Need to convert to str first, otherwise causes problems with subsequent functions which only take strings
    abrvs = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations])
    for abrv, full in abrvs:
        for ent in ents.copy(): # Iterate over a copy of the set while changing the original
            if compareStrings(full, ent) > 0.9: # Find ent matching with full form of abbreviation
                ents.remove(ent) # Remove full form
                ents.add(abrv) # Add abbreviated form                
    return ents

def mapAbrv(string, abrv_container, threshold = 0.9):
    """
    Checks if there is an abbreviation in a string given an abbreviation
    container and maps it to the abbreviation if present
    Returns original string if no matches
    """
    print(abrv_container)
    for abrv, full in abrv_container:
        if compareStrings(full, string) > threshold:
            return abrv
    return string


#%% Build graph from items
df_origin = pd.read_excel("gpt3_output_formatted.xlsx")
abrv_container = set()

for index, row in df_origin.iterrows():
    print(index)
    text = row["Extracted_Text"]
    items = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
    for item in items: # Collect abbreviations 
        abrv_container.update(extractAbrvs(item))
#%%
factor_counter = Counter()
outcome_counter = Counter()
edge_counter = Counter()

for index, row in df_origin.iterrows():
    # Use sets for containers so multiple mentions within same paper are not recounted 
    print(index)
    factors = set()
    outcomes = set()
    relationships = set()
    text = row["Extracted_Text"]
    items = [item.strip() for item in text.split("\n") if re.search(r"\w", item) != None] # Only include those that have word characters
    for item in items: # Counting items
        factor, outcome, size = list(filter(None, item.split("|"))) # Filter with none to get rid of empty strings
        # Can add additional resolution parsing within the if statements
        if re.search(r"\w", factor) != None: # Given that this cell is not empty
            factor_ents = nlpString(factor)
            factor_ents = {mapAbrv(ent, abrv_container) for ent in factor_ents} # Map any abbreviable strings to their abbreviations
            factors.update(factor_ents)
        if re.search(r"\w", outcome) != None:
            outcome_ents = nlpString(outcome)
            outcome_ents = {mapAbrv(ent, abrv_container) for ent in outcome_ents} # Map any abbreviable strings to their abbreviations
            outcomes.update(outcome_ents)
        if re.search(r"\w", factor) != None and re.search(r"\w", outcome) != None:
            for factor_ent in factor_ents: # Add connection between a factor and all outcomes
                for outcome_ent in outcome_ents:
                    relationships.add((factor_ent, outcome_ent))
                    relationships.add((outcome_ent, factor_ent)) # Add bidirectional relationship
                    # Remember to enumerate here to avoid repeating connections
    for factor in factors:
        factor_counter[factor] += 1
    for outcome in outcomes:
        outcome_counter[outcome] += 1
    for edge in relationships:
        edge_counter[edge] += 1
del factor_counter["mortality"]
# print(factor_counter)
# print(outcome_counter)
# print(edge_counter)

#%% Build Networkx graph
# Reminder that nx nodes can have abitrary attributes that don't contribute to rendering, need to manually adjust visual parameters with drawing methods
# nx.Graph is just a way to store data
# Data can 
graph = nx.Graph()
T = 1
for entity in factor_counter:
    count = factor_counter[entity]
    if count > T: # Only add if there is more than 1 mention
        graph.add_node(entity, color = "#1E6091", size = count)
for entity in outcome_counter:
    count = outcome_counter[entity]
    if count > T:
        graph.add_node(entity, color = "#76C893", size = count)
for (node1, node2) in edge_counter:
    count = edge_counter[(node1, node2)]
    if (factor_counter[node1] > T or outcome_counter[node1] > T) and\
        (factor_counter[node2] > T or outcome_counter[node2] > T): # Need to each node in all sets of counters
        graph.add_edge(node1, node2, weight = count) # "weight" argument is built-in but does not affect width rendering
        print(node1, node2)

node_sizes = [size for (node, size) in graph.nodes(data="size")]
node_colors = [color for (node, color) in graph.nodes(data="color")]
edge_width_true = [width for (node1, node2, width) in graph.edges(data="weight")]
edge_widths = [log(width, 2) for width in edge_width_true]
edge_widths = np.clip(edge_widths, 0.2, None) # Set lower bound of width to 1
edge_transparency = [(0.7*(width/max(edge_width_true)))**(1/3) for width in edge_width_true] # Scaled to max width times 0.7 to avoid solid lines, cube root to reduce right skewness 
edge_transparency = np.clip(edge_transparency, 0.05, None) # Use np to set lower bound for edges
label_sizes = node_sizes

# Preview distributions
data = edge_widths
bins = np.arange(min(data), max(data), 1) # fixed bin size
plt.xlim([min(data), max(data)])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Test')
plt.xlabel('variable X')
plt.ylabel('count')

plt.show()

#%% Networkx visualization (multiple elements)
# nx uses matplotlib.pyplot for figures, can use plt manipulation to modify size
plt.figure(1, figsize = (12, 12), dpi = 600)
layout = nx.kamada_kawai_layout(graph) # Different position solvers available: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html
nx.draw_networkx_nodes(graph, 
    pos = layout,
    alpha = 0.8,
    node_size = node_sizes,
    node_color = node_colors,
    )
nx.draw_networkx_edges(graph,
    pos = layout,
    alpha = edge_transparency,
    width = edge_widths,
    )
## Manually draw labels with different sizes: https://stackoverflow.com/questions/62649745/is-it-possible-to-change-font-sizes-according-to-node-sizes
for node, (x, y) in layout.items():
    label_size = log(graph.nodes[node]["size"], 2) # Retrieve size information via node identity in graph
    plt.text(x, y, node, fontsize = label_size, ha = "center", va = "center") # Manually draw text

#%% Networkx visualization (single command)
# nx uses matplotlib.pyplot for figures, can use plt manipulation to modify size
plt.figure(1, figsize = (12, 12), dpi = 600)
nx.draw_networkx(graph, 
    pos = nx.kamada_kawai_layout(graph), # Different position solvers available: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html
    alpha = 0.5,
    node_size = node_sizes,
    node_color = node_colors,
    width = edge_widths,
    font_size = 2,
    )
#%% Pyvis visualization 
net = Network()

T = 1
for entity in factor_counter:
    count = factor_counter[entity]
    if count > T: # Only add if there is more than 1 mention
        net.add_node(entity, color = "cyan", size = count, mass = count)
for entity in outcome_counter:
    count = outcome_counter[entity]
    if count > T:
        net.add_node(entity, color = "blue", size = count, mass = count)
for (node1, node2) in edge_counter:
    count = edge_counter[(node1, node2)]
    if (factor_counter[node1] > T or outcome_counter[node1] > T) and\
        (factor_counter[node2] > T or outcome_counter[node2] > T): # Need to each node in all sets of counters
        net.add_edge(node1, node2, width = count)
        print(node1, node2)

# Log relationships
# for entity in factor_counter:
#     if factor_counter[entity] > 1: # Only add if there is more than 1 mention
#         net.add_node(entity, color = "red", size = math.log(factor_counter[entity]), mass = math.log(factor_counter[entity]))
# for entity in outcome_counter:
#     if outcome_counter[entity] > 1:
#         net.add_node(entity, color = "blue", size = math.log(outcome_counter[entity]), mass = math.log(outcome_counter[entity]))
# for (node1, node2) in edge_counter:
#     if factor_counter[entity] > 1 and outcome_counter[entity] > 1: # Should make sure that nodes exist as per previous statements
#         net.add_edge(node1, node2, width = math.log(edge_counter[(node1, node2)]))

net.toggle_physics(True)
# net.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65) # For smaller graphs 
# net.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100) # For larger graphs 
net.repulsion()
net.inherit_edge_colors(False)
net.show_buttons(filter_=['physics'])
net.show("Network.html")


# %%
