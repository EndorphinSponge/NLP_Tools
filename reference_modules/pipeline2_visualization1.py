# -*- coding: utf-8 -*-

from global_functions import *

#%% 
import matplotlib.pyplot as plt;
from collections import Counter;
import networkx as nx;
from pyvis.network import Network;
        
#%% Frequency
independent = "Frequency (Hz)";
x = [];
y = [];
s = [];
for (doc, context) in docs:
    if doc._.sample_size > 0:
        container = doc._.frequency;
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl));
                y.append(0);
                s.append(doc._.sample_size*30);
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append((float(low)+float(high))/2); # Takes the average of the min and max of the range 
                y.append(0);
                s.append(doc._.sample_size*30);
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)); # Takes specified number in the comparator expression, will have to refine later
                y.append(0);
                s.append(doc._.sample_size*30);

plt.figure(figsize=(30, 3.5));
plot = plt.scatter(x,y,s=s, alpha = 0.15); # Alpha to set transparency, otherwise points are overlapping
# plt.xlim(-10,800); # Manually set axes to exclude outliers
# plt.ylim(-1,1);
plt.xlabel(independent);
plot.axes.get_yaxis().set_visible(False);
plt.savefig("./figures/%s.png" % "Frequency", bbox_inches="tight", dpi=300); # Save function has to be called before the show() function

#%% Voltage
independent = "Voltage (V)"
x = []
y = []
s = []
for (doc, context) in docs:
    if doc._.sample_size > 0:
        container = doc._.voltage;
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl))
                y.append(0)
                s.append(doc._.sample_size*30)
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append((float(low)+float(high))/2) # Takes the average of the min and max of the range 
                y.append(0)
                s.append(doc._.sample_size*30)
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
                y.append(0)
                s.append(doc._.sample_size*30)

plt.figure(figsize=(30, 3.5));
plot = plt.scatter(x,y,s=s, alpha = 0.15) # Alpha to set transparency, otherwise points are overlapping
# plt.xlim(-1,25) # Manually set axes to exclude outliers, based on manual inspection of data (max was 40V in rats, but 24 in humans)
# plt.ylim(-1,1)
plt.xlabel(independent);
plot.axes.get_yaxis().set_visible(False)
plt.savefig("./figures/%s.png" % "Voltage", bbox_inches="tight", dpi=300) # Save function has to be called before the show() function

#%% Amperage
independent = "Amperage (mA)"
x = []
y = []
s = []
for (doc, context) in docs:
    if doc._.sample_size > 0:
        container = doc._.amperage;
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl))
                y.append(0)
                s.append(doc._.sample_size*30)
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append((float(low)+float(high))/2) # Takes the average of the min and max of the range 
                y.append(0)
                s.append(doc._.sample_size*30)
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
                y.append(0)
                s.append(doc._.sample_size*30)

plt.figure(figsize=(30, 3.5));
plot = plt.scatter(x,y,s=s, alpha = 0.15) # Alpha to set transparency, otherwise points are overlapping
plt.xlim(-1, 30) # Manually set axes to exclude outliers
plt.ylim(-1,1)
plt.xlabel(independent);
plot.axes.get_yaxis().set_visible(False)
plt.savefig("./figures/%s.png" % "Amperage", bbox_inches="tight", dpi=300) # Save function has to be called before the show() function

#%% Plotting stimulation location 


# Based on number of patients    
# location_counter = Counter();
# for (doc, context) in docs:
#     size = doc._.sample_size;
#     for location in doc._.targets_text:
#         location = location.lower().strip();
#         location_counter[location] += size;
        
# location_top = location_counter.most_common(25);
# location_df = pd.DataFrame.from_records(location_top, columns=["Terms", "Count"]);
# location_graph = location_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
# plt.xlabel("Location of interest");
# plt.ylabel("Total number of patients from studies")
# location_graph = plt.savefig("./figures/%s.png" % "LocationsSample", bbox_inches="tight", dpi=300);
                
            
# Based on number of papers 

location_counter = Counter();
for (doc, context) in docs:
    for location in doc._.targets_text:
        location = location.lower().strip();
        location_counter[location] += 1;
        
location_top = location_counter.most_common(25);
location_df = pd.DataFrame.from_records(location_top, columns=["Terms", "Count"]);
location_graph = location_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
plt.xlabel("Location of interest");
plt.ylabel("Total number of papers");
location_graph = plt.savefig("./figures/%s.png" % "LocationsPaper", bbox_inches="tight", dpi=300);

#%% Modality

# Based on sample size
# modality_counter = Counter();
# for (doc, context) in docs:
#     size = doc._.sample_size;
#     for modality in doc._.modality:
#         modality = modality.lower().strip();
#         modality_counter[modality] += size;
        
# modality_top = modality_counter.most_common(25);
# modality_df = pd.DataFrame.from_records(modality_top, columns=["Terms", "Count"]);
# modality_graph = modality_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
# plt.xlabel("Modality of neuromodulation");
# plt.ylabel("Total number of patients from studies");
# modality_graph = plt.savefig("./figures/%s.png" % "ModalitiesSample", bbox_inches="tight", dpi=300);

# Based on number of papers
modality_counter = Counter();
for (doc, context) in docs:
    for modality in doc._.modality:
        modality = modality.lower().strip();
        modality_counter[modality] += 1;
        
modality_top = modality_counter.most_common(25);
modality_df = pd.DataFrame.from_records(modality_top, columns=["Terms", "Count"]);
modality_graph = modality_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
plt.xlabel("Modality of neuromodulation");
plt.ylabel("Total number of papers");
modality_graph = plt.savefig("./figures/%s.png" % "ModalitiesPaper", bbox_inches="tight", dpi=300);

#%% Disease model - broad

# Based on sample size
# disease_counter = Counter();
# for (doc, context) in docs:
#     size = doc._.sample_size;
#     for disease in doc._.disease_broad:
#         disease = disease.lower().strip();
#         disease_counter[disease] += size;
        
# disease_top = disease_counter.most_common(25);
# disease_df = pd.DataFrame.from_records(disease_top, columns=["Terms", "Count"]);
# disease_graph = disease_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
# plt.xlabel("Epilepsy type (broad)");
# plt.ylabel("Total number of patients from studies");
# disease_graph = plt.savefig("./figures/%s.png" % "DiseasesBroadSample", bbox_inches="tight", dpi=300);

# Based on number of papers

disease_counter = Counter();
for (doc, context) in docs:
    for disease in doc._.disease_broad:
        disease = disease.lower().strip();
        disease_counter[disease] += 1;
        
disease_top = disease_counter.most_common(25);
disease_df = pd.DataFrame.from_records(disease_top, columns=["Terms", "Count"]);
disease_graph = disease_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
plt.xlabel("Epilepsy type (broad)");
plt.ylabel("Total number of papers");
disease_graph = plt.savefig("./figures/%s.png" % "DiseasesBroadPaper", bbox_inches="tight", dpi=300);

#%% Disease model - narrow

# Based on sample size
# disease_counter = Counter();
# for (doc, context) in docs:
#     size = doc._.sample_size;
#     for disease in doc._.disease_narrow:
#         disease = disease.lower().strip();
#         disease_counter[disease] += size;
        
# disease_top = disease_counter.most_common(25);
# disease_df = pd.DataFrame.from_records(disease_top, columns=["Terms", "Count"]);
# disease_graph = disease_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
# plt.xlabel("Epilepsy type (narrow)");
# plt.ylabel("Total number of patients from studies");
# disease_graph = plt.savefig("./figures/%s.png" % "DiseasesNarrowSample", bbox_inches="tight", dpi=300);

# Based on number of papers

disease_counter = Counter();
for (doc, context) in docs:
    for disease in doc._.disease_narrow:
        disease = disease.lower().strip();
        disease_counter[disease] += 1;
        
disease_top = disease_counter.most_common(25);
disease_df = pd.DataFrame.from_records(disease_top, columns=["Terms", "Count"]);
disease_graph = disease_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5));
plt.xlabel("Epilepsy type (narrow)");
plt.ylabel("Total number of papers");
disease_graph = plt.savefig("./figures/%s.png" % "DiseasesNarrowPaper", bbox_inches="tight", dpi=300);

#%% Pyvis - Screening by node size

edges = Counter();
nodes_modalities = Counter();
nodes_locations = Counter();
nodes_diseases = Counter();
# nodes_parameters = Counter();

# Store edges separately from nodes since you can characherize nodes by type if separated 
for (doc, context) in docs:
    node_list = set();
    # Modality
    for modality in doc._.modality:
        modality = modality.lower().strip();
        nodes_modalities[modality] += 1;
        node_list.add(modality);
    
    # Location 
    for location in doc._.targets_text:
        location = location.lower().strip();
        nodes_locations[location] += 1;
        node_list.add(location);
    # Diseases - narrow
    for disease in doc._.disease_narrow:
        disease = disease.lower().strip();
        nodes_diseases[disease] += 1;
        node_list.add(disease);
    
    # Parameters
    # Interconnect all nodes of a document together collected within a document per loop
    node_list = [*node_list];
    for (i, node_start) in enumerate(node_list): # Enumerate used to simulatenously return index for loop
        for (j, node_end) in enumerate(node_list[i+1:]): # To prevent repeating already enumarated nodes
            edges[(node_start, node_end)] += 1;
            edges[(node_end, node_start)] += 1;


# Add nodes and edges
net = Network();

# Add each type of node separately as a color 
for modality in nodes_modalities:
    net.add_node(modality, color = "red", size = nodes_modalities[modality], mass = math.log(nodes_modalities[modality], 2)); # Need to scale down in order to match sizes of locations
    # Number argument at end is for scaling factor (?)
    
for location in nodes_locations:
    if nodes_locations[location] >= 10: # Only add the node if there is 10 or more hits 
        net.add_node(location, color = "green", size = nodes_locations[location], mass = math.log(nodes_modalities[modality], 2));
        
for disease in nodes_diseases:
    if nodes_diseases[disease] >= 10: # Only add the node if there is 10 or more hits 
        net.add_node(disease, color = "blue", size = nodes_diseases[disease], mass = math.log(nodes_modalities[modality], 2));
    
for (node1, node2) in edges:
    if (nodes_locations[node1] >= 10 or nodes_modalities[node1] >= 10 or nodes_diseases[node1] >= 10) \
        and (nodes_locations[node2] >= 10 or nodes_modalities[node2] >= 10 or nodes_diseases[node2] >= 10):
        # Need to check nodes in both sets of counters
        # 10 as threshold for visualization of node 
        net.add_edge(node1, node2, width = math.log(edges[(node1, node2)],4)); # Width needs to be scaled

net.toggle_physics(True);
# net.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65); # For smaller graphs 
net.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100); # For larger graphs 
net.show_buttons(filter_=['physics']);
net.show("Network.html");

#%% Diagnostics 

df = pd.DataFrame(columns = ["TEXT", "SIZE", "FREQMAX", "VOLTMAX", "AMPMAX"]);
for (doc, context) in docs:
    text = doc.text;
    size = doc._.sample_size;
    # Frequency
    container = doc._.frequency;
    x = [];
    if container["fl"] != []:
        for fl in container["fl"]:
            x.append(float(fl));
    if container["rg"] != []:
        for (low, high) in container["rg"]:
            x.append(float(low)); # Takes the average of the min and max of the range
            x.append(float(high));
    if container["cp"] != []:
        for (op, num) in container["cp"]:
            x.append(float(num)); # Takes specified number in the comparator expression, will have to refine later
    freq = None;
    if x != []:
        amp = max(x);
    
    # Voltage
    container = doc._.voltage;
    x = [];
    if container["fl"] != []:
        for fl in container["fl"]:
            x.append(float(fl));
    if container["rg"] != []:
        for (low, high) in container["rg"]:
            x.append(float(low)); # Takes the average of the min and max of the range
            x.append(float(high));
    if container["cp"] != []:
        for (op, num) in container["cp"]:
            x.append(float(num)); # Takes specified number in the comparator expression, will have to refine later
    volt = None;
    if x != []:
        volt = max(x);
    
    # Amperage
    container = doc._.amperage;
    x = [];
    if container["fl"] != []:
        for fl in container["fl"]:
            x.append(float(fl));
    if container["rg"] != []:
        for (low, high) in container["rg"]:
            x.append(float(low)); # Takes the average of the min and max of the range
            x.append(float(high));
    if container["cp"] != []:
        for (op, num) in container["cp"]:
            x.append(float(num)); # Takes specified number in the comparator expression, will have to refine later
    amp = None
    if x != []:
        amp = max(x);
    df = df.append({"TEXT": text, "SIZE": size, "FREQMAX": freq, "VOLTMAX": volt, "AMPMAX": amp}, ignore_index=True);

df.to_csv("Embase Data Diagnostics.csv");


#%% Reference


# x = []
# y = []
# s = []

# for (index, row) in df.iterrows():
#     if row["FLOAT"] != []:
#         for fl in row["FLOAT"]:
#             x.append(float(fl))
#             y.append(0)
#             s.append(row["PATIENTS"]*30)
#     if row["RANGE"] != []:
#         for (low, high) in row["RANGE"]:
#             x.append((float(low)+float(high))/2) # Takes the average of the min and max of the range 
#             y.append(0)
#             s.append(row["PATIENTS"]*30)
#     if row["COMPARATOR"] != []:
#         for (op, num) in row["COMPARATOR"]:
#             x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
#             y.append(0)
#             s.append(row["PATIENTS"]*30)

# plot = plt.scatter(x,y,s=s, alpha = 0.15) # Alpha to set transparency, otherwise points are overlapping
# # plot = plt.xlim(-10,160) # Manually set axes
# plot = plt.ylim(-1,1)
# plot = plt.savefig("./figures/"+input("Figure title: ")+".png", bbox_inches="tight", dpi=300) # Save function has to be called before the show() function
# plot = plt.show()


