#%% 
from spacy.tokens import Doc
from typing import List, Tuple

from math import log

import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
from pyvis.network import Network
import pandas as pd
        
#%% Frequency
class Visualizer:
    
    def __init__(self, docs: List[Tuple[Doc, str]]):
        self.docs = docs
    

    def visHeatmap(self, extension: str, xlabel: str):
        """
        Visualize numerical information contained inside a Doc custom extension using a heatmap

        extension: the exact identifier of the Doc extension containing the numberical information to be visualized
        xlabel: label for the x-axis of the output
        """
        x = []
        y = []
        s = []
        for (doc, context) in self.docs:
            if doc._.sample_size > 0:
                container = doc._.get(extension) # Obtains the value contained in the specified extension, also has corresponding Doc._.set method: https://spacy.io/usage/processing-pipelines#custom-components-attributes
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

        plt.figure(figsize=(15, 7.5))
        plot = plt.scatter(x,y,s=s, alpha = 0.15) # Alpha to set transparency, otherwise points are overlapping
        # plt.xlim(-10,800) # Manually set axes to exclude outliers
        # plt.ylim(-1,1)
        plt.xlabel(xlabel)
        plot.axes.get_yaxis().set_visible(False)
        plt.savefig(f"figures/{extension}.png", bbox_inches="tight", dpi=300) # Save function has to be called before the show() function

    def visBargraph(self, extension: str, xlabel: str, by_pts: bool=False):
        """
        by_pts: scale numbers by number of patients rather than number of papers
        """
        location_counter = Counter()
        for (doc, context) in self.docs:        
            size = by_pts*doc._.sample_size + (not by_pts)*1 # Branchless way to set sample size based on by_patients argument
            for location in doc._.get(extension):
                location = location.lower().strip()
                location_counter[location] += size
                
        location_top = location_counter.most_common(25)
        location_df = pd.DataFrame.from_records(location_top, columns=["Terms", "Count"])
        location_graph = location_df.plot(kind="bar", x="Terms", y="Count", figsize=(10,5))
        plt.xlabel(xlabel)
        plt.ylabel("Total number of" + by_pts*"patients from studies" + (not by_pts)*"papers") # Branchless way to set y-axis label based on by_patients
        graph_name = f"{extension}_" + by_pts*"patients" + (not by_pts)*"papers" # Branchless naming of file
        location_graph = plt.savefig(f"figures/{graph_name}.png", bbox_inches="tight", dpi=300) 

    def visPyvis(self, min_count: int=10):
        """
        Visualize relationships between modality, targets, and type of disease within docs
        min_count: minimum number of hits for node in order for it to be included within the network
        """
        edges = Counter()
        nodes_modalities = Counter()
        nodes_locations = Counter()
        nodes_diseases = Counter()
        # nodes_parameters = Counter()
        for (doc, context) in self.docs:
            node_list = set()
            # Modality
            for modality in doc._.modality:
                modality = modality.lower().strip()
                nodes_modalities[modality] += 1
                node_list.add(modality)
            
            # Location 
            for location in doc._.targets_text:
                location = location.lower().strip()
                nodes_locations[location] += 1
                node_list.add(location)
            # Diseases - narrow
            for disease in doc._.disease_narrow:
                disease = disease.lower().strip()
                nodes_diseases[disease] += 1
                node_list.add(disease)
            
            # Parameters
            
            node_list = [*node_list]
            for (i, node_start) in enumerate(node_list):
                for (j, node_end) in enumerate(node_list[i+1:]):
                    edges[(node_start, node_end)] += 1
                    edges[(node_end, node_start)] += 1

        net = Network()

        for modality in nodes_modalities:
            if nodes_modalities[modality] >= min_count: # Only add the node if there is min_count or more hits 
                net.add_node(modality, color = "red", 
                    size = log(nodes_modalities[modality], 1.2), 
                    mass = log(nodes_modalities[modality], 2)
                    ) # Need to scale down in order to match sizes of locations
            
        for location in nodes_locations:
            if nodes_locations[location] >= min_count: # Only add the node if there is min_count or more hits 
                net.add_node(location, color = "green", 
                    size = log(nodes_locations[location], 1.2), 
                    mass = log(nodes_modalities[modality], 2)
                    )
                
        for disease in nodes_diseases:
            if nodes_diseases[disease] >= min_count: # Only add the node if there is min_count or more hits 
                net.add_node(disease, color = "blue",
                    size = log(nodes_diseases[disease], 1.2), 
                    mass = log(nodes_modalities[modality], 2)
                    )
            
        for (node1, node2) in edges:
            if (nodes_locations[node1] >= min_count 
                or nodes_modalities[node1] >= min_count 
                or nodes_diseases[node1] >= min_count
                ) \
                and (nodes_locations[node2] >= min_count 
                or nodes_modalities[node2] >= min_count 
                or nodes_diseases[node2] >= min_count
                ):
                # Need to check nodes in both sets of counters
                net.add_edge(node1, node2, width = log(edges[(node1, node2)],4)) # Width needs to be scaled

        net.toggle_physics(True)
        # net.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65) # For smaller graphs 
        net.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100) # For larger graphs 
        net.show_buttons(filter_=['physics'])
        net.show(f"figures/pyvis_thresh{min_count}.html")

   

if __name__ == "__main__":
    pass    
    #%% Diagnostics 

    df = pd.DataFrame(columns = ["TEXT", "SIZE", "FREQMAX", "VOLTMAX", "AMPMAX"])
    for (doc, context) in docs:
        text = doc.text
        size = doc._.sample_size
        # Frequency
        container = doc._.frequency
        x = []
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl))
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append(float(low)) # Takes the average of the min and max of the range
                x.append(float(high))
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
        freq = None
        if x != []:
            amp = max(x)
        
        # Voltage
        container = doc._.voltage
        x = []
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl))
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append(float(low)) # Takes the average of the min and max of the range
                x.append(float(high))
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
        volt = None
        if x != []:
            volt = max(x)
        
        # Amperage
        container = doc._.amperage
        x = []
        if container["fl"] != []:
            for fl in container["fl"]:
                x.append(float(fl))
        if container["rg"] != []:
            for (low, high) in container["rg"]:
                x.append(float(low)) # Takes the average of the min and max of the range
                x.append(float(high))
        if container["cp"] != []:
            for (op, num) in container["cp"]:
                x.append(float(num)) # Takes specified number in the comparator expression, will have to refine later
        amp = None
        if x != []:
            amp = max(x)
        df = df.append({"TEXT": text, "SIZE": size, "FREQMAX": freq, "VOLTMAX": volt, "AMPMAX": amp}, ignore_index=True)

    df.to_csv("Embase Data Diagnostics.csv")



