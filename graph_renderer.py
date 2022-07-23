#%% Imports 
# General
from collections import Counter
from math import log
import os
from typing import Union

# Data science
import numpy as np
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt


#%% Constants

#%% Classes & Functions

        
class GraphVisualizer:
    """
    Contains 
    """
    def __init__(self, graphml_path: Union[str, bytes, os.PathLike]):
        self.graph = nx.read_graphml(graphml_path)


    def importGraph(self, path):
        """
        Imports a new NX graph from an XML file into the object and replaces old
        """
        # Import graph https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.read_graphml.html#networkx.readwrite.graphml.read_graphml
        self.graph = nx.read_graphml(path)
        return None

    def resetGraph(self,):
        self.graph = nx.Graph()

    def renderGraphNX(self, 
        width_log = 2, width_min = 0.2, 
        alpha_max = 0.95, alpha_min = 0.01, alpha_root = 1, 
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

def roundNum(num, base):
    """
    For creating integer numbers for legend and scales in visualization 
    Rounds a number to a base, will return 1/5 of the base if the number
    is less than half the base (i.e., rounding with min being half of base)
    """
    if num > base/2:
        return base * round(num/base)
    elif num <= base/2 and num > base/5: # Catch cases in between and round up (normally it would round down)
        return base
    else:
        return int(base/5)

#%% Rendering from exported XMLs
if __name__ == "__main__":
    DIR = r"figures\network_v0.9.1 (colorbar, node legend, no auto size)\exportXML"
    visualizer = GraphVisualizer()
    for i in range(0, 5): # Rendering with threshold = 1
        visualizer.importGraph(os.path.join(DIR, f"tbi_topic{i}_t3_graph.xml"))
        visualizer.renderGraphNX(save_prefix = f"tbi_topic{i}_t3_graph.xml", cmap = True)
    for i in range(5, 6): # Rendering with threshold = 2
        visualizer.importGraph(os.path.join(DIR, f"tbi_topic{i}_t2_graph.xml"))
        visualizer.renderGraphNX(save_prefix = f"tbi_topic{i}_t2_graph.xml", cmap = True)
    for i in range(6, 11): # Rendering with threshold = 3
        visualizer.importGraph(os.path.join(DIR, f"tbi_topic{i}_t1_graph.xml"))
        visualizer.renderGraphNX(save_prefix = f"tbi_topic{i}_t1_graph.xml", cmap = True)

