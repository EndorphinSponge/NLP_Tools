#%% Imports 
# General
from cgitb import text
from collections import Counter
from math import log
import os, re
from typing import Union, Hashable

# Data science
import numpy as np
from pyvis.network import Network
import networkx as nx
from networkx.classes.reportviews import NodeView
from networkx import Graph, MultiDiGraph, DiGraph, MultiGraph
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from adjustText import adjust_text

# Adjusttext package to resolve overlapping labels https://adjusttext.readthedocs.io/en/latest/Examples.html

# Internals
from internals import LOG

#%% Constants

#%% Classes & Functions

        
class GraphVisualizer:
    """
    Contains 
    """
    def __init__(self, graphml_path: Union[str, bytes, os.PathLike]):
        self.graph: Union[Graph, MultiDiGraph] = nx.read_graphml(graphml_path)
        self.graph_root_name = os.path.splitext(graphml_path)[0]
        self.nodes: list[Hashable] = list(self.graph.nodes)
        self.edges: list[tuple[Hashable, Hashable]] = list(self.graph.edges)
        
        self.fig_size: int
        self.scaling: float
        self.true_node_sizes: list[int]
        self.node_sizes: list[int]
        self.node_colors: list[str] # List of hexacedimal color strings
        self.true_edge_widths: list[int]
        self.edge_widths: list[float]
        self.edge_widths_alpha: list[float]
        self.edge_probs: list[float]
        self.edge_zeroes: list[int]
        
        self.args_log: str
        self.legend: dict[str, list[Union[str, PathCollection]]] # Property for storing legend points and their labels


    def importGraph(self, graphml_path):
        """
        Imports a new NX graph from an XML file into the object and replaces old
        """
        # Import graph https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.read_graphml.html#networkx.readwrite.graphml.read_graphml
        self.graph = nx.read_graphml(graphml_path)
        self.graph_root_name: str = os.path.splitext(graphml_path)[0]

    def resetGraph(self,):
        self.graph = nx.Graph()
        
    def genRenderArgs(self,
                         fig_size = 10, width_log = 2, width_min = 0.2,
                         alpha_max = 0.95, alpha_min = 0.01, alpha_root = 1, 
                         ):
        
        self.args_log = f"(width[log{str(width_log)}_min{str(width_min)}]alpha[max{str(alpha_max)}min{str(alpha_min)}root{str(alpha_root)}])"
        self.r_args = dict()
        self.fig_size = fig_size
        
        # NOTE THAT AUTOSCALING ONLY WORKS WHEN SAVING FIGURE, doesn't work with matplotlib display
        node_sizes_true = [size for (node, size) in self.graph.nodes(data="size")]
        self.true_node_sizes = node_sizes_true
        
        scaling = fig_size*40/max(node_sizes_true) # Use max node value 
        self.scaling = scaling
        
        node_sizes = [scaling*size for (node, size) in self.graph.nodes(data="size")]
        self.node_sizes = node_sizes
        
        edge_width_true = [width for (node1, node2, width) in self.graph.edges(data="width")]
        self.true_edge_widths = edge_width_true
        
        node_colors = []
        for node, data in self.graph.nodes(data=True):
            if data["ent_type"] == "factor":
                node_colors.append("#8338ec")
            elif data["ent_type"] == "outcome":
                node_colors.append("#f72585")
            else:
                node_colors.append("#000000")
        self.node_colors = node_colors
        
        
        edge_widths = [log(scaling*width, width_log) for width in edge_width_true]
        edge_widths = np.clip(edge_widths, width_min, None) # Set lower bound of width to 1
        self.edge_widths = edge_widths
        
        edge_alphas = [alpha_max*(width/max(edge_width_true))**(1/alpha_root) for width in edge_width_true] # Scaled to max width times 0.7 to avoid solid lines, cube root if you want to reduce right skewness 
        edge_alphas = np.clip(edge_alphas, alpha_min, None) # Use np to set lower bound for edges
        self.edge_widths_alpha = edge_alphas # For manually width to transparency 
        
        edge_probs = [prob for (node1, node2, prob) in self.graph.edges(data="prob")]
        self.edge_probs = edge_probs
        
        edge_zeroes = [0 for i in edge_width_true]
        self.edge_zeroes = edge_zeroes # Array with zero for each edge
        
    def genLegend(self):
        def _roundNum(num, base):
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
        
        # Draw legend: https://stackoverflow.com/questions/29973952/how-to-draw-legend-for-scatter-plot-indicating-size
        # Same scaling factor but different rounding thresholds
        scaling = self.scaling
        color = '#8338ec'
        max_original = max(self.node_sizes)/scaling # Get original max node size by dividing by scaling factor
        l1 = _roundNum(0.02*max_original, 5) # Reference of 5 for max of 250
        l2 = _roundNum(0.08*max_original, 10) # Reference of 20 for max of 250 
        l3 = _roundNum(0.4*max_original, 20) # Reference of 100 for max of 250
        l4 = _roundNum(max_original, 50) # Reference of 250 for max of 250
        p1 = plt.scatter([],[], s=l1*scaling, marker='o', color=color, alpha = 0.8) # Set actual sizes to match scaled rendered nodes
        p2 = plt.scatter([],[], s=l2*scaling, marker='o', color=color, alpha = 0.8)
        p3 = plt.scatter([],[], s=l3*scaling, marker='o', color=color, alpha = 0.8)
        p4 = plt.scatter([],[], s=l4*scaling, marker='o', color=color, alpha = 0.8)

        self.legend = {"points": [p1, p2, p3, p4], "labels": [l1, l2, l3, l4]}

    def renderGraphNX(self, title = "Figure", save_suffix = "", dpi = 800, 
                      display = False, cmap = True, arrows = False,
                      adjust_shell = True,
                      ):
        """
        Renders the graph contained within the object using NX
        ----
        save_prefix: prefix for saving figures
        cmap: use color mapping in the stead of transparency 
        """
       
        layout: dict[Hashable, tuple[float, float]] = nx.shell_layout(self.graph) # Different position solvers available: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html
        
        fig_size = self.fig_size
        
        # Networkx visualization (multiple elements)
        # nx uses matplotlib.pyplot for figures, can use plt manipulation to modify size
        sns.set_theme()
        plt.figure(figsize=(fig_size*1.1, fig_size), dpi=dpi)
        plt.title(title)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, 
            pos = layout,
            alpha = 0.8,
            node_size = self.node_sizes,
            node_color = self.node_colors,
            )
        # Manually draw labels with different sizes: https://stackoverflow.com/questions/62649745/is-it-possible-to-change-font-sizes-according-to-node-sizes
        labels_adjust = []
        y_coords = [y for node, (x, y) in layout.items()]
        upper_bound = np.percentile(y_coords, 90)
        lower_bound = np.percentile(y_coords, 10)
        for node, (x, y) in layout.items():
            label_size = log(self.scaling*self.graph.nodes[node]["size"], 2) # Retrieve size information via node identity in graph
            label = plt.text(x, y, node, fontsize = label_size, ha = "center", va = "center", alpha = 0.7) # Manually draw text
            if y > upper_bound or y < lower_bound: # Only adjust labels that are outside of bounds
                labels_adjust.append(label)
        if adjust_shell:
            adjust_text(labels_adjust,
                        autoalign='',
                        only_move={'points':'y', 'text':'y'},
                        arrowprops=dict(arrowstyle="-", color="black"),
                        )

        # Plot legend
        plt.legend(self.legend["points"], self.legend["labels"],
            scatterpoints=1, ncol=4,
            title="Number of articles reporting factor/outcome", title_fontsize=fig_size,
            loc='lower left', prop={'size': fig_size},borderpad = 0.8,
            )
        # scatterpoints = number of points in each size demo
        # ncol = number of columns that each size demo will be split into
        # prop = container deciding properties of text relating to the size demos
        # 10 is default font size
        
        plt_cmap = plt.cm.Wistia
        plt_cmap = plt.cm.summer
        
        widths = self.true_edge_widths
        if max(widths) <= 10: # If range is less than 10, need to use static transparency
            alphas = [0.95*(width/max(widths))**(2) for width in widths] # Right skew data to make lighter lines easier to see
        else:
            alphas = self.edge_widths_alpha
        
        if cmap: # Will map values (in proportion to min/max) to a color spectrum
            rendered_edges = nx.draw_networkx_edges(self.graph,
                pos=layout,
                alpha=alphas, # Can add transparency on top to accentuate
                width=self.edge_widths,
                edge_color=self.edge_widths_alpha, # Map color to transparency (calculated based on true widths)
                edge_cmap=plt_cmap, # Colors edges but doesn't generate colorbar scale legend
                arrows=arrows
                )
            cbar_edges = nx.draw_networkx_edges(self.graph, # Dummy variable for if color assignment is cube rooted, (transparency set to zero)
                pos=layout,
                alpha=self.edge_zeroes, # Array of zeroes
                edge_color=self.true_edge_widths, # NOTE THAT THIS IS NOT EXACTLY THE SAME SCALE (due to cube root)
                edge_cmap=plt_cmap, 
                arrows=False, # Need to disable arrows, otherwise draw_edges method returns a FancyArrowPatch for every edge, unable to
                )
            # Colorbar legend solution: https://groups.google.com/g/networkx-discuss/c/gZmr-YgvIQs
            # Alternative solution using FuncFormatter here: https://stackoverflow.com/questions/38309171/colorbar-change-text-value-matplotlib
            plt.sci(cbar_edges) # Set current image to edges created by nx 
            colorbar = plt.colorbar() # Creats actual colorbar legend with ticks coresponding to true_edge_widths
            colorbar.set_label("Number of articles reporting association")
            # Available colormaps: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
            # Tested colormaps: GnBu is too similar to node color scheme, [YlOrRd, PuRd, Wistia] makes small edges too light, 
        else:
            nx.draw_networkx_edges(self.graph,
                pos=layout,
                alpha=self.edge_probs,
                width=self.edge_widths,
                )
            
        root_name = self._getSimplifiedName()
            
        if display:
            plt.show()
        else:
            output_name = f"{root_name}{save_suffix}_net{self.args_log}.png"
            plt.savefig(output_name, bbox_inches='tight')
            print("Exported rendered graph to", output_name)
    

    def renderGraphPyvis(self, solver = "repulsion"):
        """
        Builds graph from counters and renders it using Pyvis
        """
        graphpy = Network()
        
        if type(self.graph) in [MultiDiGraph, DiGraph, MultiGraph]: # Need convertion to simple Graph for pyvis to work
            graph = self.graph.to_undirected()
            LOG.warning("Imported graph was not a simple Graph type, may not render correctly")
            # FIXME find way to convert to Graph whie enabling visualization
        else:
            graph = self.graph
            
        graphpy.from_nx(graph)

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
        output_name = self._getSimplifiedName() + "_pyvis.html"
        graphpy.show(output_name)
        print(f"Exported rendered graph to {output_name}")

    def _getSimplifiedName(self):
        # Function for returning a simplified version of the graph's name 
        if re.search(R"F_entsF_t\d+$", self.graph_root_name): # If it has matched naming conventions (remember to stay model agnostic)
            thresh = re.search(R"F_entsF(_t\d+)$", self.graph_root_name).group(1)
            root_name = re.sub(R"F_entsF_t\d+$", "", self.graph_root_name) # Remove annotations to leave root E.g., test_gpt3 as root
            return root_name + thresh # Add threshold annotation to root (can't use lookahead with variable length search)
        else:
            return self.graph_root_name 
        
    def renderBarGraph(self, ent_types:list[str], top_n=15):
        if type(self.graph) == DiGraph:
            for ent_type in ent_types:
                if len(self.graph.nodes) < top_n:
                    top_n = len(self.graph.nodes) # Re-assign top_n to length of nodes
                alt_ent_type = [t for t in ent_types if t != ent_type][0] # Get the other item in a list

                nodes = [node for node, data in self.graph.nodes(data=True) 
                         if data["ent_type"] == ent_type]
                
                nodes_counts: list[tuple[Hashable, int]] = list(self.graph.nodes(data="size"))
                nodes_counts = [(node, data["size"])for node, data in self.graph.nodes(data=True) 
                                if data["ent_type"] == ent_type]
                nodes_counts.sort(key=lambda x: x[1]) # Sort by count
                nodes_counts = nodes_counts[-top_n:] # Take slice starting from end (since hbar plots from bottom of y-axis)
                
                degrees_raw: list[tuple[Hashable, int]] = list(self.graph.in_degree(nodes))
                degrees_raw.sort(key=lambda x: x[1]) # Sort by degree
                degrees_raw = degrees_raw[-top_n:] 
                
                degrees_weighted: list[tuple[Hashable, int]] = list(self.graph.in_degree(nodes, weight="width"))
                degrees_weighted.sort(key=lambda x: x[1]) 
                degrees_weighted = degrees_weighted[-top_n:] 
                
                sns.set_theme()
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig: Figure
                ax1: Axes
                ax2: Axes
                
                fig.set_size_inches(15, 10) # set_size_inches(w, h=None, forward=True)
                
                ax1.barh([p[0] for p in nodes_counts],
                    [p[1] for p in nodes_counts])
                ax1.set_xlabel(f"Number of articles that report association of the {ent_type} with a {alt_ent_type}")
                ax1.set_ylabel(ent_type.capitalize())
                
                ax2.set_yticklabels([]) # Hide the left y-axis tick-labels
                ax2.set_yticks([]) # Hide the left y-axis ticks
                ax2.invert_xaxis() # Invert x-axis
                ax2t = ax2.twinx() # Create twin x-axis
                ax2t.barh([p[0] for p in degrees_raw],
                    [p[1] for p in degrees_raw])
                ax2t.set_ylabel(ent_type.capitalize())
                ax2.set_xlabel(f"Number of unique {alt_ent_type}s associated with {ent_type}")
                
                if ent_type == "outcome":
                    fig.suptitle(f"Top {str(top_n)} TBI prognosis outcome measures over entire corpora")
                elif ent_type == "factor":
                    fig.suptitle(f"Top {str(top_n)} TBI prognostic factors over entire corpora")
            
            
        else: # Undirected graph parsing 
            LOG.warning("Undirected graph detected, this pipeline has not been built yet")
            print(self.graph.degree())
            print(self.graph.degree(weight="width"))
            pass

    
    def renderScatter(self):
        x = self.true_edge_widths
        y = self.edge_probs
        labels = self.edges
        
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 15)
        ax.scatter(x, y)
        ax.set_xlabel("Number of articles supporting association between factor and outcome")
        ax.set_ylabel("Specificity of the association")
        ax.set_title("Relationship between specificity of a factor-outcome/outcome-factor relationship and how often the relationship is reported", fontsize=15, pad=15)
        
        for i, label in enumerate(labels):
            annotation = f"{label[0]} -> {label[1]}"
            if x[i] > 25 or y[i] > 0.45: # Only label extreme points
                ax.annotate(annotation, (x[i], y[i]))


if __name__ == "__main__": # For testing
    a = GraphVisualizer("data/gpt3_output_gpt3F_entsF_t10.xml")
    a = GraphVisualizer("data/gpt3_output_gpt3F_entsF_topics2_t3.xml")
    a.genRenderArgs()
    a.genLegend()
    # a.renderScatter()
    # a.renderBarGraph(ent_types=["factor", "outcome",])
    title = "Network graph of factors (purple nodes) and outcomes (pink nodes) and \nassociations between them extracted over entire corpora of 1412 abstracts"
    title = f"Network graph of factors (purple nodes) and outcomes (pink nodes) and associations between them \n"
    title += f"extracted over topic 3 (Markers for TBI of lesser severity)"

    a.renderGraphNX(title, adjust_shell=True)




# %%
