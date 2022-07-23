#%% Imports
import pandas as pd

# Internal 
from graph_builder import extractAbrvCont, GraphBuilder
from graph_renderer import GraphVisualizer

#%%

#%% Build graphs from output
if __name__ == "__main__" and True: 
    df_origin = pd.read_excel("testdata.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
    abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
    builder = GraphBuilder(abrvs)
    builder.populateCounters(df_origin, col_input = "Extracted_Text")
    builder.buildGraph(thresh = 1)
    builder.exportGraph(f"tbi_ymcombined_t1_graph.xml")
    renderer = GraphVisualizer("tbi_ymcombined_t1_graph.xml")
    renderer.renderGraphNX(save_prefix = f"tbi_ymcombined_t1", alpha_root = 3, cmap = True)
