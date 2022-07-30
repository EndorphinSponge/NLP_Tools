#%% Imports
import os
import pandas as pd

# Internal 
from models_api import CloudModel
from graph_builder import extractAbrvCont, GraphBuilder
from graph_renderer import GraphVisualizer






#%% Build graphs from output
if __name__ == "__main__" and True: 
    DIRPATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(DIRPATH)
    df_origin = pd.read_excel("testdata.xlsx", engine='openpyxl') # For colab support after installing openpyxl for xlsx files
    abrvs = extractAbrvCont(df_origin, col_input = "Extracted_Text")
    builder = GraphBuilder(abrvs)
    builder.popCountersMulti(df_origin, col = "Extracted_Text")
    builder.buildGraph(thresh = 1)
    builder.exportGraph(f"tbi_ymcombined_t1_graph.xml")
    renderer = GraphVisualizer("tbi_ymcombined_t1_graph.xml")
    renderer.renderGraphNX(save_prefix = f"tbi_ymcombined_t1", alpha_root = 3, cmap = True)
