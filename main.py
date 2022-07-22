#%% Imports
import pandas as pd

# Internal 
from graph_builder import extractAbrvCont, GraphBuilder

#%% Build graphs from output
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
