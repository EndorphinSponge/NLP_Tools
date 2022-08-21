import os



DIRPATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRPATH)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if 1:
    ROOT_PATH = "data/test2/Data.xls"
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]
    THRESH = 5
    
    if 0:
        pass
    
        
    from components_diseases import visHeatmapParams
    visHeatmapParams(F"{ROOT_NAME}_userdata_kw.csv",
                    col="frequency",
                    col_sampsize="sample_size")
    
    from models_spacy import GeneralExtractor
    extractor = GeneralExtractor()
    extractor.addPipeSampleSize()
    extractor.addPipeNmParams()
    extractor.addPipeCnsLocs()
    extractor.parseCorpora(ROOT_PATH, "AB", export_userdata=True, skiprows=1) # *_userdata.csv
    
    from components_diseases import procKeywordsEpilep
    procKeywordsEpilep(F"{ROOT_NAME}_userdata.csv") # *_userdata_kw.csv
    
    GeneralExtractor.convColsToStmts(F"{ROOT_NAME}_userdata_kw.csv",
                              cols=["cns_locs", "modalities", "diseases_broad"]) # *_userdata_kw_entsF.csv
    
    from graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.popCountersMulti(f"{ROOT_NAME}_userdata_kw_entsF.csv")
    builder.buildGraph(thresh=THRESH, multidi=False)
    builder.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    visualizer = GraphVisualizer(f"{ROOT_NAME}_userdata_kw_entsF_t{THRESH}.xml")
    visualizer.renderBarGraph(ent_types=["cns_locs", "modalities"])
    visualizer.genRenderArgs()
    visualizer.genLegend()
    visualizer.renderGraphNX() # *_gpt3_t{int}_net(<rendering info>).png
    visualizer.renderGraphPyvis() # *_gpt3_t{int}_pyvis.html, ONLY WORKS with undirected Graphs

        
    




if 0: # Full GPT3/JUR1 entity detection pipeline example using test.xlsx
    ROOT_PATH = "data/test/test.xlsx"
    MODEL = "gpt3"
    THRESH = 2
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]
    RUN_CLOUD = 0 # Extra guard against running cloud model
    
    if RUN_CLOUD:
        from models_api import CloudModel
        cloudmodel = CloudModel(ROOT_PATH)
        cloudmodel.mineTextGpt3() # *_gpt3R.xlsx
        cloudmodel.exportOutputFormatted(MODEL) # *_gpt3F.xlsx
    
    from models_spacy import EntityExtractor
    from abrvs_syns import refineAbrvs
    localmodel = EntityExtractor()
    localmodel.extractEnts(f"{ROOT_NAME}_{MODEL}F.xlsx") # *_gpt3F_entsR.xlsx
    localmodel.extractAbrvCont(ROOT_PATH) # *_abrvs.json
    print(localmodel.empty_log)
    refineAbrvs(f"{ROOT_NAME}_abrvs.json") # *_abrvs_rfn.json, *_abrvs_trans.json
    
    from graph_builder import EntProcessor
    from components_diseases import TBICore
    core = TBICore(abrv_path=f"{ROOT_NAME}_abrvs_rfn.json",
                   common_trans_path=f"{ROOT_NAME}_abrvs_trans.json")
    processor = EntProcessor(ent_processor_core=core)
    processor.procDfEnts(f"{ROOT_NAME}_{MODEL}F_entsR.xlsx") # *_gpt3F_entsF.xlsx
    processor.printLogs()
    
    from graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.popCountersMulti(f"{ROOT_NAME}_{MODEL}F_entsF.xlsx")
    builder.buildGraph(thresh=THRESH, multidi=True)
    builder.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    visualizer = GraphVisualizer(f"{ROOT_NAME}_{MODEL}F_entsF_t{THRESH}.xml")
    visualizer.renderBarGraph(ent_types=["factor", "outcome"])
    visualizer.genRenderArgs()
    visualizer.genLegend()
    visualizer.renderGraphNX() # *_gpt3_t{int}_net(<rendering info>).png
    visualizer.renderGraphPyvis() # *_gpt3_t{int}_pyvis.html, ONLY WORKS with undirected Graphs
    
# 0 guard to keep linting active without running code
if 0: # Pipeline to render graphs for each topic 
    ROOT_PATH = "data/gpt3_output.xlsx"
    SUFFIX = "_topics"
    MODEL = "gpt3"
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]
    
    from graph_builder import EntProcessor, GraphBuilder
    from graph_renderer import GraphVisualizer
    
    for i in range(11):
        
        THRESH = 3
        if i > 4: # Topic 6 (Index 5) and above 
            THRESH = 2 
        if i > 5: # Topic 7 (Index 6) and above 
            THRESH = 1

        builder = GraphBuilder()
        builder.popCountersMulti(f"{ROOT_NAME}_{MODEL}F_entsF{SUFFIX}.xlsx",
                                 col_sub="Topic", subset=i)
        builder.buildGraph(thresh=THRESH, multidi=True)
        builder.exportGraph(f"{ROOT_NAME}_{MODEL}F_entsF{SUFFIX}{i}_t{THRESH}.xml") # *_gpt3F_entsF_t{int}.xml
        
        visualizer = GraphVisualizer(f"{ROOT_NAME}_{MODEL}F_entsF{SUFFIX}{i}_t{THRESH}.xml")
        visualizer.genRenderArgs()
        visualizer.genLegend()
        if i == 0:
            title = f"Network graph of factors (purple nodes) and outcomes (pink nodes) and associations between them \n"
            title += f"extracted over topic {i+1} (Markers for TBI of greater severity)"
        elif i == 1:
            title = f"Network graph of factors (purple nodes) and outcomes (pink nodes) and associations between them \n"
            title += f"extracted over topic {i+1} (Serum markers of neuronal damage/inflammation)"
        elif i == 2:
            title = f"Network graph of factors (purple nodes) and outcomes (pink nodes) and associations between them \n"
            title += f"extracted over topic {i+1} (Markers for TBI of lesser severity)"
        else: 
            title = "Network graph of factors (purple nodes) and outcomes (pink nodes) and associations between them"
        visualizer.renderGraphNX(title, adjust_shell=True)

if 0: # Full GPT3/JUR1 entity detection pipeline example using test.xlsx
    ROOT_PATH = "data/test/test.xlsx"
    MODEL = "gpt3"
    THRESH = 2
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]
    RUN_CLOUD = 0 # Extra guard against running cloud model
    

    
    from graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.popCountersMulti(f"{ROOT_NAME}_{MODEL}F_entsF.xlsx")
    builder.buildGraph(thresh=THRESH, multidi=True)
    builder.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    visualizer = GraphVisualizer(f"{ROOT_NAME}_{MODEL}F_entsF_t{THRESH}.xml")
    visualizer.renderBarGraph(ent_types=["factor", "outcome"])
    visualizer.genRenderArgs()
    visualizer.genLegend()
    visualizer.renderGraphNX() # *_gpt3_t{int}_net(<rendering info>).png
    visualizer.renderGraphPyvis() # *_gpt3_t{int}_pyvis.html, ONLY WORKS with undirected Graphs