import os

F = False # To guard against running of test code while still keeping linting active

DIRPATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRPATH)

   

if 1:
    ROOT_PATH = "test/gpt3_output.xlsx"
    MODEL = "gpt3"
    THRESH = 15
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]

    from graph_builder import EntProcessor, GraphBuilder
    builder = GraphBuilder()
    builder.popCountersMulti(f"{ROOT_NAME}_{MODEL}F_entsF.xlsx")
    builder.buildGraph(thresh=THRESH, multidi=True)
    builder.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    visualizer = GraphVisualizer(f"{ROOT_NAME}_{MODEL}F_entsF_t{str(THRESH)}.xml")
    visualizer.genRenderArgs()
    visualizer.genLegend()
    visualizer.renderGraphNX()

if F: # Full pipeline example using test.xlsx
    ROOT_PATH = "test/test.xlsx"
    MODEL = "gpt3"
    THRESH = 2
    ROOT_NAME = os.path.splitext(ROOT_PATH)[0]
    
    from models_api import CloudModel
    cloudmodel = CloudModel(ROOT_PATH)
    cloudmodel.mineTextGpt3() # *_gpt3R.xlsx
    cloudmodel.exportOutputFormatted(MODEL) # *_gpt3F.xlsx
    
    from models_spacy import SpacyModelTBI, refineAbrvs
    localmodel = SpacyModelTBI()
    localmodel.extractEntsTBI(f"{ROOT_NAME}_{MODEL}F.xlsx") # *_gpt3F_entsR.xlsx
    localmodel.extractAbrvCont(ROOT_PATH) # *_abrvs.json
    print(localmodel.empty_log)
    refineAbrvs(f"{ROOT_NAME}_abrvs.json") # *_abrvs_rfn.json, *_abrvs_trans.json
    
    from graph_builder import EntProcessor, GraphBuilder
    from components_tbi import TBICore
    core = TBICore(abrv_path=f"{ROOT_NAME}_abrvs_rfn.json",
                   common_trans_path=f"{ROOT_NAME}_abrvs_trans.json")
    processor = EntProcessor(ent_processor_core=core)
    processor.procDfEnts(f"{ROOT_NAME}_{MODEL}F_entsR.xlsx") # *_gpt3F_entsF.xlsx
    processor.printLogs()
    
    builder = GraphBuilder()
    builder.popCountersMulti(f"{ROOT_NAME}_{MODEL}F_entsF.xlsx")
    builder.buildGraph(thresh=THRESH, multidi=True)
    builder.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    visualizer = GraphVisualizer(f"{ROOT_NAME}_{MODEL}F_entsF_t{str(THRESH)}.xml")
    visualizer.renderBarGraph(ent_types=["factor", "outcome"])
    visualizer.genRenderArgs()
    visualizer.genLegend()
    visualizer.renderGraphNX() # *_gpt3_t{int}_net(<rendering info>).png
    visualizer.renderGraphPyvis() # *_gpt3_t{int}_pyvis.html
    
    


