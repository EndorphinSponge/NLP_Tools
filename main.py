import os

DIRPATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRPATH)



   
if False: # Full pipeline using test.xlsx
    from models_api import CloudModel
    fetcher = CloudModel("test/test.xlsx")
    fetcher.mineTextGpt3() # *_gpt3R.xlsx
    fetcher.exportOutputFormatted("gpt3") # *_gpt3F.xlsx
    
    from models_spacy import SpacyModelTBI, refineAbrvs
    model = SpacyModelTBI()
    model.extractEntsTBI(R"test/test_gpt3F.xlsx") # *_gpt3F_entsR.xlsx
    model.extractAbrvCont(R"test/test.xlsx") # *_abrvs.json
    print(model.empty_log)
    refineAbrvs("test/test_abrvs.json") # *_abrvs_rfn.json, *_abrvs_trans.json
    
    from graph_builder import EntProcessor, GraphBuilder
    a = EntProcessor()
    a.procDfEnts("test/test_gpt3F_entsR.xlsx") # *_gpt3F_entsF.xlsx
    a.printLogs()
    b = GraphBuilder()
    b.popCountersMulti("test/test_gpt3F_entsF.xlsx")
    thresh = 2
    b.buildGraph(thresh=thresh)
    b.exportGraph() # *_gpt3F_entsF_t{int}.xml
    
    from graph_renderer import GraphVisualizer
    c = GraphVisualizer(f"test/test_gpt3F_entsF_t{str(thresh)}.xml")
    c.renderGraphNX() # *_gpt3_t{int}_net(<rendering info>).png

    


