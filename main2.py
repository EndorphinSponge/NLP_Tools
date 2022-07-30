from internal_globals import importData
import pandas as pd
import os

DIRPATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRPATH)

if True:
    # from models_spacy import SpacyModelTBI
    # nlpmodel = SpacyModelTBI()
    # # nlpmodel.extractEntsTBI(R"test/gpt3_output_fmt.xlsx")
    # nlpmodel.extractAbrvCont(R"test/gpt3_output_fmt.xlsx")
    # print(nlpmodel.empty_log)
    
    # from models_spacy import refineAbrvs
    # refineAbrvs("test/gpt3_output_fmt_abrvs.json")
    
    from graph_builder import EntProcessor
    a = EntProcessor()
    a.procDfEnts("test/gpt3_output_fmt_ents.xlsx")
    a.printLogs()

if False:
    
    
    from graph_builder import EntProcessor
    a = EntProcessor()
    list_ents = [{"factor": ["mechanical ventilation"], "outcome": ["neurological"]}, {"factor": ["severity", "head injury"], "outcome": ["neurological"]}, {"factor": ["blood transfusion"], "outcome": ["neurological"]}, {"factor": ["neurosurgical intervention"], "outcome": ["neurological"]}, {"factor": ["mechanical ventilation"], "outcome": ["non-neurological", "complication"]}, {"factor": ["glasgow coma scale"], "outcome": ["non-neurological", "complication"]}, {"factor": ["blood transfusion"], "outcome": ["non-neurological", "complication", "neurosurgical intervention"]}, {"factor": ["injury", "concomitant"], "outcome": ["non-neurological", "complication", "gcs"]}]
    a._procEnts(list_ents)
    a._sepConfEnts(list_ents)
    a.printLogs()
    


    import time
    from models_api import CloudModel
    fetcher = CloudModel("tbi_ymcombined_subset25.csv")
    fetcher.mineTextGpt3((16, 19))
    fetcher.storeOutputFormatted("gpt3")

    print("Pause")
    time.sleep(30)
    print("Resume")

    importer = CloudModel("")
    importer.importRaw("tbi_ymcombined_subset25(16, 19)_gpt3raw.xlsx")
    importer.storeOutputFormatted("gpt3")
