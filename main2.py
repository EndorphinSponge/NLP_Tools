from global_functions import importData
import pandas as pd
import os



if True:
    from models_spacy import SpacyModel

    nlpmodel = SpacyModel()
    print(nlpmodel.model)
    nlpmodel.importCorpora(R"test\test.xlsx", "Abstract", ["Title", "Tags", "Extracted"])
    for doc in nlpmodel.doclist:
        print(doc.user_data["Title"])
    nlpmodel.exportDocs()

    model2 = SpacyModel()
    print(model2.model)
    model2.importDocs(R"test\test(en_core_web_sm).spacy")
    for doc in model2.doclist:
        print(">", doc.user_data["Extracted"])


if False:
    import time
    from models_api import CloudModel
    fetcher = CloudModel("tbi_ymcombined_subset25.csv")
    fetcher.extractRawGpt3((16, 19))
    fetcher.processRaw("gpt3")

    print("Pause")
    time.sleep(30)
    print("Resume")

    importer = CloudModel("")
    importer.importRaw("tbi_ymcombined_subset25(16, 19)_gpt3raw.xlsx")
    importer.processRaw("gpt3")
