from internal_globals import importData
import pandas as pd
import os

if True:
    from models_spacy import SpacyModelTBI
    model = SpacyModelTBI()
    model.extractAbrvCont(R"test\test_fmt.xlsx")

if False:
    from models_spacy import checkAbrvs
    checkAbrvs("test_fmt_abrvs.json")
    

if False:
    from models_spacy import SpacyModelTBI

    nlpmodel = SpacyModelTBI("en_core_sci_scibert")
    print(nlpmodel.model)
    nlpmodel.extractEntsTBI(R"test\test_fmt.xlsx", "Model_output", "Ents")


if False:
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
