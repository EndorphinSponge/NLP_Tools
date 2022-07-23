
# %%
import time
from models_api import CloudModel
fetcher = CloudModel("tbi_ymcombined_subset25.csv")
fetcher.extractRawGpt3((16, 19))
fetcher.processRaw("gpt3")

print("Pause")
time.sleep(30)
print("Resume")

fetcher2 = CloudModel("")
fetcher2.importRaw("tbi_ymcombined_subset25(16, 19)_gpt3raw.xlsx")
fetcher2.processRaw("gpt3")

# %%
