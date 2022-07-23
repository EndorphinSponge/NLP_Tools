#%%
from global_functions import mergeDfSlices
mergeDfSlices("gpt3raw", "test")

#%% Converting text to JSON strings within each row
import json

import pandas as pd
from pandas import DataFrame


df: DataFrame = pd.read_excel("testdata.xlsx", engine="openpyxl")
df_extracted = DataFrame()
# print(df["Extracted_Text"])
for index, row in df.iterrows():
    print(index)
    row_text: str = row["Extracted_Text"]
    processed_text = json.dumps(row_text.split("|"))
    print(processed_text)
    new_row = DataFrame({"testcol": [processed_text]})
    df_extracted = pd.concat([df_extracted, new_row])
df_extracted = df_extracted.reset_index(drop = True) # Drop variable avoids adding old index as a column
df = pd.concat([df, df_extracted], axis = 1) # Concat on columns instead of rows
df.to_excel("testdataOUT.xlsx")



# %%
