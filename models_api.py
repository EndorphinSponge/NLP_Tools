#%% Imports
# General 
import os
import json
import re

# Data Science
import pandas as pd

# NLP
import openai

from global_functions import importData
#%% Constants
openai.api_key = "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" # Personal
# openai.api_key = "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" # Throwaway
# openai.api_key = "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" # Sea
DATA = importData("data/tbi_ymcombined.csv")

#%% Execution ACTIVE

df_slice = DATA[0:10].reset_index(drop = True) # Start bound is included, end bound is not
df_new = pd.DataFrame() # Don't need to initiate columns since it is done automatically when adding entries
for index, row in df_slice.iterrows():
    print(index)
    abstract = row["Abstract"]
    extracted = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"A table summarizing each of the predictors, what they predict, and the total number of subjects in the study:\n\n{abstract}\n\n| Predictor | Outcome being predicted | Number of subjects |",
        temperature=0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    new_row = pd.DataFrame({"Extracted": [extracted]})
    df_new = pd.concat([df_new, new_row])
df_new = df_new.reset_index(drop = True) # Drop variable avoids adding old index as a column
df_slice = pd.concat([df_slice, df_new], axis = 1) # Axis 1 to concat on columns
print("-------")
print(df_slice)

df_slice.to_excel("extracted6.xlsx")

#%% Read single object from GPT3 output
df = pd.read_excel("gpt3_output")
container = json.loads(df["Extracted"][0]) # Pick first line
text = container["choices"][0]["text"] # Constant expression to get text from extracted object
#%% Extract text from gpt3 json output in excel file
df_origin = pd.read_excel("gpt3_output.xlsx")
df_extracted = pd.DataFrame()
for index, row in df_origin.iterrows():
    print(index)
    json_container = json.loads(row["Extracted"])
    text = json_container["choices"][0]["text"] # Constant expression to get text from extracted object
    new_row = pd.DataFrame({"Extracted_Text": [text]}) # Create new row for appending 
    df_extracted = pd.concat([df_extracted, new_row])
df_extracted = df_extracted.reset_index(drop = True) # Drop variable avoids adding old index as a column
df_merged = pd.concat([df_origin, df_extracted], axis = 1) # Concat on columns instead of rows
df_merged.to_excel("gpt3_output_formatted.xlsx")


#%% Merge multiple outputs
file_prefix = "extracted"
list_num = list(range(1,7))
df_merged = pd.DataFrame()
for num in list_num:
    df = pd.read_excel(f"data/{file_prefix}{str(num)}.xlsx") # Must all follow same prefix
    df_merged = pd.concat([df_merged, df])

df_merged.to_excel("merged.xlsx")
#%% Test
df_slice = DATA[3:6].reset_index(drop = True) # Start bound is included, end bound is not
df_new = pd.DataFrame() # Don't need to initiate columns since it is done automatically when adding entries
print(df_slice)
for index, row in df_slice.iterrows():
    new_row = pd.DataFrame({"Extracted": [row["Title"].lower()]})
    df_new = pd.concat([df_new, new_row])
df_new = df_new.reset_index(drop = True) # Drop variable avoids adding old index as a column
print(df_new)
df_slice = pd.concat([df_slice, df_new], axis = 1) # Axis 1 to concat on columns
print("-------")
print(df_slice)

#%%
