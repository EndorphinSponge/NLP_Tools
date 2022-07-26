#%% Imports
# General 
import os
import json
import re
from typing import Union

# Data Science
import pandas as pd
from pandas import DataFrame, Series

# NLP
import openai
from openai.openai_object import OpenAIObject # For type assumption check

from global_functions import importData
#%% Constants
openai.api_key = "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" # Personal
# openai.api_key = "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" # Throwaway
# openai.api_key = "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" # Sea
TEST = """{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "\n| --- | --- | --- |\n| Glasgow coma scale score | Glasgow Outcome Scale (GOS) | 95 |\n| Standard mortality rate (SMR) | Mortality rate | 34/95 |\n| GOS at 1-year | GOS 10-15 years post-trauma | Good (GOS 4-5): 68<br>Poor (GOS 2-3): 27 |"
    }
  ],
  "created": 1658561404,
  "id": "cmpl-5X3sKAOI3o6SPzo3MjyyBnRA7YEUE",
  "model": "text-davinci-002",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 88,
    "prompt_tokens": 451,
    "total_tokens": 539
  }
}"""
#%% Classes & Functions

class CloudModel:

    def __init__(self, df_path: Union[str, bytes, os.PathLike] = "",
                 input_col: str = "Abstract",
                 output_col: str = "Model_output") -> None:
        self.init_path = df_path
        self.df_file_name = os.path.splitext(df_path)[0] # Split root from file extension
        self.df_raw: DataFrame = importData(df_path, screen_dupl=[input_col], screen_text=[input_col]) # Empty string as df_path will return empty dataframe
        self.input_col: str = input_col
        self.output_col: str = output_col
        self.df: DataFrame = DataFrame() # Placeholder for last DF worked on 
    
    def mineTextGpt3(self, subset: tuple[int, int] = ()):
        """
        Will save a version of the intermediate raw output by modifying the original df name

        Args:
            subset: Provide a slice range if you want to only process a subset of the df. Defaults to ().
        """
        if subset: # If slice is not empty
            df = self.df_raw[subset[0]:subset[1]] # Take a subset of the input DF using the range of the subset
            self.df_file_name = self.df_file_name + str(subset) # Append subset slice info to root name
        else: 
            df = self.df_raw
            self.df_file_name = self.init_path # Reset filename to initial path name
        df_out_raw = DataFrame() # Placeholder for new output
        for index, row in df.iterrows():
            print("Extraction using GPT3, entry: ", index)
            query_insert = row[self.input_col]
            query_complete = f"A table summarizing each of the predictors, what they predict, and the total number of subjects in the study:\n\n{query_insert}\n\n| Predictor | Outcome being predicted | Number of subjects |",
            output_raw = openai.Completion.create(
                engine="text-davinci-002",
                prompt=query_complete,
                temperature=0,
                max_tokens=2000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            new_row = DataFrame({"Gpt3_raw": [str(output_raw)]}) # Convert openai.openai_object.OpenAIObject to str so that it can be used for subequent steps
            new_row.index = pd.RangeIndex(start=index, stop=index+1, step=1) # Reassign index of new row by using current index 
            df_out_raw = pd.concat([df_out_raw, new_row])
        # df_out_raw = df_out_raw.reset_index(drop = True) # Drop variable avoids adding old index as a column
        df_merged = pd.concat([df, df_out_raw], axis=1) # Axis 1 to concat on columns
        df_merged.to_excel(f"{self.df_file_name}_gpt3raw.xlsx", index=False)
        self.df = df_merged # Set current DF to the df that was just exported
    

        
    def importRaw(self, df_path: Union[str, bytes, os.PathLike]):
        """Import a df into self.df manually, mainly for troubleshooting

        Args:
            df_path (Union[str, bytes, os.PathLike]): Path to DF with model output
        """
        self.df = importData(df_path) # Only import using pandas without pre-processing
        self.df_file_name = os.path.splitext(df_path)[0] # Split root from file extension
    
        # Add final output here 
    
    def storeOutputFormatted(self, raw_type: str):
        """Looks at current df for raw mined output of given format (GPT3 vs JUR1),
        extracts the generated portion of output, formats them into a JSON string, 
        appends them to the current DF

        Args:
            raw_type (str): type of mined output ("gpt3" or "jur1")
        """
        df = self.df
        df_out = pd.DataFrame()
        if self.df_file_name.endswith("raw"): # Replace raw with fmt if it exists at end of filename
            self.df_file_name = re.sub(R"raw$", "fmt", self.df_file_name)
        else: # Otherwise append fmt to end
            self.df_file_name = self.df_file_name + "fmt"
            
        if raw_type.lower() == "gpt3":
            for index, row in df.iterrows():
                output_json = json.loads(row["Gpt3_raw"], strict=False) # non-strict mode to allow control characters (e.g., \n) in strings
                raw_str: str = output_json["choices"][0]["text"] # Constant expression to get generated output from extracted object
                article_stmts_str = [item.strip() for item in raw_str.split("\n") if re.search(r"\w", item) != None] # Split by newline, only include lines that have word characters
                article_stmts_items = [list(filter(None, statement.split("|"))) for statement in article_stmts_str] # Filter with none to get rid of empty strings, should only return 3 items corresponding to the 3 columns of output
                stments_json_str = json.dumps(article_stmts_items) # Shape of list[list[str]] for each article, inner lists are statements, each str is an item
                new_row = pd.DataFrame({self.output_col: [stments_json_str]}) # Create new row for appending 
                new_row.index = pd.RangeIndex(start=index, stop=index+1, step=1) # Reassign index of new row by using current index 
                df_out = pd.concat([df_out, new_row])
            df_merged = pd.concat([df, df_out], axis = 1) # Concat on columns instead of rows
            df_merged.to_excel(f"{self.df_file_name}.xlsx", index=False)
            
        elif raw_type == "jur1":
            pass


