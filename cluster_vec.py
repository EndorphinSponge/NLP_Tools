#%% Imports
# General
import os
import matplotlib.pyplot as plt

# Data science
import pandas as pd

# NLP
from top2vec import Top2Vec

# Local imports
from internal_globals import importData

#%% Constants 
FILESOURCE = "tbi_ymcombined.csv"
FILEROOTNAME = FILESOURCE.split(".")[0] # Get root name without extension 
DIR = os.path.dirname(__file__)
DATA = importData(os.path.join(DIR, F"data/{FILESOURCE}"))["Abstract"] # Imports series of abstracts


#%% Functions


#%% Execution 

os.chdir(os.path.dirname(__file__))
print("Running")
corpus = DATA.tolist()
model = Top2Vec(corpus)

#%% Generate wordcloud images 
model.get_num_topics()
topic_words, word_scores, topic_nums = model.get_topics()
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
    plt.savefig(F"figures/{FILEROOTNAME}_{topic}.png")
#%% Annotate
df_annot = pd.DataFrame()
topic_sizes, topic_nums = model.get_topic_sizes()
for topic_size, topic_num in zip(topic_sizes, topic_nums):
    documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_num, num_docs=topic_size)
    for doc, score, doc_id in zip(documents, document_scores, document_ids):
        print(doc[:40])
        entry = pd.DataFrame({
            "Abstract": [doc],
            "Topic": [topic_num],
        })
        df_annot = pd.concat([df_annot, entry])

df_annot.to_excel("topic_annotations.xlsx")

#%% Merge abstract df with annotated df https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html#:~:text=Join%20DataFrames%20using%20their%20indexes.&text=If%20we%20want%20to%20join,have%20key%20as%20its%20index.&text=Another%20option%20to%20join%20using,to%20use%20the%20on%20parameter.

df_origin = pd.read_excel("gpt3_output_formatted.xlsx").set_index("Abstract")
df_annot = pd.read_excel("topic_annotations.xlsx").set_index("Abstract")
df_merged = df_origin.join(df_annot, lsuffix='_left', rsuffix='_right')
df_merged.to_excel("gpt3_output_formatted_annotated.xlsx")

# %%
os.chdir(os.path.dirname(__file__))
print("Running")
corpus = DATA.tolist()
model2 = Top2Vec(corpus)
# %%
model2.get_topic_sizes()
# %%
