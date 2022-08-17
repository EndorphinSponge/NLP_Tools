#%% Imports
# General
import os, sys
from typing import Union

# Logging 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")

# Logging CLI output stream
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Logging file output stream
fh = logging.FileHandler("mylog.log", "w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Data science
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# NLP
from top2vec import Top2Vec

# Local imports
from internal_globals import importData

#%% Constants 

class Clusterer:
    def __init__(self) -> None:
        self.file_path = ""
        self.root_name = ""
        self.root_base = ""
        self.df = DataFrame()
        self.col_corpora = ""
        self.topic_model: Top2Vec = None
        
    
    def importCorpora(self, df_path: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        self.file_path = df_path
        self.root_name = os.path.splitext(df_path)[0] # Get root name without extension 
        self.root_base = os.path.splitext(os.path.basename(df_path))[0] # Get base name without extension
        self.df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Can't have any data type apart from str in list
        self.col_corpora = col
        logger.info(F"Imported data from {df_path}")
        
    def clusterTopicsVec(self, save = False):
        corpus_df = self.df[self.col_corpora]
        corpus_list = list(corpus_df) # Need list format 
        self.topic_model = Top2Vec(corpus_list) # Need a minimum number of present topics to cluster, otherwise throws error
        num_topics = self.topic_model.get_num_topics()
        logger.info(F"Successfully built topic model with {num_topics} topics")
        if save:
            self.topic_model.save(F"{self.root_name}_top2vec.dat")
            logger.info(F"Saved trained model in {self.root_name}_top2vec.dat")
            
    def importTopicsVec(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Basically combines importCorpora with loading of corresponding top2vec model
        # Meant to be a starting point after cluterTopicsVec has been run 
        self.importCorpora(df_path=df_of_origin, col=col)
        self.topic_model = Top2Vec.load(F"{self.root_name}_top2vec.dat")
    
    def genTopicWordClouds(self):
        model = self.topic_model
        topic_words, word_scores, topic_nums = model.get_topics()
        logger.info(F"Found {len(topic_nums)} topics")
        for topic in topic_nums:
            model.generate_topic_wordcloud(topic)
            plt.savefig(F"figures/{self.root_base}_topic{topic}.png")
            
    def annotateCorpora(self, save = False):
        # Uses a generated or imported top2vec model to annotated topics to its original df
        model = self.topic_model
        df_annot = DataFrame()
        topic_sizes, topic_nums = model.get_topic_sizes()
        for topic_size, topic_num in zip(topic_sizes, topic_nums): # Transform arrays into tuples
            documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_num, num_docs=topic_size)
            for doc, score, doc_id in zip(documents, document_scores, document_ids):
                entry = DataFrame({
                    "Abstract": [doc],
                    "Topic": [topic_num],
                })
                df_annot = pd.concat([df_annot, entry])

        if save:
            df_annot.to_csv(F"{self.root_name}_t2v_annotations.csv", index=False)
            logger.info(F"Exported annotations to {self.root_name}_t2v_annotations.csv")


        df_origin = importData(self.file_path)
        df_origin = df_origin.set_index(self.col_corpora) # Set index to col containing corpora
        df_annot = df_annot.set_index(self.col_corpora)
        
        # Merge abstract df with annotated df https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html#:~:text=Join%20DataFrames%20using%20their%20indexes.&text=If%20we%20want%20to%20join,have%20key%20as%20its%20index.&text=Another%20option%20to%20join%20using,to%20use%20the%20on%20parameter.
        df_merged = df_origin.join(df_annot, lsuffix='_left', rsuffix='_right')
        df_merged.to_csv(F"{self.root_name}_annotated.csv")
        logger.info(F"Successfully appended annotations to {self.root_name}_annotated.csv")
        
    

#%% Plot generation 
if 0:
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = [("TBI of greater severity", 271),
            ("Serum markers of neuronal damage/inflammation", 263),
            ("TBI of lesser severity", 253),
            ("Prognostic model performance benchmarking", 164),
            ("MRI imaging markers", 118),
            ("Hemodynamic markers", 110),
            ("Genetic markers", 63),
            ("Metabolic and other common serum markers", 61),
            ("CT imaging markers", 49),
            ("Coagulation markers", 38),
            ("EEG markers", 23)
                ]

    sns.set_theme()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.bar([p[0] for p in data],
        [p[1] for p in data])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylabel("Number of articles in cluster")
    ax.set_xlabel("Topic cluster")
    ax.set_title("Latent topic clusters identified in the corpora via Top2Vec")

