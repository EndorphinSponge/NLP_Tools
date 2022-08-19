#%% Imports
# General
import os, sys, pickle
from typing import Dict, Union


# Data science
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


# NLP
from top2vec import Top2Vec
import gensim
from gensim.models import CoherenceModel, TfidfModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

# Local imports
from internal_globals import importData
from models_spacy import SpacyModel
from components_tbi import LDA_STOPWORDS

#%% Logging 
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

#%% Classes

class Clusterer:
    def __init__(self) -> None:
        self.file_path = ""
        self.root_name = ""
        self.root_base = ""
        self.df = DataFrame()
        self.col_corpora = ""
        
        self.vocab: gensim.corpora.Dictionary = None
        self.corpus_bow: list[list[tuple[int, int]]] = []
        self.model_lda: LdaModel = None
        
        self.model_vec: Top2Vec = None
        
        
    
    def importCorpora(self, df_path: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        self.file_path = df_path
        self.root_name = os.path.splitext(df_path)[0] # Get root name without extension 
        self.root_base = os.path.splitext(os.path.basename(df_path))[0] # Get base name without extension
        self.df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Can't have any data type apart from str in list
        self.col_corpora = col
        logger.info(F"Imported data from {df_path}")
        
    def genVocabAndBow(self, save = False):
        # Only for LDA pipeline, generates vocab including bi/trigrams and BOWs of every corpora
        # Uses TFD-IDF to process BOWs to remove irrelevant terms
        
        nlpmodel = SpacyModel(disable=["parser", "ner"])
        
        pos_tags = ["NOUN", "ADJ", "VERB", "ADV"]
        docs_lemma: list[str] = nlpmodel.lemmatizeCorpora(df_path=self.file_path,
                                                       col=self.col_corpora,
                                                       pos_tags=pos_tags,
                                                       stopwords=LDA_STOPWORDS)

        docs_tokens = [gensim.utils.simple_preprocess(doc, deacc=True)
                    for doc in docs_lemma]
        # data_words should be corpora_tokens_processed
        
        # Bi/tri-grams

        bigram_phrases = gensim.models.Phrases(docs_tokens, min_count=5, threshold=50) 
        # Results in Phrases object whose index can be used to merge two tokens that are often found adjacent - hence bigrams
        bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases) 
        # Extracts the phraser portion of the Phrases oject for better performance https://radimrehurek.com/gensim/models/phrases.html
        data_bigrams = [bigram_phraser[doc] for doc in docs_tokens] # Likely combines tokens that fit bigram phrase 
        
        trigram_phrases = gensim.models.Phrases(bigram_phraser[docs_tokens], min_count=5, threshold=50)
        # Usess tokenized corpora, first merges bigrams and then uses output tokens to then detect trigrams
        trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
        # Same as bigram phraser, only using phraser portion of Phrases object
        
        
        data_bigrams_trigrams = [trigram_phraser[doc] for doc in data_bigrams]
        # Mrges any bigrams and adjacent tokens into trigrams if possible to get both tri and bigrams
        # Can also go straight from single tokens into bi+trigrams by wrappering each doc in bigram phraser and then trigram phraser
        
        # TF-IDF 
        vocab = gensim.corpora.Dictionary(data_bigrams_trigrams) # Generates a vocabulary by mapping unique tokens in corpora to ID
        corpus_bow: list[list[tuple[int, int]]] = [vocab.doc2bow(text) for text in data_bigrams_trigrams] # Turn corpora into bag-of-words via vocab
        tfidf = TfidfModel(corpus_bow, id2word=vocab)
        
        corpus_bow_processed = [] # Container for corpus bows after being processed by tfidf method 
        
        low_value = 0.03
        for ind, doc_bow in enumerate(corpus_bow):
            low_value_words = [id for id, value in tfidf[doc_bow] if value < low_value]
            new_bow = [(id, count) for (id, count) in doc_bow 
                    if id not in low_value_words]
            corpus_bow_processed.append(new_bow)
            
        self.vocab = vocab
        self.corpus_bow = corpus_bow_processed
        
        if save:
            vocab_bow_obj = (vocab, corpus_bow_processed)
            with open(F"{self.root_name}_vocab_bow.dat", "w+b") as file:
                pickle.dump(vocab_bow_obj, file)
            
        

    def importVocabBow(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Uses importCorpora to re-instantiate df info
        self.importCorpora(df_path=df_of_origin, col=col)
        with open(F"{self.root_name}_vocab_bow.dat", "rb") as file:
            vocab_bow_obj = pickle.load(file)
            
        self.vocab, self.corpus_bow = vocab_bow_obj # Unpack objects into class instance

    def clusterTopicsLda(self, num_topics: int, save = False):
        
        lda_model = LdaModel(corpus=self.corpus_bow,
                                id2word=self.vocab,
                                num_topics=num_topics,
                                random_state=100,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha="auto")
        
        self.model_lda = lda_model
        
        if save:
            lda_model.save(F"{self.root_name}_lda") # Creates this main file without an extension, has other files with extensions that is linked to this main file
            
    def importTopicsLda(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Uses importVocabBow to instantiate df info and populate self.vocab and self.corpus_bow
        self.importVocabBow(df_of_origin=df_of_origin, col=col)
        self.model_lda: LdaModel = LdaModel.load(F"{self.root_name}_lda")
        assert self.vocab == self.model_lda.id2word # Check that imported vocab is the same as one contained in model
        

    def visLdaTopics(self):
        num_topics = len(self.model_lda.get_topics())
        vis = pyLDAvis.gensim_models.prepare(topic_model=self.model_lda,
                                             corpus=self.corpus_bow,
                                             dictionary=self.vocab,
                                             mds="mmds", R=30)
        pyLDAvis.save_html(vis, f"figures/{self.root_base}_lda_n{num_topics}.html")
        
    def clusterTopicsVec(self, save = False):
        corpus_df = self.df[self.col_corpora]
        corpus_list = list(corpus_df) # Need list format 
        self.model_vec = Top2Vec(corpus_list) # Need a minimum number of present topics to cluster, otherwise throws error
        num_topics = self.model_vec.get_num_topics()
        logger.info(F"Successfully built topic model with {num_topics} topics")
        if save:
            self.model_vec.save(F"{self.root_name}_top2vec.dat")
            logger.info(F"Saved trained model in {self.root_name}_top2vec.dat")
            
    def importTopicsVec(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Basically combines importCorpora with loading of corresponding top2vec model
        # Meant to be a starting point after cluterTopicsVec has been run 
        self.importCorpora(df_path=df_of_origin, col=col)
        self.model_vec = Top2Vec.load(F"{self.root_name}_top2vec.dat")
    
    def genTopicWordClouds(self):
        # Only has support for top2vec models
        model = self.model_vec
        topic_words, word_scores, topic_nums = model.get_topics()
        logger.info(F"Found {len(topic_nums)} topics")
        for topic in topic_nums:
            model.generate_topic_wordcloud(topic)
            plt.savefig(F"figures/{self.root_base}_topic{topic}.png")
            
    def annotateCorpora(self, save = False):
        # Uses a generated or imported top2vec model to annotated topics to its original df
        # save -> save annotations byproduct in a separate container
        model = self.model_vec
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
        
#%%


#%% Plot generation (vec clusters for TBI)
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

