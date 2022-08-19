#%% Imports 
# General
import os

from pandas import DataFrame

# NLP
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, TfidfModel
from gensim.models.ldamodel import LdaModel


# Vis
import pyLDAvis
import pyLDAvis.gensim_models

# Local imports
from internal_globals import importData
from models_spacy import SpacyModel
from components_tbi import LDA_STOPWORDS


#%% Constants
DIR = os.path.dirname(__file__)
# DATA = importData(os.path.join(DIR, "data/screening.xlsx"), filt = "maybe", filt_col = "Tags")["Abstract"] # Imports series of abstracts
PATH = "data/tbi_ymcombined.csv"
POS_TAGS = ["NOUN", "ADJ", "VERB", "ADV"] # POS of interest
STOPWORDS = LDA_STOPWORDS


#%% Execution 

class ClustererTfidf:
    def __init__(self) -> None:
        self.file_path = ""
        self.root_name = ""
        self.root_base = ""
        self.df = DataFrame()
        self.col_corpora = ""
        self.topic_model: LdaModel = None
        

        


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__)) # Change directory to that of python module
    
    nlpmodel = SpacyModel(disable=["parser", "ner"])
    #
    docs_lemma: list[str] = nlpmodel.lemmatizeCorpora(PATH, "Abstract", pos_tags=POS_TAGS, stopwords=STOPWORDS)

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
    vocab = Dictionary(data_bigrams_trigrams) # Generates a vocabulary by mapping unique tokens in corpora to ID
    corpus_bow: list[list[tuple[int, int]]] = [vocab.doc2bow(text) for text in data_bigrams_trigrams] # Turn corpora into bag-of-words via vocab
    tfidf = TfidfModel(corpus_bow, id2word=vocab)
    
    corpus_tfidf_bow = []
    
    low_value = 0.03
    for ind, doc_bow in enumerate(corpus_bow):
        low_value_words = [id for id, value in tfidf[doc_bow] if value < low_value]
        new_bow = [(id, count) for (id, count) in doc_bow 
                   if id not in low_value_words]
        corpus_tfidf_bow[ind] = new_bow

    


    # LDA + display 
    if 0:
        lda_model = LdaModel(corpus=corpus_tfidf_bow,
                                                        id2word=vocab,
                                                        num_topics=i,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        passes=10,
                                                        alpha="auto")
        for i in range(10, 13): # Iterate between range of number of topics and store all of them 
            lda_model = LdaModel(corpus=corpus_tfidf_bow,
                                                    id2word=vocab,
                                                    num_topics=i,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha="auto")

            vis = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf_bow, vocab, mds="mmds", R=30)
            pyLDAvis.save_html(vis, f"LDA{i}.html")
# %%
