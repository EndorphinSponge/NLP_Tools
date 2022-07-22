#%% Imports 
# General
import os
import pandas as pd

# NLP
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, TfidfModel


# Vis
import pyLDAvis
import pyLDAvis.gensim_models

# Local imports
from global_functions import importData, lemmatizeText

#%% Constants
DIR = os.path.dirname(__file__)
# DATA = importData(os.path.join(DIR, "data/screening.xlsx"), filt = "maybe", filt_col = "Tags")["Abstract"] # Imports series of abstracts
DATA = importData(os.path.join(DIR, "data/tbi_ymcombined.csv"), col = "Abstract")["Abstract"] # Imports series of abstracts

#%% Local Functions

def getTokens(texts) -> list:
    final = []
    counter = 1
    for text in texts:
        print(counter)
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
        counter += 1
    return (final)

def getBigrams(texts):
    return ([bigram[doc] for doc in texts])

def getTrigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

#%% Execution 
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__)) # Change directory to that of python module
    lemmatized_texts = lemmatizeText(DATA)

    # Bi/tri-grams
    data_words = getTokens(lemmatized_texts)

    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50)
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=50)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = getBigrams(data_words)
    data_bigrams_trigrams = getTrigrams(data_bigrams)

    # TF-IDF 
    id2word = corpora.Dictionary(data_bigrams_trigrams) # Assign words to dictionary 
    corpus = [id2word.doc2bow(text) for text in data_bigrams_trigrams]
    tfidf = TfidfModel(corpus, id2word = id2word)

    low_value = 0.03
    words = []
    words_excluded = []
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_excluded
        for item in drops:
            words.append(id2word[item])
        words_excluded = [id for id in bow_ids if id not in tfidf_ids]
        
        new_bow = [word for word in bow if word[0] not in low_value_words and word[0] not in words_excluded]
        corpus[i] = new_bow


    # LDA + display 
    for i in range(10, 13): # Iterate between range of number of topics and store all of them 
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=i,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis, f"LDA{i}.html")
# %%
