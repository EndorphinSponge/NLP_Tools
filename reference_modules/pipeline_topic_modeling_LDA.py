# -*- coding: utf-8 -*-

from global_functions import *

#%%

import glob

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, TfidfModel



from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim_models


#%%

data = importExcelData3("data/TBIExport.xls")["AB"] # Imports series of abstracts
stopwords = stopwords.words("english")
other_stops = ["use"]
stopwords += other_stops

def lemmatizeText(texts, pos_tags=["NOUN", "ADJ", "VERB", "ADV"]) -> list:
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    texts_out = []
    count = 1
    for text in texts:
        print(count)
        if type(text) == str:
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in pos_tags and token.lemma_ not in stopwords and token.text not in stopwords:
                    new_text.append(token.lemma_)
            final = " ".join(new_text)
            texts_out.append(final)
        count += 1
    return (texts_out)

lemmatized_texts = lemmatizeText(data)

#%%

def getTokens(texts) -> list:
    final = []
    counter = 1
    for text in texts:
        print(counter)
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
        counter += 1
    return (final)

data_words = getTokens(lemmatized_texts)

#%% Bigrams and trigrams

bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=50)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def getBigrams(texts):
    return ([bigram[doc] for doc in texts])

def getTrigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = getBigrams(data_words)
data_bigrams_trigrams = getTrigrams(data_bigrams)

# print(data_bigrams_trigrams)

#%% TF-IDF to remove common terms from corpus

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
    


#%% Processing words through dictionary, replaced by above cell

# corpus = []
# id2word = corpora.Dictionary(data_words)
# for text in data_words:
#     new = id2word.doc2bow(text)
#     corpus.append(new)

 
#%% Generating and displying LDA model 

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto")

vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
# pyLDAvis.enable_notebook()
pyLDAvis.save_html(vis, "LDA.html")
