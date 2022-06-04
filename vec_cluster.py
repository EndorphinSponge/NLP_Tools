#%% Imports

# General
import os
import matplotlib.pyplot as plt

# NLP
from top2vec import Top2Vec

# Local imports
from global_functions import importData

#%% Constants 
FILESOURCE = "tbi_all.csv"
FILEROOTNAME = FILESOURCE.split(".")[0] # Get root name without extension 
DIR = os.path.dirname(__file__)
DATA = importData(os.path.join(DIR, F"data/{FILESOURCE}"))["Abstract"] # Imports series of abstracts


#%% Functions


#%% Execution 

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    print("Running")
    corpus = DATA.tolist()
    model = Top2Vec(corpus)

# %%
model.get_num_topics()
topic_words, word_scores, topic_nums = model.get_topics()
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
    plt.savefig(F"figures/{FILEROOTNAME}_{topic}.png")
# %%
