"""
For process that handle more than one doc at a time
"""
#%% Imports
import spacy 
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from spacy.tokens import Span, Doc, DocBin

import pandas as pd
import numpy as np
from pyvis.network import Network


from collections import Counter
import csv
import math
import re

from global_functions import importData
#%% Constants
# spacy.require_gpu() # Needs manual memory management, more conventient to use CPU and store results
# NLPV = spacy.load("en_core_sci_lg") # Doesn't require GPU, but has vectors
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector")
Doc.set_extension("title", default = "", force = True); # Force true to avoid having to restart kernel every debug cycle
doc_bin = DocBin()
# %% Process and save
counter = 0
texts = importData("data/tbi_ymcombined_subset100.csv")
docs = [(texts["Abstract"][index], texts["Title"][index]) for index in texts.index] # Convert texts into tuples of abstract and title
for (doc, title) in NLP.pipe(docs, as_tuples=True): # Batch process docs from the list of tuples containing abstracts and titles 
    doc._.title = title
    doc_bin.add(doc)
    print(counter)
    counter += 1
doc_bin.to_disk("tbi_ymcombined_subset100_scibert.spacy") # Saves content using hashes based on a model's vocab

#%% Load docbin
doc_bin = DocBin().from_disk("tbi_ymcombined_scibert.spacy")
doc_bin = list(doc_bin.get_docs(NLP.vocab)) # Retrieves content by mapping hashes back to words by using the vocab dictionary 

#%% Load list from csv
with open("list.csv", "r") as file:
    reader = csv.reader(file)
    docs = list(reader)[0] # Items are contained in one row
    docs = [NLP(doc) for doc in docs]

#%% Functions
def mapEntity(token):
    """
    Maps the token to the highest available resolution entity by
    finding the respective span of a token using the parent doc object
    If no matches are found, will return original token
    """
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk
    for span in token.doc.ents:
        if token.i >= span.start and token.i < span.end:
            return span
    return token


def colAllChildren(token, dep = ["conj"]):
    """ 
    Recursive function that gathers all tokens connected by the same dep relationship
    Useful for gathering entities in a comma separated list
    dep: list of dep tags used in recursive search 
    """
    children = []
    for child in token.children:
        if child.dep_ in dep:
            children.append(mapEntity(child))
            children += colAllChildren(child) # Collate all variables to flatten
    return children

def colChildCases(token):
    """
    Gathers the dependent cases of a token and returns the full entity
    """

def checkChildDep(token, dep_list: list):
    """
    Check if the children of a given token match any items in a dep list
    """
    children_dep = [child.dep_ for child in token.children]
    for dep in dep_list:
        for child_dep in children_dep:
            if child_dep == dep:
                return True # Only return true if the dep of the children of root matches one of the given deps
    return False 

def checkChildWord(token, dep_list: list, word_list: list):
    """
    Check if the children of a given token match any combinatino of items in the dep list and item list
    dep_list: list of deps for children that are used for check
    words: list of words to check for 
    """
    children_words = [child.text.lower() for child in token.children if child.dep_ in dep_list]
    for child in children_words:
        for word in word_list:
            if child == word:
                return True # True if the child token is both a listed dep and matches a word in the word list
    return False 
def checkChildText(token, text_list: list):
    """
    Check if children of a given token match the strings in the given list regardless of DEP
    """
    for child in token.children:
        for text in text_list:
            if child.text.lower() == text:
                return True # True if the child token is both a listed dep and matches a word in the word list
    return False 

def extractType1(root_token):
    subj = []
    obj = []
    for child in root_token.children:
        if child.dep_ in ["nsubj"]: # For subject half
            subj.append(mapEntity(child))
            subj += colAllChildren(child) # Recursively collect children with "conj" dep
        elif child.dep_ in ["dobj"]: # For object half
            obj.append(mapEntity(child))
            obj += colAllChildren(child) # Recursively collect children with "conj" dep
    # print (subj, obj)
    return (subj, obj)

def extractType2(root_token):
    factors = []
    outcomes = []
    for child in root_token.children:
        if child.dep_ in ["nsubjpass", "nsubj"]: # For factor
            factors.append(mapEntity(child))
            factors += colAllChildren(child) # Recursively collect children with "conj" dep
        elif child.dep_ in ["nmod"]: # For outcome
            if checkChildWord(child, ["case"], ["with", "by", "to"]): # Only add outcome if token has a nmod child with its own child of case dep
                # Not all tokens with dep case are valid (e.g., nmod with "for" with dep case)
                outcomes.append(mapEntity(child))
                outcomes += colAllChildren(child) # Recursively collect children with "conj" dep
    # print (subj, obj)
    return (factors, outcomes)

def extractType3(root_token):
    factors = []
    outcomes = []
    for child in root_token.children:
        if child.dep_ in ["nsubj", "nsubjpass"]: # For factor
            factors.append(mapEntity(child))
            factors += colAllChildren(child) # Recursively collect children with "conj" dep
        elif child.dep_ in ["xcomp"]: # For outcome
            for subchild in child.children:
                if subchild.dep_ in ["nmod"]:
                    outcomes.append(mapEntity(subchild))
                    outcomes += colAllChildren(subchild) # Recursively collect children with "conj" dep
    # print (subj, obj)
    return (factors, outcomes)

# Visualization
def visSentStruct(docs):
    """
    Visualize the sentence structure (root, children, dep tags of children)
    """
    for doc in docs:
        print("---------------------------")
        print(doc)
        for sent in doc.sents:
            print("Root: " + sent.root.text)
            print("Children: " + str([word.text for word in sent.root.children]))
            print("Dep tags: " + str([word.dep_ for word in sent.root.children]))
            for child in sent.root.children:
                if child.dep_ in ["cop", "auxpass"]: # If root is not centered around a state of being word
                    print("True root: " + child.text)
            print(extractType1(sent.root))
    return

def visSpecChildren(docs, target_dep = "nmod"):
    """
    Visualize specific children of subj and obj of sentence based on dep attribute
    target_dep: relationship for visualization 
    """
    for doc in docs:
        print("---------------------------")
        print(doc)
        for sent in doc.sents:
            print("Root: " + sent.root.text)
            subj = [word for word in sent.root.children if word.dep_ == "nsubj"]
            subnmod = []
            for subject in subj: # Should only have one
                for child in subject.children:
                    if child.dep_ == target_dep:
                        subnmod.append(child)
            obj = [word for word in sent.root.children if word.dep_ == "dobj"]
            objnmod = []
            for object in obj: # Should only have one
                for child in object.children:
                    if child.dep_ == target_dep:
                        objnmod.append(child)
            print("Subject: " + str(subj) + F" {target_dep.upper()}: " + str(subnmod))
            print("Object: " + str(obj) + F" {target_dep.upper()}: " + str(objnmod))
    return

def visEntities(docs):
    """
    Visualize entities and noun chunks in a document 
    """
    for doc in docs:
        print("---------------------------")
        print(doc)
        for sent in doc.sents:
            print("Entities: " + str(sent.ents))
            print("Noun chunks: " + str(list(sent.noun_chunks)))
    return 

# Dev


#%% Dataframe generation for SVO relationships
df = pd.DataFrame(columns = ["Title", "Abstract", "Included sentences", "Roots", "Subjects", "Objects", "Noun chunks", "Entities"])
for doc in doc_bin:
    sentences = []
    roots = []
    factors = []
    outcomes = []
    for sent in doc.sents:
        # Each if statement is a separate pipeline 
        for word in sent: # Decouple from root word (i.e., root can start from any token)
            # if word.text.lower() in ["predicted", "predict"] \
            #     and not checkChildDep(word, ["cop", "auxpass"]) \
            #     and checkChildDep(word, ["nsubj"]) \
            #     and checkChildDep(word, ["dobj"]):
            #     sentences.append(sent.text)
            #     roots.append(word.text)
            #     subj, obj = extractType1(word)
            #     factors.append(subj)
            #     outcomes.append(obj)
            # if word.text.lower() in ["predicted", "associated", "correlated", "related", "linked", "connected"] \
            #     and checkChildWord(word, ["cop", "auxpass"], ["was", "were"]) \
            #     and checkChildDep(word, ["nsubj", "nsubjpass"]) \
            #     and checkChildDep(word, ["nmod"]):
            #     sentences.append(sent.text)
            #     roots.append(word.text)
            #     subj, obj = extractType2(word)
            #     factors.append(subj)
            #     outcomes.append(obj)
            if word.text.lower() in ["shown", "suggested", "demonstrated", "determined"] \
                and checkChildWord(word, ["cop", "auxpass", "aux"], ["was", "were", "have", "has", "been"]) \
                and checkChildWord(word, ["xcomp"], ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators"]):
                sentences.append(sent.text)
                roots.append(word.text)
                subj, obj = extractType3(word)
                factors.append(subj)
                outcomes.append(obj)


    # Root, subj, obj using just forward sentences
    # Remember that to create a df from dict, need to have values in lists, otherwise creates empty df
    entry = pd.DataFrame({"Title": [doc._.title], 
        "Abstract": [doc.text], 
        "Included sentences": [sentences],
        "Roots": [roots], 
        "Subjects": [factors], 
        "Objects": [outcomes], 
        "Noun chunks": [str(list(doc.noun_chunks))], 
        "Entities": [str(list(doc.ents))],
        })
    df = pd.concat([df, entry])

df.to_excel("output.xlsx")

#%% Dataframe generation for word lemmas 
df = pd.DataFrame(columns = ["Title", "Abstract", "Included sentences", "Word"])
variations = set()
for doc in doc_bin:
    sentences = []
    words = []
    for sent in doc.sents:
        if "predict" in sent.text.lower():
            sentences.append(sent.text)
            match = re.search(r"predict\w*", sent.text.lower()).group()
            words.append(match)
            variations.add(match) # Add to variations in outer list to keep track of all word variations
    # Root, subj, obj using just forward sentences
    # Remember that to create a df from dict, need to have values in lists, otherwise creates empty df
    entry = pd.DataFrame({"Title": [doc._.title], 
        "Abstract": [doc.text], 
        "Included sentences": [sentences],
        "Word": [words], 
        })
    df = pd.concat([df, entry])
    
print(variations)
df.to_excel("output.xlsx")
#%% Visualize entities as graph

# Counters used to track number of occurences, later used for display 
edges = Counter()
entities = Counter() # Track how many times each entity was mentioned by a sentence
num_doc = 0
for doc in doc_bin:
    print(num_doc)
    for sent in doc.sents:
        # Interconnect all nodes of a sentence together as a measure of co-mentions 
        node_list = set([str(ent) for ent in sent.ents]) # Unpack in set to remove duplicates
        # Make sure to process all entities to string to avoid issues in graphing nodes
        node_list = [*node_list] # Unpack set into list to allow iteration
        for entity in node_list:
            entities[entity] += 1
        for (i, node_start) in enumerate(node_list): # Enumerate used to simulatenously return index for loop
            for (j, node_end) in enumerate(node_list[i+1:]): # To prevent repeating already enumarated nodes
                # Add edges in duplicate for undirected graph
                edges[(node_start, node_end)] += 1
                edges[(node_end, node_start)] += 1
    num_doc += 1

# To scale to documents, should make counts relative to document rather than sentences 
# Add nodes and edges
net = Network()

for entity in entities:
    net.add_node(entity, mass = math.log(entities[entity]))
    # net.add_node(entity, size = entities[entity], mass = math.log(entities[entity]))
for (node1, node2) in edges:
    net.add_edge(node1, node2, width = math.log(edges[(node1, node2)]))

net.toggle_physics(True)
# net.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65) # For smaller graphs 
# net.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100) # For larger graphs 
net.repulsion()
net.show_buttons(filter_=['physics'])
net.show("Network.html")



#%% Diagnostics
TEXT = "Further, good outcome (high GOS) at 1-year was associated with high GOS 10-15 years post-trauma (P < 0.0001)."
TEXT = """
BACKGROUND: Blunt cerebrovascular injuries (BCVIs) and cervical spinal injuries (CSIs) are not uncommon injuries in patients with severe head injury and may affect patient recovery. We aimed to assess the independent relationship between BCVI, CSI, and outcome in patients with severe head injury., METHODS: We identified patients with severe head injury from the Helsinki Trauma Registry treated during 2015-2017 in a large level 1 trauma hospital. We assessed the association between BCVI and SCI using multivariable logistic regression, adjusting for injury severity. Our primary outcome was functional outcome at 6 months, and our secondary outcome was 6-month mortality., RESULTS: Of 255 patients with a cervical spine CT, 26 patients (10%) had a CSI, and of 194 patients with cervical CT angiography, 16 patients (8%) had a BCVI. Four of the 16 BCVI patients had a BCVI-related brain infarction, and four of the CSI patients had some form of spinal cord injury. After adjusting for injury severity in multivariable logistic regression analysis, BCVI associated with poor functional outcome (odds ratio [OR] = 6.0, 95% CI [confidence intervals] = 1.4-26.5) and mortality (OR = 7.9, 95% CI 2.0-31.4). We did not find any association between CSI and outcome., CONCLUSIONS: We found that BCVI with concomitant head injury was an independent predictor of poor outcome in patients with severe head injury, but we found no association between CSI and outcome after severe head injury. Whether the association between BCVI and poor outcome is an indirect marker of a more severe injury or a result of treatment needs further investigations.
"""
doc = NLP(TEXT)
count = 0
for sent in doc.sents:
    print(count, sent)
    for word in sent: # Decouple from root word (i.e., root can start from any token)
        if word.text.lower() in ["shown", "suggested", "demonstrated", "determined"] \
            and checkChildWord(word, ["cop", "auxpass", "aux"], ["was", "were", "have", "has", "been"]) \
            and checkChildWord(word, ["xcomp"], ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators"]):
            print(word, *extractType3(word))
    count += 1
sentence = list(doc.sents)[9]
dep_figure = displacy.render(sentence,style="dep", jupyter=True, options={"compact":True, "distance":100})
# %%
