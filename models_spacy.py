"""
For process that handles more than one doc at a time
"""
#%% Imports
# General 
from collections import Counter
import csv
import math
import re
import difflib

# NLP
import spacy 
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from spacy.tokens import Span, Doc, DocBin
from nltk.corpus import stopwords

# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np
from pyvis.network import Network


from global_functions import importData
#%% Constants
# spacy.require_gpu() # Needs manual memory management, more conventient to use CPU and store results
# NLPV = spacy.load("en_core_sci_lg") # Doesn't require GPU, but has vectors
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector")
Doc.set_extension("title", default = "", force = True); # Force true to avoid having to restart kernel every debug cycle
doc_bin = DocBin()

SPACY_MODEL = "en_core_web_trf" # Word model for SpaCy
STOPWORDS = stopwords.words("english")
STOPWORDS += ["patient", "outcome", "mortality", "year", "month", "day", "hour", "predict", "factor", "follow", \
    "favorable", "adult", "difference", "tbi", "score", "auc", "risk", "head", "associate", \
    "significantly", "group", "unfavorable", "outcome", "accuracy", "probability", "median", "mean", \
    "average", "high", "analysis",] # List of other stop words to include 
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
    Check if the children of a given token match any combination of items in the dep list and item list
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

def extractType4(root_token):
    factors = []
    outcomes = []
    for child in root_token.children:
        if child.dep_ in ["nsubj", "nsubjpass"]: # For factor
            factors.append(mapEntity(child))
            factors += colAllChildren(child) # Recursively collect children with "conj" dep
        elif child.dep_ in ["nmod"]: # For outcome
            if checkChildWord(child, ["case"], ["of", "for", "in"]): # Only add outcome if token has a nmod child with its own child of case dep
                outcomes.append(mapEntity(child))
                outcomes += colAllChildren(child) # Recursively collect children with "conj" dep
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

def lemmatizeText(texts, pos_tags=["NOUN", "ADJ", "VERB", "ADV"]) -> list:
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    texts_out = []
    count = 1
    for text in texts:
        print(count)
        if type(text) == str:
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in pos_tags and token.lemma_ not in STOPWORDS and token.text not in STOPWORDS:
                    new_text.append(token.lemma_)
            final = " ".join(new_text)
            texts_out.append(final)
        count += 1
    return (texts_out)

#%% Dataframe generation for SVO relationships
df = pd.DataFrame(columns = ["Title", "Abstract", "Included sentences", "Roots", "Subjects", "Objects", "Noun chunks", "Entities"])
for doc in doc_bin:
    sentences = []
    roots = []
    factors = []
    outcomes = []
    for sent in doc.sents:
        for word in sent: # Decouple from root word (i.e., root can start from any token)
            # Each if statement is a separate pipeline 
            if word.text.lower() in ["predicted", "predict"] \
                and not checkChildDep(word, ["cop", "auxpass"]) \
                and checkChildDep(word, ["nsubj"]) \
                and checkChildDep(word, ["dobj"]):
                sentences.append(sent.text)
                roots.append(word.text)
                subj, obj = extractType1(word)
                factors.append(subj)
                outcomes.append(obj)
            if word.text.lower() in ["predicted", "associated", "correlated", "related", "linked", "connected"] \
                and checkChildWord(word, ["cop", "auxpass"], ["was", "were"]) \
                and checkChildDep(word, ["nsubj", "nsubjpass"]) \
                and checkChildDep(word, ["nmod"]):
                sentences.append(sent.text)
                roots.append(word.text)
                subj, obj = extractType2(word)
                factors.append(subj)
                outcomes.append(obj)
            if word.text.lower() in ["shown", "suggested", "demonstrated", "determined"] \
                and checkChildText(word, ["was", "were", "have", "has", "been"]) \
                and checkChildText(word, ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators"]):
                sentences.append(sent.text)
                roots.append(word.text)
                subj, obj = extractType3(word)
                factors.append(subj)
                outcomes.append(obj)
            if word.text.lower() in ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators", "factor", "factors"] \
                and checkChildDep(word, ["cop"]) \
                and checkChildDep(word, ["nmod"]):
                sentences.append(sent.text)
                roots.append(word.text)
                subj, obj = extractType4(word)
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

df.to_excel("manualdep_output.xlsx")

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
Determining a patient's prognosis after severe traumatic brain injury remains difficult and complex. The purpose of the present study was following up patients with severe traumatic brain injury by correlating their clinical outcome and sequential computer tomography (CT) findings. We investigated 51 patients who survived the first year following an accident. All patients underwent successive CT examinations within a maximum period of 2 years. The patients' outcomes depended on the underlying brain damage and are presented by the Glasgow Outcome Scale. Based on the investigated data we concluded that the worst outcomes were experienced by patients with initial massive cerebral edema, extensive subdural hematoma and intraventricular hemorrhage, followed by stroke as subacute CT finding and cerebral atrophy as chronic finding visible at follow-up CT scans. The majority of lesions identified by CT scan were found in the frontal lobes, basal ganglia, and temporal lobes. We suggest that CT examination still represents a simple and useful tool in attempting to predict the clinical outcome in patients with severe traumatic brain injury.
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
sentence = list(doc.sents)[5]
dep_figure = displacy.render(sentence,style="dep", jupyter=True, options={"compact":True, "distance":100})
# %%

#%% Constants
# NLPV = spacy.load("en_core_sci_lg") # Doesn't require GPU, but has vectors
NLP = spacy.load("en_core_sci_scibert") # Requires GPU
NLP.add_pipe("abbreviation_detector")
TEXT = """Older patients had a higher mortality, with the highest mortality (37.5%) among those over 50 years old (p = 0.009)"""
TEXT = """Traumatic Brain Injury (TBI) is a major cause of death and disability; the leading cause of mortality and morbidity in previously healthy people aged under 40 in the United Kingdom (UK). There are currently little official Irish statistics regarding TBI or outcome measures following TBI, although it is estimated that over 2000 people per year sustain TBI in Ireland. We performed a retrospective cohort study of TBI patients who were managed in the intensive care unit (ICU) at CUH between July 2012 and December 2015. Demographic data were compiled by patients' charts reviews. Using the validated Glasgow outcome scale extended (GOS-E) outcome measure tool, we interviewed patients and/or their carers to measure functional outcomes. Descriptive statistical analyses were performed. Spearman's correlation analysis was used to assess association between different variables using IBM's Statistical Package for the Social Sciences (SPSS) 20. In the 42-month period, 102 patients were identified, mainly males (81%). 49% had severe TBI and 56% were referred from other hospitals. The mean age was 44.7 and a most of the patients were previously healthy, with 65% of patients having ASA I or II. Falls accounted for the majority of the TBI, especially amongst those aged over 50. The 30-day mortality was 25.5% and the mean length of hospital stay (LOS-H) was 33 days. 9.8% of the study population had a good recovery (GOS-E 8), while 7.8% had a GOS-E score of 3 (lower sever disability). Patients with Extra-Dural haemorrhage had better outcomes compared with those with SDH or multi-compartmental haemorrhages (p = 0.007). Older patients had a higher mortality, with the highest mortality (37.5%) among those over 50 years old (p = 0.009). TBI is associated with significant morbidity and mortality. Despite the young mean age and low ASA the mortality, morbidity and average LOS-H were significant, highlighting the health and socioeconomic burden of TBI."""
TEXT = """BACKGROUND: Evidence from the last 25 years indicates a modest reduction of mortality after severe traumatic head injury (sTBI). This study evaluates the variation over time of the whole Glasgow Outcome Scale (GOS) throughout those years., METHODS: The study is an observational cohort study of adults (>= 15 years old) with closed sTBI (GCS <= 8) who were admitted within 48 h after injury. The final outcome was the 1-year GOS, which was divided as follows: (1) dead/vegetative, (2) severely disabled (dependent patients), and (3) good/moderate recovery (independent patients). Patients were treated uniformly according to international protocols in a dedicated ICU. We considered patient characteristics that were previously identified as important predictors and could be determined easily and reliably. The admission years were divided into three intervals (1987-1995, 1996-2004, and 2005-2012), and the following individual CT characteristics were noted: the presence of traumatic subarachnoid or intraventricular hemorrhage (tSAH, IVH), midline shift, cisternal status, and the volume of mass lesions (A x B x C/2). Ordinal logistic regression was performed to estimate associations between predictors and outcomes. The patients' estimated propensity scores were included as an independent variable in the ordinal logistic regression model (TWANG R package)., FINDINGS: The variables associated with the outcome were age, pupils, motor score, deterioration, shock, hypoxia, cistern status, IVH, tSAH, and epidural volume. When adjusting for those variables and the propensity score, we found a reduction in mortality from 55% (1987-1995) to 38% (2005-2012), but we discovered an increase in dependent patients from 10 to 21% and just a modest increase in independent patients of 6%., CONCLUSIONS: This study covers 25 years of management of sTBI in a single neurosurgical center. The prognostic factors are similar to those in the literature. The improvement in mortality does not translate to better quality of life."""
TEXT = """An unfavorable GOS score (1-3) at 1 year was predicted by higher Day 7 GFAP levels (above 9.50 ng/ml; AUC 0.82, sensitivity 78.6%, and specificity 82.4%)."""
TEXT = """Presence of coagulopathy, anticoagulant drug use, GCS of 13-14 and increased age predicted further deterioration."""

TEXT = "Functioning and HRQoL postinjury in older people"
TEXT = "Care pathway and treatment variables, and 6-month measures of functional outcome, health-related quality of life (HRQoL), post-concussion symptoms (PCS), and mental health symptoms"
TEXT = "90-day mortality"
TEXT = "TBI outcome"
TEXT = "The severity of traumatic brain injury (TBI)"
TEXT = "Hospital mortality"
TEXT = "CSF and serum Lac, NSE, and BBB index"
TEXT = "Condition and prognosis after a severe TBI"
TEXT = "PCS at 30 days"
TEXT = "Being under the influence of drugs or alcohol at the time of injury"
TEXT = "6-month Glasgow-Outcome-Scale score"
#%% Functions

def compareStrings(str1, str2):
    return difflib.SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio()

def nlpString(string) -> list:
    """
    Takes a string and returns a tuple of a set of entities and a list of abbreviations
    """
    doc = NLP(string)
    ents = {ent.text.lower().strip() for ent in list(doc.ents)} # Need to convert to str first, otherwise causes problems with subsequent functions which only take strings
    abrvs = [(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) for abrv in doc._.abbreviations]
    for abrv, full in abrvs:
        for ent in ents.copy(): # Iterate over a copy of the set while changing the original
            if compareStrings(full, ent) > 0.9: # Find ent matching with full form of abbreviation
                ents.remove(ent) # Remove full form
                ents.add(abrv) # Add abbreviated form                
    return (ents, abrvs)

print(nlpString("Care pathway and treatment variables, and 6-month measures of functional outcome, health-related quality of life (HRQoL), post-concussion symptoms (PCS), and mental health symptoms"))


#%% Display abbreviations, entities, DEP

TEXT = "patients"

doc = NLP(TEXT)

print("Length: " + str(len(doc)))
for ent in doc.ents:
    print(ent.lemma_)
sent_no = 0
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
for sentence in doc.sents:
    print(sent_no)
    displacy_image = displacy.render(sentence, jupyter = True, style = "ent")
    sent_no += 1
print(list(doc.noun_chunks))
dep_figure = displacy.render(doc,style="dep", options={"compact":True, "distance":100})



#%% DEP rendering + save
num = 1 
text_cont = []
#%% DEP rendering STOP
TEXT = """
, CONCLUSIONS: In our group of patients, the V/C parameter and BP have been suggested to be predictors of outcome.
"""
doc = NLP(TEXT.strip())
dep_figure = displacy.render(doc,style="dep", jupyter=True, options={"compact":True, "distance":100})

## Below code for saving
# dep_figure = displacy.render(doc,style="dep", jupyter=False, options={"compact":True, "distance":100})
# output_path = Path(f"plot{str(num)}.html") # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script 
# output_path.open("w", encoding="utf-8").write(dep_figure)
# text_cont.append(TEXT.strip())
# num += 1 # Look above for definition


# %% Top similar words based on vectors 
word = "associated"
word_vector = np.asarray([NLPV.vocab.vectors[NLPV.vocab.strings[word]]])
similar_vectors = NLPV.vocab.vectors.most_similar(word_vector, n=10)
similar_words = [NLPV.vocab.strings[i] for i in similar_vectors[0][0]]
print(similar_words)

#%% Visualize graph of entities by co-mentions in a sentence 
keywords = ["augments", "increased", "decreased", "increases", "decreases", "more", "less", "higher", "lower", "greater", "lesser", "improved", "worsened", "improves", "worsens", "predict", "predicts", "predicted", "predictor", "predictors", "predictive", "factor", "factors", "variable", "variables", "marker", "markers", "biomarker", "biomarkers", "correlate", "correlates", "correlated ", "correlation", "correlations", "associates", "associated ", "association", "associations", "related", "relationship", "relationships ", "link", "linked", "linkage", "connected", "connection", "connections"]
TEXT = """
BACKGROUND: Between 20-50% of those suffering a mild traumatic brain injury (MTBI) will suffer symptoms beyond 3 months or post-concussive disorder (PCD). Researchers in Sydney conducted a prospective controlled study which identified that bedside recordings of memory impairment together with recordings of moderate or severe pain could predict those who would suffer PCS with 80% sensitivity and specificity of 76%., PRIMARY OBJECTIVE: This study is a cross-validation study of the Sydney predictive model conducted at Montreal General Hospital, Montreal, Canada., METHODS: One hundred and seven patients were assessed in the Emergency Department following a MTBI and followed up by phone at 3 months. The Rivermead Post-Concussive Questionnaire was the main outcome measure., RESULTS: Regression analysis showed that immediate verbal recall and quantitative recording of headache was able to predict PCD with a sensitivity of 71.4% and a specificity of 63.3%. In the combined MTBI groups from Sydney and Montreal the sensitivity was 70.2% and the specificity was 64.2%., CONCLUSION: This is the first study to compare populations from different countries with diverse language groups using a predictive model for identifying PCD following MTBI. The model may be able to identify an 'at risk' population to whom pre-emptive treatment can be offered.
"""
doc = NLP(TEXT.strip())
# Counters used to track number of occurences, later used for display 
edges = Counter()
entities = Counter() # Track how many times each entity was mentioned by a sentence
for sent in doc.sents:
    contains_keyword = False
    for keyword in keywords: # Loop through all keywords
        match = re.search(rf"\b{keyword}\b", sent.text.lower())
        if match != None: # Only process sentence if it contains the keyword 
            print(match.group())
            contains_keyword = True
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
    if contains_keyword:
        print(sent.text.lower())

# To scale to documents, should make counts relative to document rather than sentences 
# Add nodes and edges
net = Network()

for entity in entities:
    net.add_node(entity, mass = math.log(entities[entity]))
    # net.add_node(entity, size = entities[entity], mass = math.log(entities[entity]))
for (node1, node2) in edges:
    # net.add_edge(node1, node2, width = math.log(edges[(node1, node2)]))
    net.add_edge(node1, node2, width = 100)

net.toggle_physics(True)
# net.force_atlas_2based(damping = 1, gravity = -20, central_gravity = 0.05, spring_length = 65) # For smaller graphs 
# net.force_atlas_2based(damping = 1, gravity = -12, central_gravity = 0.01, spring_length = 100) # For larger graphs 
net.repulsion()
net.show_buttons(filter_=['physics'])
net.show("Network.html")

#%% Save list as CSV
with open("list.csv", 'w', newline='') as file:
     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
     wr.writerow(text_cont)
#%% Load list 
with open("list.csv", "r") as file:
    reader = csv.reader(file)
    TEXTS = list(reader)
# %%
TEXT = """BACKGROUND: Severe traumatic brain injury (sTBI) can be divided into primary and secondary injuries. Intensive care protocols focus on preventing secondary injuries. This prospective cohort study was initiated to investigate outcome, including mortality, in patients treated according to the Lund Concept after a sTBI covering 10-15 years post-trauma., METHODS: Patients were included during 2000-2004 when admitted to the neurointensive care unit, Sahlgrenska University Hospital. Inclusion criteria were: Glasgow coma scale score of <=8, need for artificial ventilation and intracranial monitoring. Glasgow Outcome Scale (GOS) was used to evaluate outcome both at 1-year and 10-15 years post-trauma., RESULTS: Ninety-five patients, (27 female and 68 male), were initially included. Both improvement and deterioration were noted between 1- and 10-15 years post-injury. Mortality rate (34/95) was higher in the studied population vs. a matched Swedish population, (Standard mortality rate (SMR) 9.5; P < 0.0001). When dividing the cohort into Good (GOS 4-5) and Poor (GOS 2-3) outcome at 1-year, only patients with Poor outcome had a higher mortality rate than the matched population (SMR 7.3; P < 0.0001). Further, good outcome (high GOS) at 1-year was associated with high GOS 10-15 years post-trauma (P < 0.0001). Finally, a majority of patients demonstrated symptoms of mental fatigue., CONCLUSION: This indicates that patients with severe traumatic brain injury with Good outcome at 1-year have similar survival probability as a matched Swedish population and that high Glasgow outcome scale at 1-year is related to good long-term outcome. Our results further emphasise the advantage of the Lund concept. Copyright © 2017 The Acta Anaesthesiologica Scandinavica Foundation. Published by John Wiley & Sons Ltd."""
doc = NLP(TEXT)
for sent in doc.sents:
    pass
# %%
