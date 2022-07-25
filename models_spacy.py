"""
For process that handles more than one doc at a time
"""
#%% Imports
# General 
import os, re, math, csv, difflib
from collections import Counter
from typing import Union

# NLP
import spacy 
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from spacy.tokens import Span, Doc, DocBin


# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np
from pyvis.network import Network

# Internals 
from global_functions import importData
#%% Constants


#%% Classes and functions

class SpacyModel:
    def __init__(self, model: str = "en_core_web_sm", disable: list[str] = []):
        self.model = model
        self.NLP = spacy.load(model, disable=disable)
        self.doclist: list[Doc] = [] # List of processed text, directly usable 
        self.lastimportsrc: str = "" # Tracker of source path (no extension) for last import to docbin for export naming 
        
    
    def importCorpora(self,
                      corpora_path: Union[str, bytes, os.PathLike],
                      col: str,
                      annotation_cols: list[str] = []):
        """Imports Excel or CSV file with corpora in the specified column and processes 
        it using the loaded SpaCy model. Processed results are appended to the class 
        instance in both doclist (for immediate use) and docbin (for export)
        Is the main function for batch processing texts

        Args:
            corpora_path: path to the dataframe containing the corpora
            col: Column label for column containing text to parse 
            annotation_cols: Optional list of column labels of columns to be passed as annotations to the processed text 
        """
        df = importData(corpora_path, screen_dupl=[col], screen_text=[col]) # Screen for duplicates and presence of text for the column containing the text 
        self.lastimportsrc = os.path.splitext(corpora_path)[0] # Assign path without extension to tracker in case it's needed for export naming
        text_list: list[tuple[str, dict]] = []
        
        for index, row in df.iterrows():
            text: str = row[col]
            context = dict()
            for annot_col in annotation_cols: # Will not loop if list is empty 
                context[annot_col] = row[annot_col] # Add new entry for every annotation column by using data from corresponding column cell of the current row
            text_list.append((text, context)) # Append text together with the context dictionary
        
        counter = 0
        for (doc, context) in self.NLP.pipe(text_list, as_tuples=True): # Pass in (text, context) tuples to get (doc, context) tuples
            doc.user_data = context
            self.doclist.append(doc) # Add processed doc to list for immediate use
            counter += 1 
            print("NLP processing text no: ", counter)
            
        return self.doclist
    
    def exportDocs(self, custom_name: Union[str, bytes, os.PathLike] = ""):
        """Exports current Doclist object to a file. Names export based on last import, can override this behaviour by passing a custom name

        Args:
            custom_name : Manually set file path prefix (directory and file name without extension) for the save file, extension will be added in function.
        """
        docbin = DocBin(store_user_data=True) # Need store_user_data to save Doc.user_data
        for doc in self.doclist:
            docbin.add(doc) # Serielize processed doc for exporting
        if custom_name: # If custom name not empty, use custom name
            docbin.to_disk(f"{custom_name}({self.model}).spacy") # Saves content using hashes based on a model's vocab, will need this vocab to import it back
        else: # Otherwise use prefix of last import to name output
            docbin.to_disk(f"{self.lastimportsrc}({self.model}).spacy") # Saves content using hashes based on a model's vocab, will need this vocab to import it back
            
            

    def importDocs(self, path: Union[str, bytes, os.PathLike]):
        """Imports processed docs from a docbin .spacy file
        Adds imported docs to doclist

        Args:
            path (Union[str, bytes, os.PathLike]): Path to docbin object
        """
        docbin = DocBin().from_disk(path) # Don't need to set store_user_data=True for import
        doclist = list(docbin.get_docs(self.NLP.vocab)) # Retrieves content by mapping hashes back to words by using the vocab dictionary 
        self.doclist += doclist # Concat imported doclist to current doclist 
        
    def resetDocs(self):
        self.doclist = []
        
    
    def printAvailableModels():
        print("en_core_sci_scibert", spacy.util.is_package("en_core_sci_scibert"))
        print("en_core_sci_lg", spacy.util.is_package("en_core_sci_lg"))
        print("en_core_web_trf", spacy.util.is_package("en_core_web_trf"))
        print("en_core_web_sm", spacy.util.is_package("en_core_web_sm"))
        print("en_core_web_lg", spacy.util.is_package("en_core_web_lg"))
    
    def lemmatizeText(self,
                      df_path: Union[str, bytes, os.PathLike],
                      col: str,
                      pos_tags: list[str] = ["NOUN", "ADJ", "VERB", "ADV"],
                      stopwords: list[str] = [],
                      ) -> list[str]:
        """Imports XLS/CSV with texts in a specified column, processes, 
        and returns texts in a list with every token in their lemmatized form.

        Args:
            df_path (Union[str, bytes, os.PathLike]): Source path
            col (str): Column that contains texts to be lemmatized
            pos_tags (list, optional): Tags to be considered for lemmatization Defaults to ["NOUN", "ADJ", "VERB", "ADV"].

        Returns:
            list: list of strings of lemmatized texts
        """
        df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Screen for duplicates and presence of text for the column containing the text
        texts_lemmatized = []
        for index, row in df.iterrows():
            print("Lemmatizing row: ", index)
            if type(row[col]) == str:
                doc = self.NLP(row[col])
                row_lemmas = []
                for token in doc:
                    if token.pos_ in pos_tags and token.lemma_ not in stopwords and token.text not in stopwords:
                        row_lemmas.append(token.lemma_)
                row_lemmatized = " ".join(row_lemmas)
                texts_lemmatized.append(row_lemmatized)
        return texts_lemmatized
    
class DocParse:
    """
    Container for functions that are performed on processed docs 
    """

class DocVis:
    """
    Container for visualization functions performed on processed docs 
    """
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
    

class ManualExtractor:
    """
    Container for functions used in manual DEP information extraction
    Mostly deprecated
    """
    def mapEntity(cls, token):
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


    def colAllChildren(cls, token, dep = ["conj"]):
        """ 
        Recursive function that gathers all tokens connected by the same dep relationship
        Useful for gathering entities in a comma separated list
        dep: list of dep tags used in recursive search 
        """
        children = []
        for child in token.children:
            if child.dep_ in dep:
                children.append(cls.mapEntity(child))
                children += cls.colAllChildren(child) # Collate all variables to flatten
        return children


    def checkChildDep(cls, token, dep_list: list):
        """
        Check if the children of a given token match any items in a dep list
        """
        children_dep = [child.dep_ for child in token.children]
        for dep in dep_list:
            for child_dep in children_dep:
                if child_dep == dep:
                    return True # Only return true if the dep of the children of root matches one of the given deps
        return False 

    def checkChildWord(cls, token, dep_list: list, word_list: list):
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
    def checkChildText(cls, token, text_list: list):
        """
        Check if children of a given token match the strings in the given list regardless of DEP
        """
        for child in token.children:
            for text in text_list:
                if child.text.lower() == text:
                    return True # True if the child token is both a listed dep and matches a word in the word list
        return False 

    def extractType1(cls, root_token):
        subj = []
        obj = []
        for child in root_token.children:
            if child.dep_ in ["nsubj"]: # For subject half
                subj.append(cls.mapEntity(child))
                subj += cls.colAllChildren(child) # Recursively collect children with "conj" dep
            elif child.dep_ in ["dobj"]: # For object half
                obj.append(cls.mapEntity(child))
                obj += cls.colAllChildren(child) # Recursively collect children with "conj" dep
        # print (subj, obj)
        return (subj, obj)

    def extractType2(cls, root_token):
        factors = []
        outcomes = []
        for child in root_token.children:
            if child.dep_ in ["nsubjpass", "nsubj"]: # For factor
                factors.append(cls.mapEntity(child))
                factors += cls.colAllChildren(child) # Recursively collect children with "conj" dep
            elif child.dep_ in ["nmod"]: # For outcome
                if cls.checkChildWord(child, ["case"], ["with", "by", "to"]): # Only add outcome if token has a nmod child with its own child of case dep
                    # Not all tokens with dep case are valid (e.g., nmod with "for" with dep case)
                    outcomes.append(cls.mapEntity(child))
                    outcomes += cls.colAllChildren(child) # Recursively collect children with "conj" dep
        # print (subj, obj)
        return (factors, outcomes)

    def extractType3(cls, root_token):
        factors = []
        outcomes = []
        for child in root_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]: # For factor
                factors.append(cls.mapEntity(child))
                factors += cls.colAllChildren(child) # Recursively collect children with "conj" dep
            elif child.dep_ in ["xcomp"]: # For outcome
                for subchild in child.children:
                    if subchild.dep_ in ["nmod"]:
                        outcomes.append(cls.mapEntity(subchild))
                        outcomes += cls.colAllChildren(subchild) # Recursively collect children with "conj" dep
        # print (subj, obj)
        return (factors, outcomes)

    def extractType4(cls, root_token):
        factors = []
        outcomes = []
        for child in root_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]: # For factor
                factors.append(cls.mapEntity(child))
                factors += cls.colAllChildren(child) # Recursively collect children with "conj" dep
            elif child.dep_ in ["nmod"]: # For outcome
                if cls.checkChildWord(child, ["case"], ["of", "for", "in"]): # Only add outcome if token has a nmod child with its own child of case dep
                    outcomes.append(cls.mapEntity(child))
                    outcomes += cls.colAllChildren(child) # Recursively collect children with "conj" dep
        # print (subj, obj)
        return (factors, outcomes)
    
    def genDFSVO(cls):
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
                        and not cls.checkChildDep(word, ["cop", "auxpass"]) \
                        and cls.checkChildDep(word, ["nsubj"]) \
                        and cls.checkChildDep(word, ["dobj"]):
                        sentences.append(sent.text)
                        roots.append(word.text)
                        subj, obj = cls.extractType1(word)
                        factors.append(subj)
                        outcomes.append(obj)
                    if word.text.lower() in ["predicted", "associated", "correlated", "related", "linked", "connected"] \
                        and cls.checkChildWord(word, ["cop", "auxpass"], ["was", "were"]) \
                        and cls.checkChildDep(word, ["nsubj", "nsubjpass"]) \
                        and cls.checkChildDep(word, ["nmod"]):
                        sentences.append(sent.text)
                        roots.append(word.text)
                        subj, obj = cls.extractType2(word)
                        factors.append(subj)
                        outcomes.append(obj)
                    if word.text.lower() in ["shown", "suggested", "demonstrated", "determined"] \
                        and cls.checkChildText(word, ["was", "were", "have", "has", "been"]) \
                        and cls.checkChildText(word, ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators"]):
                        sentences.append(sent.text)
                        roots.append(word.text)
                        subj, obj = cls.extractType3(word)
                        factors.append(subj)
                        outcomes.append(obj)
                    if word.text.lower() in ["predictor", "predictors", "marker", "markers", "biomarker", "biomarkers", "indictor", "indicators", "factor", "factors"] \
                        and cls.checkChildDep(word, ["cop"]) \
                        and cls.checkChildDep(word, ["nmod"]):
                        sentences.append(sent.text)
                        roots.append(word.text)
                        subj, obj = cls.extractType4(word)
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





#%% Snippets
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
#%% Add to NLP processing


def listSimilar():
    # Top similar words based on vectors 
    word = "associated"
    word_vector = np.asarray([NLPV.vocab.vectors[NLPV.vocab.strings[word]]])
    similar_vectors = NLPV.vocab.vectors.most_similar(word_vector, n=10)
    similar_words = [NLPV.vocab.strings[i] for i in similar_vectors[0][0]]
    print(similar_words)

#%% Post-processing

def visAbrvs():
    # Display abbreviations, entities, DEP
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





if __name__ == "__main__":
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

    #%% Generate edges and entities for subsequent generation
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
