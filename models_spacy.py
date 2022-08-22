"""
For process that handles more than one doc at a time
"""
#%% Imports
# General 
import os, re, json
from collections import Counter
from typing import Union


# NLP
import spacy 
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector # Added via NLP.add_pipe("abbreviation_detector")
from scispacy.linking import EntityLinker
from spacy.tokens import Span, Doc, DocBin


# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np

# Internals 
from internals import importData, LOG
#%% Constants


#%% Classes and functions

class SpacyModel:
    # Base class for other classes using SpaCy functions 
    def __init__(self, model: str = "en_core_web_sm", disable: list[str] = []):
        self.model = model
        self.NLP = spacy.load(model, disable=disable)
        self.doclist: list[Doc] = [] # List of processed text, directly usable 
        self.df_root_name: str = "" # Tracker of source path (no extension) for last import to docbin for export naming 
    
    @classmethod
    def printAvailableModels(cls):
        LOG.info(F"en_core_sci_scibert: {spacy.util.is_package('en_core_sci_scibert')}")
        LOG.info(F"en_core_sci_lg: {spacy.util.is_package('en_core_sci_lg')}")
        LOG.info(F"en_core_web_trf: {spacy.util.is_package('en_core_web_trf')}")
        LOG.info(F"en_core_web_sm: {spacy.util.is_package('en_core_web_sm')}")
        LOG.info(F"en_core_web_lg: {spacy.util.is_package('en_core_web_lg')}")
    
    def parseCorpora(self,
                      corpora_path: Union[str, bytes, os.PathLike],
                      col: str,
                      annotation_cols: list[str] = [],
                      export_userdata: bool = False,
                      **kwargs,
                      ):
        """Imports Excel or CSV file with corpora in the specified column and processes 
        it using the loaded SpaCy model. Processed results are appended to the class 
        instance in both doclist (for immediate use) and docbin (for export)
        Is the main function for batch processing texts

        Args:
            corpora_path: path to the dataframe containing the corpora
            col: Column label for column containing text to parse 
            annotation_cols: Optional list of column labels of columns to be passed as annotations to the processed text, data added to doc.user_data as dict
        """
        df = importData(corpora_path, screen_dupl=[col], screen_text=[col], **kwargs) # Screen for duplicates and presence of text for the column containing the text 
        self.df_root_name = os.path.splitext(corpora_path)[0] # Assign path without extension to tracker in case it's needed for export naming
        
        df_userdata = DataFrame()
        
        counter = 0
        for ind, row in df.iterrows():
            text: str = row[col]
            context = dict()
            for annot_col in annotation_cols: # Will not loop if list is empty 
                context[annot_col] = row[annot_col] # Add new entry for every annotation column by using data from corresponding column cell of the current row
                
            doc = self.NLP(text)
            
            # Parse user data from pipelines, assumes that all objects within user_data is JSON serializable 
            user_data_dict = {key: [json.dumps(value)] 
                         for key, value in doc.user_data.items()} # Package userdata values in list so that they can be instantiated in DataFrame
            new_entry = DataFrame(user_data_dict)
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_userdata = pd.concat([df_userdata, new_entry])
            
            doc.user_data.update(context) # Add annotation columns to .user_data attribute after user data from pipeline has been extracted
            
            self.doclist.append(doc) # Add processed doc to list for immediate use
            counter += 1
            LOG.debug(F"NLP processing text no: {counter}")
            if counter % 10 == 0: 
                LOG.info(F"NLP processing text no: {counter}") # Only print out every 10 texts
        
        if export_userdata:
            df_merged = pd.concat([df, df_userdata], axis=1)
            df_merged.to_csv(F"{self.df_root_name}_userdata.csv", index=False)
        
        
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
            LOG.info(f"Exported DocBin to {custom_name}({self.model}).spacy")
        else: # Otherwise use prefix of last import to name output
            docbin.to_disk(f"{self.df_root_name}({self.model}).spacy") # Saves content using hashes based on a model's vocab, will need this vocab to import it back
            LOG.info(f"Exported DocBin to {self.df_root_name}({self.model}).spacy")
            
            

    def importDocs(self, path: Union[str, bytes, os.PathLike]):
        """Imports processed docs from a docbin .spacy file
        Adds imported docs to doclist

        Args:
            path (Union[str, bytes, os.PathLike]): Path to docbin object
        """
        docbin = DocBin().from_disk(path) # Don't need to set store_user_data=True for import
        doclist = list(docbin.get_docs(self.NLP.vocab)) # Retrieves content by mapping hashes back to words by using the vocab dictionary 
        self.doclist += doclist # Concat imported doclist to current doclist in case exports were done in batches
        
        
    @staticmethod # Don't need to load self, hence bypasses need to load language model
    def convColsToStmts(df_path, cols: list[str], col_out: str = "Processed_ents"):
        # Converts JSON string items in cols to JSON string in format of statements used by graph_builder
        root_name = os.path.splitext(df_path)[0] # Store root name
        df = importData(df_path)
        df_stmts = DataFrame()
        for ind, row in df.iterrows():
            ents_dict: dict[str, list[str]] = {c: json.loads(row[c]) for c in cols}
            ents_dict_fmt = json.dumps([ents_dict]) # Wrap in list to keep consistent formatting (GPT3 output has multiple output statements)
            new_entry = DataFrame({col_out: [ents_dict_fmt]})
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_stmts = pd.concat([df_stmts, new_entry])
            
        df_merged = pd.concat([df, df_stmts], axis=1)
        df_merged.to_csv(F"{root_name}_entsF.csv")
        LOG.info(F"Exported to {root_name}_entsF.csv")
        
        
        
    def resetDocs(self):
        self.doclist = []
    
    def lemmatizeCorpus(self,
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
            LOG.info(F"Lemmatizing row: {index}")
            if type(row[col]) == str:
                doc = self.NLP(row[col])
                row_lemmas = []
                for token in doc:
                    if token.pos_ in pos_tags and token.lemma_ not in stopwords and token.text not in stopwords:
                        row_lemmas.append(token.lemma_)
                row_lemmatized = " ".join(row_lemmas)
                texts_lemmatized.append(row_lemmatized)
        return texts_lemmatized
    
    # Start of single doc operations 
    
    def lemmatizeDoc(self, text: str,
                     pos_tags: list[str] = ["NOUN", "ADJ", "VERB", "ADV"], 
                     stopwords: list[str] = [],
                     ) -> str:
        # Takes document in form of string and returns all tokens (filtered by POS tags)
        # in their lemmatized form
        if isinstance(text, str): # Only parse if doc is string
            doc = self.NLP(text)
            doc_lemmas = []
            for token in doc:
                if all([token.pos_ in pos_tags,
                       token.lemma_ not in stopwords,
                       token.text not in stopwords]):
                    doc_lemmas.append(token.lemma_)
            doc_lemmatized = " ".join(doc_lemmas) # Join all lemmatized tokens
            return doc_lemmatized
        else:
            return ""
        
    def listSimilar(self, word: str, top_n: int = 10):
        """Prints top n similar words based on word vectors in the loaded model

        Args:
            word (str): Word to look up
            top_n (int, optional): How many similar words to print. Defaults to 10.
        """
        word_vector = np.asarray([self.NLP.vocab.vectors[self.NLP.vocab.strings[word]]])
        similar_vectors = self.NLP.vocab.vectors.most_similar(word_vector, n=top_n)
        similar_words = [self.NLP.vocab.strings[i] for i in similar_vectors[0][0]]
        LOG.info(similar_words)
    
class GeneralExtractor(SpacyModel):
    def __init__(self, model: str = "en_core_web_sm", disable: list[str] = []):
        super().__init__(model, disable)
        
    def addPipeSampleSize(self):
        from components_spacy import component_sample_size
        self.NLP.add_pipe("extractSampleSize")
    
    def addPipeNmParams(self):
        from components_spacy import component_parameters
        self.NLP.add_pipe("extractNmParams")
    
    def addPipeCnsLocs(self):
        from components_spacy import component_location
        self.NLP.add_pipe("extractCnsLocations")
    

class EntityExtractor(SpacyModel):
    # Postprocessing using SpaCy for outputs for large NLP models (e.g., GPT3)
    def __init__(self, model: str = "en_core_sci_scibert",
                 disable: list[str] = []): 
        super().__init__(model, disable)
        self.NLP.add_pipe("abbreviation_detector") # Requires AbbreviationDetector to be imported first 
        self.empty_log = Counter()
    
    def extractEnts(self, 
                     df_path: Union[str, bytes, os.PathLike], 
                     col: str = "Model_output", 
                     col_out: str = "Ents",):
        """
        Takes DF containing formatted GPT3/JUR1 output in form of list[list[str]] for 
        each article in JSON string format. Each inner list represents a statement 
        within the article, each string represents an element in the statement (e.g., 
        factor, outcome, study size)
        
        Args: 
            df_path: path for input DF
            col: column in input DF that has formatted output 
            col_out: column in new DF to export extracted ents to 
        Returns: 
            Exports processed entities for each row in a new Excel file
        """
        root_name = os.path.splitext(df_path)[0]
        df = importData(df_path, screen_text=[col]) # Screen for presence of text for the column containing the text
        df_out = DataFrame() # Placeholder for output of lemmatized/abbreviated entities
        for index, row in df.iterrows():
            LOG.debug(F"NLP extracting ents for: {index}")
            statements: list[list[str]] = json.loads(row[col]) # list[str] of items for each statement 
            
            statements_ents: list[tuple[list[str], list[str]]] = [] # Container for ents
            """
            Unpacks statements so that NLP can be called on each of the statements separately
            REFACTOR THIS WHOLE SECTION AS A COMPONENT INTO components_tbi?"""
            for items in statements:
                factors = items[0]
                if re.search(R"\w", factors):
                    factor_ents = self._gatherEnts(factors)
                else:
                    factor_ents = set()
                    self.empty_log[tuple(items)] += 1
                    
                outcomes = items[1]
                if re.search(R"\w", outcomes):
                    outcome_ents = self._gatherEnts(outcomes)
                else:
                    outcome_ents = set()
                    self.empty_log[tuple(items)] += 1
                """CAN UNPACK MORE FACTORS HERE"""
                statements_ents.append({"factor": list(factor_ents), "outcome": list(outcome_ents)}) # Need convertion to list since sets can't be serialized into JSON
            
            stmts_ents_json = json.dumps(statements_ents)
            new_row = DataFrame({col_out: [stmts_ents_json]})
            new_row.index = pd.RangeIndex(start=index, stop=index+1, step=1) # Reassign index of new row by using current index 
            df_out = pd.concat([df_out, new_row])
        df_merged = pd.concat([df, df_out], axis=1)
        df_merged.to_excel(f"{root_name}_entsR.xlsx")
        LOG.info(f"Exported raw ents to {root_name}_entsR.xlsx")
                
    
    def _gatherEnts(self, string: str):
        # Extracts entities in lemmatized and abbreviated form (if they are available)
        doc = self.NLP(string.strip()) # Need to strip whitespace, otherwise recognition is suboptimal esp for shorter queries
        if len(doc) > 1: # Only process if there is more than one token
            ents = set()
            for ent in list(doc.ents):
                if len(ent.text.lower().strip()) <= 5:
                    ents.add(ent.text.lower().strip())
                else: # Only add lemma if word is bigger than 5 characters (lemmas on abbreviations tend to be buggy)
                    ents.add(ent.lemma_.lower().strip())
            return ents
        else: # Otherwise there will be only one token, return its lemma 
            if len(doc[0].text.lower().strip()) <= 5:
                return {doc[0].text.lower().strip(),}
            else: # Only add lemma if word is bigger than 5 characters (lemmas on abbreviations tend to be buggy)
                return {doc[0].lemma_.strip().lower(),} 

    def extractAbrvCont(self,
                        df_path: Union[str, bytes, os.PathLike],
                        col = "Abstract",
                        ):
        """
        Separated from main counter process because some tasks may want to use
        NLP to look ahead and pre-process all documents in a corpora for 
        metadata (e.g., corpora-wide abbreviations) that can be used to help
        process subsets of the corpora 
        """
        # Note that this function used to only look at Model_output
        root_name = os.path.splitext(df_path)[0]
        df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Screen for duplicates and presence of text for the column of interest
        abrv_counter = Counter()
        for index, row in df.iterrows():
            LOG.info(F"Extracting abbreviations for: {index}")
            text: str = row[col]
            doc = self.NLP(text.strip())
            abrvs: set[tuple[str, str]] = set([(abrv.text.lower().strip(), abrv._.long_form.text.lower().strip()) 
                                               for abrv in doc._.abbreviations])
            for abrv in abrvs:
                abrv_counter[abrv] += 1 # Add unique abbreviation into acounter
                
        abrv_items: list[tuple[tuple[str, str], int]] = list(abrv_counter.items()) # Gives list of tuples of the counted object (tuple[str, str]) and its count in dict form
        for item in abrv_items.copy(): # Screen items
            (short, long), count = item
            short_alnum = re.sub(R"[^a-zA-Z0-9]", "", short) # Get alphanumeric version of short
            if len(short_alnum) < 2: # Screen for invalid abbreviations
                # Can add low count filter though uncommonly defined abbreviations may help in short form convergence during fuzzy matching 
                abrv_items.remove(item)
        abrv_items.sort(key=lambda x: (x[1], len(x[0][1])), reverse=True) # Sort by counts and then by length of long form, will be translated in this priority
        with open(f"{root_name}_abrvs.json", "w") as file:
            json.dump(abrv_items, file) # Items converted to list for serialization 
        LOG.info(f"Exported extracted abbreviations to {root_name}_abrvs.json")
        return abrv_items

#%% Extraneous classes & functions
class DocParse:
    """
    Container for functions that are performed on processed docs 
    """

class DocVis:
    """
    Container for visualization functions performed on processed docs 
    """
    @classmethod
    def visSentStruct(cls, docs):
        """
        Visualize the sentence structure (root, children, dep tags of children)
        """
        for doc in docs:
            LOG.info("---------------------------")
            LOG.info(doc)
            for sent in doc.sents:
                LOG.info("Root: " + sent.root.text)
                LOG.info("Children: " + str([word.text for word in sent.root.children]))
                LOG.info("Dep tags: " + str([word.dep_ for word in sent.root.children]))
                for child in sent.root.children:
                    if child.dep_ in ["cop", "auxpass"]: # If root is not centered around a state of being word
                        LOG.info("True root: " + child.text)
        return

    @classmethod
    def visSpecChildren(cls, docs, target_dep = "nmod"):
        """
        Visualize specific children of subj and obj of sentence based on dep attribute
        target_dep: relationship for visualization 
        """
        for doc in docs:
            LOG.info("---------------------------")
            LOG.info(doc)
            for sent in doc.sents:
                LOG.info("Root: " + sent.root.text)
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
                LOG.info("Subject: " + str(subj) + F" {target_dep.upper()}: " + str(subnmod))
                LOG.info("Object: " + str(obj) + F" {target_dep.upper()}: " + str(objnmod))
        return
    
    @classmethod
    def visEntities(cls, docs):
        """
        Visualize entities and noun chunks in a document 
        """
        for doc in docs:
            LOG.info("---------------------------")
            LOG.info(doc)
            for sent in doc.sents:
                LOG.info("Entities: " + str(sent.ents))
                LOG.info("Noun chunks: " + str(list(sent.noun_chunks)))
        return 
    
    @classmethod
    def visAbrvsEntsDep(cls, doc: Doc):
        # Display abbreviations, entities, DEP
        LOG.info("Length: " + str(len(doc)))
        for ent in doc.ents:
            LOG.info(ent.lemma_)
        sent_no = 0
        for abrv in doc._.abbreviations:
            LOG.info(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
        for sentence in doc.sents:
            LOG.info(sent_no)
            displacy_image = displacy.render(sentence, jupyter = True, style = "ent")
            sent_no += 1
        LOG.info(list(doc.noun_chunks))
        dep_figure = displacy.render(doc,style="dep", options={"compact":True, "distance":100})


class ManualExtractor:
    """
    Container for functions used in manual DEP information extraction
    Mostly deprecated
    """
    @classmethod
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

    @classmethod
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

    @classmethod
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
    
    @classmethod
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
    
    @classmethod
    def checkChildText(cls, token, text_list: list):
        """
        Check if children of a given token match the strings in the given list regardless of DEP
        """
        for child in token.children:
            for text in text_list:
                if child.text.lower() == text:
                    return True # True if the child token is both a listed dep and matches a word in the word list
        return False 

    @classmethod
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

    @classmethod
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

    @classmethod
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

    @classmethod
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
    
    @classmethod
    def genDFSVO(cls, doc_bin):
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
