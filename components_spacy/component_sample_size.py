# Pipeline component to extract sample size from noun chunks in abstract
#%% Imports
from word2number import w2n

from spacy.tokens import Span, Doc
from spacy.language import Language

from internals import LOG

#%% Extracting sample size from abstract

DEPRAC = False # Guard while keeping linting active
# Reminder that set_extension is a classmethod, will affect all instances of Doc
if DEPRAC:
    Doc.set_extension("sample_size", default = 0, force = True) # Force true to avoid having to restart kernel every debug cycle


@Language.component("extractSampleSize")
def extractSampleSize(doc: Doc):
    patients = [chunk for chunk in doc.noun_chunks if ("patient" in str(chunk)) or ("subject" in str(chunk))]
    numbers = [0] # 0 as default
    for entry in patients:
        if (str(entry).lower() == "a patient") or \
            (str(entry).lower() == "the patient") or \
            (str(entry).lower() == "a subject") or \
            (str(entry).lower() == "the subject"): # these arguments have to come first, otherwise the second set of conditions will catch them before being processed by these arguments
            numbers.append(1)
        entities = entry.ents
        for word in entities:            
            if ((word.label_ == "CARDINAL") or (word.label_ == "QUANTITY")) and (word.text != ""):
                try: numbers.append(w2n.word_to_num(word.text)) #Sample data has "six6" in entry ORN 19 which can't be parsed, this is to catch similar exceptions
                except ValueError: LOG.debug(F"\"{word.text}\" could not be converted to a numer in \"{doc}\"")
                finally: pass
    doc.user_data["sample_size"] = max(numbers)
    return doc

#%% Debug
# NLP.add_pipe("extractSampleSize")
