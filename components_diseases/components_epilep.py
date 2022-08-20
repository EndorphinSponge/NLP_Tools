# Post-processing component for keyword associations in documents
#%% Imports
from word2number import w2n

from spacy.tokens import Span, Doc

#%% 
# Reminder that set_extension is a classmethod, will affect all instances of Doc
Doc.set_extension("modality", default = None, force = True) # Force true to avoid having to restart kernel every debug cycle
modalities = ["brain depth stimulation",
              "transcranial magnetic stimulation",
              "vagus nerve stimulation"]
Doc.set_extension("disease_broad", default = None, force = True)
diseases_broad = ["drug resistant epilepsy",
            "focal epilepsy",
            "intractable epilepsy",
            "focal epilepsy",
            "generalized epilepsy"
            ]
Doc.set_extension("disease_narrow", default = None, force = True)
diseases_narrow = ["complex partial seizure",
                   "frontal lobe epilepsy",
                   "gelastic seizure",
                   "panayiotopoulos syndrome",
                   "Rasmussen syndrome",
                   "rolandic epilepsy",
                   "simple partial seizure",
                   "temporal lobe epilepsy"] \
                + ["absence",
                   "Alpers disease",
                   "atonic seizure",
                   "benign childhood epilepsy",
                   "clonic seizure",
                   "grand mal epilepsy",
                   "hypsarrhythmia",
                   "infantile spasm",
                   "Lennox Gastaut syndrome",
                   "MERRF syndrome",
                   "myoclonic astatic epilepsy",
                   "myoclonus",
                   "myoclonus epilepsy",
                   "myoclonus seizure",
                   "nodding syndrome",
                   "tonic clonic seizure",
                   "tonic seizure"]
                # Split into focal and generalized epilepsy terms
                
def processKeywords(docs):
    for (doc, context) in docs:
        # Cleanup keywords
        keywords = context["mh"].split("\n\n")
        for i in range(len(keywords)):
            if keywords[i].find("/") != -1: # Split based on presence of forwardslash which precedes the modifier
                keywords[i] = keywords[i].split("/")[0] # Only store the main keyword, exclude the modifier
        doc._.modality = [*{*[keyword for keyword in keywords 
                              if keyword.strip("*") in modalities]}]
        doc._.disease_broad = [*{*[keyword for keyword in keywords 
                                   if keyword.strip("*") in diseases_broad]}]
        doc._.disease_narrow = [*{*[keyword for keyword in keywords 
                                    if keyword.strip("*") in diseases_narrow]}]
        # Only get main keywords
        # doc._.modality = [item.strip("*") for item in doc._.modality if item.startswith("*")]
        # doc._.disease_broad = [item.strip("*") for item in doc._.disease_broad if item.startswith("*")]
        # doc._.disease_narrow = [item.strip("*") for item in doc._.disease_narrow if item.startswith("*")]
        # Strip asterisk
        doc._.modality = [item.strip("*") for item in doc._.modality]
        doc._.disease_broad = [item.strip("*") for item in doc._.disease_broad]
        doc._.disease_narrow = [item.strip("*") for item in doc._.disease_narrow]
        print(doc._.modality)
        print(doc._.disease_broad)
        print(doc._.disease_narrow)

    # Additional screening based on keywords

    docs_new = []
    for (doc, context) in docs:
        if doc._.modality != [] and (doc._.disease_broad != [] or doc._.disease_narrow != []):
            docs_new.append((doc, context))
    return docs_new

