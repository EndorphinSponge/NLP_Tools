# Custom definitions/components for TBI prognostication 

# LDA custom components
from nltk.corpus import stopwords

TBI_LDA_STOPWORDS = stopwords.words("english") + ["patient", "outcome", "mortality", "year", "month", "day", "hour", "predict", "factor", "follow", \
    "favorable", "adult", "difference", "tbi", "score", "auc", "risk", "head", "associate", \
    "significantly", "group", "unfavorable", "outcome", "accuracy", "probability", "median", "mean", \
    "average", "high", "analysis", "also",
    ] # List of other stop words to include 

# NLP entity processing custom components 
common_ignore = {"patient", "patient\'", "patients", 
    "increase", "favorable", "favourable", "good", "high", "low", 
    "decrease", "bad", "poor", "unfavorable", "unfavourable", "reduced", "worse",
    "rate", "associated",
    "prevalence", "presence", "result", "ratio", "in-hospital",
    "use of", "development",
    "clinical trial", "significance", "finding", "score", "analysis", "isolate",
    "early", "adult", "study", "background", "conclusion", "compare", "time",
    "hour", "day", "week", "month", "year", "level", 
    "hours", "days", "weeks", "months", "years", "rates",
    "prognosis",
    "persistent",
    "acute",
    "characteristic",
    } # Words common to both factors and outcomes
common_tbi_ignore = {"tbi", "mtbi", "stbi", "csf", "serum", "blood", "plasma", "pl", "mild",
    "moderate", "severe", "concentration", "risk", "traumatic", "finding", "post-injury",
    "injury", "injuries", "hi", "trauma"
    } # Specific to TBI in both factors and outcomes

# Exclusions
factors_ignore = {"problem", "mortality rate", "outcome", "mortality",
                  "poor outcome", "in-hospital mortality"
                  } | common_ignore | common_tbi_ignore
outcomes_ignore = {"age", "improved", "reduced", "trauma"} | common_ignore | common_tbi_ignore

# Translations
factors_trans = {
    "snps": "snp",
    "rotterdam ct score": "rotterdam",
    "rotterdam score": "rotterdam",
    "marshall ct score": "marshall",
    "marshall score": "marshall",
    "gc score": "gcs",
    "impact": "impact model",
    "crash": "crash model",

}
outcomes_trans = {
    "hospital mortality": "in-hospital mortality",
    "death": "mortality",
    "mortality rate": "mortality",
    "survival": "mortality",
    "functional": "functional outcome",
    "clinical outcome": "unspecified clinical outcome",
    "outcomes": "unspecified clinical outcome",
    "outcome": "unspecified clinical outcome",
    "drs": "disability",
    "ufo": "poor outcome",
    "poor prognosis": "poor outcome",
}

import os
from typing import Union

from graph_builder import EntProcessorCore

class TBICore(EntProcessorCore):
    # Core component of EntProcessor, returns 
    def __init__(self,
                 abrv_path: Union[str, bytes, os.PathLike],
                 common_trans_path: Union[str, bytes, os.PathLike],
                 ):
        super().__init__(abrv_path, common_trans_path)
        # Exclusions and translations are overrided by custom component 
        self.exclusions = {
            "factor": factors_ignore,
            "outcome": outcomes_ignore,
        }
        self.translations = {
            "factor": self.trans_json | factors_trans, # Merge custom translations with automatically generated ones
            "outcome": self.trans_json | outcomes_trans, # Add custom translations to translation json so that custom translations override it
        }