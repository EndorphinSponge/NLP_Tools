# Custom definitions/components for TBI prognostication 

common_ignore = {"patient", "patient\'", "patients", "rate", "associated", "hour", "day", "month", "year", "level", 
    "favorable", "favourable", "good", "high", "low", "prevalence", "presence", "result", "ratio", "in-hospital",
    "decrease", "bad", "poor", "unfavorable", "unfavourable", "reduced", "use of", "development",
    "clinical trial", "significance", "finding", "score", "analysis", "isolate"
    "early", "adult", "study", "background", "conclusion", "compare", "time"
    "hours", "days", "months", "years", "rates",
    } # Words common to both factors and outcomes
common_tbi_ignore = {"tbi", "mtbi", "stbi", "csf", "serum", "blood", "plasma", "mild",
    "moderate", "severe", "concentration", "risk", "traumatic", "finding", "post-injury",
    "injury", "injuries",
    } # Specific to TBI in both factors and outcomes

# Exclusions
factors_ignore = {"problem", "mortality rate", "outcome"} | common_ignore | common_tbi_ignore
outcomes_ignore = {"age", "improved", "reduced", "trauma"} | common_ignore | common_tbi_ignore

# Translations
factors_trans = {
    "snps": "snp",
    "rotterdam ct score": "rotterdam",
    "rotterdam score": "rotterdam",
    "marshall ct score": "marshall",
    "marshall score": "marshall",

}
outcomes_trans = {
    "hospital mortality": "in-hospital mortality",
    "clinical outcome": "outcome",
    "death": "mortality",
    "mortality rate": "mortality",
    "survival": "mortality",
    "functional": "functional outcome",
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
            "factor": factors_trans | self.trans_json, # Merge custom translations with automatically generated ones
            "outcome": outcomes_trans | self.trans_json,
        }