# Custome definitions/components for TBI prognostication 

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

#%% Exclusions
factors_ignore = {"problem","mortality rate"} | common_ignore | common_tbi_ignore
outcomes_ignore = {"age", "improved", "reduced", "trauma"} | common_ignore | common_tbi_ignore

#%% Translations
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
