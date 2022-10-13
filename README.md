*Under construction*<br />
_Note that source code documentation is only partially complete_

**Organization/Architecture**
- This repo features two main pipelines:
1. Extraction of TBI prognostication factors and their associated outcomes
    - Uses GPT3/Jurassic-1 for unstructured data parsing
    - SciBERT via SpaCy for subsequent entity post-processing 
    - Regex/rule-based post-processing of entity to achieve semantic convergence of entities
    - Matplotlib, Networkx, Pyvis for visualization of relationships between prognostication factors and their associated outcomes
    - Optional clustering of abstracts by TF-IDF or Top2Vec into subgroups on which the above pipeline can be applied to for generating higher intra-topic entity relationships 
2. Extraction of neuromodulation/neurostimulation parameters for epilepsy
    - Uses SpaCy's pre-trained large NLP model and regex/rule-based tools to extract neurostimulation parameters (i.e., frequency, amperage, voltage, location, modality, disease type) and study sample size
    - Matplotlib, Networkx, Pyvis for visualization of extracted neurostimulation parameters scaled by sample size and relationships between neurostimulation location, modality, and epilepsy type
    

**Usage**
- In `main.py` there are `if` statement guards on separate pipelines which can be activated by changing the guard's `if` statement evaluation to `True`
    - Comments adjacent to the `if` guard give a brief description of what each pipeline does
    - Parameters immediately below the `if` statement guards can be changed to adjust how the pipeline generates its outputs without having to understand the underlying functions/classes
- Some graphical outputs are saved to the `figures/` directory which needs to be created locally before these outputs can be generated (some graphical outputs are quite large so `figures/` is included in `.gitignore` to avoid tracking these files)
- Note that to use GPT3 parsing, an API key needs to be generated and put into `private/keys.py` with the variable name `KEY1` so that it can be imported from `models_api.py`

**Demo figures**
![Figure](/data/zdemo1.png)
![Figure](/data/zdemo1-1.png)
![Figure](/data/zdemo2.png)
![Figure](/data/zdemo3.png)
![Figure](/data/zdemo4.png)
![Figure](/data/zdemo5.png)
![Figure](/data/zdemo6.png)