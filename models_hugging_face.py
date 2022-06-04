# Tests using hugging face module
#%% Imports
from transformers import pipeline, AutoTokenizer, AutoModel
from pprint import pprint

#%% Constants SciBert
# Something about using autoclasses to import models doesn't allow execution, should use model name in the pipeline call if the model is available in the hub
# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
# model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
#%% Mask fill 
model_name = 'allenai/scibert_scivocab_uncased'
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
unmasker = pipeline("fill-mask", model=model_name, tokenizer=model_name)
text = f"""Patient presented with five days of increased urinary frequency, {unmasker.tokenizer.mask_token} and dysuria as well as 48 hours of fever and rigors. He was hypotensive and tachycardic upon arrival to the emergency department. The internal medicine service was consulted. The following issues were addressed during the hospitalization: 
"""
output = unmasker(text)
pprint(output)
# %% NER
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
model_name = 'allenai/scibert_scivocab_uncased'
ner_pipe = pipeline("ner", model=model_name, tokenizer=model_name)

sequence = """For the subset of 37 patients lacking neuropsychiatric and substance abuse history, MRI surpassed all other predictors for both 3- and 6-month outcome prediction. This is the first study to compare DTI in individual mTBI patients to conventional imaging, clinical, and demographic/socioeconomic characteristics for outcome prediction. DTI demonstrated utility in an inclusive group of patients with heterogeneous backgrounds, as well as in a subset of patients without neuropsychiatric or substance abuse history."""
for entity in ner_pipe(sequence):
    print(entity)
# %% Other
model_name = 'allenai/scibert_scivocab_uncased'
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
pipe = pipeline("text-classification", model=model_name, tokenizer=model_name)

text = """This patient has cancer"""
output = pipe(text)
pprint(output)

# %%
