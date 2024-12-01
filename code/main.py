import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, GPT2Config
import os

sm = nn.Softmax()
head_masked_probs = {
     "joy": [],
     "sadness": [],
     "anger": [],
     "fear": [],
     "love": [],
     "surprise": []
 }

mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def apply_mask(model, head_name):
    """
    :param: model - the trained model
    :param: the attention matrix of the head to be masked. e.g. 'transformer.h.2.attn.c_proj'
    """

    weight_name = head_name + '.weights'
    bias_name = head_name + '.bias'

    mask_weight = torch.zeros((768, 768))
    mask_bias = torch.zeros(768)

    model_copy = copy.deepcopy(model)

    for name, param in model_copy.named_parameters():
        param.requires_grad = False
        if name == weight_name:
            param *= mask_weight
        elif name == bias_name:
            param *= mask_bias
    
    return model_copy

def get_one_head_masked_models(main_model):
    ## mask one attention head at a time:
    variation_list = []
    n_heads = 12 # MHA

    for i in range(n_heads):
        attention_head_name = f'transformer.h.{i}.attn.c_proj'
        variation_list.append(apply_mask(main_model, attention_head_name))
    return variation_list

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def main(text, tokenizer, variation_list):

    ## collect prob vectors
    output_dict = {}
    for i, model in enumerate(variation_list):
        inputs = tokenizer(text, return_tensors = 'pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        softmax_probs = sm(logits)
        # print(softmax_probs)
        output_dict[i] = softmax_probs
    return output_dict

def pass_text_to_model(inputs, model):

    ## collect prob vectors
    with torch.no_grad():
        logits = model(**inputs).logits
    softmax_probs = sm(logits)
    # print(softmax_probs)
    return softmax_probs

def head_masking(text, tokenizer, variation_list, ):
    inputs = tokenizer(text, return_tensors = 'pt')
    for i, model in enumerate(variation_list):
        probs = pass_text_to_model(inputs, model)
    return probs

if __name__ == '__main__':

    # trained model:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    main_model = GPT2ForSequenceClassification.from_pretrained("./code/model")
    tokenizer.pad_token = tokenizer.eos_token

    # get variations
    variation_list = get_one_head_masked_models(main_model=main_model)
    samples_folder = './code/samples'
    sample_files = [f for f in os.listdir(f"{samples_folder}")]

    path = 'output_logs'
    if not os.path.exists(path):
        os.mkdir(path)
                
    for v_i, model in enumerate(variation_list):
        message = f"%%%%%%%%%%%%%%%%%%%%%%%%%%% VARIANT - {v_i} %%%%%%%%%%%%%%%%%%%%%%%%%%% \n\n"
        with open(f"{path}/results.txt", 'a') as text_file:
            text_file.write(message)
        print("\n---------------------------------\n")
        print(f"Processing variation #{v_i}")
        for sample_file in sample_files:
            message = f"{sample_file} \n\n"
            with open(f"{path}/results.txt", 'a') as text_file:
                text_file.write(message)
            
            print(f"Processing file: {sample_file}")
            df = pd.read_csv(os.path.join(samples_folder, sample_file))

            labels = df['label'].values.tolist()
            texts = df['text'].tolist()

            for text, label in zip(texts, labels):
                message = f"{text}\n"
                
                with open(f"{path}/results.txt", 'a') as text_file:
                    text_file.write(message)
                inputs = tokenizer(text, return_tensors='pt')
                
                # Baseline result
                with torch.no_grad():
                    logits = main_model(**inputs).logits
                probs = sm(logits).tolist()
                class_ = mapping[int(label[-1])]

                message = f"Baseline probabilities: {probs}\n"
                with open(f"{path}/results.txt", 'a') as text_file:
                    text_file.write(message)
                
                # Get the probabilities for one variation
                diff_probs = pass_text_to_model(inputs, model)
                message = f"variant: {diff_probs}\n\n"
                with open(f"{path}/results.txt", 'a') as text_file:
                    text_file.write(message)
        message = "\n---------------------------------\n"
        with open(f"{path}/results.txt", 'a') as text_file:
            text_file.write(message)
       


    '''
    ## get examples per class:

    # text = """
    # I am happy becuase of this movie. The joy i experience is insurmountable.
    # """
    text = """i am very SAD"""
    
    ## baseline result:
    inputs = tokenizer(text, return_tensors = 'pt')
    with torch.no_grad():
        logits = main_model(**inputs).logits
    probs = sm(logits).tolist()
    print(probs)
    print()

    # print(variation_list)

    # ## get the probabilities for each variation:
    # diff_probs = main(text, tokenizer, variation_list)
    # print(diff_probs)
    '''