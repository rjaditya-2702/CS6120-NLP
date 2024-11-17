import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import pipeline
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, GPT2Config

sm = nn.Softmax()

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

def main(text, tokenizer, main_model):

    variation_list = []
    n_heads = 12 # MHA

    ## mask one attention head at a time:
    for i in range(n_heads):
        attention_head_name = f'transformer.h.{i}.attn.c_proj'
        variation_list.append(apply_mask(main_model, attention_head_name))

    ## collect prob vectors
    output_dict = {}
    for i, model in enumerate(variation_list):
        inputs = tokenizer(text, return_tensors = 'pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        softmax_probs = sm(logits)
        print(softmax_probs)
        output_dict[i] = softmax_probs
    return output_dict

if __name__ == '__main__':

    ## trained model:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2ForSequenceClassification.from_pretrained("./model")
    tokenizer.pad_token = tokenizer.eos_token

    ## get examples per class:

    # text = """
    # I am happy becuase of this movie. The joy i experience is insurmountable.
    # """
    text = """Today is a good day. A lot of people minding their own business. None of them include checking on me though!"""
    
    print(len(text))
    ## baseline result:
    inputs = tokenizer(text, return_tensors = 'pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = sm(logits).tolist()
    print(probs)
    print()

    ## get the probabilities for each variation:
    diff_probs = main(text, tokenizer, model)
    print(diff_probs)

    ## comparison:
