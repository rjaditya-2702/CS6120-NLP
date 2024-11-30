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

def main(text, tokenizer, variation_list):

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

def pass_text_to_model(inputs, model):

    ## collect prob vectors
    with torch.no_grad():
        logits = model(**inputs).logits
    softmax_probs = sm(logits)
    print(softmax_probs)
    return softmax_probs

def head_masking(text, tokenizer, variation_list, ):
    inputs = tokenizer(text, return_tensors = 'pt')
    for i, model in enumerate(variation_list):
        probs = pass_text_to_model(inputs, model)
    return probs

if __name__ == '__main__':

    ## trained model:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    main_model = GPT2ForSequenceClassification.from_pretrained("./model")
    tokenizer.pad_token = tokenizer.eos_token

    ## get variations
    variation_list = get_one_head_masked_models(main_model=main_model)

    samples_folder = '/Users/rjaditya/Documents/NEU-SEM/Fall-2024/NLP/project/CS6120-NLP/code/samples'
    sample_files = [f for f in os.listdir(samples_folder) if os.path.isfile(os.path.join(samples_folder, f))]

    exec_path_list = os.path.abspath(__file__).split('/')
    path = ''
    for i in exec_path_list:
        if i == 'code':
            break
        path += i + '/'
    path += 'output_logs/'
    if not os.path.exists(path):
        os.mkdir(path)
    for v_i, model in enumerate(variation_list):
        message = ""
        for sample_file in sample_files:
            if True:
                print(f"Processing file: {sample_file}")
                df = pd.read_csv(os.path.join(samples_folder, sample_file))
                labels = df['labels'].values().tolist()
                text = df['text']
                dataset = Dataset(text.values())
                data_loader = DataLoader(dataset) # [batch, <>]
                
                inputs = tokenizer(data_loader, return_tensors='pt') #[batch, d_model]
                
                # Baseline result
                with torch.no_grad():
                    logits = main_model(**inputs).logits # [batch, 6]
                probs = sm(logits).tolist() # [batch, 6]
                class_ = [mapping[i] for i in labels]
                message += f"Baseline probabilities for {sample_file}: {probs} \n\n"
                
                # Get the probabilities for one variation
                diff_probs = pass_text_to_model(inputs, model) # [batch, 6]
                message += f"Variation probabilities for variation #{v_i}: {diff_probs}\n"
        message += "\n---------------------------------\n"
        with open(f"{path}dummy.txt", 'w') as text_file:
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