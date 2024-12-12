import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from transformers import pipeline
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn import metrics
from datasets import Dataset

from head_masking_analysis import head_masking_analysis

if torch.cuda.is_available():
    device = torch.device('cude')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

dataset_path = "./code/test.csv"
df = pd.read_csv(dataset_path)
df = df.drop("Unnamed: 0", axis=1)

test_dataset = Dataset.from_pandas(df)

sm = nn.Softmax()

def evaluate(model, tokenizer):
    """
    This function evaluates the model on test dataset (already loaded). 

    :param: model - a transformer model suited for sentiment classification.
    :param: tokenizer - tokenizer of the model that returns the word embeddings of each word/token in the document  
    :return: Accuracy score.
    """

    test_strings = [d["text"] for d in test_dataset]
    
    pipe = pipeline("text-classification", model = model, tokenizer = tokenizer, device=device)
    test_pred = pipe(test_strings)
    test_pred_class = [int(p["label"][-1]) for p in test_pred]
    test_true_class = df['label'].tolist()
    test_acc = metrics.accuracy_score(test_true_class, test_pred_class)
    return test_acc


def apply_mask(model, head_name):
    """
    Applys a mask by multiplying the parameters of a head with 0.

    :param: model - the trained model
    :param: the attention matrix of the head to be masked. e.g. 'transformer.h.2.attn.c_proj'
    :return: a copy of the model passed with the mask applied.
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

def get_multi_head_masked_models(main_model, mask_list):
    """
    Given a list of specific heads to mask, this function calls apply_mask() to apply and get a new variant.
    
    :param: main_model - transformer model suited for sentiment classification (usually a baseline).
    :param: mask_list - a list of heads to mask.
    :return: one modified model where all the heads mentioned in mask_list are masked.
    """

    final_model = copy.deepcopy(main_model)
    for i in mask_list:
        attention_head_name = f'transformer.h.{i}.attn.c_proj'
        final_model = apply_mask(final_model, attention_head_name)
    return final_model

def get_one_head_masked_models(main_model):
    """
    Given a main_model, this function calls apply_mask() to apply and get a new variant.
    
    :param: main_model - transformer model suited for sentiment classification (usually a baseline).
    :return: a list of variant models. each element in the list is a model with one head masked.
    """
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

def pass_text_to_model(inputs, model):
    """
    This function takes in the tokenized inputs and 
    the model to return the probability distribution of each class
    """

    ## collect prob vectors
    with torch.no_grad():
        logits = model(**inputs).logits
    softmax_probs = sm(logits)
    # print(softmax_probs)
    return softmax_probs

def head_masking_inference(tokenizer, main_model, variation_list, path):
    """
    For every test example, fetch the baseline pronbability distribution and the variant models' probability distribution.
    :param: tokenizer - converts the text into embeddings
    :param: main_model - baseline model
    :param: variation_list - list of variant models
    :return: a pandas dataframe of consolidated results 
    """

    d = {
        "Text": [],
        'True':[],
        'mask':[],
        '0':[],
        '1':[],
        '2':[],
        '3':[],
        '4':[],
        '5':[]
    }
    samples_folder = './code/samples'
    sample_files = [f for f in os.listdir(f"{samples_folder}")]

    if not os.path.exists(path):
        os.mkdir(path)
                
    for v_i, model in enumerate(variation_list):

        for sample_file in sample_files:
            
            print(f"Processing file: {sample_file}")
            df = pd.read_csv(os.path.join(samples_folder, sample_file))

            labels = df['label'].values.tolist()
            texts = df['text'].tolist()

            for text, label in zip(texts, labels):
                d['Text'].append(text)
                d['True'].append(label)
                inputs = tokenizer(text, return_tensors='pt')
                
                # Baseline result
                with torch.no_grad():
                    logits = main_model(**inputs).logits
                probs = sm(logits).tolist()[0]
                d['mask'].append(-1)
                d['0'].append(probs[0])
                d['1'].append(probs[0])
                d['2'].append(probs[0])
                d['3'].append(probs[0])
                d['4'].append(probs[0])
                d['5'].append(probs[0])

                
                diff_probs = pass_text_to_model(inputs, model).tolist()[0]
                d['Text'].append(text)
                d['True'].append(label)
                d['mask'].append(v_i)
                d['0'].append(diff_probs[0])
                d['1'].append(diff_probs[0])
                d['2'].append(diff_probs[0])
                d['3'].append(diff_probs[0])
                d['4'].append(diff_probs[0])
                d['5'].append(diff_probs[0])
    
    df = pd.DataFrame(d)
    return df


if __name__ == "__main__":

    # load the trained model:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    main_model = GPT2ForSequenceClassification.from_pretrained("./code/model")
    tokenizer.pad_token = tokenizer.eos_token

    # get variations
    variation_list = get_one_head_masked_models(main_model=main_model)
    
    # create a folder to save files.
    path = 'output_logs'

    if os.path.exists(path+'/results.txt'):
        os.remove(path+'/results.txt')

    # df_head_masking = head_masking_inference(tokenizer, main_model, variation_list, path)
    # df_head_masking.to_csv(path+'/head_masking.csv')

    # Analyse each variant's performance.
    head_masking_analysis(path+'/head_masking.csv', path+'/head_masking_class_prob_diff.npy')
    read_dictionary = np.load(path+'/head_masking_class_prob_diff.npy',allow_pickle='TRUE').item()
    with open(f"{path}/results.txt", 'a') as text_file:
        text_file.write('Average correct label drop in probability from for each variant w.r.t baseline:\n')
        text_file.write(str(read_dictionary))
        text_file.write("\n\n")

    # mask all heads except 0.
    mask_list = [i for i in range(1, 12)]
    new_model = get_multi_head_masked_models(main_model, mask_list)
    df_multi_head_masking = head_masking_inference(tokenizer, main_model, [new_model], path)
    df_multi_head_masking.to_csv(path+'/multi_head_masking.csv')
    head_masking_analysis(path+'/multi_head_masking.csv', path+'/multi_head_masking_class_prob_diff.npy')
    read_dictionary = np.load(path+'/multi_head_masking_class_prob_diff.npy',allow_pickle='TRUE').item()
    with open(f"{path}/results.txt", 'a') as text_file:
        text_file.write("Average correct label drop in probability from for multi head masked w.r.t baseline:\n")
        text_file.write(str(read_dictionary))
    