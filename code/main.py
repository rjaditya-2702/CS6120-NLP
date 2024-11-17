import copy
import torch
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, GPT2Config


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
