import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LanguageModel(nn.Module):
    def __init__(self, model_name):
        super(LanguageModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

