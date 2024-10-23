import os
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from HMM.hmm_model import *
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

#BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen' # a gpt2-large checkpoint domain adapted to the common-gen corpus
HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_32768' # alternatively 'ctrlg/hmm_gpt2-large_common-gen_32768' for better quality

#base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
#base_model.eval()
#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hmm_model = HMM.from_pretrained(HMM_MODEL_PATH).to(device)
