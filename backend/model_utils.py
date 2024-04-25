from HMM.hmm_model import *
from transformers import LogitsProcessor
import torch
import json
from string import Template
import string

def get_operation(prefix, prior, suffix, llama_insertion = False):
    # if llama_insertion is true, use Insertion for the insertion prompt
    have_prefix = prefix.strip() != ""
    have_prior = prior.strip() != ""
    have_suffix = suffix.strip() != ""
    if have_prior: # +prior, +-prefix, +-suffix
        operation = "Rewrite"
    elif have_suffix:  # -prior, +-prefix, +suffix
        # Insertion
        if llama_insertion:
            operation = "Insertion"
        else:
            operation = "Continuation"
    elif have_prefix: # -prior, +prefix, -suffix
        operation = "Continuation"
    else: # -prior, -prefix, -suffix
        operation = "Write"
    return operation

# Vector: TypeAlias = list[float]
# def get_gpt_prompt(prefix, prior, suffix, keyword_constraint: list[str], word_range: list[int], token_range: list[int], instruction: str):
def get_gpt_prompt(prefix, prior, suffix, keyword_constraint, word_range, token_range, instruction):
    # Including 3 basic prompts: [Continuation, Insertion, Rewrite]
    # Also includes 6 types of constraints: [Freeform, Keyword, Wordlength, Tokenlength(Deprecated), Keyword+Wordlength, Keyword+Tokenlength(Deprecated)]
    # We do not do token constaints in our interface, so the types of constraints will be [Freeform, Keyword, Wordlength, Keyword+Wordlength]
    prompt_templates = json.load(open("Prompt_Templates.json", "r"))
    # First decide the basic operations.
    have_prior = prior.strip() != ""
    have_suffix = suffix.strip() != ""
    if have_prior: # +prior, +-prefix, +-suffix
        operation = "Rewrite"
    elif have_suffix:  # -prior, +-prefix, +suffix
        operation = "Insertion"
    else: # -prior, +prefix, -suffix
        operation = "Continuation"
    # Check the constraints
    constraints = []
    if keyword_constraint != []:
        constraints += ["Keyword"]
    if len(word_range) > 0:
        constraints += ["Wordlength"]
    else:
        word_range = ["", ""] # For later formatting
        # If not word constraints, see if use token constraints
        if len(token_range) > 0:
            constraints += ["Tokenlength"]
    if not ("Tokenlength" in constraints):
        # Reset token range
        token_range = ["", ""]

    constraints_str = "+".join(constraints)
    if constraints_str == "": # no constraints:
        constraints_str = "Freeform"
    print(f"Operation: {operation}, constraints_str: {constraints_str}")
    GPT_prompt_template = Template(prompt_templates[operation][constraints_str])
    # Format the prompt
    GPT_prompt = GPT_prompt_template.safe_substitute(
        keyword = ", ".join(keyword_constraint),
        word_range_0 = word_range[0],
        word_range_1 = word_range[1],
        token_range_0 = token_range[0],
        token_range_1 = token_range[1],
        prefix = prefix,
        prior = prior,
        suffix = suffix,
        instruction = instruction
    )
    return GPT_prompt



def encode_with_messages_format(Prefix, SoftControl, Suffix, Prior, tokenizer, operation):
    if operation == "Insertion":
        # Strip prefix and suffix:
        Prefix = Prefix.strip()
        Suffix = Suffix.strip()
    operation_prompts ={
        'Write' : "Write a story{SoftControl}.",
        'Continuation' : "Continue the given text{SoftControl}:\n{Prefix}",
        'Insertion' : "Generate the text{SoftControl} at [INSERT_TEXT] tag:\n{Prefix} [INSERT_TEXT] {Suffix}",
        'Rewrite' : "Continue the Prefix by rewriting \"{Prior}\"{SoftControl}:\nPrefix: {Prefix}",
        # 'RewriteOnly' : "Rewrite the given text{SoftControl}:\n{Prior}", # Paraphrase

    }
    prompt = operation_prompts[operation].format(
        Prefix = Prefix,
        SoftControl = SoftControl,
        Prior = Prior,
        Suffix = Suffix,
    )
    message_text = "<|user|>\n" + prompt.strip() + "\n"
    message_text += "<|assistant|>\n"
    tokenized_example = tokenizer.encode(message_text)
    input_ids = tokenized_example
    # Do not get labels here, no need to
    return input_ids


def hash_hmm_status(prefix_tokens, suffix_tokens,
    token_constraint, word_constraint, keyword_constraint, banword_constraint, Suffix):

    return (tuple(prefix_tokens),
        tuple(suffix_tokens),
        tuple(sorted([tuple(x) for x in token_constraint])),
        tuple(word_constraint),
        tuple(sorted(keyword_constraint)),
        tuple(sorted(banword_constraint)),
        Suffix == '')


def get_prefix_suffix_tokens_for_HMM(prefix, suffix, tokenizer):
    """To get the prefix and suffix tokens for HMM
        1. Remove additional space, new lines at the start of prefix and the end of suffix using ltrip() and rstrip()
        2. Remove additional SPIECE_UNDERLINE token(29871) at the end of prefix and the start of suffix.
    Args:
        prompt (str): The example prompt
        suffix (str): The example suffix
    """

    prefix_tokens = tokenizer.encode(prefix)[1:]
    # 3. Remove additional SPIECE_UNDERLINE token(29871)
    if len(prefix_tokens) > 0:
        if prefix_tokens[-1] == 29871:
            prefix_tokens = prefix_tokens[:-1]
    prefix_tokens = tuple(prefix_tokens)

    if suffix.strip() != '':
        # Cannot strip the suffix at the start. Need to keep the \n
        suffix = suffix.lstrip(' ')
        if suffix[0] in string.punctuation:
            # if the first character of suffix is a character (i.e. punctuation), we need to add \n to before that to get the correct token_id
            # Remove 29871 + 13, which have two tokens
            suffix_tokens = tokenizer.encode('\n' + suffix.rstrip(' '))[3:]
        else: 
            suffix_tokens = tokenizer.encode(suffix.rstrip(' '))[1:]
        if suffix_tokens[0] == 29871:
            suffix_tokens = suffix_tokens[1:]
        suffix_tokens = tuple(suffix_tokens + [2])
    else:
        # suffix_tokens = tuple([2])
        suffix_tokens = tuple([29889])
    return prefix_tokens, suffix_tokens


def get_sequence_scores(llama_model, output_ids,
    prompt_tokens, suffix_tokens, past_key_values):
    device = llama_model.device

    # preprocessing input_ids
    input_ids = []
    mask1, mask2 = [], []
    check_suffix_len = min(5, len(suffix_tokens))
    for output_id in output_ids:
        input_ids.append(list(prompt_tokens) + list(output_id) + list(suffix_tokens))
        mask1.append([0.0] * len(prompt_tokens) + [1.0] * len(output_id) + [0.0] * len(suffix_tokens))
        mask2.append([0.0] * (len(prompt_tokens)+len(output_id)) + [1.0] * check_suffix_len + [0.0] * (len(suffix_tokens)-check_suffix_len))

    max_len = max([len(x) for x in input_ids])
    input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
    mask1 = [x + [0.0] * (max_len - len(x)) for x in mask1]
    mask2 = [x + [0.0] * (max_len - len(x)) for x in mask2]
    input_ids = torch.tensor(input_ids, device=device)
    mask1 = torch.tensor(mask1, device=device)
    mask2 = torch.tensor(mask2, device=device)

    # lm forward
    n, d = input_ids.shape
    kv_len = past_key_values[0][0].shape[2]
    with torch.no_grad():
        logits = llama_model(input_ids[:, kv_len:], past_key_values=past_key_values).logits[:, :-1, :]
        logits = torch.log_softmax(logits, dim=-1)
        log_probs = logits[torch.arange(n)[:, None],
            torch.arange(d-kv_len-1)[None, :],
            input_ids[:, kv_len+1:]]

    scores1 = torch.sum(log_probs * mask1[:, kv_len+1:], dim=-1)
    scores2 = torch.sum(log_probs * mask2[:, kv_len+1:], dim=-1)

    return scores1, scores2

def update_prediction_cache(CACHE, content, prediction, is_background):
    CACHE[content['session_id']] = {
        "content": content,
        "prediction": prediction,
        "is_background": is_background,
    }

def merge_predictions(dict_1, dict_2):
    # each arg is a Dict[Dict[List]]
    original_item_num = 0
    merged_item_num = 0
    merged_dict = json.loads(json.dumps(dict_1)) # Copy
    for key in dict_1.keys(): # Dict[Dict[List]]
        for key_key in dict_1[key]: # Dict[List]
            if key_key == "beam_outputs_texts":
                original_item_num += len(merged_dict[key][key_key])
            merged_dict[key][key_key] += dict_2[key][key_key] # Merge the list
            if key_key == "beam_outputs_texts":
                merged_item_num += len(merged_dict[key][key_key])
    print(f"Merged the predictions from len: {original_item_num} to len: {merged_item_num}")
    return merged_dict

def check_have_cache(CACHE, content):
    content = json.loads(json.dumps(content))
    # Ignore these two keys
    content.pop("suggestions", None)
    content.pop("background_query", None)
    
    session_id = content['session_id']
    EXIST_CACHE = False
    if session_id in CACHE:
        CACHE[session_id]['content'].pop("suggestions", None)
        CACHE[session_id]['content'].pop("background_query", None)
        if CACHE[session_id]['content'] == content:
            EXIST_CACHE = True
    return session_id, EXIST_CACHE

def check_background_cache(CACHE, content):
    # Check this before really getting the prediction. If exist cache with is_background == True, skip the prediction
    is_background = "background_query" in content
    
    session_id, EXIST_CACHE = check_have_cache(CACHE, content)
    
    # only when currently it's not background query and content are same, we direclty use the cache stored by previous background query
    if EXIST_CACHE and (not is_background) and CACHE[session_id]['is_background']:
        # Exist and is using background cache. Skip model generation...
        return True
    else:
        # Doesn't exist background cache...
        return False
        
        
        
        
def retrieve_prediction_cache(CACHE, content, prediction):
    is_background = "background_query" in content
    
    session_id, EXIST_CACHE = check_have_cache(CACHE, content)
    if EXIST_CACHE:
        print(f"Find previous cache.")
        if len(prediction) > 0:
            # Further merge the predictions
            merged_prediction = merge_predictions(prediction, CACHE[session_id]['prediction'])
        else:
            # No current prediction, directly use the old one. When querying the server and the server have a background cache, we use the cache without generating
            merged_prediction = CACHE[session_id]['prediction']
    else: # New
        print(f"No previous cache.")
        merged_prediction = prediction
    # Update the CACHE
    update_prediction_cache(CACHE, content, merged_prediction, is_background = is_background)
    return merged_prediction

class ConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, hmm_model, hmm_config, temperature=1.0):
        self.hmm_model = hmm_model
        self.hmm_config = hmm_config
        self.temperature = temperature


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        eos_token_id, pad_token_id = 2, 0
        hmm_prompt_len = self.hmm_config['hmm_prompt_len']
        hmm_prefix = self.hmm_config['hmm_prefix']
        hmm_suffix = self.hmm_config['hmm_suffix']
        hmm_generation_offset = self.hmm_config['hmm_generation_offset']
        hmm_token_ranges = self.hmm_config['hmm_token_ranges']
        hmm_batch_size = self.hmm_config['hmm_batch_size']

        prefixes = [tuple(hmm_prefix) + tuple(prefix)
            for prefix in input_ids[:,hmm_prompt_len:].tolist()]

        if len(prefixes[0]) > 0:
            selected_idx = [i for i, prefix in enumerate(prefixes)
                if prefix[-1] != eos_token_id and prefix[-1] != pad_token_id]
        else:
            selected_idx = [i for i, _ in enumerate(prefixes)]
        selected_prefixes = [prefixes[i] for i in selected_idx]
        selected_token_ranges = [hmm_token_ranges[i] for i in selected_idx]

        hmm_logits, hmm_logits_ = self.hmm_model.compute_logits(
            selected_prefixes, hmm_suffix,
            hmm_generation_offset,
            selected_token_ranges,
            batch_size=hmm_batch_size)

        hmm_logits -= hmm_logits_
        hmm_logits = torch.cat((hmm_logits, -1e30 * torch.ones((hmm_logits.shape[0], 1), device=scores.device)), dim=1)
        logits = torch.log_softmax(scores, dim=-1)
        logits[selected_idx, :] += hmm_logits
        logits = torch.log_softmax(logits, dim=-1)
        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        return logits


class SuffixNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len, suffix_tokens, suffix_no_repeat_ngram_size=3):
        self.config = {
            'prompt_len': prompt_len,
            'suffix_tokens': suffix_tokens,
            'suffix_no_repeat_ngram_size': suffix_no_repeat_ngram_size,
        }


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        neginf_cuda = -1e30 * torch.ones(1, device=scores.device)

        generation_len = input_ids.shape[1] - self.config['prompt_len']
        if generation_len == self.config['suffix_no_repeat_ngram_size'] - 1:
            for i in range(0, input_ids.shape[0]):
                if input_ids[i, -generation_len:].tolist() == self.config['suffix_tokens'][:generation_len]:
                    scores[i, self.config['suffix_tokens'][generation_len]] = neginf_cuda

        return scores