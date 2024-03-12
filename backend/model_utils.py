from HMM.hmm_model import *
from transformers import LogitsProcessor
import torch

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
    if prefix_tokens[-1] == 29871:
        prefix_tokens = prefix_tokens[:-1]
    prefix_tokens = tuple(prefix_tokens)

    if suffix != '':
        # Cannot strip the suffix at the start. Need to keep the \n
        if suffix[0] == " ":
            suffix = suffix[1:]
        suffix_tokens = tokenizer.encode(suffix.rstrip(" "))[1:] # Strip the right side
        if suffix_tokens[0] == 29871:
            suffix_tokens = suffix_tokens[1:]
        suffix_tokens = tuple(suffix_tokens + [2])
    else:
        # suffix_tokens = tuple([2])
        suffix_tokens = tuple([29889])
    return prefix_tokens, suffix_tokens


def get_sequence_scores(llama_model, input_ids, mask1, mask2, past_key_values, batch_size=32):
    n, d = input_ids.shape
    kv_len = past_key_values[0][0].shape[2]
    print(kv_len)
    with torch.no_grad():
        logits = llama_model(input_ids[:, kv_len:], past_key_values=past_key_values).logits[:, :-1, :]
        logits = torch.log_softmax(logits, dim=-1)
        log_probs = logits[torch.arange(n)[:, None],
            torch.arange(d-kv_len-1)[None, :],
            input_ids[:, kv_len+1:]]

    scores1 = torch.sum(log_probs * mask1[:, kv_len+1:], dim=-1)
    scores2 = torch.sum(log_probs * mask2[:, kv_len+1:], dim=-1)

    return scores1, scores2


class ConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, hmm_model, hmm_config, temperature=1.0):
        self.hmm_model = hmm_model
        self.hmm_config = hmm_config
        self.temperature = temperature


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        hmm_prompt_len = self.hmm_config['hmm_prompt_len']
        hmm_prefix = self.hmm_config['hmm_prefix']
        hmm_suffix = self.hmm_config['hmm_suffix']
        hmm_generation_offset = self.hmm_config['hmm_generation_offset']
        hmm_min_tokens = self.hmm_config['hmm_min_tokens']
        hmm_max_tokens = self.hmm_config['hmm_max_tokens']
        hmm_batch_size = self.hmm_config['hmm_batch_size']

        prefixes = [tuple(hmm_prefix) + tuple(prefix)
            for prefix in input_ids[:,hmm_prompt_len:].tolist()]

        hmm_logits, hmm_logits_ = self.hmm_model.compute_logits(
            prefixes, hmm_suffix,
            hmm_generation_offset,
            hmm_min_tokens, hmm_max_tokens,
            batch_size=hmm_batch_size)

        hmm_logits -= hmm_logits_
        hmm_logits = torch.cat((hmm_logits, -1e10 * torch.ones((hmm_logits.shape[0], 1), device=scores.device)), dim=1)

        logits = torch.log_softmax(scores, dim=-1)
        logits = torch.log_softmax(hmm_logits + logits, dim=-1)
        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        return logits

