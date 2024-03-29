"""
Starts a Flask server that response with the local model
"""

from flask import Flask, render_template, request
import argparse
import os

from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import NoRepeatNGramLogitsProcessor
from transformers import BeamSearchScorer, LogitsProcessorList, LogitsProcessor, StoppingCriteriaList, StoppingCriteria, MaxLengthCriteria

from HMM.hmm_model import *
from HMM.DFA_model import *
from model_utils import (
    encode_with_messages_format,
    hash_hmm_status,
    get_prefix_suffix_tokens_for_HMM,
    get_sequence_scores,
    ConstraintLogitsProcessor,
    get_operation
)

app = Flask(__name__)

kv_cache = {}
hmm_status = None

@app.route('/prompt/',methods=['POST'])
def prompt():
    # Get the text and operation
    input_json = request.json
    print(input_json)
    return prompt_(input_json)


def prompt_(input_json):
    RawPrefix, Prior, Suffix, Instruct = input_json['Prefix'], input_json['Prior'], input_json['Suffix'], input_json['Instruct']
    Prefix = RawPrefix.rstrip(" ")

    Operation = get_operation(
        Prefix,
        Prior,
        Suffix,
        llama_insertion=args.llama_insertion # If set to true, use "Insertion" for the insertion prompt
    )
    # Get the constraints
    token_constraint, word_constraint, keyword_constraint = input_json["token_constraint"], input_json["word_constraint"], input_json["keyword_constraint"]
    max_tokens = max([token_range[1] for token_range in token_constraint])
    num_token_ranges = len(token_constraint)

    # Get generation config
    temperature = input_json['temperature']
    num_beams = input_json['num_beams']
    no_repeat_ngram_size = input_json['no_repeat_ngram_size']
    top_p = input_json['top_p']

    # TODO Maybe we should let model_server decide token_constraint based on word_constraint?
    if len(word_constraint) != 0:
        max_tokens = max(max_tokens, int(1.5 * word_constraint[1]))

    # Get prefix, suffix tokens for HMM
    prefix_tokens, suffix_tokens = get_prefix_suffix_tokens_for_HMM(Prefix, Suffix, tokenizer)

    # Construct DFA graph
    dfa_graphs = []
    if len(keyword_constraint) != 0:
        print("Build Keyword")
        dfa_graphs.append(keyphrase_builder.build(keyword_constraint))
    if len(word_constraint) != 0:
        print("Build Word Length")
        dfa_graphs.append(word_count_builder.build(word_constraint[0], word_constraint[1]))
    if Suffix == '':
        dfa_graphs.append(end_sentence_builder.build())

    if dfa_graphs != []:
        dfa_model = DFAModel(DFA_prod(dfa_graphs, mode='intersection'))
    else:
        dfa_model = DFAModel(trivial_builder.build())

    # Get input_ids
    prompt_tokens = encode_with_messages_format(
        Prefix = Prefix,
        SoftControl = Instruct,
        Prior = Prior,
        Suffix = Suffix,
        tokenizer = tokenizer,
        operation = Operation
    )

    input_ids = torch.tensor([prompt_tokens] * (num_beams*num_token_ranges), device=device)

    with torch.no_grad():
        global kv_cache
        global hmm_status

        if tuple(prompt_tokens) not in kv_cache:            
            past_key_values = llama_model(torch.tensor([prompt_tokens[:-1]], device=device)).past_key_values
            kv_cache = {tuple(prompt_tokens): past_key_values}
        else:
            print('cache hit!', num_beams*num_token_ranges)
            past_key_values = kv_cache[tuple(prompt_tokens)]

        # expand past_key_values to match the desired batch size
        past_key_values = tuple([tuple([col.expand(num_beams*num_token_ranges, -1, -1, -1).contiguous()
            for col in row]) for row in past_key_values])

        current_hmm_status = hash_hmm_status(prefix_tokens, suffix_tokens,
            token_constraint, word_constraint, keyword_constraint, Suffix)

        if current_hmm_status != hmm_status:            
            hmm_model.initialize_cache(prefix_tokens, suffix_tokens,
                token_constraint, dfa_model)
            hmm_status = current_hmm_status
        else:
            print('cache hit!')

        model_kwargs = {
            'past_key_values': past_key_values,
        }

        hmm_token_ranges = [token_range for token_range in token_constraint for _ in range(num_beams)]

        hmm_config = {
            'hmm_prompt_len': len(prompt_tokens),
            'hmm_prefix': prefix_tokens,
            'hmm_suffix': suffix_tokens,
            'hmm_generation_offset': len(prefix_tokens),
            'hmm_token_ranges': hmm_token_ranges,
            'hmm_batch_size': args.hmm_batch_size,
        }

        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(prompt_tokens)+max_tokens)])
        logits_processor = LogitsProcessorList([
            ConstraintLogitsProcessor(hmm_model, hmm_config, temperature=temperature)])
        if no_repeat_ngram_size > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))

        # If use beam search
        if args.do_beam_search:
            print("Do beam search")
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=num_beams,
                num_beam_hyps_to_keep=num_beams,
                device=llama_model.device,
            )
            outputs= llama_model.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                **model_kwargs
            )
        else:
            outputs= llama_model.sample(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                **model_kwargs
            )

        # clean up output, removing padding, eos, and (partial) suffix
        sequences = outputs.tolist()
        output_ids = []
        sequence_ids = []
        mask1, mask2 = [], []
        for seq in sequences:
            seq = seq[len(prompt_tokens):]
            while seq[-1] == 0:
                seq = seq[:-1]
            while seq[-1] == 2:
                seq = seq[:-1]

            # remove (partial) suffix from generation
            for i in range(min(len(suffix_tokens), len(seq)), 0, -1):
                if seq[-i:] == list(suffix_tokens[:i]):
                    seq = seq[:-i]
                    break

            output_ids.append(seq)
            sequence_ids.append(list(prompt_tokens) + list(seq) + list(suffix_tokens))

            check_suffix_len = min(5, len(suffix_tokens))

            mask1.append([0.0] * len(prompt_tokens) + [1.0] * len(seq) + [0.0] * len(suffix_tokens))
            mask2.append([0.0] * (len(prompt_tokens)+len(seq)) + [1.0] * check_suffix_len + [0.0] * (len(suffix_tokens)-check_suffix_len))

        # get scores
        max_len = max([len(x) for x in sequence_ids])
        sequence_ids = [x + [0] * (max_len - len(x)) for x in sequence_ids]
        mask1 = [x + [0.0] * (max_len - len(x)) for x in mask1]
        mask2 = [x + [0.0] * (max_len - len(x)) for x in mask2]
        sequence_ids = torch.tensor(sequence_ids, device=device)
        mask1 = torch.tensor(mask1, device=device)
        mask2 = torch.tensor(mask2, device=device)

        generation_scores, suffix_scores = get_sequence_scores(llama_model,
            sequence_ids, mask1, mask2, past_key_values)

        # Here to deal with the space after prefix and before suffix by adding the tokens back to decode the entire story, and remove the prefix, suffix text
        real_prefix_tokens = tokenizer.encode(RawPrefix)
        outputs_texts = []
        for output_id in output_ids:
            if Suffix != '':
                output_id = real_prefix_tokens + output_id
            else:
                output_id = real_prefix_tokens + output_id + [29889]
            output_text = tokenizer.decode(output_id, skip_special_tokens=True)
            output_text = output_text[len(RawPrefix):] # Remove prefix again
            if len(Suffix) > 0 and Suffix[0] != ' ':
                output_text = output_text + ' '
            outputs_texts.append(output_text)

        generation_scores = generation_scores.tolist()
        suffix_scores = suffix_scores.tolist()

        results = []
        for i, token_range in enumerate(token_constraint):
            results.append({
                "token_range": token_range,
                "beam_outputs_texts": [outputs_texts[i * num_beams + j] for j in range(0, num_beams)],
                "beam_outputs_sequences_scores_generation": [generation_scores[i * num_beams + j] for j in range(0, num_beams)],
                "beam_outputs_sequences_scores_suffix": [suffix_scores[i * num_beams + j] for j in range(0, num_beams)],
            })

        print(results)

    return results


def init():
    global device
    global CUDA_CORE
    global args

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--port', type=int, required=True)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='1', type=str)
    arg_parser.add_argument('--hmm_batch_size', default=256, type=int)
    arg_parser.add_argument('--hmm_model_path', default=None, type=str)
    arg_parser.add_argument('--llama_model_path', default='gpt2', type=str)
    arg_parser.add_argument('--suffix_cap', default=10000, type=int)
    arg_parser.add_argument('--do_beam_search', action='store_true')
    arg_parser.add_argument('--llama_insertion', action='store_true', help="If sepecified, provide suffix to the llama model during insertion.")
    arg_parser.add_argument('--debug', action='store_true')

    args = arg_parser.parse_args()
    device = args.device


def load_models():
    global tokenizer
    global llama_model
    global hmm_model
    global keyphrase_builder
    global end_sentence_builder
    global word_count_builder
    global trivial_builder
    try:
        print(f'loading llama2 from {args.llama_model_path} ...')
        llama_model = LlamaForCausalLM.from_pretrained(args.llama_model_path).to(device)
        llama_model.half()
        tokenizer = LlamaTokenizer.from_pretrained(args.llama_model_path)

        print(f'loading hmm from {args.hmm_model_path} ...')
        hmm_model = HMM(args.hmm_model_path)
        hmm_model.to(device)
        print(hmm_model.alpha_exp.device)

        print(f'constructing DFA builders ...')
        keyphrase_builder = KeyphraseBuilder(tokenizer, 32000)
        end_sentence_builder = EndSentenceBuilder(tokenizer, 32000)
        word_count_builder = WordCountBuilder(tokenizer, 32000)
        trivial_builder = TrivialBuilder(tokenizer, 32000)
    except Exception as e:
        print(f"Cannot Load LLamaModel {args.llama_model_path} or HMM Model {args.hmm_model_path} because of the following exception:\n {e}")
        print("Exit the process...")
        exit(0)


if __name__ == '__main__':
    init()
    load_models()

    # warmup
    print('warming up model server...')
    prompt_({
        "Prefix": 'This is a test',
        "Suffix": 'to warm up the model server.',
        "Prior": '',
        "Instruct": 'Continuation',
        'word_constraint': [],
        'keyword_constraint': [],
        'token_constraint': [[1, 8]],
        'temperature': 0.8,
        'num_beams': 4,
        'no_repeat_ngram_size': -1,
        'top_p': 1.0,
    })

    app.run(
        host='0.0.0.0',
        port=args.port,
        threaded = False, # Add this
    )
