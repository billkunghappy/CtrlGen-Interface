"""
Starts a Flask server that response with the local model
"""

from flask import Flask, render_template, request
import argparse
import os
import warnings
from tqdm import tqdm
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from transformers import NoRepeatNGramLogitsProcessor
from transformers import BeamSearchScorer, LogitsProcessorList, LogitsProcessor, StoppingCriteriaList, StoppingCriteria, MaxLengthCriteria

from model_utils import (
    encode_with_messages_format,
    # hash_hmm_status,
    get_prefix_suffix_tokens_for_HMM,
    get_sequence_scores,
    get_operation,
    SuffixNoRepeatNGramLogitsProcessor,
)

app = Flask(__name__)

kv_cache = {}
hmm_status = None

@app.route('/prompt/',methods=['POST'])
def prompt():
    # Get the text and operation
    input_json = request.json    
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
    token_constraint, word_constraint, keyword_constraint, banword_constraint = input_json["token_constraint"], input_json["word_constraint"], input_json["keyword_constraint"], input_json["banword_constraint"]
    # if len(set(tuple(elem) for elem in token_constraint)) > 1:
    #     warnings.warn("Warning: Deprecated features of token ranges in the model server. Please use only one token range.")
    num_token_ranges = len(token_constraint)
    min_new_tokens = token_constraint[0][0]
    max_new_tokens = token_constraint[0][1]
    # Get the beam size: We fix the total batch size. The number of return sequence is num_beams * num_token_ranges = args.generation_batch_size
    num_beams = args.generation_batch_size//len(token_constraint)
    real_generation_batch_size = num_beams * len(token_constraint)
    
    # Get generation config
    temperature = input_json['temperature']
    no_repeat_ngram_size = input_json['no_repeat_ngram_size']
    top_p = input_json['top_p']

    # TODO Maybe we should let model_server decide token_constraint based on word_constraint?
    if len(word_constraint) != 0:
        max_new_tokens = max(max_new_tokens, int(1.5 * word_constraint[1]))

    # Get prefix, suffix tokens for HMM
    prefix_tokens, suffix_tokens = get_prefix_suffix_tokens_for_HMM(Prefix, Suffix, tokenizer)
    suffix_tokens = suffix_tokens[:args.suffix_cap]

    # Build DFA
    dfa_graphs = []
    
    # Build Keyphrases
    if len(keyword_constraint) != 0:
        print("Build Keyword")
        for keyphrase in keyword_constraint:
            dfa_graphs.append(ac_builder.build([tokenizer.encode(keyphrase)[1:]]))
        dfa_graphs = [ctrlg.DFA_concatenate(dfa_graphs)]
    
    # WordCount Builder
    if len(word_constraint) != 0:
        print("Build Word Count")
        dfa_graphs.append(word_count_builder.build(word_constraint[0], word_constraint[1]))
    
    # EndSentence Builder
    # This can imporve performance when doing continuation. We expect the continuation should end with a endSentence token, such as period.
    if Suffix == '':
        print("Build End of Sentence")
        dfa_graphs.append(eos_builder.build())
    
    # If there is no constraints, use trivial builder
    if dfa_graphs != []:
        dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode='intersection') # logical and
        dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device) # compile for GPU inference
    else:
        dfa_model = ctrlg.DFAModel(trivial_builder.build(), vocab_size).to(device)
        

    # Add soft control in prompt when not using HMM
    if args.llama_only:
        # Enforce the keyword and word length control with soft control
        if len(keyword_constraint) != 0:
            Instruct += f" by including the following keywords: {' ,'.join(keyword_constraint)}"
        if len(word_constraint) != 0:
            Instruct = f" within {word_constraint[0]} to {word_constraint[1]} words"
    
    # Get input_ids
    prompt_tokens = encode_with_messages_format(
        Prefix = Prefix,
        SoftControl = Instruct,
        Prior = Prior,
        Suffix = Suffix,
        tokenizer = tokenizer,
        operation = Operation,
    )
    input_ids = torch.tensor([prompt_tokens] * (real_generation_batch_size), device=device)

    with torch.no_grad():
        # Cache past_key_values for faster inference
        global kv_cache
        global hmm_status

        if tuple(prompt_tokens) not in kv_cache:
            past_key_values = llama_model(torch.tensor([prompt_tokens[:-1]], device=device)).past_key_values
            kv_cache = {tuple(prompt_tokens): past_key_values}
        else:            
            past_key_values = kv_cache[tuple(prompt_tokens)]

        # expand past_key_values to match the desired batch size
        past_key_values = tuple([tuple([col.expand(real_generation_batch_size, -1, -1, -1).contiguous()
            for col in row]) for row in past_key_values])

        model_kwargs = {
            'past_key_values': past_key_values,
        }

        # Build ConstraintLogitsProcessor if using HMM
        if args.llama_only: 
            print("Skip building ConstraintLogitsProcessor because of llama_only argument.")
            logits_processor = LogitsProcessorList([])
        else:
            # initialze the constraints logits processor & pre-computes conditional probabilities            
            constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
                hmm_model, 
                dfa_model,
                min_new_tokens,
                max_new_tokens,
                prompt_tokens, 
                prefix_ids=prefix_tokens,
                suffix_ids=suffix_tokens,
                hmm_batch_size = num_beams
                )
            logits_processor = LogitsProcessorList([constraint_logits_processor])
        
        # Add additional LogitsProcessor (optional)
        if no_repeat_ngram_size > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if args.suffix_no_repeat_ngram_size > 0:
            logits_processor.append(SuffixNoRepeatNGramLogitsProcessor(len(prompt_tokens), suffix_tokens, args.suffix_no_repeat_ngram_size))

        # Start generating
        generation_kwargs = {
            'num_return_sequences': 1,
            'logits_processor': logits_processor,
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_new_tokens,
            'pad_token_id': tokenizer.eos_token_id,
            'temperature': temperature,
            'top_p': top_p
        }
        if args.do_beam_search:
            outputs= llama_model.generate(
                input_ids=input_ids,
                do_sample=False,
                num_beams=num_beams,
                **generation_kwargs,
                **model_kwargs
            )
        else: # Do Sample
            outputs= llama_model.generate(
                input_ids=input_ids,
                do_sample=True,
                num_beams=1,
                **generation_kwargs,
                **model_kwargs
            )

        # clean up output, removing padding, eos, and (partial) suffix
        sequences = outputs.tolist()
        output_ids = []
        for seq in sequences:
            # remove prompt
            seq = seq[len(prompt_tokens):]

            # remove trailing <unk> and <eos>
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

        if args.rank_without_prompt:
            prefix_past_key_values = llama_model(torch.tensor([[1,] + prefix_tokens[:-1]], device=device)).past_key_values
            prefix_past_key_values = tuple([tuple([col.expand(len(output_ids), -1, -1, -1).contiguous()
                for col in row]) for row in prefix_past_key_values])
            generation_scores, suffix_scores = get_sequence_scores(llama_model, output_ids,
                [1,] + prefix_tokens, suffix_tokens, prefix_past_key_values)
        else:
            generation_scores, suffix_scores = get_sequence_scores(llama_model, output_ids,
                prompt_tokens, suffix_tokens, past_key_values)

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
            if len(Suffix) > 0 and (not Suffix[0] in [' ', '.', ',', '!', '?', ';']):
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
        
    if args.eval:
        # Clear cache everytime for evaluation
        del kv_cache
        kv_cache = {}
        # del hmm_status
        # hmm_status = None
        # del hmm_model.cache
        # hmm_model.cache = {}
        torch.cuda.empty_cache()
    
    return results


def init():
    global device
    global CUDA_CORE
    global args

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--port', type=int, required=True)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='1', type=str)
    arg_parser.add_argument('--generation_batch_size', default=128, type=int)
    arg_parser.add_argument('--hmm_model_path', default=None, type=str)
    arg_parser.add_argument('--llama_model_path', default='gpt2', type=str)
    arg_parser.add_argument('--suffix_cap', default=10000, type=int, help="If specified, truncate the suffix to this length.")
    arg_parser.add_argument('--do_beam_search', action='store_true')
    arg_parser.add_argument('--llama_insertion', action='store_true', help="If sepecified, provide suffix to the llama model during insertion.")
    arg_parser.add_argument('--suffix_no_repeat_ngram_size', default=0, type=int)
    arg_parser.add_argument('--rank_without_prompt', action='store_true')
    arg_parser.add_argument('--eval', action='store_true', help="If sepecified, clear cache at the end of every request for evaluation. This can make the server slower.")
    arg_parser.add_argument('--llama_only', action='store_true', help="If specified, do not use the HMM at all. Will use soft control for the wordlen and keyword control.")
    arg_parser.add_argument('--debug', action='store_true')

    args = arg_parser.parse_args()
    device = args.device


def load_models():
    global tokenizer
    global llama_model
    global hmm_model
    global eos_builder
    global trivial_builder
    global vocab_size
    global ac_builder
    global word_count_builder
    try:
        print(f'loading llama2 from {args.llama_model_path} ...')
        llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_path).to(device)
        llama_model.half().eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print(f'loading hmm from {args.hmm_model_path} ...')
        hmm_model = ctrlg.HMM.from_pretrained(args.hmm_model_path).to(device)
        hmm_model.to(device)        

        print(f'constructing DFA builders ...')
        vocab_size = hmm_model.vocab_size
        eos_token_id = hmm_model.eos_token_id
        eos_builder = ctrlg.EOSBuilder(vocab_size, eos_token_id)
        trivial_builder = ctrlg.TrivialBuilder(tokenizer, vocab_size, eos_token_id)
        ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
        word_count_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)
    except Exception as e:
        print(f"Cannot Load LLamaModel {args.llama_model_path} or HMM Model {args.hmm_model_path} because of the following exception:\n")
        raise e


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
        'banword_constraint': [],
        'token_constraint': [[1, 8]],
        'temperature': 0.8,
        'no_repeat_ngram_size': -1,
        'top_p': 1.0,
    })

    app.run(
        host='0.0.0.0',
        port=args.port,
        threaded = False, # Add this
    )
