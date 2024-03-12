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
    get_prefix_suffix_tokens_for_HMM,
    get_sequence_scores,
    ConstraintLogitsProcessor,
    get_operation
)

app = Flask(__name__)

@app.route('/prompt/',methods=['POST'])
def prompt():
    # Get the text and operation
    input_json = request.json
    RawPrefix, Prior, Suffix, Instruct = input_json['Prefix'], input_json['Prior'], input_json['Suffix'], input_json['Instruct']
    Prefix = RawPrefix.rstrip(" ")
    Operation = get_operation(
        Prefix,
        Prior,
        Suffix,
        llama_insertion = args.llama_insertion # If set to true, use "Insertion" for the insertion prompt
    )
    # Get the constraints
    token_constraint, word_contraint, keyword_constraint = input_json["token_constraint"], input_json["word_contraint"], input_json["keyword_constraint"]
    max_tokens = max([token_range[1] for token_range in token_constraint])

    # Get generation config
    temperature = input_json['temperature']
    num_return_sequences = input_json['num_return_sequences']
    num_beams = input_json['num_beams']
    no_repeat_ngram_size = input_json['no_repeat_ngram_size']
    top_p = input_json['top_p']

    # TODO
    if word_contraint != []:
        max_tokens = max(max_tokens, int(1.5 * word_contraint[1]))

    # Get prefix, suffix tokens for HMM
    prefix_tokens, suffix_tokens = get_prefix_suffix_tokens_for_HMM(Prefix, Suffix, tokenizer)

    # Construct DFA graph
    dfa_graphs = []
    if len(keyword_constraint) != 0:
        print("Build Keyword")
        dfa_graphs.append(keyphrase_builder.build(keyword_constraint))
    if len(word_contraint) != 0:
        print("Build Word Length")
        dfa_graphs.append(word_count_builder.build(word_contraint[0], word_contraint[1]))
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

    input_ids = torch.tensor([prompt_tokens] * num_beams, device=device)

    with torch.no_grad():
        past_key_values = llama_model(input_ids[:, :-1], return_dict=True).past_key_values

        model_kwargs = {
            'past_key_values': past_key_values,
        }

        hmm_model.initialize_cache(prefix_tokens, suffix_tokens,
            token_constraint, dfa_model)

        results = []

        # Init logits processors
        for token_range in token_constraint:
            min_tokens_, max_tokens_ = token_range
            max_length = len(prompt_tokens) + max_tokens_

            hmm_config = {
                'hmm_prompt_len': len(prompt_tokens),
                'hmm_prefix': prefix_tokens,
                'hmm_suffix': suffix_tokens,
                'hmm_generation_offset': len(prefix_tokens),
                'hmm_min_tokens': min_tokens_,
                'hmm_max_tokens': max_tokens_,
                'hmm_batch_size': args.hmm_batch_size,
            }

            stopping_criteria = StoppingCriteriaList([
                MaxLengthCriteria(max_length=max_length)])
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

            # clean up output
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

                check_suffix_len = min(1, len(suffix_tokens))

                mask1.append([0.0] * len(prompt_tokens) + [1.0] * len(seq) + [0.0] * len(suffix_tokens))
                mask2.append([0.0] * (len(prompt_tokens)+len(seq)) + [1.0] * check_suffix_len + [0.0] * (len(suffix_tokens)-check_suffix_len))
            
            max_len = max([len(x) for x in sequence_ids])
            sequence_ids = [x + [0] * (max_len - len(x)) for x in sequence_ids]            
            mask1 = [x + [0.0] * (max_len - len(x)) for x in mask1]
            mask2 = [x + [0.0] * (max_len - len(x)) for x in mask2]
            sequence_ids = torch.tensor(sequence_ids, device=device)
            mask1 = torch.tensor(mask1, device=device)
            mask2 = torch.tensor(mask2, device=device)

            # TODO use both scores
            # Get returns
            _, sequences_scores = get_sequence_scores(llama_model,
                sequence_ids, mask1, mask2, past_key_values)

            # Here to deal with the space after prefix and before suffix by adding the tokens back to decode the entire story, and remove the prefix, suffix text
            real_prefix_tokens = tokenizer.encode(RawPrefix)
            real_suffix_tokens = tokenizer.encode(Suffix)
            outputs_texts = []
            for output_id in output_ids:
                if Suffix != '':
                    output_id = real_prefix_tokens + output_id + real_suffix_tokens
                else:
                    output_id = real_prefix_tokens + output_id + [29889] + real_suffix_tokens
                output_text = tokenizer.decode(output_id, skip_special_tokens=True)
                output_text = output_text[len(RawPrefix):] # Remove prefix again
                if len(Suffix) > 0:
                    output_text = output_text[:-len(Suffix)] # Remove suffix again
                outputs_texts.append(output_text)

            # outputs_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            sequence_rank = torch.argsort(sequences_scores, descending=True).tolist()

            output_texts_top_n = []
            sequences_scores_top_n = []
            for idx in sequence_rank[:num_return_sequences]:
                output_texts_top_n.append(outputs_texts[idx])
                sequences_scores_top_n.append(sequences_scores[idx].item())

            if args.debug:
                print("-------------------- Top-10 -----------------------")
                sequence_rank = torch.argsort(sequences_scores, descending=True).tolist()
                top_k = 64
                for sequence_idx in sequence_rank[:top_k]:
                    print(f"#{sequence_idx}: {sequences_scores[sequence_idx]}", outputs_texts[sequence_idx])

            results.append({
                "token_range": token_range,
                "beam_outputs_texts": output_texts_top_n,
                "beam_outputs_sequences_scores": sequences_scores_top_n,
            })

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
    # device = f"{args.device}:{args.cuda_core}"
    # print(device)
    # print(torch.cuda.is_available())
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core
    # torch.cuda.set_device(int(args.cuda_core))

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
    app.run(
        host='0.0.0.0',
        port=args.port,
        threaded = False, # Add this
    )
