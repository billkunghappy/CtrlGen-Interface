import os
import json
import time
import argparse

from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BeamSearchScorer, LogitsProcessorList, LogitsProcessor, StoppingCriteriaList, StoppingCriteria, MaxLengthCriteria

from hmm_model import *
from DFA_model import *

device = 'cuda'

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
        hmm_logits = torch.cat((hmm_logits, -1e10 * torch.ones((hmm_logits.shape[0], 1), device=device)), dim=1)

        logits = torch.log_softmax(scores, dim=-1)
        logits = torch.log_softmax(hmm_logits + logits, dim=-1)
        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        return logits


def init():
    global device
    global CUDA_CORE

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='1', type=str)
    arg_parser.add_argument('--hmm_batch_size', default=256, type=int)

    arg_parser.add_argument('--hmm_model_path', default=None, type=str)
    arg_parser.add_argument('--llama_model_path', default='gpt2', type=str)

    arg_parser.add_argument('--dataset_file', default='', type=str)
    arg_parser.add_argument('--output_file', default='pred.json', type=str)

    arg_parser.add_argument('--num_beams', default=32, type=int)
    arg_parser.add_argument('--length_penalty', default=0.2, type=float)

    args = arg_parser.parse_args()

    device = args.device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core
    torch.cuda.set_device(int(args.cuda_core))

    return args


def select_best(llama_model, input_ids, logits_mask, batch_size=32):
    n, d = input_ids.shape
    log_probs = []
    with torch.no_grad():
        for batch_idx in range(0, n, batch_size):
            batch_size_ = min(batch_size, n - batch_idx)

            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_]
            logits = torch.log_softmax(
                    llama_model(input_ids_batch, return_dict='True').logits, dim=-1)[:, :-1, :]
            log_probs_batch = logits[torch.arange(batch_size_)[:, None],
                torch.arange(d-1)[None, :],
                input_ids_batch[:, 1:]]

            log_probs.append(log_probs_batch)

    log_probs = torch.cat(log_probs, dim=0)
    log_probs *= logits_mask[:, 1:]
    log_probs = torch.sum(log_probs, dim=-1)

    return torch.argmax(log_probs).item()

def main():
    args = init()

    print(f'loading llama2 from {args.llama_model_path} ...')
    llama_model = LlamaForCausalLM.from_pretrained(args.llama_model_path).to(device)
    llama_model.half()
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_model_path)

    print(f'loading hmm from {args.hmm_model_path} ...')
    hmm_model = HMM(args.hmm_model_path)
    hmm_model.to(device)

    print(f'loading dataset from {args.dataset_file} ...')
    with open(args.dataset_file, 'r') as fin:
        examples = json.load(fin)

    print(f'constructing DFA builders ...')
    keyphrase_builder = KeyphraseBuilder(tokenizer, 32000)
    end_sentence_builder = EndSentenceBuilder(tokenizer, 32000)
    word_count_builder = WordCountBuilder(tokenizer, 32000)
    eos_builder = EOSBuilder(tokenizer, 32000)
    trivial_builder = TrivialBuilder(tokenizer, 32000)

    print('generating sequences ...')
    for generation_type in examples:
        examples_ = examples[generation_type]
        print(f'generating sequences {generation_type} ...')
        for example_idx in tqdm(examples_):
            example = examples_[example_idx]
            prompt, prior, suffix = example['prompt'], example['prior'], example['constraints']['suffix']

            max_tokens = 128
            if example['constraints']['wordlength'] != []:
                min_words, max_words = example['constraints']['wordlength'][0]
                max_tokens = int(1.2 * max_words)
                # max_tokens = max(max_tokens, int(1.2 * max_words))
            else:
                prior_len = len(tokenizer.encode(prior))
                max_tokens = int(prior_len * 1.2)

            prefix = prompt[prompt.find('Prefix:')+len('Prefix:')+1:]
            prefix_tokens = tuple(tokenizer.encode(prefix)[1:])
            if suffix != '':
                suffix_tokens = tuple(tokenizer.encode(suffix.strip(' '))[1:] + [2])
            else:
                suffix_tokens = tuple([2])
                # suffix_tokens = tuple()
            prefix_len = len(prefix_tokens)

            prompt = '<|user|>\n' + prompt
            prompt = prompt + '\n' + '<|assistant|>\n'

            dfa_graphs = []
            # dfa_graphs.append(eos_builder.build()) # enforce the eos constraint by default; i.e. eos must be followed by eos
            dfa_graphs.append(trivial_builder.build()) # enforce the eos constraint by default; i.e. eos must be followed by eos

            constraints = example['constraints']
            if constraints['keyword'] != []:
                dfa_graphs.append(keyphrase_builder.build(constraints['keyword'][0]))
            if constraints['wordlength'] != []:
                word_range = constraints['wordlength'][0]
                dfa_graphs.append(word_count_builder.build(word_range[0], word_range[1]))
            if suffix == '':
                dfa_graphs.append(end_sentence_builder.build())

            if dfa_graphs != []:
                dfa_graph = DFA_prod(dfa_graphs, mode='intersection')

            state_cnt, edge_cnt = DFA_size(dfa_graph)
            print(f'dfa size: {state_cnt} states, {edge_cnt} edges')

            dfa_model = DFAModel(dfa_graph)

            prompt_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_tokens] * args.num_beams, device=device)

            with torch.no_grad():
                past_key_values = llama_model(input_ids[:, :-1], return_dict=True).past_key_values

                model_kwargs = {
                    'past_key_values': past_key_values,
                }

                hmm_model.initialize_cache(prefix_tokens, suffix_tokens,
                    [(1, max_tokens)], dfa_model)

                hmm_config = {
                    'hmm_prompt_len': len(prompt_tokens),
                    'hmm_prefix': prefix_tokens,
                    'hmm_suffix': suffix_tokens,
                    'hmm_generation_offset': len(prefix_tokens),
                    'hmm_min_tokens': 1,
                    'hmm_max_tokens': max_tokens,
                    'hmm_batch_size': args.hmm_batch_size,
                }

                max_length = len(prompt_tokens) + max_tokens

                beam_scorer = BeamSearchScorer(
                    batch_size=1,
                    length_penalty=args.length_penalty,
                    num_beams=args.num_beams,
                    num_beam_hyps_to_keep=args.num_beams,
                    device=llama_model.device
                )

                stopping_criteria = StoppingCriteriaList([
                    MaxLengthCriteria(max_length=max_length)])

                logits_processor = LogitsProcessorList([
                    ConstraintLogitsProcessor(hmm_model, hmm_config, temperature=1.0)])

                outputs = llama_model.beam_search(
                    input_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    **model_kwargs
                )

                # clean up output
                sequences = outputs.tolist()
                output_ids = []
                sequence_ids = []
                logits_mask = []
                for seq in sequences:
                    seq = seq[len(prompt_tokens):]
                    while seq[-1] == 0:
                        seq = seq[:-1]
                    while seq[-1] == 2:
                        seq = seq[:-1]
                    for i in range(min(len(suffix_tokens), len(seq)), 0, -1):
                        if seq[-i:] == list(suffix_tokens[:i]):
                            seq = seq[:-i]
                            break
                    output_ids.append(seq)
                    sequence_ids.append(list(prompt_tokens) + list(seq) + list(suffix_tokens))
                    logits_mask.append([0.0] * len(prompt_tokens) + [1.0] * len(seq) + [1.0] * len(suffix_tokens))

                max_len = max([len(x) for x in sequence_ids])
                sequence_ids = [x + [0] * (max_len - len(x)) for x in sequence_ids]
                logits_mask = [x + [0.0] * (max_len - len(x)) for x in logits_mask]
                sequence_ids = torch.tensor(sequence_ids, device=device)
                logits_mask = torch.tensor(logits_mask, device=device)

                best_idx = select_best(llama_model, sequence_ids, logits_mask)
                # best_idx = 0

                output = tokenizer.decode(output_ids[best_idx],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)

                examples[generation_type][example_idx]['output'] = output

                with open(args.output_file, 'w') as fout:
                    json.dump(examples, fout, indent=2)


if __name__ == '__main__':
    main()
