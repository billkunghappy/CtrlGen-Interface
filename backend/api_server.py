"""
Starts a Flask server that handles API requests from the frontend.
"""

import os
import gc
import shutil
import random
import openai
import warnings
import numpy as np
import json
from time import time, sleep
from argparse import ArgumentParser
# For Local Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import string

from reader import (
    read_api_keys, read_log,
    read_examples, read_prompts, read_blocklist,
    read_access_codes, update_metadata,
)
from helper import (
    print_verbose, print_current_sessions,
    get_uuid, retrieve_log_paths,
    append_session_to_file, get_context_window_size,
    save_log_to_jsonl, compute_stats, get_last_text_from_log, get_config_for_log,
)
from parsing import (
    parse_prompt, parse_suggestion, parse_probability,
    filter_suggestions,
    custom_filter_suggestions
)
from model_utils import (
    get_gpt_prompt
)

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import aiohttp
import asyncio

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa

SESSIONS = dict()
app = Flask(__name__)
CORS(app)  # For Access-Control-Allow-Origin

SUCCESS = True
FAILURE = False


@app.route('/api/start_session', methods=['POST'])
@cross_origin(origin='*')
def start_session():
    print("start_session")
    content = request.json
    result = {}

    # Read latest prompts, examples, and access codes
    global examples, prompts
    examples = read_examples(config_dir)
    prompts = read_prompts(config_dir)
    allowed_access_codes = read_access_codes(config_dir)

    # Check access codes
    access_code = content['accessCode']
    print(f"Get access code {access_code},allowed_access_codes: {allowed_access_codes}")
    if access_code not in allowed_access_codes:
        if not access_code:
            access_code = '(not provided)'
        result['status'] = FAILURE
        result['message'] = f'Invalid access code: {access_code}. Please check your access code in URL.'
        print_current_sessions(SESSIONS, 'Invalid access code')
        return jsonify(result)

    config = allowed_access_codes[access_code]

    # Setup a new session
    session_id = get_uuid()  # Generate unique session ID
    verification_code = session_id

    # Information returned to user
    result = {
        'access_code': access_code,
        'session_id': session_id,

        'example_text': examples[config.example],
        'prompt_text': prompts[config.prompt],
    }
    result.update(config.convert_to_dict())

    # Information stored on the server
    SESSIONS[session_id] = {
        'access_code': access_code,
        'session_id': session_id,

        'start_timestamp': time(),
        'last_query_timestamp': time(),
        'verification_code': verification_code,
    }
    SESSIONS[session_id].update(config.convert_to_dict())

    result['status'] = SUCCESS

    session = SESSIONS[session_id]
    model_name = result['engine'].strip()
    domain = result['domain'] if 'domain' in result else ''

    append_session_to_file(session, metadata_path)
    print_verbose('New session created', session, verbose)
    print_current_sessions(SESSIONS, f'Session {session_id} ({domain}: {model_name}) has been started successfully.')

    gc.collect(generation=2)
    return jsonify(result)


@app.route('/api/end_session', methods=['POST'])
@cross_origin(origin='*')
def end_session():
    content = request.json
    session_id = content['sessionId']
    log = content['logs']

    path = os.path.join(proj_dir, session_id) + '.jsonl'

    results = {}
    results['path'] = path
    try:
        save_log_to_jsonl(path, log)
        results['status'] = SUCCESS
    except Exception as e:
        results['status'] = FAILURE
        results['message'] = str(e)
        print(e)
    print_verbose('Save log to file', {
        'session_id': session_id,
        'len(log)': len(log),
        'status': results['status'],
    }, verbose)

    # Remove a finished session
    try:
        # NOTE: Somehow end_session is called twice;
        # Do not pop session_id from SESSIONS to prevent exception
        session = SESSIONS[session_id]
        results['verification_code'] = session['verification_code']
        print_current_sessions(SESSIONS, f'Session {session_id} has been saved successfully.')
    except Exception as e:
        print(e)
        print('# Error at the end of end_session; ignore')
        results['verification_code'] = 'SERVER_ERROR'
        print_current_sessions(SESSIONS, f'Session {session_id} has not been saved.')

    gc.collect(generation=2)
    return jsonify(results)


@app.route('/api/query', methods=['POST'])
@cross_origin(origin='*')
def query():
    content = request.json
    print("Content: ", content)
    session_id = content['session_id']
    domain = content['domain']
    prev_suggestions = content['suggestions']

    results = {}
    try:
        SESSIONS[session_id]['last_query_timestamp'] = time()
    except Exception as e:
        print(f'# Ignoring an error in query: {e}')

    # Check if session ID is valid
    if session_id not in SESSIONS:
        results['status'] = FAILURE
        results['message'] = f'Your session has not been established due to invalid access code. Please check your access code in URL.'
        return jsonify(results)

    example = content['example']
    example_text = examples[example]

    # Overwrite example text if it is manually provided
    if 'example_text' in content:
        example_text = content['example_text']

    # Get configurations
    n = int(content['n'])
    max_tokens = int(content['max_tokens'])
    temperature = float(content['temperature'])
    top_p = float(content['top_p'])
    presence_penalty = float(content['presence_penalty'])
    frequency_penalty = float(content['frequency_penalty'])

    engine = content['engine'] if 'engine' in content else None
    context_window_size = get_context_window_size(engine)

    stop = [sequence for sequence in content['stop'] if len(sequence) > 0]
    if 'DO_NOT_STOP' in stop:
        stop = []

    # Remove special characters
    stop_sequence = [sequence for sequence in stop if sequence not in {'.'}]
    stop_rules = [sequence for sequence in stop if sequence in {'.'}]
    if not stop_sequence:
        stop_sequence = None

    # Parse doc
    doc = content['doc']
    prefix = doc[:content['cursor_index']]
    selected = doc[content['cursor_index']: content['cursor_index'] + content['cursor_length']]
    suffix = doc[content['cursor_index'] + content['cursor_length']:]
    results = parse_prompt(example_text + prefix, max_tokens, context_window_size)
    prompt = results['effective_prompt']
    # Get Constraints
    # Word Constraints
    # TODO: Need to implement sentence/passage constraints
    word_control_type = content['length_unit']
    user_token_constraint = None
    # Default constraints value
    word_range = []
    token_constraint = [[1, max_tokens]]
    trunc_len_list = [n]
    if word_control_type == "none":
        if selected.strip() == "" and suffix.strip() == "":
            # Don't do this when word length control is specified or we're doing rewriting
            # Add: Also don't do this when doing insertion
            # No word constraints
            # Set custom here
            token_constraint = [
                [1, 24],
                [16, 40],
                [32, 56],
                [1, 64]
            ]
            trunc_len_list = [1,1,1,2]
    elif word_control_type == "word":
        word_range = (
            min(int(content['length'][0]), int(content['length'][1])),
            max(int(content['length'][0]), int(content['length'][1]))
        )
    elif word_control_type == "token":
        token_constraint = [[
            min(int(content['length'][0]), int(content['length'][1])),
            max(int(content['length'][0]), int(content['length'][1]))
        ]]
        trunc_len_list = [n]
    else:
        print(f"ERROR! word control type {word_control_type} does not exist!")
    # Keyword & Banword Constraints
    keyword_constraint = [k.strip() for k in content['keyword'].split(";") if k.strip() != ""]
    banword_constraint = [k.strip() for k in content['banword'].split(";") if k.strip() != ""]
    
    # Instruct
    instruction = content['instruct'].strip()
    if instruction != "":
        # We need to add a space at the start of the instruction, and remove the last punctuations
        if instruction[-1] in string.punctuation:
            instruction = instruction[:-1]
        instruction = " " + instruction
    
    if engine != "local":
        # Overwrite the prompt. The old version will use the effective prompt. Here we add more descriptions in the prompt for model to do continuation
        prompt = get_gpt_prompt(
            prefix,
            selected,
            suffix,
            keyword_constraint = keyword_constraint,
            word_range = word_range,
            token_range = token_constraint[0] if word_control_type == "token" else [], # token_constraint[0] to remove the outer list structure
            instruction = instruction,
        )
        print("Querying OpenAI...")
        # Query GPT-3
        try:
            # If you want to use chat model, change here to `openai.chat.Completion.create`
            print("--------- GPT Prompt ----------")
            print(prompt)
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                # suffix=suffix,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logprobs=10,
                stop=stop_sequence,
            )
            suggestions = []
            for choice in response['choices']:
                print(f"#{choice.text}#")
                suggestion = parse_suggestion(
                    choice.text,
                    results['after_prompt'],
                    stop_rules
                )
                probability = parse_probability(choice.logprobs)
                suggestions.append((suggestion, probability, engine))
            suggestions_list = [suggestions]
            trunc_len_list = [n]
            DO_TOKEN_RANGE = False
        except Exception as e:
            results['status'] = FAILURE
            results['message'] = str(e)
            print(e)
            return jsonify(results)
    else:
        print("Querying local model...")
        # Query Local Model using model and tokenizer
        try:
            if len(content['token_range_list']) > 0 and len(content['token_range_list'][0]) == 2:
                # Have token range list. By default it is empty list []
                # If have token range list, we are doing token length control with multiple settings
                # TODO: When the len of token_range_list cannot be divided by args.num_beams, there might be some issues
                token_constraint = content['token_range_list']
                DO_TOKEN_RANGE = True
                # Set return sequence num `n` to the number of token_range
                n = len(content['token_range_list'])
                # When querying various token length, we remove the word constraints
                word_range = []
                trunc_len_list = [1] * n
            else:
                DO_TOKEN_RANGE = False
            request_json = {
                # Input Text
                "Prefix": prefix,
                "Suffix": suffix,
                "Prior": selected,
                # Constraints
                "Instruct": instruction,
                'word_constraint': word_range,
                'keyword_constraint': keyword_constraint,
                'banword_constraint': banword_constraint,
                # General Config.
                # TODO: Some of them should be set to a fix value
                'temperature': temperature, # Should be 0.8
                'num_return_sequences': args.num_beams, # We will return all the generated results, and filter here
                'num_beams': max(1, args.num_beams // len(token_constraint)), 
                'no_repeat_ngram_size': -1, 
                'top_p': top_p,
                'token_constraint': token_constraint
            }
            print(request_json)
                        # ---------- Start debug
            if args.debug_frontend:
                if not DO_TOKEN_RANGE:
                    beam_results = {}
                    beam_results['beam_outputs_texts'] = [
                        "No Space",
                        " Start Space",
                        "End Space ",
                        " Both Space ",
                        "XXX",
                        "AAA"
                    ]
                    beam_results['beam_outputs_sequences_scores'] = [0.5, 0.4, 0.3, 0.2, 0.2, 0.1]
                    beam_results_list = [beam_results]
                    trunc_len_list = [n]
                    sleep(0)
                else:
                    beam_results_list = [
                        {
                            'beam_outputs_texts': ["X"*i],
                            'beam_outputs_sequences_scores': [-i]
                        } for i in range(1, 16+1)
                    ]
                    
                    # beam_results['beam_outputs_texts'] = ["X"*i for i in range(1, 16+1)]
                    # beam_results['beam_outputs_sequences_scores'] = [-i for i in range(1, 16+1)]
                    sleep(5)
                
            # ---------- End debug 
            else:
                # https://stackoverflow.com/questions/9110593/asynchronous-requests-with-python-requests
                #  Query local models
                async def async_aiohttp_post_all(urls, data_list):
                    async with aiohttp.ClientSession() as session:
                        async def fetch(url, data):
                            print(f"Fetching {url}")
                            async with session.post(url, json=data) as response:
                                return await response.text()
                        return await asyncio.gather(*[
                            fetch(url, data) for url, data in zip(urls, data_list)
                        ])
                # Running Async
                print(f"Sending Post Request to Servers:", local_model_server_list)
                result_text_list = asyncio.run(async_aiohttp_post_all(local_model_server_list, [request_json] * len(local_model_server_list)))
                print(f"Results Fetched...")
                # --------------------- Gather results -----------------------
                # results = [{
                #     "token_range": [int, int]],
                #     "beam_outputs_texts": [str],
                #     "beam_outputs_sequences_scores_generation": [float],
                #     "beam_outputs_sequences_scores_suffix": [float],
                # }, ...]
                all_token_range_output = {}
                for result_text in result_text_list:
                    result_data = json.loads(result_text)
                    # result_data is a list, each item is the results of a token_range.
                    for result_data_per_token_range in result_data:
                        token_range_str = json.dumps(result_data_per_token_range["token_range"])
                        if not (token_range_str in all_token_range_output):
                            all_token_range_output[token_range_str] = {}
                        if 'beam_outputs_texts' in all_token_range_output[token_range_str]:
                            all_token_range_output[token_range_str]['beam_outputs_texts'] += result_data_per_token_range['beam_outputs_texts']
                            all_token_range_output[token_range_str]['beam_outputs_sequences_scores_generation'] += result_data_per_token_range['beam_outputs_sequences_scores_generation']
                            all_token_range_output[token_range_str]['beam_outputs_sequences_scores_suffix'] += result_data_per_token_range['beam_outputs_sequences_scores_suffix']
                        else: #Init
                            all_token_range_output[token_range_str]['beam_outputs_texts'] = result_data_per_token_range['beam_outputs_texts']
                            all_token_range_output[token_range_str]['beam_outputs_sequences_scores_generation'] = result_data_per_token_range['beam_outputs_sequences_scores_generation']
                            all_token_range_output[token_range_str]['beam_outputs_sequences_scores_suffix'] = result_data_per_token_range['beam_outputs_sequences_scores_suffix']
                # print("Gathered Results:", all_token_range_output)
                # Check
                retrieved_token_ranges = list(all_token_range_output.keys())
                requested_token_ranges = [json.dumps(token_range) for token_range in token_constraint]
                if requested_token_ranges != retrieved_token_ranges:
                    print("Warning, the retrieved token_range does not match the one we sent out.")
                    print("Requested Token Range:", requested_token_ranges)
                    print("Retrieved Token Range:", retrieved_token_ranges)
                else:
                    print("Correct Token Range!")
                
                # Sort the results
                def sort_results(text_list, score_A_list, score_B_list):
                    ranked_text = []
                    ranked_score = []
                    # We implement a simpler verison, that simply first multply the scores. Since it's logprob, we add it
                    score_list = [a+b for a,b in zip(score_A_list, score_B_list)]
                    # Get the rank
                    score_rank = (-np.array(score_list)).argsort()
                    for idx in score_rank:
                        ranked_text.append(text_list[idx])
                        ranked_score.append(score_list[idx])
                    return ranked_text, ranked_score
                # Now we use a list to store it
                beam_results_list = []
                for token_range_str, result_data_per_token_range in all_token_range_output.items():
                    # For each token length, we sort it
                    ranked_text, ranked_score = sort_results(
                        result_data_per_token_range['beam_outputs_texts'],
                        result_data_per_token_range['beam_outputs_sequences_scores_generation'],
                        result_data_per_token_range['beam_outputs_sequences_scores_suffix'],
                    )
                    beam_results_list.append({
                        'beam_outputs_texts' : ranked_text,
                        'beam_outputs_sequences_scores' : ranked_score,
                    })
            suggestions_list = []
            for beam_results in beam_results_list:
                suggestions = []
                
                # Only apply stop rules when we're doing continuation or writing. (suffix, selected is '')
                if (not (suffix.strip() == '' and selected.strip() == '')) and (not DO_TOKEN_RANGE):
                    stop_rules = []
                    
                for choice_text, log_prob in zip(beam_results['beam_outputs_texts'], beam_results['beam_outputs_sequences_scores']):
                    suggestion = parse_suggestion(
                        choice_text,
                        results['after_prompt'],
                        stop_rules = []
                    )
                    probability = (np.e**log_prob) * 100
                    suggestions.append((suggestion, probability, engine))
                suggestions_list.append(suggestions)
        except Exception as e:
            results['status'] = FAILURE
            results['message'] = str(e)
            print(e)
            return jsonify(results)
    # Got the results from GPT or local models, in suggestions_list        
    original_suggestions = []
    filtered_suggestions = []
    counts = {
        'empty_cnt' : 0,
        'duplicate_cnt': 0,
        'bad_cnt': 0
    }
    for suggestions, trunc_len in zip(suggestions_list, trunc_len_list):
        # Always return original model outputs
        original_suggestions_ = []
        for index, (suggestion, probability, source) in enumerate(suggestions):
            original_suggestions_.append({
                'original': suggestion,
                'trimmed': suggestion.strip(),
                'probability': probability,
                'source': source,
            })

        # Filter out model outputs for safety
        filtered_suggestions_, counts_ = filter_suggestions(
            suggestions,
            # prev_suggestions,
            filtered_suggestions,
            blocklist,
        )
        filtered_suggestions_= custom_filter_suggestions(
            filtered_suggestions_,
            suffix = suffix
        )
        # Get the num_return_sequence of highest prob sequence here
        filtered_suggestions_ = filtered_suggestions_[:trunc_len]
        # Combine the results
        original_suggestions += original_suggestions_
        filtered_suggestions += filtered_suggestions_
        counts['empty_cnt'] += counts_['empty_cnt']
        counts['duplicate_cnt'] += counts_['duplicate_cnt']
        counts['bad_cnt'] += counts_['bad_cnt']

    # print("Suggestions: ", json.dumps(filtered_suggestions, indent = 4))
    # random.shuffle(filtered_suggestions)

    suggestions_with_probabilities = []
    for index, (suggestion, probability, source) in enumerate(filtered_suggestions):
        suggestions_with_probabilities.append({
            'index': index,
            'original': suggestion,
            'trimmed': suggestion.strip(),
            'probability': probability,
            'source': source,
        })
    # Sort agiain
    original_suggestions_sorted = []
    suggestions_with_probabilities_sorted = []
    if not DO_TOKEN_RANGE:
        rank_idx_list = np.array([ - choice['probability'] for choice in suggestions_with_probabilities]).argsort().tolist()
        # We switch the first two, because the default choice in frontend is the 2nd, which should have the hightest prob
        tmp = rank_idx_list[0]
        rank_idx_list[0] = rank_idx_list[1]
        rank_idx_list[1] = tmp
        for i, rank_idx in enumerate(rank_idx_list):
            original_suggestions_sorted.append(original_suggestions[rank_idx])
            suggestions_with_probabilities_sorted.append(suggestions_with_probabilities[rank_idx])
    else:
        original_suggestions_sorted = original_suggestions
        suggestions_with_probabilities_sorted = suggestions_with_probabilities
    print("Suggestions: ", json.dumps(suggestions_with_probabilities_sorted, indent = 4))
    results['status'] = SUCCESS
    results['original_suggestions'] = original_suggestions_sorted
    results['suggestions_with_probabilities'] = suggestions_with_probabilities_sorted
    results['ctrl'] = {
        'n': n,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'stop': stop,
    }
    results['counts'] = counts
    print_verbose('Result', results, verbose)
    return jsonify(results)


@app.route('/api/get_log', methods=['POST'])
@cross_origin(origin='*')
def get_log():
    results = dict()

    content = request.json
    session_id = content['sessionId']
    domain = content['domain'] if 'domain' in content else None

    # Retrieve the latest list of logs
    log_paths = retrieve_log_paths(args.replay_dir)

    try:
        log_path = log_paths[session_id]
        log = read_log(log_path)
        results['status'] = SUCCESS
        results['logs'] = log
    except Exception as e:
        results['status'] = FAILURE
        results['message'] = str(e)

    if results['status'] == FAILURE:
        return results

    # Populate metadata
    try:
        stats = compute_stats(log)
        last_text = get_last_text_from_log(log)
        config = get_config_for_log(
            session_id,
            metadata,
            metadata_path
        )
    except Exception as e:
        print(f'# Failed to retrieve metadata for the log: {e}')
        stats = None
        last_text = None
        config = None
    results['stats'] = stats
    results['config'] = config
    results['last_text'] = last_text

    print_verbose('Get log', results, verbose)
    return results


if __name__ == '__main__':
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--proj_name', type=str, required=True)
    parser.add_argument('--use_local_model', action='store_true')
    parser.add_argument('--local_model_server_ip',
                        type=str,
                        default="127.0.0.1",
                        help="Specify the local model ip. For example, 127.0.0.1")
    parser.add_argument('--local_model_server_func',
                        type=str,
                        default="prompt/",
                        help="Specify the function/url to query the server model")
    parser.add_argument('--local_model_server_port',
                        type=str,
                        default=None,
                        help="Specify the local model port. For example, 8888")
    parser.add_argument('--local_model_server_port_file',
                        type=str,
                        default="../config/model_ports.txt",
                        help="Specify the local model port file. In this file, each line should have a number, which is the port. Should be specify together with local_model_server_port")
    # Optional arguments
    parser.add_argument('--num_beams', type=int, default = 64, help = "The beam size or the number of returned samples for local model.")
    parser.add_argument('--replay_dir', type=str, default='../logs')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_frontend', action='store_true', help = "If specifid, will use fake data instead of really query the model servers.")
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--use_blocklist', action='store_true')

    global args
    args = parser.parse_args()

    # Get Local Model Servers
    global local_model_server_list
    local_model_server_list = []
    if args.local_model_server_port != None:
        print(f"Warning: args.local_model_server_port is set to {args.local_model_server_port}. When the port is manually set, we do not use the ports in args.local_model_server_port_file, which is {args.local_model_server_port_file}")
        local_model_server_list = [f"{args.local_model_server_ip}:{args.local_model_server_port}"]
    else:
        for line in open(args.local_model_server_port_file, "r").readlines():
            port = int(line.strip())
            local_model_server_list.append(f"http://{args.local_model_server_ip}:{port}/{args.local_model_server_func}")
    # Check exist local model server if args.use_local_model is True
    if args.use_local_model:
        assert len(local_model_server_list) > 0, "ERROR: args.use_local_model is set to True, but there're no item in local_model_server_list"
        print("Use Local Model Servers: ", local_model_server_list)
    
    # Create a project directory to store logs
    global config_dir, proj_dir
    config_dir = args.config_dir
    proj_dir = os.path.join(args.log_dir, args.proj_name)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)

    # Create a text file for storing metadata
    global metadata_path
    metadata_path = os.path.join(args.log_dir, 'metadata.txt')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            f.write('')

    # Read and set API keys
    global api_keys
    api_keys = read_api_keys(config_dir)
    openai.api_key = api_keys[('openai', 'default')]

    # Read examples (hidden prompts), prompts, and a blocklist
    global examples, prompts, blocklist
    examples = read_examples(config_dir)
    prompts = read_prompts(config_dir)
    blocklist = []
    if args.use_blocklist:
        blocklist = read_blocklist(config_dir)
        print(f' # Using a blocklist: {len(blocklist)}')

    # Read access codes
    global allowed_access_codes
    allowed_access_codes = read_access_codes(config_dir)

    global session_id_history
    metadata = dict()
    metadata = update_metadata(
        metadata,
        metadata_path
    )

    global verbose
    verbose = args.verbose

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug,
    )
