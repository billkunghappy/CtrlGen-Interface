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
from time import time
from argparse import ArgumentParser
# For Local Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    filter_suggestions
)

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import requests

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa

SESSIONS = dict()
app = Flask(__name__)
CORS(app)  # For Access-Control-Allow-Origin

SUCCESS = True
FAILURE = False


@app.route('/api/start_session', methods=['POST'])
@cross_origin(origin='*')
def start_session():
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
    if args.local_model_server is None:
        print("Querying OpenAI...")
        # Query GPT-3
        try:
            if suffix != "": # If the demarcation is there, then suggest an insertion
                # If you want to use chat model, change here to `openai.chat.Completion.create`
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    suffix=suffix,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logprobs=10,
                    stop=stop_sequence,
                )
            else:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
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
        except Exception as e:
            results['status'] = FAILURE
            results['message'] = str(e)
            print(e)
            return jsonify(results)
    else:
        print("Querying local model...")
        # Query Local Model using model and tokenizer
        try:
            # TODO: Need to implement sentence/passage constraints
            word_control_type = content['length_unit']
            if word_control_type == "none":
                word_range = []
            else:
                # TODO: Implement word constraints, to get a range
                word_range = (max(0, int(content['length']) -5) , max(int(content['length']), 1))

            step = 3
            token_constraint = [[i, i + step - 1] for i in range(1, 32, step)]

            request_json = {
                # Input Text
                "Prefix": prefix,
                "Suffix": suffix,
                "Prior": selected,
                # Constraints
                "Instruct": content['instruct'],
                'word_constraint': word_range,
                'keyword_constraint': [k.strip() for k in content['keyword'].split(";") if k.strip() != ""], # TODO: Add this
                # General Config.
                # TODO: Some of them should be set to a fix value
                'temperature': temperature, # Should be 0.8
                'num_return_sequences': n,
                'num_beams': random.randint(4, 8), 
                'no_repeat_ngram_size': -1, 
                'top_p': top_p,
                'token_constraint': token_constraint
            }

            print(request_json)
            r = requests.post(f'http://{args.local_model_server}/prompt/', json=request_json)
            beam_results = json.loads(r.text)
            print('------------------------------------------------------------')
            print(beam_results)
            print('------------------------------------------------------------')
            beam_results = beam_results[0]
            # ---------- Start debug 
            # beam_results = {}
            # beam_results['beam_outputs_texts'] = [
            #     "No Space",
            #     " Start Space",
            #     "End Space ",
            #     " Both Space ",
            #     "XXX"
            # ]
            # beam_results['beam_outputs_sequences_scores'] = [0.5, 0.4, 0.3, 0.2, 0.1]
            # ---------- End debug 
            suggestions = []
            
            # Only apply stop rules when we're doing continuation or writing. (suffix, selected is '')
            if not (suffix.strip() == '' and selected.strip() == ''):
                stop_rules = []
                
            for choice_text, log_prob in zip(beam_results['beam_outputs_texts'], beam_results['beam_outputs_sequences_scores_generation']):
                suggestion = parse_suggestion(
                    choice_text,
                    results['after_prompt'],
                    stop_rules
                )
                probability = (np.e**log_prob) * 100
                suggestions.append((suggestion, probability, engine))
        except Exception as e:
            results['status'] = FAILURE
            results['message'] = str(e)
            print(e)
            return jsonify(results)

    # Always return original model outputs
    original_suggestions = []
    for index, (suggestion, probability, source) in enumerate(suggestions):
        original_suggestions.append({
            'original': suggestion,
            'trimmed': suggestion.strip(),
            'probability': probability,
            'source': source,
        })

    # Filter out model outputs for safety
    filtered_suggestions, counts = filter_suggestions(
        suggestions,
        prev_suggestions,
        blocklist,
    )

    random.shuffle(filtered_suggestions)

    suggestions_with_probabilities = []
    for index, (suggestion, probability, source) in enumerate(filtered_suggestions):
        suggestions_with_probabilities.append({
            'index': index,
            'original': suggestion,
            'trimmed': suggestion.strip(),
            'probability': probability,
            'source': source,
        })

    results['status'] = SUCCESS
    results['original_suggestions'] = original_suggestions
    results['suggestions_with_probabilities'] = suggestions_with_probabilities
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
    parser.add_argument('--local_model_server',
                        type=str,
                        default=None,
                        help="Specify the local model ip and port. For example, 127.0.0.1:8888")

    # Optional arguments
    parser.add_argument('--replay_dir', type=str, default='../logs')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--use_blocklist', action='store_true')

    global args
    args = parser.parse_args()

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
