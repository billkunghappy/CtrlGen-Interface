"""
Starts a Flask server that response with the local model
"""

from flask import Flask, render_template, request
from argparse import ArgumentParser
# For Local Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from model_utils import (
    encode_with_messages_format
)

app = Flask(__name__)

@app.route('/prompt/',methods=['POST'])
def prompt():
    # Request Data
    # input_json = {
    #     "Prefix": "",
    #     "Instruct": "",
    #     "Prior": "",
    #     "Operation": ""
    #     "max_tokens": int,
    #     "temperature": int,
    #     "num_return_sequences": int,
    #     "num_beams": int,
    #     "no_repeat_ngram_size": int,
    #     "top_p": int,
    # }
    data = request.json
    print("Operation: ", data.get("Operation"))
    with torch.inference_mode():
        input_ids = encode_with_messages_format(
            Prefix = data.get("Prefix"),
            SoftControl = data.get("Instruct"), 
            Prior = data.get("Prior"),
            tokenizer = tokenizer, 
            operation = data.get("Operation")
            ).cuda()
        # dict_keys(['sequences', 'sequences_scores', 'scores', 'beam_indices', 'attentions', 'hidden_states'])
        beam_outputs = model.generate(
            input_ids,
            max_new_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            num_return_sequences=data.get("num_return_sequences"),
            num_beams=data.get("num_beams"),
            no_repeat_ngram_size=data.get("no_repeat_ngram_size"),
            top_p=data.get("top_p"),
            early_stopping=True,
            output_scores = True, # Get sequences_scores
            return_dict_in_generate=True,
        )
    
    beam_outputs_sequences = beam_outputs.sequences.cpu().detach().tolist()
    beam_outputs_sequences_scores = beam_outputs.sequences_scores.cpu().detach().tolist()
    beam_outputs_texts = [tokenizer.decode(choice[input_ids.shape[-1]:], skip_special_tokens=True) for choice in beam_outputs_sequences]
    # This will return it as text. You can further use json.loads(r.text) to retrieve the results
    return {
        "beam_outputs_texts": beam_outputs_texts,
        "beam_outputs_sequences_scores": beam_outputs_sequences_scores,
    }
    



if __name__ == '__main__':
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model_name_or_path',
                        type=str,
                        required = True,
                        help="Specify the local model path as arguments.")
    global args
    args = parser.parse_args()

    global tokenizer
    global model
    # Try loading the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model.half().cuda()
        print(f"Load model at {args.model_name_or_path}")
    except Exception as e:
        print(f"Cannot Load args.model {args.model_name_or_path} because of the following exception:\n {e}")
        print("Exit the process...")
        exit(0)

    app.run(
        host='0.0.0.0',
        port=args.port,
    )
