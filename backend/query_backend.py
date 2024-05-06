import requests
import sys
import json
from tqdm import tqdm
import os

dataset_path = sys.argv[1]
engine = sys.argv[2]
output_dir = sys.argv[3]
output_path = os.path.join(output_dir, os.path.basename(dataset_path).replace(".json", f"_{engine}.json"))

print(f"Storing output to {output_path}")

if "local" in engine:
    repeat_query_num = 32
else:
    repeat_query_num = 1
# Every 50 data in the evaluation set belongs to a prompt, so to get 20 from each, we set the following args:
# The number of data from each prompt is specified in the CoathorProc.ipynb
prediction_per_prompt = 20 
total_prediction_per_prompt = 50

AccessCode = "demo"
eval_dataset = json.load(open(dataset_path, "r"))

# Start session
s = requests.Session()
get_json = s.post('http://127.0.0.1:5555/api/start_session', json={"domain": AccessCode, "accessCode": AccessCode}).json()
session_id = get_json['session_id']
# print(get_json)
# Query backend

ExampleContent = {
    'session_id': '1f364e181eb74457ac75589c461d899b', # Change this
    'domain': 'story',
    'example': 'na',
    'example_text': '',
    'doc': 'I have a dog ', # Change this
    'cursor_index': 9, # Change this
    'cursor_length': 0, # Always 0 here since we're not doing rewriting
    'n': '5',
    'max_tokens': '64',
    'temperature': '0.8',
    'top_p': '1',
    'presence_penalty': '0.5',
    'frequency_penalty': '0.5',
    'stop': ['.'],
    'keyword': 'evil dragon', # Change this
    'banword': '',
    'length_unit': 'word', # Change this
    'length': ['8', '14'], # Change this
    'instruct': 'horror style', # Change this
    'token_range_list': [],
    'engine': 'local', # Change this
    'suggestions': [],
    'background_query': False
}

ExampleData = {
    "b72437b88da64ad392d21dbc32ca9af0": {
        "prefix": "A woman has been dating guy after guy, but it never seems to work out. She’s unaware that she’s actually been dating the same guy over and over; a shapeshifter who’s fallen for her, and is certain he’s going to get it right this time.\n\nOne day, the shapeshifter takes another form and falls for a new girl. He's decided that the girl he had pursued with so much vigor before, isn't worth his time, and he pours himself into a relationship with this new woman.   He's so consumed with his affection for this woman, he doesn't realize the girl he had originally dated, is now pursuing him as well. This confuses him,  she'd broken up with him time and time again, no matter what approach he tried, or what physical shape he took.",
        "infix": " Now that she's trying to get back with him, he doesn't know what to do.",
        "constraints": {
            "keyword": [],
            "wordlength": [],
            "suffix": ""
        }
    }
}
# Calculate skip
skip = 0
try:
    write_data = json.load(open(output_path, "r"))
    for session, data in write_data.items():
        if 'suggestions' in data and len(data['suggestions']) > 1:
            skip += 1
        else:
            break
    eval_dataset = write_data # if exist
except:
    skip = 0 

print(f"Skip Num: {skip}")

for cnt, (session, data) in enumerate(tqdm(eval_dataset.items())):
    if cnt % total_prediction_per_prompt > prediction_per_prompt:
        continue
    if cnt < skip:
        continue
    if not ("suggestions" in data):
        try:
            for i in range(repeat_query_num):
                if i != repeat_query_num-1:
                    # Not last, use background query (Don't get the output)
                    ExampleContent['background_query'] = True
                ExampleContent['session_id'] = session_id
                ExampleContent['doc'] =  data['prefix'] + data['constraints']['suffix']
                ExampleContent['cursor_index'] = len(data['prefix'])
                ExampleContent['keyword'] = "\n".join(data['constraints']['keyword'])
                if len(data['constraints']['wordlength']) == 0:
                    ExampleContent['length_unit'] = "token"
                    ExampleContent['length'] = ['1', '64']
                else:
                    ExampleContent['length_unit'] = "word"
                    ExampleContent['length'] = [str(data['constraints']['wordlength'][0]), str(data['constraints']['wordlength'][1])]
                ExampleContent['instruct'] = ""
                ExampleContent['engine'] = "local" if "local" in engine else engine
                query_results = s.post('http://127.0.0.1:5555/api/query', json=ExampleContent).json()
                suggestions_with_probabilities = query_results['suggestions_with_probabilities']
                suggestions_with_probabilities = sorted(suggestions_with_probabilities, key=lambda d: d['probability'], reverse=True)
                all_suggestions = [sg['original'] for sg in suggestions_with_probabilities]
            eval_dataset[session]["suggestions"] = all_suggestions
            with open(output_path, "w") as F:
                json.dump(eval_dataset, F, indent = 4, ensure_ascii=False)
        except Exception as e:
            with open(os.path.join(output_dir, "Done.txt"), "a") as F:
                F.write(f"Missing:{output_path}:{cnt}. ID: {session}\n")
            print(e)

with open(os.path.join(output_dir, "Done.txt"), "a") as F:
    F.write(f"Done:{output_path}\n")
    