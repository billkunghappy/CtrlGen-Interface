import json
from string import Template

# ----- Construct the gpt prompt based on user request
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

# ----- Cacheing in backend server.
# For writing session, if a user ask for AI suggestion with the exact same constraints as the last one, we will combine the new prediction and old results for ranking, to select the best predictions.
# Note: This cacheing only happens when querying local models. When querying the model server, doing background query will waste user's money.

# There are multiple stages:
# (1) User query the model with a new query Q1. --> (2)
# (2) We return answer A1 to the user. --> (3)
# (3) We automatically send a background query Q2 to the model and get A2 in our server. Merge with A1 to put in Cache.
# (4) If the user ask for the same query Q1 --> (5)
# --> Otherwise, go back to (1)
# (5) Check the cache and get a match. Return Cache directly without querying model_server. Go to (3)



def check_have_cache(CACHE, content):
    # Cache     --> Prior Query Info
    # content   --> Current Query Info
    content = json.loads(json.dumps(content))
    session_id = content['session_id']
    
    # Check cache exist
    EXIST_CACHE = False
    # Check if this session has previous query by checking whether the session_id is in Cache
    if session_id in CACHE:
        # Ignore these two keys: [suggestions, background_query]. We do not store these two values in cache.
        content.pop("suggestions", None)
        content.pop("background_query", None)
        CACHE[session_id]['content'].pop("suggestions", None)
        CACHE[session_id]['content'].pop("background_query", None)
        # Check if the rest of the content is the same. If so, the user is asking for exact same query as previous one.
        if CACHE[session_id]['content'] == content:
            EXIST_CACHE = True
    return session_id, EXIST_CACHE

def update_prediction_cache(CACHE, content, prediction, is_background):
    CACHE[content['session_id']] = {
        "content": content,
        "prediction": prediction,
        "is_background": is_background,
    }

# For inference efficiency, we will automatically send a background query following the user query. This background query does not present to the user, but will be cached in case the user ask for the same query again.
# check_background_cache will check if the current query is a background query. 
# If current query is a user query (not the background query we automatically send out) and there exist background query cache in the CACHE, we will skip the model generation and directly return the cache.
def check_background_cache(CACHE, content):
    is_background = "background_query" in content
    
    session_id, EXIST_CACHE = check_have_cache(CACHE, content)
    
    # only when currently it's not background query and content are same, we direclty use the cache stored by previous background query
    if EXIST_CACHE and (not is_background) and CACHE[session_id]['is_background']:
        # Exist and is using background cache. Skip model generation...
        return True
    else:
        # Doesn't exist background cache...
        return False        

# If we have a background cache, we will merge the current prediction with the background cache.
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

# If we have a cache and the prediction for this step, we will merge them and return both
# Otherwise, we will only return the cache. This happens when the user ask for the repeat query, we will directly use the cache from the previous background query instead of generating for this step.
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