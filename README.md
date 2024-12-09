<div align="center">

<img src="./Ctrl-G-Logo.png" width="350px"/>

**Adaptable Logical Control for Large Language Models**

</div>

## Overview
This repository contains the code for the interface of interactive text editing using [Ctrl-G](https://billkunghappy.github.io/Ctrl-G/). The interface includes three parts: (1) Frontend interface for interactive text editing, including a control panel; (2) Backend server that preprocess user requests and query the model servers; (3) Model server that host the models for Ctrl-G and generate suggestions.

- **Ctrl-G** Paper: 
[Adaptable Logical Control for Large Language Models](https://arxiv.org/abs/2406.13892)
([Honghua Zhang](https://web.cs.ucla.edu/~hzhang19/), [Po-Nien Kung](https://billkunghappy.github.io/ponien-kung/), [Masahiro Yoshida](https://github.com/masathehero), [Guy Van den Broeck](https://web.cs.ucla.edu/~guyvdb/), and [Nanyun(Violet) Peng](https://violetpeng.github.io/), Neurips 2024)

### Credits
This interface is extended from the **interface** of [CoAuthor](https://coauthor.stanford.edu) by [Mina Lee](https://minalee.info/), which already allows interactive text editing with OpenAI models. We suggest you use the [CoAuthor Interface](https://coauthor.stanford.edu) if you're not planning to use the additional features described below.

**Added Features in Ctrl-G Interface:**
1. Added control panel for controllable generation.
2. Allow the host of local model servers.
3. Allow the backend to query local model servers.
4. Improved pre-processing/post-processing for getting suggestions.
5. Allow getting suggestions for rewriting text.

---

## Contents
- [Overview](#overview)
- [Contents](#contents)
- [Setup](#Model)
- [Model](#Model)
- [Backend](#backend)
- [Frontend](#frontend)
- [Advanced Usage](#advanced-usage)

---

## Setup

**1. Clone this Github repository**
First clone this repository in any directory.
```
git clone https://github.com/???
```
Inside the `ctrl-g-interface` directory, run the following to install the required packages:

```
pip install -r requirements.txt
```

Install `ctrl-g` as a package.

```
git clone https://github.com/joshuacnf/Ctrl-G.git
cd Ctrl-G/
pip install -e .
```

**2. Start all the servers**
Follow the provided instructions to start the **Model Server**, **Backend Server** and **Frontend Server** sequentially.

---

## Model Server
The model server is a Flask app that host the local language models to generate suggestions based on reqursts from the backend server. The model server allows multi-gpu inference.

By default, the model server is setup to host [Ctrl-G](https://billkunghappy.github.io/Ctrl-G/), which includes a Llama2 model and an attached HMM for controlled generation. By changing the input arguments, you can easily host any other language models.

**To use the interface with [OpenAI models](https://platform.openai.com/docs/models), you do not need to setup the model server.**



**1. Prepare local models**
To try [Ctrl-G](https://billkunghappy.github.io/Ctrl-G/) models, you can download the llama2 models and the attatched HMM from [huggingface](https://huggingface.co/ctrlg). 
We recommend using these two models:
```python
{
    "base_model": "ctrlg/tulu2-7b_writing-prompts",
    "hmm_models": "ctrlg/hmm_tulu2-7b_writing-promptss_32768"
}
```

This model includes a 7B llama2 models with a HMM, which should be able to run under 40GB GPU memory. More model choices can be refer to [Ctrl-G Training Repo](https://github.com/joshuacnf/Ctrl-G?tab=readme-ov-file).

> Note: You do not need to download them explicitly. You can directly input the name as argument to `model/model_server.py`.

**2. Start the model server**
Enter the `model` directory.
```
cd ctrl-g-interface/model
```
**Run the model server with *Single GPU*** by typing the following command:
```bash
# Setup arguments. For single GPU, please only provide one GPU index to CUDA argument
export CUDA=0
export PORT=8400
export MODEL="ctrlg/tulu2-7b_writing-prompts"
export HMM_MODEL="ctrlg/hmm_tulu2-7b_writing-prompts_32768"

# Write the port into `../config/model_ports.txt`.
# -- Backend server will query the model server based on this txt file.
printf "%s\n" "${PORT}" > ../config/model_ports.txt

# Start the model server
CUDA_VISIBLE_DEVICES=${CUDA} python3 model_server.py \
    --port ${PORT} \
    --llama_model_path $MODEL \
    --hmm_model_path $HMM_MODEL \
    --generation_batch_size 128 \
    --suffix_cap 32
```

**Run the model server with *Multiple GPUs*** by typing the following command:
```bash
# Setup arguments. For multiple GPU, specify the GPUs in a list. 
# PORT_START specifies the http port we're using. For n GPUs, we will use the port from PORT_START to PORT_START + n.
export GPUS=( 0 1 )
export PORT_START=8400
export MODEL="ctrlg/tulu2-7b_writing-prompts"
export HMM_MODEL="ctrlg/hmm_tulu2-7b_writing-prompts_32768"

PORT_LIST=()

# Get the ports for each GPU process
for GPU in "${GPUS[@]}"; do
    PORT_LIST+=($((PORT_START + GPU)))
done

# Write the port into `../config/model_ports.txt`.
# -- Backend server will query the model server based on this txt file.
printf "%s\n" "${PORT_LIST[@]}" > ../config/model_ports.txt

# Setup multiple model servers backend. Each on a GPU.
(
    trap 'kill 0' SIGINT;
    for i in "${!GPUS[@]}"
    do
        #Run in background
        CUDA_VISIBLE_DEVICES=${GPUS[i]} python3 model_server.py \
            --port ${PORT_LIST[i]} \
            --llama_model_path $MODEL \
            --hmm_model_path $HMM_MODEL \
            --suffix_cap 32 \
            --generation_batch_size 128 \
            &
    done
    wait
)
```

**Additional Arguments**
**1. `--suffix_cap`:** When inserting in a long document, sometimes the suffix can be very long, which might effect the insertion quality. Specify this argument to truncate it into a specified token length.
**2. `--llama_only`:** When specified, only use the language model without the attatched HMM. This argument will usually combine with `--llama_insertion`.
**3. `--llama_insertion`:** When specified, provide the suffix to the language model. For Ctrl-G, since HMM will process the suffix, the language model does not see the suffix.
**4. `--generation_batch_size`:** The number of suggestions to return per GPU. Higher generation batch size will lead to better performance. (With higher GPU memory usages)

---

## Backend 

The backend is a Flask app that serves requests from users, manages sessions, and stores logs for future replays.

The backend is setup to support both local model servers setup by model_server.py and [OpenAI models](https://platform.openai.com/docs/models) via OpenAI API. The backend will query different servers based on **URL Parameters**.

**1. Run the server on your local machine or on a server**

Run the server in `./backend` with basic parameters as follows:
```
python3 api_server.py \
    --config_dir '../config' \
    --log_dir ../logs \
    --proj_name 'ctrl-g' \
    --port 4567 \
    --use_local_model \
    --local_model_server_ip 127.0.0.1 \
    --local_model_server_port_file ../config/model_ports.txt
	
```
**Arguments**
**1. `--port`:** The http port for the backend server. Make sure it does not use the same port as model servers.
**2. `--use_local_model`:** When specify this, it will allow the backend server to query local models. Otherwise, the backend server can only query OpenAI models.
**3. `--local_model_server_ip`:** When `--use_local_model` is set, this specify the ip address of the model servers. If the model servers and backend server are setup on the same machine, simply give 127.0.0.1 (localhost).
**4. `--local_model_server_port_file`:**: When `--use_local_model` is set, this specify the .txt file that includes the ports used by the model servers.

> Note: If you only want to use the interface with **OpenAI models**, you do not need to specify the `--use_local_model` and its following arguments. However, if you specify this, you can still query the OpenAI models by changing the `URL Parameters` after the backend server starts.

The backend initializes sessions using access codes that are read from `data/access\_codes.csv`. When you enter the frontend, the access code provided needs to match one of the created codes here.  

The choice of models, examples (prompts that are hidden from users), and prompts (prompts that are shown to users in the text editor) can be specified when you create `data/access\_codes.csv`. 

**2. Add your API key(s) to use OpenAI models**
If you want to use OpenAI models, you will need to add the API keys.

Create a file `./config/api_keys.csv` and add your API key(s) as follows:

| host | domain | key |
| ---- | ------ | --- |
| openai | default | sk-*************************************** |

Replace the `sk-***************************************` with your OpenAI API key. If you don't have it, you can get one [here](https://openai.com/pricing).

For `host` and `domain`, you can simply use `openai` and `default`.

---

## Frontend

**1. Run the frontend**

You can run the frontend using a simple Python server or host it on a third-party server.

To run the frontend on a local machine, run the following command in the `./frontend` directory:

```
python -m http.server 8000
```

**2. Set the server URL**

Update `./frontend/js/config.js` to have the correct URL of the frontend and backend server. For instance, if your ***backend server*** is running on `http://127.0.0.1:5555` and your ***frontend server*** is running on `http://127.0.0.1:8000` then the following two lines in the config file should look like:

```
const serverURL = 'http://127.0.0.1:5555'
const frontendURL = 'http://127.0.0.1:8000' 
```

**3. Access the frontend**

Now, you can access the frontend server on your browser as follows:

```
FRONTEND_URL/index.html?ctrl=show&access_code=ACCESS_CODE&engine=ENGINE
```

where `FRONTEND_URL` is the URL of the frontend server (e.g. `http://127.0.0.1:8000`) and `ACCESS_CODE` is one of the access codes you defined in `./config/access_codes.csv`. The `ENGINE` is the AI model to use. If using OpenAI models, directly specify the model name such as `engine=gpt-3.5-turbo-instruct`. If you have started the local model server (Ctrl-G models), specify `engine=local` to query the `model_server.py`. If you have followed the instructions above, you should be able to access the frontend at [here](http://127.0.0.1:8000/index.html?access_code=demo):

```
http://127.0.0.1:8000/index.html?access_code=demo
```

**4. Use the frontend**

- **Get suggestions from AI**: While writing in the text editor, press the `tab` key whenever you want to get suggestions. You can get suggestions multiple times in a row if you want more; you can navigate the suggestions using `arrow` keys and press the `enter` key to select a suggestion; to reopen the previous suggestions, press the `shift` key and `tab` key at the same time.
- **Save your writing session**: If you want to save the writing session (to share it with others or to replay it later), press the "Save your work" button on the bottom of the page and save the `SESSION_ID` you get; otherwise, your session will not be saved.
<!-- - **Replay your writing session**: To view the replay of your writing session, you can access it at `FRONTEND_URL/replay.html?session_id=SESSION_ID` where `FRONTEND_URL` is the URL of the frontend server and `SESSION_ID` is the session ID you received when you saved your writing session. -->

<!-- <div align="center">

<img src="https://p-lambda.github.io/coauthor/assets/images/pig_0.75_clip.gif" width="500px"/>

</div> -->

---

## Advanced Usage

**Blocklist**

You can block certain words or phrases from being generated by the model by adding them to `./config/blocklist.txt` and setting `--use_blocklist` to be true when running the backend.

**Additional Note**
To start the writing assistant, you need to start frontend, backend, and the model server.
* Frontend: `cd frontend`, `sh start_frontend.sh`
* Backend: `cd backend`, `sh start_backend.sh`
* Model: `cd backend`, `sh start_model.sh` 

<!-- **TODO**
### FIX:
1. The interface error message is quite intrivial. Might need to fix this part.

### Upgrade:
1. Add an example prompt when the user changed the constraints -->
