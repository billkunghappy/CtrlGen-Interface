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
printf "%s\n" "${PORT_LIST[@]}" > ./config/model_ports.txt

# Setup multiple model servers backend. Each on a GPU.
(
    trap 'kill 0' SIGINT;
    for i in "${!GPUS[@]}"
    do
        #Run in background
        CUDA_VISIBLE_DEVICES=${GPUS[i]} python model/model_server.py \
            --port ${PORT_LIST[i]} \
            --llama_model_path $MODEL \
            --hmm_model_path $HMM_MODEL \
            --suffix_cap 32 \
            --generation_batch_size 128 \
            &
    done
    wait
)