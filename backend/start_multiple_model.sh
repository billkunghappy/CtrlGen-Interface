#!/bin/bash

HMM_MODEL_PATH="/local1/hzhang19/matcha/models/hmm_llama-story-pretrain-finetune_32768_64/checkpoint-140.weight.th"
LLAMA_MODEL_PATH="/local1/ponienkung/CtrlGen/output/NewFinetunePretrained_Filtered_StoryPretrain-TULU-LLAMA2"

GPUS=( 0 1 )
PORT_START=8400

PORT_LIST=()

# Iterate over the original array and copy to the new array
for GPU in "${GPUS[@]}"; do
    PORT_LIST+=($((PORT_START + GPU)))
done

# Write Ports
printf "%s\n" "${PORT_LIST[@]}" > ../config/model_ports.txt

(
    trap 'kill 0' SIGINT;
    for i in "${!GPUS[@]}"
    do
        #Run in background
        CUDA_VISIBLE_DEVICES=${GPUS[i]} python3 model_server.py \
            --port ${PORT_LIST[i]} \
            --device cuda \
            --hmm_batch_size 32 \
            --hmm_model_path $HMM_MODEL_PATH \
            --llama_model_path $LLAMA_MODEL_PATH \
            &
    done
    wait
)