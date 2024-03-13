# python3 model_server.py \
# 	--port 8888 \
# 	--model_name_or_path /local1/ponienkung/CtrlGen/output/NewFinetune_cont_para_2K_8K_2K_StoryPretrain-TULU-LLAMA2
export CUDA=$1
export PORT=$2

CUDA_VISIBLE_DEVICES=$CUDA python3 model_server.py \
	--port $PORT \
	--device cuda \
    --hmm_batch_size 16 \
    --hmm_model_path /local1/hzhang19/matcha/models/hmm_llama-story-pretrain-finetune_32768_64/checkpoint-50.weight.th \
    --llama_model_path /local1/ponienkung/CtrlGen/output/NewFinetunePretrained_Filtered_StoryPretrain-TULU-LLAMA2
