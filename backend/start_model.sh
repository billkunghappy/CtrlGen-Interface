# python3 model_server.py \
# 	--port 8888 \
# 	--model_name_or_path /local1/ponienkung/CtrlGen/output/NewFinetune_cont_para_2K_8K_2K_StoryPretrain-TULU-LLAMA2

CUDA_VISIBLE_DEVICES=6 python3 model_server.py \
	--port 8888 \
	--device cuda \
	--cuda_core 6 \
    --hmm_batch_size 16 \
    --debug \
    --hmm_model_path /local1/hzhang19/matcha/models/hmm_llama-story-pretrain-finetune_32768_64/checkpoint-50.weight.th \
    --llama_model_path /local1/ponienkung/CtrlGen/output/NewFinetunePretrained_Filtered_StoryPretrain-TULU-LLAMA2
