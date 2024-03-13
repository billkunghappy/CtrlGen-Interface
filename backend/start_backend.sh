python3 api_server.py \
	--config_dir '../config' \
	--log_dir ../logs \
	--port 5555 \
	--proj_name 'pilot' \
	--use_local_model \
	--local_model_server_ip 127.0.0.1 \
	--local_model_server_port_file ../config/model_ports.txt \
	--debug

	#--model_name_or_path /local1/ponienkung/CtrlGen/output/NewFinetune_cont_para_2K_8K_2K_StoryPretrain-TULU-LLAMA2 \
