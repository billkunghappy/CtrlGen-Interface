python3 api_server.py \
	--config_dir '../config' \
	--log_dir ../logs \
	--port 4567 \
	--proj_name 'ctrl-g' \
	--use_local_model \
	--local_model_server_ip 127.0.0.1 \
	--local_model_server_port_file ../config/model_ports.txt
