mkdir -p ./logs

python ./backend/api_server.py \
    --config_dir './config' \
    --log_dir ./logs \
    --proj_name 'ctrl-g' \
    --port 32000 \
    --use_local_model \
    --local_model_server_ip 127.0.0.1 \
    --local_model_server_port_file ./config/model_ports.txt
	