#!/bin/bash

# Get the directory of the script
script_dir=$(dirname "$(realpath "$0")")

# Get the current working directory
current_dir=$(pwd)

# Check if the current directory is the same as the script directory
if [ "$script_dir" != "$current_dir" ]; then
    echo "Error: Current working directory is not the script directory."
    echo "Please change to the directory: $script_dir"
    exit 1
fi

# 定义PID文件路径
PID_FILE="localqxsim_service.pid"

# 检查PID文件是否存在
if [ -f "$PID_FILE" ]; then
    # 读取PID文件中的PID
    OLD_PID=$(cat "$PID_FILE")
    
    # 检查PID对应的进程是否存在并杀死它
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Killing old process with PID $OLD_PID"
        kill -9 $OLD_PID
    fi
fi

# 启动服务并使用nohup运行，重定向输出到日志文件
nohup ./train_sim --conf config.yaml  > output.log 2>&1 &

# 获取新的PID并保存到PID文件中
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"

echo "Service started with PID $NEW_PID"
