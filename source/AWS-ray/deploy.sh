#!/bin/bash
# deploy.sh — copy worker.py to each worker and start it
# Usage: bash deploy.sh

KEY="~/.ssh/ray-autoscaler_us-east-2.pem"
EFS_PATH="/mnt/efs/dataset"
PORT=5556

WORKERS=(
    "172.31.25.172"
    "172.31.31.16"
    "172.31.23.98"
    "172.31.21.20"
    "172.31.20.37"
    "172.31.17.255"
    "172.31.16.43"
    "172.31.16.250"
    "172.31.19.104"
    "172.31.25.86"
    "172.31.17.182"
    "172.31.30.198"
    "172.31.16.99"
    "172.31.25.75"
    "172.31.29.172"
    "172.31.24.143"
    "172.31.24.32"
)

for i in "${!WORKERS[@]}"; do
    IP="${WORKERS[$i]}"
    echo "Deploying to worker $i at $IP..."
    scp -i $KEY -o StrictHostKeyChecking=no worker.py ec2-user@$IP:~/worker.py
    ssh -i $KEY -o StrictHostKeyChecking=no ec2-user@$IP \
    "sudo fuser -k 5555/tcp; pkill -9 -f worker.py; sleep 2; pip3 install pyzmq -q; nohup python3 ~/worker.py $i $PORT > ~/worker.log 2>&1 & disown"
done

echo "All workers deployed."
