HEAD="3.12.196.0"
KEY="~/.ssh/ray-autoscaler_us-east-2.pem"
scp -i $KEY worker.py ec2-user@$HEAD:~/worker.py
scp -i $KEY deploy.sh ec2-user@$HEAD:~/deploy.sh
scp -i $KEY coordinator.py ec2-user@$HEAD:~/coordinator.py
