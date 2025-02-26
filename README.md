# Docker for Deep Patch Visual Odometry

## Install Dependencies
You can install Docker [here](https://docs.docker.com/engine/install/)

Also add the [nvidia-docker](https://nvidia.github.io/nvidia-docker/) repository
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
Install the Nvidia container/docker toolkits
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```

## Container Setup
Build the docker container

```
sudo ./build_container.sh <cuda-version>.1
```
_Example: If_ `nvidia-smi` _shows_ `CUDA Version: 11.6` _then run_ `sudo ./build_container.sh 11.6.1`

You should now be able to run DPVO
```
./download_models_and_data.sh
docker exec -i dpvo /bin/bash -c "source activate dpvo && python3 /DPVO/extract_traj.py --imagedir=/DPVO/custom_data/{your_folder_name} --calib=/DPVO/calib/{your_txt_camera_config} --stride=5 --plot" 
```
