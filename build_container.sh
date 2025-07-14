
# if [ $# -eq 0 ]
#   then
#     echo "Please specify a cuda version (e.g. \`sudo ./build_container.sh 11.7.1\`)"
#     exit 1
# fi

CONTAINER_NAME="dpvo_new"

# # echo "Removing dpvo docker image if already exists..."
# docker rm -f $CONTAINER_NAME 2> /dev/null
# docker rmi -f dpvo_docker_img 2> /dev/null
# docker build --tag dpvo_docker_img --build-arg CUDA_VERSION=$1 .

# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker

# Create a new container
docker run -td --privileged --net=host --ipc=host \
    --name=$CONTAINER_NAME \
    --gpus=all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v /etc/group:/etc/group:ro \
    -v /mnt/vol2_raid/shared_data/:/home/captain/DPVO/data \
    -v ./calib:/home/captain/DPVO/calib \
    -v ./trajectory_plots:/home/captain/DPVO/trajectory_plots \
    dpvo_new bash

docker exec -i $CONTAINER_NAME "sudo chmod +x /home/captain/DPVO/install_dpvo.sh && conda activate dpvo && bash /home/captain/DPVO/install_dpvo.sh"
