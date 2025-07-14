FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

ENV PATH="/home/captain/miniconda3/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y unzip sudo git wget && \
    rm -rf /var/lib/apt/lists/*


# Make non-root user
RUN adduser --disabled-password --gecos '' captain \
    && adduser captain sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER captain
WORKDIR /home/captain/
RUN chmod a+rwx /home/captain/


RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir .conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh


RUN git clone https://github.com/princeton-vl/DPVO.git --recursive
COPY install_dpvo.sh /home/captain/DPVO/install_dpvo.sh
WORKDIR /home/captain/DPVO

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
RUN unzip eigen-3.4.0.zip -d thirdparty

RUN conda init bash && . ~/.bashrc
RUN conda env create -f environment.yml

RUN bash ./download_models_and_data.sh

COPY scripts/extract_traj.py /home/captain/DPVO/extract_traj.py

# This command runs your application, comment out this line to compile only
CMD ["bash"]
