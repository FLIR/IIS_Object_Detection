sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  openssh-server \
  build-essential \
  cmake \
  checkinstall \
  pkg-config \
  gnupg-agent \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  wget \
  git \
  curl \
  unzip \
  yasm \
  pkg-config \
  nano \
  vim \
  tigervnc-standalone-server \
  python3-dev \
  python3-pip # might need to handle this

echo "COMMAND : sudo rm /etc/apt/sources.list.d/cuda*"
sudo rm -rf /etc/apt/sources.list.d/cuda*

echo "COMMAND : sudo apt remove --autoremove nvidia-*"
sudo apt remove --autoremove nvidia-*

echo "COMMAND : sudo apt update"
sudo apt update

echo "sudo add-apt-repository ppa:graphics-drivers/ppa"
# echo | implicitly hits enter
echo | sudo add-apt-repository ppa:graphics-drivers/ppa

echo "COMMAND : sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'"
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'

echo "COMMAND : sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda_learn.list'"
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda_learn.list'

echo "COMMAND : sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub"
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

echo "COMMAND : sudo apt update && sudo apt install cuda-11-4 libcudnn8"
sudo apt update && sudo apt install -y cuda-11-4 libcudnn8 nvidia-driver-470 nvidia-cuda-toolkit

curl -s -k -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -k -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

echo "distribution=${distribution}"
echo "COMMAND : curl -s -k -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list"
curl -s -k -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

echo "COMMAND : sudo apt update && sudo apt install nvidia-container-runtime"
sudo apt update && sudo apt install -y nvidia-container-runtime

echo "sudo add-apt-repository deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

echo "COMMAND curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

echo "COMMAND : sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io"
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io

echo "COMMAND : sudo groupadd docker"
sudo groupadd docker

echo "COMMAND : sudo usermod -aG docker $USER"
sudo usermod -aG docker $USER

echo "COMMAND : newgrp docker"
newgrp docker

echo "============================= REBOOT ME PLEASE! ============================="
# sudo reboot
