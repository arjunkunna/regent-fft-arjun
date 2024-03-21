#!/bin/bash

set -e
set -x

$SUDO_COMMAND apt-get update -qq
$SUDO_COMMAND apt-get install -qq software-properties-common
wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$SUDO_COMMAND mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$SUDO_COMMAND apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
$SUDO_COMMAND add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
$SUDO_COMMAND apt-get update -qq
$SUDO_COMMAND apt-get install -qq cuda-compiler-11.6