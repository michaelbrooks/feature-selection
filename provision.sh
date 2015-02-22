#!/bin/bash

set -e

apt-get update -y
apt-get upgrade -y

apt-get install -y libatlas-dev libblas-dev liblapack-dev gfortran
apt-get install -y python-numpy python-scipy

su --login vagrant <<EOF
mkvirtualenv -a /home/vagrant/feature-selection -r /home/vagrant/feature-selection/requirements.txt --system-site-packages feature-selection
ipython test.py
EOF
