#!/bin/bash

set -e

apt-get update -y
apt-get upgrade -y

apt-get install -y libatlas-dev libblas-dev liblapack-dev gfortran
apt-get install -y python-numpy python-scipy

VAGRANT_HOME=/home/vagrant
VENV_NAME=feature-selection
PROJECT_ROOT=$VAGRANT_HOME/feature-selection

su --login vagrant <<EOF
mkvirtualenv -a $PROJECT_ROOT -r $PROJECT_ROOT/requirements.txt --system-site-packages $VENV_NAME
ipython test.py
EOF

# Add the workon command to the bashrc
echo "Augmenting user's bashrc file..."

if grep -q 'workon' ${VAGRANT_HOME}/.bashrc; then
    echo "workon already in bashrc"
else
    echo "workon $VENV_NAME" >> ${VAGRANT_HOME}/.bashrc
    echo "added workon to bashrc"
fi

if grep -q 'remount' ${VAGRANT_HOME}/.bashrc; then
    echo "remount already in bashrc"
else
    echo "alias remount_vagrant='sudo mount -o remount home_vagrant_textvisdrg'" >> ${VAGRANT_HOME}/.bashrc
    echo "added remount_vagrant to bashrc"
fi
