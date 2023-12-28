#!/bin/bash

venv_path=~/ailab/.venv
venv_bin=~/ailab/.venv/bin

# sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv $venv_path

$venv_bin/pip install python-dotenv
$venv_bin/pip install langchain --upgrade
$venv_bin/pip install pypdf
$venv_bin/pip install pinecone-client
$venv_bin/pip install nlpcloud
# $venv_bin/pip install transformers
# $venv_bin/pip install tensorflow
# $venv_bin/pip install flask