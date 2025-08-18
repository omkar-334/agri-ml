#!/bin/bash
export PYTHONPATH=$(pwd)

python training_scripts/d3/soyatrans_mendeley.py
python training_scripts/d3/coreplant_mendeley.py
python training_scripts/d3/convnext_mendeley.py
python training_scripts/d3/maianet_mendeley.py
python training_scripts/d3/tswinf_mendeley.py

