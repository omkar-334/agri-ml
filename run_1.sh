#!/bin/bash
export PYTHONPATH=$(pwd)

python training_scripts/d1/convnext_nirmal.py
# python training_scripts/d1/coreplant_nirmal.py
python training_scripts/d1/tswinf_nirmal.py
python training_scripts/d1/soyatrans_nirmal.py
python training_scripts/d1/maianet_nirmal.py