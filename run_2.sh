#!/bin/bash
export PYTHONPATH=$(pwd)

python training_scripts/d2/convnext_pungliya.py
# python training_scripts/d2/coreplant_pungliya.py
python training_scripts/d2/tswinf_pungliya.py
python training_scripts/d2/soyatrans_pungliya.py
python training_scripts/d2/maianet_pungliya.py