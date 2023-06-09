#!/bin/bash

cd dct-policy
source venv/bin/activate
python3 exp$1.py
cd ..

