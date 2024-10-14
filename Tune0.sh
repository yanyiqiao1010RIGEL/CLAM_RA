#!/bin/bash

#PBS -P au38
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -l storage=gdata/au38
source /g/data/au38/yy3740/miniconda3/bin/activate clam_latest
python3 /g/data/au38/CLAM/create_patches_fp.py --source /g/data/au38/yy3740/Eval/Carolina --save_dir /g/data/au38/yy3740/Eval/Carolina_feats --patch_size 256 --seg --patch --stitch

python3 /g/data/au38/CLAM/create_patches_fp.py --source /g/data/au38/yy3740/Eval/Georgia --save_dir /g/data/au38/yy3740/Eval/Georgia_feats --patch_size 256 --seg --patch --stitch
