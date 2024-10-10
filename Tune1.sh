#!/bin/bash

#PBS -P au38
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -l storage=gdata/au38

source /g/data/au38/yy3740/miniconda3/bin/activate clam_latest

# Extract features for Georgia
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Georgia_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Georgia \
--csv_path /g/data/au38/yy3740/Labels/Georgia.csv \
--feat_dir /g/data/au38/yy3740/Eval/Georgia_feats \
--batch_size 512 --slide_ext .svs

# Extract features for Carolina
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Carolina_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Carolina \
--csv_path /g/data/au38/yy3740/Labels/Carolina.csv \
--feat_dir /g/data/au38/yy3740/Eval/Carolina_feats \
--batch_size 512 --slide_ext .svs

# Extract features for Florida_26
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Florida_26_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Florida_26 \
--csv_path /g/data/au38/yy3740/Labels/lorrida.csv \
--feat_dir /g/data/au38/yy3740/Eval/Florida_26_feats \
--batch_size 512 --slide_ext .svs

