#!/bin/bash

#PBS -P au38
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -l storage=gdata/au38

source /g/data/au38/yy3740/miniconda3/bin/activate clam_latest

# Extract features for Ohio
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Ohio_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Ohio \
--csv_path /g/data/au38/yy3740/Labels/Ohio.csv \
--feat_dir /g/data/au38/yy3740/Eval/Ohio_feats \
--batch_size 512 --slide_ext .svs

# Extract features for Minnesota
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Minnesota_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Minnesota \
--csv_path /g/data/au38/yy3740/Labels/Minnesota.csv \
--feat_dir /g/data/au38/yy3740/Eval/Minnesota_feats \
--batch_size 512 --slide_ext .svs

# Extract features for Utah
python3 /g/data/au38/CLAM/extract_features_fp.py \
--data_h5_dir /g/data/au38/yy3740/Eval/Utah_feats \
--data_slide_dir /g/data/au38/yy3740/Eval/Utah \
--csv_path /g/data/au38/yy3740/Labels/Utah.csv \
--feat_dir /g/data/au38/yy3740/Eval/Utah_feats \
--batch_size 512 --slide_ext .svs

