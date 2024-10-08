#!/bin/bash

#PBS -P au38
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -l storage=gdata/au38

source /g/data/au38/yy3740/miniconda3/bin/activate clam_latest

# Minnesota_feats evaluation
python3 /g/data/au38/CLAM/eval.py \
  --task task_3_tgca \
  --data_root_dir /g/data/au38/yy3740/Eval/Minnesota_feats/h5_files \
  --csv_path /g/data/au38/yy3740/Labels/Minnesota.csv \
  --models_exp_code task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1 \
  --results_dir /g/data/au38/yy3740/results \
  --save_exp_code Minnesota_eval_run \
  --split all

# Ohio_feats evaluation
python3 /g/data/au38/CLAM/eval.py \
  --task task_3_tgca \
  --data_root_dir /g/data/au38/yy3740/Eval/Ohio_feats/h5_files \
  --csv_path /g/data/au38/yy3740/Labels/Ohio.csv \
  --models_exp_code task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1 \
  --results_dir /g/data/au38/yy3740/results \
  --save_exp_code Ohio_eval_run \
  --split all

# Utah_feats evaluation
python3 /g/data/au38/CLAM/eval.py \
  --task task_3_tgca \
  --data_root_dir /g/data/au38/yy3740/Eval/Utah_feats/h5_files \
  --csv_path /g/data/au38/yy3740/Labels/Utah.csv \
  --models_exp_code task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1 \
  --results_dir /g/data/au38/yy3740/results \
  --save_exp_code Utah_eval_run \
  --split all
