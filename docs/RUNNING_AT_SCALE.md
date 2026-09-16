# Running training at scale

The closed-set heads and mask-transformers are trained on a single GPU
(V100/A100 class). Cluster-specific job scripts are intentionally omitted from
this repository; run the training entry points directly, e.g.:

```bash
export OFFROAD_OOD_ROOT=$(pwd)
export PYTHONPATH=offroad_ood
export HF_HOME=$OFFROAD_OOD_ROOT/.hf_cache      # cache backbones locally

python offroad_ood/m2f_train.py   --dataset goose --seed 0
python offroad_ood/m2f_rba_eval.py --dataset goose --seed 0
```

Seeds `{0,1,2}` reproduce the seed mean±std reported for the operating-point
analysis. Per-pixel feature baselines and all calibration/analysis steps run on
CPU from cached scores (`results/*.npz`), so they do not need a GPU once
features are dumped.
