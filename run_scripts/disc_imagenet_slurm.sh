#!/bin/bash
#SBATCH --job-name=lbo
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH -o /ceph_data/guhyun9454/logs/lbo_%A_%a.out
# Official LinBa (BPTT + memory addressing; Deng & Russakovsky, NeurIPS 2022) on DISC's ImageNet-subset benchmark,
# ICLR 2027 rebuttal C4. Runs on ariel account guhyun9454 (QOS ugrad: 1 GPU, 1 day) -> one subset per job; training
# resumes from $SAVE/ckpt_latest.pt, so a subset that needs more than a day is continued by resubmitting the same command.
#
# Settings = the official README line "BPTT with memory addressing" (CIFAR-10), except:
#   dataset ImageNet_<subset> 128x128 (DISC preprocessing), budget IPC 0.5 via --fixed_budget 1, backbone ConvNetD5
#   (DISC/DDiF train+eval architecture), outer iterations N_ITERS (paper: 50k on CIFAR; compute-bound here).
#
# Array index = subset (0 nette, 1 woof, 2 fruits, 3 yellow, 4 cats, 5 birds).
# STAGE=train (default): data prep (once) + training.  STAGE=eval: DISC-protocol evaluation of synset_it<N_ITERS>.pt
#   (DISC repo rebuttal/eval_linba_synset.py: ConvNetD5 x NEVAL, 1000 ep, DSA, DISC test split).
# Env: N_ITERS, MB (images/class per inner step, official 32), GENQ (real/class in the generalization loss, official 64),
#      INNER (BPTT steps, official 150), EVAL_EVERY (checkpoint interval), TAG, NEVAL.
set -u
SUBS=(nette woof fruits yellow cats birds)
S=${SUBS[${SLURM_ARRAY_TASK_ID:?}]}
STAGE=${STAGE:-train}
N_ITERS=${N_ITERS:-2000}; MB=${MB:-32}; GENQ=${GENQ:-64}; INNER=${INNER:-150}; EVAL_EVERY=${EVAL_EVERY:-20}; TAG=${TAG:-main}
ROOT=/ceph_data/guhyun9454/g
LINBA=$ROOT/LinBa
DISC=$ROOT/DISC_linba
PY=/nas2/data/guhyun9454/anaconda3/envs/disc-cu121/bin/python
NAS=/nas2/data/guhyun9454
DDIR=$NAS/linba_data                       # DISC-preprocessed tensors (NAS, ~3 GB per subset)
DATA=$NAS/ImageNet/extracted/ImageNet_${S} # extracted once from $NAS/ImageNet/ImageNet_<s>.tar.gz
SAVE=$ROOT/LinBa/runs/${S}_ipc0.5_${TAG}   # checkpoints + exported synsets (< 3 MB each)
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 LINBA_DISC_DATA=$DDIR LINBA_RES=128
export TMPDIR=$NAS/tmp; mkdir -p "$TMPDIR" "$DDIR" "$SAVE" "$NAS/ImageNet/extracted"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} stage=$STAGE subset=$S linba=$(git -C $LINBA rev-parse --short=7 HEAD) disc=$(git -C $DISC rev-parse --short=7 HEAD) N_ITERS=$N_ITERS MB=$MB GENQ=$GENQ INNER=$INNER save=$SAVE"
$PY -c "import torch;torch.zeros(1).cuda();print(torch.__version__,torch.cuda.get_device_name(0))" || { echo "FATAL: CUDA init"; exit 3; }

if [ "$STAGE" = "train" ]; then
  # 1) data: extract the subset (once, NAS) and dump DISC-preprocessed tensors (once)
  if [ ! -d "$DATA/train" ] || [ ! -d "$DATA/val" ]; then
    tar -xzf $NAS/ImageNet/ImageNet_${S}.tar.gz -C $NAS/ImageNet/extracted/ || { echo "FATAL: extract"; exit 5; }
  fi
  if [ ! -f "$DDIR/ImageNet_${S}.pt" ]; then
    (cd "$DISC" && $PY -u rebuttal/export_imagenet_tensors.py --subset $S --data_path $DATA --res 128 --out $DDIR/ImageNet_${S}.pt) || { echo "FATAL: export"; exit 5; }
  fi
  # 2) official LinBa training (resumes from $SAVE/ckpt_latest.pt)
  cd "$LINBA" || exit 3
  export PYTHONPATH=$LINBA  # srcs/main.py imports utils/, runners/ from the repo root
  $PY -u srcs/main.py --note disc --backbone_name ConvNetD5 --config compressors_bptt_interventions.yml \
    --ipc 0.5 --compressor_optimizer SGD --bptt_inner_steps $INNER --inner_optimizer SGD --inner_momentum 0.9 \
    --dataset ImageNet_${S} --task_sampler_nc 10 --compressor_minibatch_size $MB --n_basis 16 \
    --compressor imgs_embedding_basis_minibatch --downsample_scale 2 --zca 0 --fixed_budget 1 \
    --coeff_reg l2 --coeff_reg_alpha 1e-4 --interventions none --validation_ratio 0 \
    --n_iters $N_ITERS --eval_every $EVAL_EVERY --num_eval 0 --gen_queries $GENQ --save_dir $SAVE
  echo "TRAIN_EXIT=$?"
  [ -f "$SAVE/synset_it$(printf %06d $N_ITERS).pt" ] && echo "TRAIN_DONE" || echo "TRAIN_INCOMPLETE (resubmit to continue; tracebacks above if any)"
elif [ "$STAGE" = "eval" ]; then
  FINAL=$SAVE/synset_it$(printf %06d $N_ITERS).pt
  [ -f "$FINAL" ] || { echo "FATAL: no $FINAL"; exit 6; }
  cd "$DISC" && $PY -u rebuttal/eval_linba_synset.py --synset $FINAL --subset $S --data_path $DATA --res 128 --ipc 0.5 \
    --num_eval ${NEVAL:-5} --out $SAVE/eval_it$(printf %06d $N_ITERS).json
  echo "EVAL_EXIT=$?"
fi
