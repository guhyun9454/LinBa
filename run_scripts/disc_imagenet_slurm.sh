#!/bin/bash
#SBATCH --job-name=lbo
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=3-0
#SBATCH -o /ceph_data/jihye4118/reb_iclr/logs/lbo_%A_%a.out
# Official LinBa (BPTT + memory addressing; Deng & Russakovsky, NeurIPS 2022) on DISC's ImageNet-subset benchmark,
# ICLR 2027 rebuttal C4. Settings = the official README "BPTT with memory addressing" line for CIFAR-10, except:
#   dataset ImageNet_<subset> 128x128 (DISC preprocessing), budget IPC 0.5 via --fixed_budget 1, backbone ConvNetD5
#   (the architecture DISC/DDiF are trained and evaluated with), outer iterations N_ITERS (paper: 50k; compute-bound here).
# Array index = subset (0 nette, 1 woof, 2 fruits, 3 yellow, 4 cats, 5 birds). Env overrides: N_ITERS, MB (images/class per
# inner step, official 32), GENQ (real images/class in the generalization loss, official 64), INNER (BPTT steps, official 150),
# EVAL_EVERY (checkpoint interval), TAG.
# After training: DISC-protocol evaluation of the final synset (DISC repo rebuttal/eval_linba_synset.py: ConvNetD5 x5, 1000 ep, DSA).
set -u
SUBS=(nette woof fruits yellow cats birds)
S=${SUBS[${SLURM_ARRAY_TASK_ID:?}]}
N_ITERS=${N_ITERS:-2000}; MB=${MB:-32}; GENQ=${GENQ:-64}; INNER=${INNER:-150}; EVAL_EVERY=${EVAL_EVERY:-50}; TAG=${TAG:-main}
LINBA=/ceph_data/jihye4118/LinBa
DISC=/ceph_data/jihye4118/DISC_linba2
PY=/ceph_data/jihye4118/miniconda3/envs/ddif/bin/python
DDIR=/nas2/data/jihye4118/reb_iclr/linba_data
SAVE=/ceph_data/jihye4118/reb_iclr/linba_official/${S}_ipc0.5_${TAG}
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 LINBA_DISC_DATA=$DDIR LINBA_RES=128
export TMPDIR=/ceph_data/jihye4118/reb_iclr/tmp; mkdir -p "$TMPDIR" "$DDIR" "$SAVE"
DATA=/nas2/data/jihye4118/datasets/ImageNet_${S}
for L in /data2/local_datasets/ImageNet_${S} /data2/local_datasets/jh/ImageNet_${S} /data3/local_datasets/jh/ImageNet_${S} \
         /data3/local_datasets/jihye4118/ImageNet_${S} /data2/local_datasets/jihye4118/ImageNet_${S} \
         /data3/local_datasets/ImageNet_${S} /data1/local_datasets/ImageNet_${S} /data/local_datasets/ImageNet_${S}; do
  [ -d "$L/train" ] && [ -d "$L/val" ] && { DATA=$L; break; }
done
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} subset=$S linba=$(git -C $LINBA rev-parse --short=7 HEAD) disc=$(git -C $DISC rev-parse --short=7 HEAD) data=$DATA N_ITERS=$N_ITERS MB=$MB GENQ=$GENQ INNER=$INNER save=$SAVE"
$PY -c "import torch;torch.zeros(1).cuda();print(torch.__version__,torch.cuda.get_device_name(0))" || { echo "FATAL: CUDA init"; exit 3; }

# 1) DISC-preprocessed tensors (once per subset, NAS)
if [ ! -f "$DDIR/ImageNet_${S}.pt" ]; then
  (cd "$DISC" && $PY -u rebuttal/export_imagenet_tensors.py --subset $S --data_path $DATA --res 128 --out $DDIR/ImageNet_${S}.pt) || { echo "FATAL: export"; exit 5; }
fi

# 2) official LinBa training (resumes from $SAVE/ckpt_latest.pt)
cd "$LINBA" || exit 3
for TRY in 1 2 3; do
  $PY -u srcs/main.py --note disc --backbone_name ConvNetD5 --config compressors_bptt_interventions.yml \
    --ipc 0.5 --compressor_optimizer SGD --bptt_inner_steps $INNER --inner_optimizer SGD --inner_momentum 0.9 \
    --dataset ImageNet_${S} --task_sampler_nc 10 --compressor_minibatch_size $MB --n_basis 16 \
    --compressor imgs_embedding_basis_minibatch --downsample_scale 2 --zca 0 --fixed_budget 1 \
    --coeff_reg l2 --coeff_reg_alpha 1e-4 --interventions none --validation_ratio 0 \
    --n_iters $N_ITERS --eval_every $EVAL_EVERY --num_eval 0 --gen_queries $GENQ --save_dir $SAVE
  RC=$?; echo "TRAIN_EXIT=$RC (try $TRY)"
  [ -f "$SAVE/synset_it$(printf %06d $N_ITERS).pt" ] && break
  case $RC in 137|143) sleep 30 ;; *) [ $RC -ne 0 ] && exit $RC ;; esac
done
FINAL=$SAVE/synset_it$(printf %06d $N_ITERS).pt
[ -f "$FINAL" ] || { echo "FATAL: no final synset (training printed tracebacks above?)"; exit 6; }

# 3) DISC-protocol evaluation
cd "$DISC" && $PY -u rebuttal/eval_linba_synset.py --synset $FINAL --subset $S --data_path $DATA --res 128 --ipc 0.5 \
  --num_eval ${NEVAL:-5} --out $SAVE/eval_it$(printf %06d $N_ITERS).json
echo "EVAL_EXIT=$?"
