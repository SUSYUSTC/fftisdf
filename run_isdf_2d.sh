#!/bin/bash
# ISDF-GDF fit for 2D systems (nk x nk x 1 mesh); same arguments as run_isdf.sh
system=$1
nk=$2
basis=$3
factor=$4
suffix=$5
c=$6
shift 6

mkdir -p log_isdf

block=full
abs_tag=
for arg in "$@"; do
    if [ "$arg" = "--ov" ]; then
        block=ov
    fi
    if [ "$arg" = "--abs" ]; then
        abs_tag=_abs
    fi
done

echo host $(hostname)
echo date $(date)
echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK
logfile=log_isdf/isdf${block}_${system}_${nk}x${nk}x1_c${c}_factor${factor}${abs_tag}
echo python -u generate_isdf_gdf.py $system $nk $nk 1 $basis $c $factor -suffix $suffix "$@" --save
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} python -u generate_isdf_gdf.py $system $nk $nk 1 $basis $c $factor -suffix $suffix "$@" --save | tee $logfile
