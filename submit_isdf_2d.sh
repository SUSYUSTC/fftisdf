#!/bin/bash
# ISDF generation for 2D systems on central (bare only; DGX never stores GDF)
# ov: reference c = 12, 16 (factor 1.5), 20 (factor 3.0), keep smallest with ref error <= 1e-3 at all nk
#     initial guesses c = 4, 5, 6, 8 (factor 1.5)
# full bare: reference c = 36 (--abs, 5e-5), initial guesses c = 4, 5, 6 (factor 1.5)
nthreads=8
time_limit=2400
suffix=final
basis=gth-dzvp-molopt-sr
reservation=changroup_standingres
constraint=icelake
logdir=log_isdf

mkdir -p $logdir

function submit_one(){
    system=$1
    nk=$2
    c=$3
    factor=$4
    shift 4
    gdf=data_${system}_${basis}_${suffix}/GDF_${nk}x${nk}x1.chk
    mem_gb=$(( ($(stat -c%s $gdf) * 4 + 1073741823) / 1073741824 + 40 ))
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
    logfile=$logdir/slurm_isdf${block}_${system}_${nk}x${nk}x1_c${c}_factor${factor}${abs_tag}.out
    echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=${mem_gb}GB -t $time_limit -J isdf${block}_${system}_${nk}_c${c} -o $logfile ./run_isdf_2d.sh $system $nk $basis $factor $suffix $c "$@"
    sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=${mem_gb}GB -t $time_limit -J isdf${block}_${system}_${nk}_c${c} -o $logfile ./run_isdf_2d.sh $system $nk $basis $factor $suffix $c "$@"
}

for system in MoSe2D; do
    for nk in 2 3 4 5 6 7 8; do
        for c in 4 5 6 8 12 16; do
            submit_one $system $nk $c 1.5 --ov
        done
        submit_one $system $nk 20 3.0 --ov
        for c in 4 5 6; do
            submit_one $system $nk $c 1.5
        done
        submit_one $system $nk 36 5e-5 --abs
    done
done
