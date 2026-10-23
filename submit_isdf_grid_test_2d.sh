#!/bin/bash
# test the 2D candidate grid (z spacing ratio*hxy decaying as n_AO^-power, hxy set by the 20000-point budget)
nthreads=16
time_limit=600
suffix=final
basis=gth-dzvp-molopt-sr
reservation=changroup_standingres
constraint=icelake
logdir=log_isdf_gridtest
chkdir=chk_gridtest

mkdir -p $logdir $chkdir

function submit_one(){
    system=$1
    nk=$2
    block=$3
    c=$4
    ratio=$5
    power=$6
    ov_flag=
    if [ $block = ov ]; then
        ov_flag=--ov
    fi
    tag=${block}_${system}_${nk}x${nk}x1_c${c}_r${ratio}_p${power}
    cmd="OMP_NUM_THREADS=$nthreads python -u generate_isdf_gdf.py $system $nk $nk 1 $basis $c 1.5 -suffix $suffix $ov_flag -grid_ratio $ratio -grid_power $power -chk $chkdir/$tag.isdf"
    echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=96GB -t $time_limit -J gt_$tag -o $logdir/$tag.out --wrap "$cmd"
    sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=96GB -t $time_limit -J gt_$tag -o $logdir/$tag.out --wrap "$cmd"
}

# first scan (fixed hxy 0.12-0.25 A, cholesky_max 60000): insensitive within ~20%
for system in MoSe2D blackP2D; do
    for fit in "ov 16" "full 36"; do
        for ratio in 1.0 0.75 0.5; do
            for power in 0.3333 0.1667 0.0; do
                submit_one $system 3 $fit $ratio $power
            done
        done
    done
done
