#!/bin/bash
# test the 2D candidate grid (uniform xy spacing hxy, z spacing ratio*hxy decaying as n_AO^-power)
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
    hxy=$5
    ratio=$6
    power=$7
    ov_flag=
    if [ $block = ov ]; then
        ov_flag=--ov
    fi
    tag=${block}_${system}_${nk}x${nk}x1_c${c}_hxy${hxy}_r${ratio}_p${power}
    cmd="OMP_NUM_THREADS=$nthreads python -u generate_isdf_gdf.py $system $nk $nk 1 $basis $c 1.5 -suffix $suffix $ov_flag -cholesky_max 60000 -grid_hxy $hxy -grid_ratio $ratio -grid_power $power -chk $chkdir/$tag.isdf"
    echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=160GB -t $time_limit -J gt_$tag -o $logdir/$tag.out --wrap "$cmd"
    sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=160GB -t $time_limit -J gt_$tag -o $logdir/$tag.out --wrap "$cmd"
}

for system in MoSe2D blackP2D; do
    for fit in "ov 16" "full 36"; do
        for hxy in 0.25 0.19 0.15 0.12; do
            submit_one $system 3 $fit $hxy 1.0 0.3333
        done
        for ratio in 0.5 0.75; do
            submit_one $system 3 $fit 0.19 $ratio 0.3333
        done
        for power in 0.1667 0.5 0.0; do
            submit_one $system 3 $fit 0.19 1.0 $power
        done
    done
done
