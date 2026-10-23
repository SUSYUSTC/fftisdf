#!/bin/bash
# c_ref scan on a medium-large k-mesh: reference ISDF should reach rel_error <= 1e-3 (ov) and ~5e-5 (full)
nthreads=16
time_limit=2400
suffix=final
basis=gth-dzvp-molopt-sr
reservation=changroup_standingres
constraint=icelake
logdir=log_isdf

mkdir -p $logdir

function submit_one(){
    runscript=$1
    system=$2
    nk=$3
    c=$4
    factor=$5
    mem=$6
    dep=$7
    shift 7
    dep_flag=
    if [ $dep != none ]; then
        dep_flag=--dependency=afterok:$dep
    fi
    tag=$(echo "$@" | tr -d ' -')
    logfile=$logdir/slurm_cref_${system}_${nk}_c${c}_factor${factor}_${tag}.out
    echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=$mem -t $time_limit $dep_flag -J cref_${system}_${nk}_c${c} -o $logfile ./$runscript $system $nk $basis $factor $suffix $c "$@"
    sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=$mem -t $time_limit $dep_flag -J cref_${system}_${nk}_c${c} -o $logfile ./$runscript $system $nk $basis $factor $suffix $c "$@"
}

# 2D (bare full ERI): MoSe2D now, blackP2D after its 6x6x1 DFT/GDF (job 5355813)
for sd in "MoSe2D none" "blackP2D 5355813"; do
    set -- $sd
    system=$1
    dep=$2
    submit_one run_isdf_2d.sh $system 6 16 1.5 120GB $dep --ov
    for c in 20 24 30 36; do
        submit_one run_isdf_2d.sh $system 6 $c 3.0 120GB $dep --ov
    done
    for c in 36 48 60 72; do
        submit_one run_isdf_2d.sh $system 6 $c 5e-5 120GB $dep --abs
    done
done

# 3D zinc blende CdSe (previous settings: bare ov, screened full)
for c in 12 16; do
    submit_one run_isdf.sh CdSezb 4 $c 1.5 160GB none --ov
done
for c in 20 24; do
    submit_one run_isdf.sh CdSezb 4 $c 3.0 160GB none --ov
done
for c in 24 36 48; do
    submit_one run_isdf.sh CdSezb 4 $c 5e-5 160GB none --screen --abs
done
