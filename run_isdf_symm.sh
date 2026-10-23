#!/bin/bash

system=$1
nk=$2
basis=$3
factor=$4
suffix=$5
c=$6
shift 6

mkdir -p log_isdf_symm

block=full
for arg in "$@"; do
    if [ "$arg" = "--ov" ]; then
        block=ov
    fi
done

logfile=log_isdf_symm/isdf${block}_${system}_${nk}_c${c}_factor${factor}
echo python -u generate_isdf_gdf_symm.py $system $nk $nk $nk $basis $c $factor -suffix $suffix "$@" --save
python -u generate_isdf_gdf_symm.py $system $nk $nk $nk $basis $c $factor -suffix $suffix "$@" --save | tee $logfile
