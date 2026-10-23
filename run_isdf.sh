#!/bin/bash

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
screen_tag=
for arg in "$@"; do
    if [ "$arg" = "--ov" ]; then
        block=ov
    fi
    if [ "$arg" = "--abs" ]; then
        abs_tag=_abs
    fi
    if [ "$arg" = "--screen" ]; then
        screen_tag=_screen
    fi
done

logfile=log_isdf/isdf${block}${screen_tag}_${system}_${nk}_c${c}_factor${factor}${abs_tag}
echo python -u generate_isdf_gdf.py $system $nk $nk $nk $basis $c $factor -suffix $suffix "$@" --save
python -u generate_isdf_gdf.py $system $nk $nk $nk $basis $c $factor -suffix $suffix "$@" --save | tee $logfile
