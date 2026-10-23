#!/bin/bash

guide=${1:-symm_systems.txt}
mkdir -p log_symm_mf

while read system basis suffix max_nk; do
    if [ -z "$system" ]; then
        continue
    fi
    case "$system" in
        \#*) continue ;;
    esac

    for nk in $(seq 2 $max_nk); do
        klabel=${nk}x${nk}x${nk}
        if [ "$suffix" = "-" ]; then
            suffix_label=nosuffix
            cmd="python -u symmetrize_mf.py $system $nk $nk $nk $basis"
        else
            suffix_label=$suffix
            cmd="python -u symmetrize_mf.py $system $nk $nk $nk $basis -suffix $suffix"
        fi
        logfile=log_symm_mf/symm_${system}_${basis}_${suffix_label}_${klabel}.log
        echo sbatch -n 1 -c 8 --mem=40GB -t 2400 -o $logfile --wrap "$cmd"
        sbatch -n 1 -c 8 --mem=40GB -t 2400 -o $logfile --wrap "$cmd"
    done
done < $guide
