#!/bin/bash

system=$1
kx=$2
ky=$3
kz=$4
basis=$5
suffix=$6
shift 6

nthreads=${SLURM_CPUS_PER_TASK:-32}
max_memory=280000
if [ -n "$SLURM_MEM_PER_NODE" ]; then
	max_memory=$((SLURM_MEM_PER_NODE - 4000))
fi

klabel=${kx}x${ky}x${kz}
name=${system}_${basis}
suffix_flag=
if [ "$suffix" != none ]; then
	name=${name}_${suffix}
	suffix_flag="-suffix $suffix"
fi

mkdir -p log_dft
logfile=log_dft/dft_${name}_${klabel}.log

echo command OMP_NUM_THREADS=$nthreads PYSCF_MAX_MEMORY=$max_memory python -u do_dft_gdf.py $system $kx $ky $kz $basis $suffix_flag "$@"
OMP_NUM_THREADS=$nthreads PYSCF_MAX_MEMORY=$max_memory python -u do_dft_gdf.py $system $kx $ky $kz $basis $suffix_flag "$@" | tee $logfile
