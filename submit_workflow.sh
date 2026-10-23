#!/bin/bash

nthreads_dft=32
nthreads_vxc=32
nthreads_screen=32
nthreads_bse=32
nmpi_gw=32
mem_dft=250GB
mem_vxc=250GB
mem_gw=320GB
mem_screen=250GB
mem_bse=250GB
time_limit=2400
pyscf_max_memory_dft=100000
pyscf_max_memory_vxc=100000
pyscf_max_memory_gw=8000
reservation=changroup_standingres
constraint=icelake

system=$1
nk=$2
basis=$3
shift 3

suffix=
while [ $# -gt 0 ]; do
    if [ "$1" = --suffix ]; then
        suffix=$2
        shift 2
    else
        echo "Unknown argument: $1"
        exit 1
    fi
done

klabel=${nk}x${nk}x${nk}
name=${system}_${basis}
if [ -n "$suffix" ]; then
    name=${name}_${suffix}
fi

logdir=log
if [ -n "$suffix" ]; then
    logdir=log/$suffix
fi
mkdir -p $logdir

suffix_flag=
if [ -n "$suffix" ]; then
    suffix_flag="-suffix $suffix"
fi

cmd_dft="bash -lc 'set -o pipefail; mkdir -p $logdir; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_dft OMP_NUM_THREADS=$nthreads_dft python -u do_dft_gdf.py $system $nk $nk $nk $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_dft OMP_NUM_THREADS=$nthreads_dft python -u do_dft_gdf.py $system $nk $nk $nk $basis $suffix_flag | tee $logdir/dft_${name}_${klabel}.log'"
cmd_vxc="bash -lc 'set -o pipefail; mkdir -p $logdir; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_vxc OMP_NUM_THREADS=$nthreads_vxc python -u do_vxc.py $system $nk $nk $nk $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_vxc OMP_NUM_THREADS=$nthreads_vxc python -u do_vxc.py $system $nk $nk $nk $basis $suffix_flag | tee $logdir/vxc_${name}_${klabel}.log'"
cmd_gw="bash -lc 'set -o pipefail; mkdir -p $logdir; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_gw mpiexec -n $nmpi_gw python -u do_gw.py $system $nk $nk $nk $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_gw mpiexec -n $nmpi_gw python -u do_gw.py $system $nk $nk $nk $basis $suffix_flag | tee $logdir/gw_${name}_${klabel}.log'"
cmd_screen="bash -lc 'set -o pipefail; mkdir -p $logdir; echo OMP_NUM_THREADS=$nthreads_screen python -u compute_screening.py $system $nk $nk $nk $basis $suffix_flag; OMP_NUM_THREADS=$nthreads_screen python -u compute_screening.py $system $nk $nk $nk $basis $suffix_flag | tee $logdir/screening_${name}_${klabel}.log'"
cmd_bse="bash -lc 'set -o pipefail; mkdir -p $logdir; echo OMP_NUM_THREADS=$nthreads_bse python -u bse_df.py $system $nk $nk $nk $basis $suffix_flag; OMP_NUM_THREADS=$nthreads_bse python -u bse_df.py $system $nk $nk $nk $basis $suffix_flag | tee $logdir/bse_direct_${name}_${klabel}.log; status=\${PIPESTATUS[0]}; if [ \$status -ne 0 ]; then exit \$status; fi; echo OMP_NUM_THREADS=$nthreads_bse python -u bse_df.py $system $nk $nk $nk $basis $suffix_flag --indirect; OMP_NUM_THREADS=$nthreads_bse python -u bse_df.py $system $nk $nk $nk $basis $suffix_flag --indirect | tee $logdir/bse_indirect_${name}_${klabel}.log'"

logfile_dft=$logdir/slurm_dft_${name}_${klabel}.out
logfile_vxc=$logdir/slurm_vxc_${name}_${klabel}.out
logfile_gw=$logdir/slurm_gw_${name}_${klabel}.out
logfile_screen=$logdir/slurm_screening_${name}_${klabel}.out
logfile_bse=$logdir/slurm_bse_${name}_${klabel}.out

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_dft --mem=$mem_dft -t $time_limit -J dft_${name}_${klabel} -o $logfile_dft --wrap "$cmd_dft"
dftid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_dft --mem=$mem_dft -t $time_limit -J dft_${name}_${klabel} -o $logfile_dft --wrap "$cmd_dft" | awk '{print $4}')
echo dft_jobid $dftid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_vxc --mem=$mem_vxc -t $time_limit -J vxc_${name}_${klabel} -o $logfile_vxc --dependency=afterok:$dftid --wrap "$cmd_vxc"
vxcid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_vxc --mem=$mem_vxc -t $time_limit -J vxc_${name}_${klabel} -o $logfile_vxc --dependency=afterok:$dftid --wrap "$cmd_vxc" | awk '{print $4}')
echo vxc_jobid $vxcid

echo sbatch --reservation=$reservation --constraint=$constraint -N 1 --ntasks-per-node=$nmpi_gw -c 1 --mem=$mem_gw -t $time_limit -J gw_${name}_${klabel} -o $logfile_gw --dependency=afterok:$vxcid --wrap "$cmd_gw"
gwid=$(sbatch --reservation=$reservation --constraint=$constraint -N 1 --ntasks-per-node=$nmpi_gw -c 1 --mem=$mem_gw -t $time_limit -J gw_${name}_${klabel} -o $logfile_gw --dependency=afterok:$vxcid --wrap "$cmd_gw" | awk '{print $4}')
echo gw_jobid $gwid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_screen --mem=$mem_screen -t $time_limit -J screening_${name}_${klabel} -o $logfile_screen --dependency=afterok:$gwid --wrap "$cmd_screen"
screenid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_screen --mem=$mem_screen -t $time_limit -J screening_${name}_${klabel} -o $logfile_screen --dependency=afterok:$gwid --wrap "$cmd_screen" | awk '{print $4}')
echo screen_jobid $screenid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_bse --mem=$mem_bse -t $time_limit -J bse_${name}_${klabel} -o $logfile_bse --dependency=afterok:$screenid --wrap "$cmd_bse"
bseid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_bse --mem=$mem_bse -t $time_limit -J bse_${name}_${klabel} -o $logfile_bse --dependency=afterok:$screenid --wrap "$cmd_bse" | awk '{print $4}')
echo bse_jobid $bseid
