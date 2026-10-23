#!/bin/bash

nthreads_dft=8
nthreads_vxc=8
nthreads_screen=8
nthreads_isdf=8
nmpi_gw=8
mem_dft=120GB
mem_vxc=120GB
mem_gw=120GB
mem_screen=120GB
mem_isdf=120GB
time_limit=2400
pyscf_max_memory_dft=80000
pyscf_max_memory_vxc=80000
pyscf_max_memory_gw=12000
reservation=changroup_standingres
constraint=icelake
system=diamond
c_ov=16
c_screen=24
factor=1.2
suffix=final

kx=$1
ky=$2
kz=$3
basis=$4

klabel=${kx}x${ky}x${kz}
name=${system}_${basis}
suffix_flag=
logdir=log
if [ -n "$suffix" ]; then
    name=${name}_${suffix}
    suffix_flag="-suffix $suffix"
    logdir=log/$suffix
fi
mkdir -p $logdir

cmd_dft="bash -lc 'set -o pipefail; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_dft OMP_NUM_THREADS=$nthreads_dft python -u do_dft_gdf.py $system $kx $ky $kz $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_dft OMP_NUM_THREADS=$nthreads_dft python -u do_dft_gdf.py $system $kx $ky $kz $basis $suffix_flag | tee $logdir/dft_${name}_${klabel}.log'"
cmd_vxc="bash -lc 'set -o pipefail; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_vxc OMP_NUM_THREADS=$nthreads_vxc python -u do_vxc.py $system $kx $ky $kz $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_vxc OMP_NUM_THREADS=$nthreads_vxc python -u do_vxc.py $system $kx $ky $kz $basis $suffix_flag | tee $logdir/vxc_${name}_${klabel}.log'"
cmd_gw="bash -lc 'set -o pipefail; echo PYSCF_MAX_MEMORY=$pyscf_max_memory_gw mpiexec -n $nmpi_gw python -u do_gw.py $system $kx $ky $kz $basis $suffix_flag; PYSCF_MAX_MEMORY=$pyscf_max_memory_gw mpiexec -n $nmpi_gw python -u do_gw.py $system $kx $ky $kz $basis $suffix_flag | tee $logdir/gw_${name}_${klabel}.log'"
cmd_screen="bash -lc 'set -o pipefail; echo OMP_NUM_THREADS=$nthreads_screen python -u compute_screening.py $system $kx $ky $kz $basis $suffix_flag; OMP_NUM_THREADS=$nthreads_screen python -u compute_screening.py $system $kx $ky $kz $basis $suffix_flag | tee $logdir/screening_${name}_${klabel}.log'"
cmd_isdf_ov="bash -lc 'set -o pipefail; echo OMP_NUM_THREADS=$nthreads_isdf python -u generate_isdf_gdf.py $system $kx $ky $kz $basis $c_ov $factor $suffix_flag --ov --save; OMP_NUM_THREADS=$nthreads_isdf python -u generate_isdf_gdf.py $system $kx $ky $kz $basis $c_ov $factor $suffix_flag --ov --save | tee $logdir/isdfov_bare_${name}_${klabel}_c${c_ov}_factor${factor}.log'"
cmd_isdf_screen="bash -lc 'set -o pipefail; echo OMP_NUM_THREADS=$nthreads_isdf python -u generate_isdf_gdf.py $system $kx $ky $kz $basis $c_screen $factor $suffix_flag --screen --save; OMP_NUM_THREADS=$nthreads_isdf python -u generate_isdf_gdf.py $system $kx $ky $kz $basis $c_screen $factor $suffix_flag --screen --save | tee $logdir/isdffull_screen_${name}_${klabel}_c${c_screen}_factor${factor}.log'"

logfile_dft=$logdir/slurm_dft_${name}_${klabel}.out
logfile_vxc=$logdir/slurm_vxc_${name}_${klabel}.out
logfile_gw=$logdir/slurm_gw_${name}_${klabel}.out
logfile_screen=$logdir/slurm_screening_${name}_${klabel}.out
logfile_isdf_ov=$logdir/slurm_isdfov_bare_${name}_${klabel}_c${c_ov}_factor${factor}.out
logfile_isdf_screen=$logdir/slurm_isdffull_screen_${name}_${klabel}_c${c_screen}_factor${factor}.out

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_dft --mem=$mem_dft -t $time_limit -o $logfile_dft --wrap "$cmd_dft"
dftid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_dft --mem=$mem_dft -t $time_limit -o $logfile_dft --wrap "$cmd_dft" | awk '{print $4}')
echo dft_jobid $dftid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_vxc --mem=$mem_vxc -t $time_limit -o $logfile_vxc --dependency=afterok:$dftid --wrap "$cmd_vxc"
vxcid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_vxc --mem=$mem_vxc -t $time_limit -o $logfile_vxc --dependency=afterok:$dftid --wrap "$cmd_vxc" | awk '{print $4}')
echo vxc_jobid $vxcid

echo sbatch --reservation=$reservation --constraint=$constraint -N 1 --ntasks-per-node=$nmpi_gw -c 1 --mem=$mem_gw -t $time_limit -o $logfile_gw --dependency=afterok:$vxcid --wrap "$cmd_gw"
gwid=$(sbatch --reservation=$reservation --constraint=$constraint -N 1 --ntasks-per-node=$nmpi_gw -c 1 --mem=$mem_gw -t $time_limit -o $logfile_gw --dependency=afterok:$vxcid --wrap "$cmd_gw" | awk '{print $4}')
echo gw_jobid $gwid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_screen --mem=$mem_screen -t $time_limit -o $logfile_screen --dependency=afterok:$gwid --wrap "$cmd_screen"
screenid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_screen --mem=$mem_screen -t $time_limit -o $logfile_screen --dependency=afterok:$gwid --wrap "$cmd_screen" | awk '{print $4}')
echo screen_jobid $screenid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_isdf --mem=$mem_isdf -t $time_limit -o $logfile_isdf_ov --dependency=afterok:$screenid --wrap "$cmd_isdf_ov"
isdfovid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_isdf --mem=$mem_isdf -t $time_limit -o $logfile_isdf_ov --dependency=afterok:$screenid --wrap "$cmd_isdf_ov" | awk '{print $4}')
echo isdfov_jobid $isdfovid

echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_isdf --mem=$mem_isdf -t $time_limit -o $logfile_isdf_screen --dependency=afterok:$screenid --wrap "$cmd_isdf_screen"
isdffullid=$(sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads_isdf --mem=$mem_isdf -t $time_limit -o $logfile_isdf_screen --dependency=afterok:$screenid --wrap "$cmd_isdf_screen" | awk '{print $4}')
echo isdffull_jobid $isdffullid
