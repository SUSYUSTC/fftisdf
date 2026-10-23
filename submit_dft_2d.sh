#!/bin/bash
# DFT + GDF for 2D systems (analytic 2D truncated Coulomb, Lz = 30 A), nk x nk x 1 up to GDF ~50 GB
basis=gth-dzvp-molopt-sr
suffix=final
reservation=changroup_standingres
constraint=icelake
time=2400

mkdir -p log_dft

function submit(){
	system=$1
	nk=$2
	nthreads=$3
	mem=$4
	logfile=log_dft/slurm_dft_${system}_${basis}_${suffix}_${nk}x${nk}x1_${nthreads}threads
	echo sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=$mem -t $time -J dft_${system}_${nk}x${nk}x1 -o $logfile ./run_dft_gdf.sh $system $nk $nk 1 $basis $suffix
	sbatch --reservation=$reservation --constraint=$constraint -n 1 -c $nthreads --mem=$mem -t $time -J dft_${system}_${nk}x${nk}x1 -o $logfile ./run_dft_gdf.sh $system $nk $nk 1 $basis $suffix
}

# GDF size estimate (GB): blackP2D/MoSe2D 0.011*nk^4, CdSe2D 0.186*nk^4
# for system in blackP2D MoSe2D; do
# 	for nk in 2 3 4; do
# 		submit $system $nk 16 64GB
# 	done
# 	for nk in 5 6 7 8; do
# 		submit $system $nk 32 250GB
# 	done
# done

# submit CdSe2D 2 16 64GB
# for nk in 3 4; do
# 	submit CdSe2D $nk 32 250GB
# done

# # rerun MoSe2D with gdf j2c_eig_always (settings.pydata); old files moved to old_cd/
# for system in MoSe2D; do
# 	for nk in 2 3 4; do
# 		submit $system $nk 16 64GB
# 	done
# 	for nk in 5 6 7 8; do
# 		submit $system $nk 32 250GB
# 	done
# done

# rerun blackP2D and CdSe2D with gdf j2c_eig_always (settings.pydata); old files moved to old_cd/
for nk in 2 3 4; do
	submit blackP2D $nk 16 64GB
done
for nk in 5 6 7 8; do
	submit blackP2D $nk 32 250GB
done
submit CdSe2D 2 16 64GB
for nk in 3 4; do
	submit CdSe2D $nk 32 250GB
done
