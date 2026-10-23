#!/bin/bash
cuda=$1
system=$2
basis=$3
shift 3
cref_ov=$(grep -e $system cref_system_ov | awkn 2)
cref_full=$(grep -e $system cref_system_full | awkn 2)
for nk in $@
do
	echo nk = $nk
	echo
	#python bse_df.py $system $nk $nk $nk $basis -suffix final -cuda $cuda --nofc | grep -e "binding Q=0"
	#echo
	python bse_thc.py $system $nk $nk $nk $basis -suffix final -ov_ref ${cref_ov} -full_ref ${cref_full} -cuda $cuda | grep -e "binding Q=0"
	echo
	python bse_thc.py $system $nk $nk $nk $basis -suffix final -ov_opt c5_cref${cref_ov}_norm0.1p2 -full_ref ${cref_full} -cuda $cuda | grep -e "binding Q=0"
	echo
done
