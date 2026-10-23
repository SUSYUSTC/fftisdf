#!/bin/bash
set -e
echo ./run_mp2_convert.sh cuda system basis c suffix
cuda=$1
system=$2
basis=$3
c=$4
suffix=$5
shift 5
for nk in $@
do
	cref_ov=$(grep -e $system cref_system_ov | awkn 2)
	echo nk = $nk
	echo

	#python custom_mp2_df.py $system $nk $nk $nk $basis -suffix final -cuda $cuda | grep -e "MP2 energy"
	#echo
	
	echo python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_ref ${cref_ov} --convert 
	python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_ref ${cref_ov} --convert  | grep -e "MP2 energy"
	echo
	
	echo python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_opt c${c}_cref${cref_ov}_${suffix} --convert 
	python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_opt c${c}_cref${cref_ov}_${suffix} --convert  | grep -e "MP2 energy"
	echo
done
