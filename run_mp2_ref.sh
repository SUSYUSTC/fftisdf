#!/bin/bash
set -e
echo ./run_mp2_ref.sh cuda system basis
cuda=$1
system=$2
basis=$3
shift 3
for nk in $@
do
	cref_ov=$(grep -e $system cref_system_ov | awkn 2)
	echo nk = $nk
	echo

	echo python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_ref ${cref_ov} --convert 
	python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_ref ${cref_ov} --convert  | grep -e "MP2 energy"
	echo
done
