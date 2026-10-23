#!/bin/bash
echo ./run_mp2.sh cuda system basis c norm power
cuda=$1
system=$2
basis=$3
c=$4
norm=$5
p=$6
shift 6
for nk in $@
do
	cref_ov=$(grep -e $system cref_system_ov_symm | awkn 2)
	echo nk = $nk
	echo
	#python custom_mp2_df.py $system $nk $nk $nk $basis -suffix final -cuda $cuda | grep -e "MP2 energy"
	#echo
	echo python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda --symm --real -ov_ref ${cref_ov}
	python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda --symm --real -ov_ref ${cref_ov} | grep -e "MP2 energy"
	echo
	echo python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda --symm --real -ov_opt c${c}_cref${cref_ov}_norm${norm}p${p}
	python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda --symm --real -ov_opt c${c}_cref${cref_ov}_norm${norm}p${p} | grep -e "MP2 energy"
	#python custom_mp2_thc.py $system $nk $nk $nk $basis -suffix final -cuda $cuda -ov_opt c8_cref${cref_ov}_norm0.02p2 | grep -e "MP2 energy"
	echo
done
