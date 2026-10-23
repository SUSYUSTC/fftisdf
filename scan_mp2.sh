#!/bin/bash
cuda=3
run=1
symm=0
real=0

[[ $run =~ ^[01]$ ]] || { echo "run must be 0 or 1"; exit 1; }
[[ $symm =~ ^[012]$ ]] || { echo "symm must be 0, 1, or 2"; exit 1; }
[[ $real =~ ^[012]$ ]] || { echo "real must be 0, 1, or 2"; exit 1; }

mkdir -p mp2_energy

for chk in data_*_final/ISDFov_opt_*.chk
do
	[ -f "$chk" ] || continue

	data_dir=${chk%%/*}
	setting=${data_dir#data_}
	setting=${setting%_final}
	system=${setting%%_gth-*}
	basis=gth-${setting#*_gth-}

	name=$(basename "$chk" .chk)
	body=${name#ISDFov_opt_bareGDF_}
	extra=()
	file_symm=0
	file_real=0
	if [[ $body == symm_* ]]; then
		file_symm=1
		extra+=(--symm)
		body=${body#symm_}
	fi
	if [[ $body == real_* ]]; then
		file_real=1
		extra+=(--real)
		body=${body#real_}
	fi
	[ "$symm" != 2 ] && [ "$symm" != "$file_symm" ] && continue
	[ "$real" != 2 ] && [ "$real" != "$file_real" ] && continue
	klabel=${body%%_*}
	pattern=${body#${klabel}_}
	IFS=x read -r kx ky kz <<< "$klabel"

	output=mp2_energy/${setting}_${name}.txt
	[ -s "$output" ] && continue

	echo python -u custom_mp2_thc.py $system $kx $ky $kz $basis -suffix final -cuda $cuda -ov_opt $pattern --convert --verbose ${extra[@]}
	if [ "$run" = 1 ]; then
		rm -f ${output}.tmp
		python -u custom_mp2_thc.py $system $kx $ky $kz $basis -suffix final -cuda $cuda -ov_opt $pattern --convert --verbose ${extra[@]} \
			| awk -v output=${output}.tmp '{print; fflush()} /custom THC MP2 energy/ {print $NF > output; close(output)}'
		[ -s ${output}.tmp ] && mv ${output}.tmp $output
	fi
done

for cref_file in cref_system_ov cref_system_ov_symm
do
	extra=()
	symm_tag=
	file_symm=0
	if [ "$cref_file" = cref_system_ov_symm ]; then
		file_symm=1
		extra+=(--symm)
		symm_tag=_symm
	fi
	[ "$symm" != 2 ] && [ "$symm" != "$file_symm" ] && continue

	while read -r system cref
	do
		for data_dir in data_${system}_gth-*_final
		do
			[ -d "$data_dir" ] || continue
			setting=${data_dir#data_}
			setting=${setting%_final}
			basis=gth-${setting#*_gth-}

			for chk in ${data_dir}/ISDFov_bareGDF${symm_tag}_*x*x*_c${cref}.chk
			do
				[ -f "$chk" ] || continue
				name=$(basename "$chk" .chk)
				klabel=${name#ISDFov_bareGDF${symm_tag}_}
				klabel=${klabel%_c${cref}}
				IFS=x read -r kx ky kz <<< "$klabel"

				output=mp2_energy/${setting}_${name}.txt
				[ -s "$output" ] && continue

				echo python -u custom_mp2_thc.py $system $kx $ky $kz $basis -suffix final -cuda $cuda -ov_ref $cref --convert --verbose ${extra[@]}
				if [ "$run" = 1 ]; then
					rm -f ${output}.tmp
					python -u custom_mp2_thc.py $system $kx $ky $kz $basis -suffix final -cuda $cuda -ov_ref $cref --convert --verbose ${extra[@]} \
						| awk -v output=${output}.tmp '{print; fflush()} /custom THC MP2 energy/ {print $NF > output; close(output)}'
					[ -s ${output}.tmp ] && mv ${output}.tmp $output
				fi
			done
		done
	done < $cref_file
done
