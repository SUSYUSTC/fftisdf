#!/bin/bash
# bulk zinc blende CdSe, full workflow (DFT/GDF -> vxc -> GW -> screening -> BSE); GDF ~4.6e-3*nk^6 GB
basis=gth-dzvp-molopt-sr
suffix=final

for nk in 2 3 4 5; do
    echo ./submit_workflow.sh CdSezb $nk $basis --suffix $suffix
    ./submit_workflow.sh CdSezb $nk $basis --suffix $suffix
done
