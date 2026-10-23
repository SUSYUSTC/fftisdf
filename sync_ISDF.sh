#!/bin/bash
#opts="--dry-run"
ops=""
rsync $opts --itemize-changes -rPa --include='data_*_final/' --include='data_*_final/ISDF*.chk' --exclude='*' central:/resnick/groups/changroup/members/jsun3/xprize/THC_general/ /home/susyustc/changroup/THC_general/
#rsync $opts -rPa --include='data_*_final/' --include='data_*_final/DFT*symm*.pkl' --exclude='*' central:/resnick/groups/changroup/members/jsun3/xprize/THC_general/ /home/susyustc/changroup/THC_general/
#rsync $opts -rPa --include='data_*_final/' --include='data_*_final/ISDF*opt*.chk' --exclude='*' /home/susyustc/changroup/THC_general/ central:/resnick/groups/changroup/members/jsun3/xprize/THC_general/
