#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
CONFIGS=(
    # "rvlcdip_convnext_0 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=0 end_idx=4000"
    # "rvlcdip_convnext_1 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=4000 end_idx=8000"
    # "rvlcdip_convnext_2 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=8000 end_idx=12000"
    "rvlcdip_convnext_3 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=12000 end_idx=16000"
    # "rvlcdip_convnext_4 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=16000 end_idx=20000"
    # "rvlcdip_convnext_5 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=20000 end_idx=24000"
    # "rvlcdip_convnext_6 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=24000 end_idx=28000"
    # "rvlcdip_convnext_7 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=28000 end_idx=32000"
    # "rvlcdip_convnext_8 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=32000 end_idx=36000"
    # "rvlcdip_convnext_9 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext_no_aug start_idx=36000 end_idx=40000"
)
