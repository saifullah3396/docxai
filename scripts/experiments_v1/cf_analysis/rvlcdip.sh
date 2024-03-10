#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
CONFIGS=(
    # "rvlcdip_alexnet $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/alexnet"
    # "rvlcdip_resnet50 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/resnet50"
    # "rvlcdip_inception_v3 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/inception_v3"
    # "rvlcdip_vgg16 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/vgg16"
    # "rvlcdip_efficientnet-b4 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/efficientnet-b4"
    # "rvlcdip_convnext $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/convnext"
    # "rvlcdip_densenet121 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/densenet121"
    # "rvlcdip_mobilenetv3_large_100 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/mobilenetv3_large_100"
    "rvlcdip_res2net50_26w_8s $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/res2net50_26w_8s"
    "rvlcdip_dm_nfnet_f1 $ANALYSIS_SCRIPT +analysis=cf_analysis/rvlcdip/dm_nfnet_f1"
)

# set -- ${CONFIGS[9]}
# EXP_NAME=$1
# SCRIPT="${@:2}"

# echo "=================================================="
# echo "CID: $CID"
# echo "Experiment: $EXP_NAME"
# echo "Task: $SCRIPT"
# echo "=================================================="
# $SCRIPT