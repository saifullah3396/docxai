#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
CONFIGS=(
    "tobacco3482_alexnet $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/alexnet"
    "tobacco3482_resnet50 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/resnet50"
    "tobacco3482_inception_v3 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/inception_v3"
    "tobacco3482_vgg16 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/vgg16"
    "tobacco3482_efficientnet-b4 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/efficientnet-b4"
    "tobacco3482_convnext $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/convnext"
    "tobacco3482_densenet121 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/densenet121"
    "tobacco3482_mobilenetv3_large_100 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/mobilenetv3_large_100"
    "tobacco3482_res2net50_26w_8s $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/res2net50_26w_8s"
    "tobacco3482_dm_nfnet_f1 $ANALYSIS_SCRIPT +analysis_v1=base_analysis/tobacco3482/dm_nfnet_f1"
)