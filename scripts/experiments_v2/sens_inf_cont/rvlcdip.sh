#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
CONFIGS=(
    # "rvlcdip_alexnet $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/alexnet"
    # "rvlcdip_resnet50 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/resnet50"
    # "rvlcdip_inception_v3 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/inception_v3"
    # "rvlcdip_vgg16 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/vgg16"
    # "rvlcdip_efficientnet-b4 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/efficientnet-b4"
    "rvlcdip_efficientnet-b41 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/efficientnet-b4_multi start_idx=0 end_idx=1000"
    "rvlcdip_efficientnet-b42 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/efficientnet-b4_multi start_idx=1000 end_idx=2000"
    "rvlcdip_efficientnet-b43 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/efficientnet-b4_multi start_idx=2000 end_idx=3000"
    "rvlcdip_efficientnet-b44 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/efficientnet-b4_multi start_idx=3000 end_idx=4000"
    # "rvlcdip_convnext $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/convnext"
    "rvlcdip_convnext1 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/convnext_multi start_idx=0 end_idx=1000"
    "rvlcdip_convnext2 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/convnext_multi start_idx=1000 end_idx=2000"
    "rvlcdip_convnext3 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/convnext_multi start_idx=2000 end_idx=3000"
    "rvlcdip_convnext4 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/convnext_multi start_idx=3000 end_idx=4000"
    # "rvlcdip_densenet121 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/densenet121"
    # "rvlcdip_mobilenetv3_large_100 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/mobilenetv3_large_100"
    # "rvlcdip_res2net50_26w_8s $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/res2net50_26w_8s"
    # "rvlcdip_dm_nfnet_f1 $ANALYSIS_SCRIPT +analysis_v2=sens_inf_cont/rvlcdip/dm_nfnet_f1"
)