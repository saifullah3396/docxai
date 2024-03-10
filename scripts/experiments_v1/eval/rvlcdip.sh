# #!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/alexnet
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/resnet50
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/inception_v3
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/vgg16
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/efficientnet-b4
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/convnext
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/densenet121
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/mobilenetv3_large_100
# $ANALYSIS_SCRIPT +analysis=eval/rvlcdip/res2net50_26w_8s
$ANALYSIS_SCRIPT +analysis=eval/rvlcdip/dm_nfnet_f1