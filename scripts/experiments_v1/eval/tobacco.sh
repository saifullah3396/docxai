# #!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/../../analyze.sh"
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/alexnet args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/resnet50 args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/inception_v3 args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/vgg16 args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/efficientnet-b4 args/data_args=tobacco3482_clean_384
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/convnext args/data_args=tobacco3482_clean_384
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/densenet121 args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/mobilenetv3_large_100 args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/res2net50_26w_8s args/data_args=tobacco3482_clean_224
$ANALYSIS_SCRIPT +analysis=eval/tobacco3482/dm_nfnet_f1 args/data_args=tobacco3482_clean_224