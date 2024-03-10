#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=densenet121 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/densenet121.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=dpn107 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/dpn107.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=mobilenetv3_large_100 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/mobilenetv3_large_100.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=res2net101_26w_4s args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/res2net101_26w_4s.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=res2net50_26w_8s args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/res2net50_26w_8s.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=res2netres2next50 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/res2netres2next50.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet18 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet18.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet34 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet34.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet50 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet50.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet101 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet101.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet152 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet152.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet50_32x4d args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet50_32x4d.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnet_bitv2_101x1_bitm args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnet_bitv2_101x1_bitm.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=resnext101_32x8d args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/resnext101_32x8d.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=senet154 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/senet154.pt\]\]
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=squeezenet1_0 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/squeezenet1_0.pt\]\]

# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=densenet121 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/densenet121.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=mobilenetv3_large_100 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/mobilenetv3_large_100.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=res2net50_26w_8s args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/res2net50_26w_8s.pt\]\] #
# ./scripts/analyze.sh +evaluation=evaluate_model args.model_args.config.model_type=senet154 args.analyzer_args.model_checkpoints=\[\[\"baseline\",/netscratch/saifullah/resources/pretrained_models/rvlcdip/senet154.pt\]\]

./scripts/analyze.sh +evaluation=evaluate_model_384 args.model_args.config.model_type=convnext_base args.analyzer_args.model_checkpoints=\[\[\"baseline\",https://cloud.dfki.de/owncloud/index.php/s/tPwa2dXaywB6dMy/download/docshap_convnext_rvlcdip.ckpt\]\]
